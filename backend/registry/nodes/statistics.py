"""
backend/registry/nodes/statistics.py

Statistical analysis node types (Tier 2).

These nodes compute statistical tests on EEG epochs or on metrics dicts and
return results as metrics dicts. They are designed to be placed at the end of
an analysis pipeline to assess the significance of observed effects.

All nodes return a `metrics` dict compatible with the batch processor CSV
export and the post-MVP report generator.

execute_fn contract:
  - Never mutate input.
  - Use verbose=False on all MNE calls.
  - Return a plain Python dict (metrics handle type).

Dependencies:
  - mne.stats  (included with MNE — no extra install)
  - scipy.stats (included with MNE's dependency tree)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import mne
import mne.stats

from backend.registry.nodes._channel_utils import resolve_channel

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Cluster Permutation Test
# ---------------------------------------------------------------------------

def _execute_cluster_permutation_test(
    epochs: mne.Epochs, params: dict
) -> dict[str, Any]:
    """
    Runs a 1-sample spatiotemporal cluster permutation test on epoch data.

    The test asks: is the EEG amplitude at any channel × time point
    significantly different from zero (or from the baseline mean)?

    Method:
      1. Extract epoch data in the requested time window.
      2. For each channel, compute the within-window mean amplitude per epoch.
         Shape: (n_epochs, n_channels).
      3. Run mne.stats.permutation_cluster_1samp_test:
         - H0: population mean = 0 for each channel.
         - Cluster correction: channels are linked by adjacency to form clusters.
         - n_permutations label-shuffles the data to build the null distribution.
      4. Return cluster statistics as a metrics dict.

    This test is the standard approach for ERP and resting-state EEG because it
    controls the family-wise error rate across many simultaneous channel-level tests
    without requiring Gaussian assumptions.

    Parameters note: 'tmin_ms' and 'tmax_ms' define the analysis window; the test
    is run on the mean amplitude within this window per channel per epoch.
    """
    tmin_s = float(params["tmin_ms"]) / 1000.0
    tmax_s = float(params["tmax_ms"]) / 1000.0
    n_permutations = int(params["n_permutations"])
    alpha = float(params["alpha"])

    # Crop to requested window and compute mean per epoch per channel
    epochs_cropped = epochs.copy().crop(tmin=tmin_s, tmax=tmax_s, include_tmax=True)
    # Shape: (n_epochs, n_channels, n_times_window)
    data = epochs_cropped.get_data()
    # Reduce to (n_epochs, n_channels) by averaging over time
    data_mean = data.mean(axis=-1)  # (n_epochs, n_channels)

    # Run 1-sample cluster test across channels
    # Input to permutation_cluster_1samp_test: (n_observations, n_tests)
    t_obs, clusters, cluster_pv, h0 = mne.stats.permutation_cluster_1samp_test(
        data_mean,
        n_permutations=n_permutations,
        threshold=None,  # automatic t-threshold via t-dist (recommended)
        tail=0,          # two-tailed
        n_jobs=1,
        verbose=False,
        seed=42,
    )

    n_sig = int(np.sum(cluster_pv < alpha))
    cluster_pv_list = [round(float(p), 6) for p in cluster_pv]

    return {
        "n_clusters": len(clusters),
        "n_significant_clusters": n_sig,
        "cluster_p_values": cluster_pv_list,
        "alpha": alpha,
        "n_permutations": n_permutations,
        "tmin_ms": float(params["tmin_ms"]),
        "tmax_ms": float(params["tmax_ms"]),
        "test_type": "permutation_cluster_1samp",
    }


CLUSTER_PERMUTATION_TEST = NodeDescriptor(
    node_type="cluster_permutation_test",
    display_name="Cluster Permutation Test",
    category="Analysis",
    description=(
        "Runs a spatiotemporal cluster permutation test on EEG epoch data. "
        "Tests whether the mean amplitude across channels in a specified time window "
        "is significantly different from zero. Controls the family-wise error rate "
        "by forming clusters of adjacent channels and comparing cluster-level statistics "
        "against a permuted null distribution. The standard statistical test for ERP "
        "and resting-state EEG analyses in cognitive neuroscience (Maris & Oostenveld 2007). "
        "Output is a metrics dict with cluster counts and p-values."
    ),
    tags=[
        "statistics", "cluster", "permutation", "test", "erp",
        "significance", "fwer", "maris", "oostenveld", "analysis",
    ],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Cluster Stats"),
    ],
    parameters=[
        ParameterSchema(
            name="tmin_ms",
            label="Window Start",
            type="float",
            default=0.0,
            min=-500.0,
            max=2000.0,
            step=10.0,
            unit="ms",
            description=(
                "Start of the analysis window in milliseconds. "
                "For ERP: typically 0 ms (stimulus onset) to avoid baseline in the test window."
            ),
        ),
        ParameterSchema(
            name="tmax_ms",
            label="Window End",
            type="float",
            default=500.0,
            min=-500.0,
            max=2000.0,
            step=10.0,
            unit="ms",
            description=(
                "End of the analysis window in milliseconds. "
                "Include the full range of the component of interest (e.g., 0–600 ms for P300)."
            ),
        ),
        ParameterSchema(
            name="n_permutations",
            label="Permutations",
            type="int",
            default=1000,
            min=100,
            max=10000,
            step=100,
            description=(
                "Number of permutations to build the null distribution. "
                "1000 is standard for exploratory analyses. "
                "Use 5000–10 000 for publication-quality results."
            ),
        ),
        ParameterSchema(
            name="alpha",
            label="Alpha Level",
            type="float",
            default=0.05,
            min=0.001,
            max=0.1,
            step=0.005,
            description=(
                "Significance threshold for cluster p-values. "
                "Clusters with p < alpha are reported as significant. "
                "Standard: 0.05. Use 0.01 for strict control."
            ),
        ),
    ],
    execute_fn=_execute_cluster_permutation_test,
    code_template=lambda p: f't_obs, clusters, cluster_pv, h0 = mne.stats.permutation_cluster_1samp_test(\n    data_mean, n_permutations={p.get("n_permutations", 1000)}, tail=0, verbose=False, seed=42\n)',
    methods_template=lambda p: f'A cluster-based permutation test ({p.get("n_permutations", 1000)} permutations, alpha = {p.get("alpha", 0.05)}) was used to assess statistical significance while controlling for multiple comparisons (Maris & Oostenveld, 2007; MNE-Python).',
    docs_url="https://mne.tools/stable/generated/mne.stats.permutation_cluster_1samp_test.html",
)


# ---------------------------------------------------------------------------
# Compute T-Test
# ---------------------------------------------------------------------------

def _execute_compute_t_test(epochs: mne.Epochs, params: dict) -> dict[str, Any]:
    """
    Computes a 1-sample t-test on the mean epoch amplitude at a specified channel.

    For each trial, extracts the mean amplitude in [tmin_ms, tmax_ms] at the
    given channel.  Then runs a 1-sample t-test against popmean (default 0).

    Use case:
      - Single-condition analysis: "Is the P300 amplitude > 0 at Pz?"
      - Baseline comparison: "Is resting-state beta power different from 0?"

    For a paired t-test comparing two conditions (e.g., target vs standard),
    compute the difference wave first using Compute Difference Wave, epoch it,
    and feed the single-condition result here.

    Returns t statistic, p-value, and degrees of freedom in a metrics dict.
    """
    from scipy import stats as scipy_stats  # scipy is a standard MNE dependency

    channel = str(params["channel"]).strip()
    tmin_s = float(params["tmin_ms"]) / 1000.0
    tmax_s = float(params["tmax_ms"]) / 1000.0
    popmean = float(params["popmean"])

    channel = resolve_channel(channel, epochs.ch_names)

    ch_idx = epochs.ch_names.index(channel)

    # Crop to window and take mean per epoch
    epochs_cropped = epochs.copy().crop(tmin=tmin_s, tmax=tmax_s, include_tmax=True)
    data = epochs_cropped.get_data()          # (n_epochs, n_ch, n_times)
    amplitudes = data[:, ch_idx, :].mean(axis=-1) * 1e6  # µV, shape (n_epochs,)

    t_stat, p_val = scipy_stats.ttest_1samp(amplitudes, popmean=popmean)
    df = len(amplitudes) - 1
    ci_low, ci_high = scipy_stats.t.interval(
        0.95, df=df,
        loc=float(np.mean(amplitudes)),
        scale=float(scipy_stats.sem(amplitudes)),
    )

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "degrees_of_freedom": df,
        "n_epochs": len(amplitudes),
        "mean_amplitude_uv": round(float(np.mean(amplitudes)), 4),
        "std_amplitude_uv": round(float(np.std(amplitudes, ddof=1)), 4),
        "ci_95_low_uv": round(float(ci_low), 4),
        "ci_95_high_uv": round(float(ci_high), 4),
        "channel": channel,
        "tmin_ms": float(params["tmin_ms"]),
        "tmax_ms": float(params["tmax_ms"]),
        "popmean_uv": popmean,
        "test_type": "ttest_1samp",
    }


COMPUTE_T_TEST = NodeDescriptor(
    node_type="compute_t_test",
    display_name="Compute T-Test",
    category="Analysis",
    description=(
        "Computes a one-sample t-test on the mean EEG amplitude at a specified channel "
        "within a time window. Tests whether the mean amplitude across trials is "
        "significantly different from zero (or a specified comparison value). "
        "Returns t statistic, p-value, degrees of freedom, mean amplitude (µV), "
        "and 95% confidence interval. "
        "For condition comparisons, first compute a difference wave and then test it here. "
        "Output is a metrics dict compatible with Apply FDR Correction downstream."
    ),
    tags=["t-test", "statistics", "significance", "amplitude", "p-value", "epochs", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="T-Test Results"),
    ],
    parameters=[
        ParameterSchema(
            name="channel",
            label="Channel",
            type="string",
            default="Cz",
            description=(
                "Electrode name at which to extract the amplitude for the t-test. "
                "P300: Pz. N200/N400: Cz or FCz. Use the electrode where the component "
                "of interest is maximal based on the topographic map."
            ),
            channel_hint="single",
        ),
        ParameterSchema(
            name="tmin_ms",
            label="Window Start",
            type="float",
            default=250.0,
            min=-500.0,
            max=2000.0,
            step=10.0,
            unit="ms",
            description="Start of the amplitude averaging window in milliseconds.",
        ),
        ParameterSchema(
            name="tmax_ms",
            label="Window End",
            type="float",
            default=500.0,
            min=-500.0,
            max=2000.0,
            step=10.0,
            unit="ms",
            description="End of the amplitude averaging window in milliseconds.",
        ),
        ParameterSchema(
            name="popmean",
            label="Comparison Value",
            type="float",
            default=0.0,
            min=-1000.0,
            max=1000.0,
            step=0.1,
            unit="µV",
            description=(
                "Value to test against (population mean under H0). "
                "0.0 (default): tests whether the amplitude is different from zero. "
                "Set to a specific µV value to test against a known baseline."
            ),
        ),
    ],
    execute_fn=_execute_compute_t_test,
    code_template=lambda p: f'from scipy.stats import ttest_1samp\nt_stat, p_val = ttest_1samp(amplitudes, popmean={p.get("popmean", 0.0)})',
    methods_template=lambda p: f'A one-sample t-test was performed on mean amplitude at electrode {p.get("channel", "Cz")} in the {p.get("tmin_ms", 250)}–{p.get("tmax_ms", 500)} ms window (comparison value = {p.get("popmean", 0.0)} µV).',
    docs_url="https://mne.tools/stable/generated/mne.stats.permutation_cluster_1samp_test.html",
)


# ---------------------------------------------------------------------------
# Apply FDR Correction
# ---------------------------------------------------------------------------

def _execute_apply_fdr_correction(metrics: dict, params: dict) -> dict[str, Any]:
    """
    Applies Benjamini–Hochberg False Discovery Rate (FDR) correction to p-values
    in a metrics dict.

    This node looks for p-value entries in the input metrics dict:
      1. If the dict contains a "p_values" key (list of floats from multiple tests),
         it corrects all of them simultaneously.
      2. If the dict contains a "p_value" key (single float from compute_t_test),
         it wraps it in a list, corrects, and reports the result.

    FDR controls the expected proportion of false positives among rejected
    hypotheses — less stringent than Bonferroni (FWER) but appropriate when
    testing multiple channels, frequency bands, or brain regions simultaneously.

    Uses mne.stats.fdr_correction which implements the BH procedure.

    Returns the original metrics dict augmented with:
      - "reject_h0": list[bool] — True means the null hypothesis is rejected
      - "corrected_p_values": list[float] — BH-corrected p-values
      - "fdr_alpha": float — the alpha threshold used
      - "n_rejected": int — number of hypotheses rejected
    """
    alpha = float(params["alpha"])

    # Extract p-values from the metrics dict
    if "p_values" in metrics:
        pvals = [float(p) for p in metrics["p_values"]]
        multi_input = True
    elif "p_value" in metrics:
        pvals = [float(metrics["p_value"])]
        multi_input = False
    elif "cluster_p_values" in metrics:
        pvals = [float(p) for p in metrics["cluster_p_values"]]
        multi_input = True
    else:
        raise ValueError(
            "Apply FDR Correction requires the input metrics dict to contain "
            "'p_value', 'p_values', or 'cluster_p_values'. "
            "Connect the output of Compute T-Test or Cluster Permutation Test."
        )

    if len(pvals) == 0:
        raise ValueError("p-values list is empty — cannot apply FDR correction.")

    reject, pvals_corrected = mne.stats.fdr_correction(
        np.array(pvals), alpha=alpha, method="indep"
    )

    result = dict(metrics)  # copy — never mutate input
    result["reject_h0"] = [bool(r) for r in reject]
    result["corrected_p_values"] = [round(float(p), 8) for p in pvals_corrected]
    result["fdr_alpha"] = alpha
    result["n_tested"] = len(pvals)
    result["n_rejected"] = int(np.sum(reject))
    result["fdr_method"] = "benjamini-hochberg"

    # For single p-value input, also expose top-level keys
    if not multi_input:
        result["corrected_p_value"] = result["corrected_p_values"][0]
        result["reject_h0_single"] = result["reject_h0"][0]

    return result


APPLY_FDR_CORRECTION = NodeDescriptor(
    node_type="apply_fdr_correction",
    display_name="Apply FDR Correction",
    category="Analysis",
    description=(
        "Applies Benjamini–Hochberg False Discovery Rate (FDR) correction to p-values "
        "in a metrics dict. Reduces the expected proportion of false positives when "
        "testing multiple hypotheses simultaneously (multiple channels, frequency bands, "
        "or brain regions). Less conservative than Bonferroni correction — recommended "
        "when you want to detect true effects while accepting a small FDR. "
        "Accepts output from Compute T-Test (single p_value) or Cluster Permutation Test "
        "(cluster_p_values). Adds corrected p-values and rejection flags to the metrics dict."
    ),
    tags=["fdr", "correction", "multiple-comparison", "statistics", "benjamini", "hochberg", "analysis"],
    inputs=[
        HandleSchema(id="metrics_in", type="metrics", label="Metrics (with p-values)"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="FDR-Corrected Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="alpha",
            label="FDR Alpha",
            type="float",
            default=0.05,
            min=0.001,
            max=0.5,
            step=0.005,
            description=(
                "False Discovery Rate threshold. The expected proportion of false positives "
                "among all rejected hypotheses is controlled at this level. "
                "Standard: 0.05 (5% FDR). Use 0.10 for exploratory analyses "
                "where power is more important than strict control."
            ),
        ),
    ],
    execute_fn=_execute_apply_fdr_correction,
    code_template=lambda p: f'reject, pvals_corrected = mne.stats.fdr_correction(p_values, alpha={p.get("alpha", 0.05)}, method="indep")',
    methods_template=lambda p: f'False Discovery Rate correction was applied using the Benjamini-Hochberg procedure at alpha = {p.get("alpha", 0.05)} (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.stats.fdr_correction.html",
)


# ---------------------------------------------------------------------------
# Compute Noise Floor
# ---------------------------------------------------------------------------

def _execute_compute_noise_floor(evoked: mne.Evoked, params: dict) -> dict[str, Any]:
    """
    Estimates the pre-stimulus noise floor from an Evoked (averaged ERP) object.

    The noise floor is quantified as the standard deviation of EEG amplitude
    across channels during a quiet baseline period (before stimulus onset).
    This provides a data-driven estimate of how much background noise is present
    in the recording, which determines the minimum detectable ERP amplitude.

    Computes:
      - Global noise floor (RMS across per-channel stds) in µV
      - Per-channel noise floors: identifies the noisiest and cleanest channels
      - Signal-to-noise ratio (SNR) in dB: peak post-stimulus amplitude / noise floor
        A positive dB SNR means the signal exceeds the noise floor.

    Use this node to:
      - Flag poor-quality recordings before reporting ERP amplitudes
      - Compare noise levels across sessions or conditions
      - Sanity-check that preprocessing (ICA, reference) reduced noise
    """
    tmin_s = float(params["tmin_ms"]) / 1000.0
    tmax_s = float(params["tmax_ms"]) / 1000.0

    evoked_copy = evoked.copy()
    times = evoked_copy.times

    # Locate baseline time points
    mask = (times >= tmin_s) & (times <= tmax_s)
    if not np.any(mask):
        raise ValueError(
            f"No time points found in baseline window [{tmin_s * 1000:.0f}, "
            f"{tmax_s * 1000:.0f}] ms. "
            f"Evoked time range: [{times[0] * 1000:.0f}, {times[-1] * 1000:.0f}] ms."
        )

    # Baseline data: (n_channels, n_times_in_window)
    baseline_data = evoked_copy.data[:, mask]

    # Per-channel noise floor in µV (std across baseline time points)
    channel_noise_uv = np.std(baseline_data, axis=-1, ddof=1) * 1e6

    global_noise = float(np.sqrt(np.mean(channel_noise_uv ** 2)))  # RMS
    max_noise = float(np.max(channel_noise_uv))
    min_noise = float(np.min(channel_noise_uv))
    mean_noise = float(np.mean(channel_noise_uv))

    # Peak post-stimulus amplitude and SNR
    post_stim_mask = times > 0.0
    if np.any(post_stim_mask):
        post_stim_data = evoked_copy.data[:, post_stim_mask]
        peak_amplitude_uv = float(np.max(np.abs(post_stim_data))) * 1e6
        if global_noise > 0:
            snr_db = round(20.0 * float(np.log10(peak_amplitude_uv / global_noise)), 2)
        else:
            snr_db = None
    else:
        peak_amplitude_uv = None
        snr_db = None

    worst_ch_idx = int(np.argmax(channel_noise_uv))
    best_ch_idx = int(np.argmin(channel_noise_uv))

    return {
        "noise_floor_global_uv": round(global_noise, 4),
        "noise_floor_max_uv": round(max_noise, 4),
        "noise_floor_min_uv": round(min_noise, 4),
        "noise_floor_mean_uv": round(mean_noise, 4),
        "peak_amplitude_uv": round(peak_amplitude_uv, 4) if peak_amplitude_uv is not None else None,
        "snr_db": snr_db,
        "worst_channel": evoked_copy.ch_names[worst_ch_idx],
        "best_channel": evoked_copy.ch_names[best_ch_idx],
        "n_channels": len(evoked_copy.ch_names),
        "baseline_tmin_ms": float(params["tmin_ms"]),
        "baseline_tmax_ms": float(params["tmax_ms"]),
        "measurement": "noise_floor",
    }


COMPUTE_NOISE_FLOOR = NodeDescriptor(
    node_type="compute_noise_floor",
    display_name="Compute Noise Floor",
    category="Analysis",
    description=(
        "Estimates the pre-stimulus baseline noise floor from an averaged ERP (Evoked) object. "
        "The noise floor is computed as the RMS of per-channel standard deviations during the "
        "specified baseline window — the quieter the baseline, the higher the sensitivity to "
        "true ERP components. "
        "Also computes Signal-to-Noise Ratio (SNR) in dB: the ratio of peak post-stimulus "
        "amplitude to the baseline noise floor. A positive SNR confirms the ERP is visible above "
        "noise; typical good-quality ERP recordings show SNR of 10–20 dB. "
        "Output is a metrics dict with global noise floor, per-channel extremes, SNR, and "
        "the noisiest and cleanest channel names."
    ),
    tags=[
        "noise", "floor", "snr", "signal-to-noise", "baseline", "quality",
        "erp", "evoked", "preprocessing", "qc", "analysis",
    ],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked (ERP)"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Noise Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="tmin_ms",
            label="Baseline Start",
            type="float",
            default=-200.0,
            min=-2000.0,
            max=0.0,
            step=10.0,
            unit="ms",
            description=(
                "Start of the pre-stimulus baseline window in milliseconds. "
                "This must be within the Evoked time range. "
                "Typical ERP baselines: -200 to 0 ms."
            ),
        ),
        ParameterSchema(
            name="tmax_ms",
            label="Baseline End",
            type="float",
            default=0.0,
            min=-2000.0,
            max=0.0,
            step=10.0,
            unit="ms",
            description=(
                "End of the pre-stimulus baseline window in milliseconds. "
                "Must be ≤ 0 ms (pre-stimulus). "
                "Noise is computed across all time points in [tmin_ms, tmax_ms]."
            ),
        ),
    ],
    execute_fn=_execute_compute_noise_floor,
    code_template=lambda p: f'# Noise floor: std of baseline [{p.get("tmin_ms", -200)}, {p.get("tmax_ms", 0)}] ms\nbaseline_data = evoked.data[:, baseline_mask]\nchannel_noise_uv = np.std(baseline_data, axis=-1) * 1e6\nglobal_noise = np.sqrt(np.mean(channel_noise_uv ** 2))',
    methods_template=lambda p: f'The pre-stimulus noise floor was estimated as the RMS of per-channel standard deviations during the [{p.get("tmin_ms", -200)}, {p.get("tmax_ms", 0)}] ms baseline window.',
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html",
)
