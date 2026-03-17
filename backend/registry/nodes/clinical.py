"""
backend/registry/nodes/clinical.py

Clinical + qEEG node types: specialized analysis for clinical neurologists,
qEEG/neurofeedback practitioners, and epileptologists.

All nodes here output `metrics` (dict[str, Any]) — a named collection of
scalar clinical values. These can be collected by the POST /pipeline/report
endpoint to generate a PDF report.

Added in Tier 3.
"""

from __future__ import annotations

import numpy as np
import mne

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)
from backend.registry.nodes._channel_utils import resolve_channel


# ---------------------------------------------------------------------------
# Compute Alpha Peak (Individual Alpha Frequency)
# ---------------------------------------------------------------------------

def _execute_compute_alpha_peak(
    spectrum: "mne.time_frequency.Spectrum",
    params: dict,
) -> dict:
    """
    Computes the Individual Alpha Frequency (IAF) from a PSD spectrum.

    'cog' (center of gravity): IAF = Σ(f × P(f)) / Σ(P(f)) — robust to
    noisy spectra; preferred in clinical and neurofeedback contexts.
    'peak': IAF = frequency of the maximum PSD value in the alpha range.

    Returns a metrics dict with iaf_hz rounded to 2 decimal places.
    Does not modify the input spectrum.
    """
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])
    method = str(params["method"])

    freqs = spectrum.freqs
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not mask.any():
        raise ValueError(
            f"No frequency bins found between {fmin} and {fmax} Hz. "
            f"Check that fmin/fmax fall within the PSD range "
            f"({freqs[0]:.1f}–{freqs[-1]:.1f} Hz)."
        )

    band_freqs = freqs[mask]
    data = spectrum.get_data()  # (n_ch, n_freqs_all)
    band_data = data[:, mask]   # (n_ch, n_freqs_in_band)

    # Average across channels for global IAF estimate
    mean_psd = band_data.mean(axis=0)  # (n_freqs_in_band,)

    if method == "peak":
        iaf = float(band_freqs[np.argmax(mean_psd)])
    else:  # "cog" — center of gravity
        total_power = mean_psd.sum()
        if total_power < 1e-30:
            iaf = float(band_freqs.mean())  # fallback for zero-power edge case
        else:
            iaf = float(np.sum(band_freqs * mean_psd) / total_power)

    return {
        "iaf_hz": round(iaf, 2),
        "method": method,
        "alpha_range_hz": f"{fmin}–{fmax}",
        "n_channels_averaged": int(band_data.shape[0]),
    }


COMPUTE_ALPHA_PEAK = NodeDescriptor(
    node_type="compute_alpha_peak",
    display_name="Alpha Peak (IAF)",
    category="Clinical",
    description=(
        "Computes the Individual Alpha Frequency (IAF) — the dominant oscillatory "
        "frequency in the alpha band (~8–13 Hz). IAF is a stable individual trait "
        "linked to processing speed and cognitive performance. "
        "Connect a Compute PSD node first. "
        "Use the center-of-gravity method for noisy recordings; use peak for clean spectra."
    ),
    tags=["alpha", "iaf", "individual-alpha-frequency", "clinical", "qeeg", "neurofeedback"],
    inputs=[
        HandleSchema(id="psd_in", type="psd", label="PSD"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Alpha Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin",
            label="Alpha Band Min",
            type="float",
            default=7.0,
            min=1.0,
            max=20.0,
            step=0.5,
            unit="Hz",
            description=(
                "Lower edge of the alpha search window. "
                "7 Hz captures the low end of alpha including slow alpha variants. "
                "Increase to 8 Hz for a strict canonical alpha band."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="fmax",
            label="Alpha Band Max",
            type="float",
            default=13.0,
            min=5.0,
            max=30.0,
            step=0.5,
            unit="Hz",
            description=(
                "Upper edge of the alpha search window. "
                "13 Hz is the standard upper boundary. "
                "Extend to 14–15 Hz if the patient has a high IAF."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="method",
            label="IAF Method",
            type="select",
            default="cog",
            options=["cog", "peak"],
            description=(
                "cog (center of gravity): weighted mean frequency — more robust "
                "when the spectrum has multiple local peaks or is noisy. "
                "Preferred for clinical and neurofeedback use. "
                "peak: frequency of the maximum PSD value — simple and intuitive, "
                "best for clean spectra with a clear dominant peak."
            ),
            exposed=True,
        ),
    ],
    execute_fn=_execute_compute_alpha_peak,
    code_template=lambda p: (
        f"freqs = spectrum.freqs\n"
        f"mask = (freqs >= {p.get('fmin', 7.0)}) & (freqs <= {p.get('fmax', 13.0)})\n"
        f"band_freqs = freqs[mask]\n"
        f"mean_psd = spectrum.get_data()[:, mask].mean(axis=0)\n"
        + (
            f"iaf = float(band_freqs[np.argmax(mean_psd)])"
            if p.get("method", "cog") == "peak"
            else f"iaf = float(np.sum(band_freqs * mean_psd) / mean_psd.sum())"
        )
    ),
    methods_template=lambda p: (
        f"The individual alpha frequency was computed using the "
        f"{'peak detection' if p.get('method', 'cog') == 'peak' else 'center-of-gravity'} "
        f"method over the {p.get('fmin', 7.0)}\u2013{p.get('fmax', 13.0)} Hz range "
        f"(MNE-Python; Gramfort et al., 2013)."
    ),
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html",
)


# ---------------------------------------------------------------------------
# Compute Frontal Alpha Asymmetry
# ---------------------------------------------------------------------------

def _execute_compute_asymmetry(
    spectrum: "mne.time_frequency.Spectrum",
    params: dict,
) -> dict:
    """
    Computes frontal alpha asymmetry: ln(right_alpha) − ln(left_alpha).

    Positive values indicate right-frontal alpha dominance (associated with
    approach motivation and left-hemisphere activation). Negative values
    indicate left-frontal alpha dominance. Standard measure in affective
    neuroscience and clinical depression research.

    Raises ValueError if specified channels are not present in the PSD.
    """
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])
    left_ch = str(params["left_channel"])
    right_ch = str(params["right_channel"])

    ch_names = list(spectrum.ch_names)

    left_ch = resolve_channel(left_ch, ch_names)
    right_ch = resolve_channel(right_ch, ch_names)

    freqs = spectrum.freqs
    mask = (freqs >= fmin) & (freqs <= fmax)

    if not mask.any():
        raise ValueError(
            f"No frequency bins between {fmin} and {fmax} Hz in this PSD."
        )

    data = spectrum.get_data()  # (n_ch, n_freqs_all)
    band_data = data[:, mask]   # (n_ch, n_freqs_in_band)

    left_idx = ch_names.index(left_ch)
    right_idx = ch_names.index(right_ch)

    left_power = float(band_data[left_idx].mean())
    right_power = float(band_data[right_idx].mean())

    # Standard FAA: natural log of mean band power
    left_ln = float(np.log(left_power + 1e-30))
    right_ln = float(np.log(right_power + 1e-30))
    asymmetry_index = right_ln - left_ln

    return {
        "asymmetry_index": round(asymmetry_index, 4),
        f"{left_ch}_ln_alpha": round(left_ln, 4),
        f"{right_ch}_ln_alpha": round(right_ln, 4),
        "channel_left": left_ch,
        "channel_right": right_ch,
        "band_hz": f"{fmin}–{fmax}",
        "interpretation": (
            "right-dominant (approach)" if asymmetry_index > 0 else "left-dominant (withdrawal)"
        ),
    }


COMPUTE_ASYMMETRY = NodeDescriptor(
    node_type="compute_asymmetry",
    display_name="Alpha Asymmetry (FAA)",
    category="Clinical",
    description=(
        "Computes Frontal Alpha Asymmetry (FAA): the difference in natural-log "
        "alpha power between homologous right and left frontal electrodes. "
        "Positive values indicate right-frontal alpha dominance (left-hemisphere "
        "activity, approach motivation). Associated with depression, anxiety, and "
        "affective regulation. Connect a Compute PSD node first."
    ),
    tags=["asymmetry", "faa", "alpha", "frontal", "clinical", "depression", "affective"],
    inputs=[
        HandleSchema(id="psd_in", type="psd", label="PSD"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Asymmetry Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="left_channel",
            label="Left Channel",
            type="string",
            default="F3",
            description=(
                "Name of the left frontal electrode. Standard options: F3 (10-20), "
                "AF3, Fp1. Must exactly match a channel name in your recording."
            ),
            exposed=True,
            channel_hint="single",
        ),
        ParameterSchema(
            name="right_channel",
            label="Right Channel",
            type="string",
            default="F4",
            description=(
                "Name of the right frontal electrode. Standard options: F4 (10-20), "
                "AF4, Fp2. Must exactly match a channel name in your recording."
            ),
            exposed=True,
            channel_hint="single",
        ),
        ParameterSchema(
            name="fmin",
            label="Alpha Band Min",
            type="float",
            default=8.0,
            min=1.0,
            max=20.0,
            step=0.5,
            unit="Hz",
            description="Lower frequency bound of the alpha band. Canonical value: 8 Hz.",
        ),
        ParameterSchema(
            name="fmax",
            label="Alpha Band Max",
            type="float",
            default=13.0,
            min=5.0,
            max=30.0,
            step=0.5,
            unit="Hz",
            description="Upper frequency bound of the alpha band. Canonical value: 13 Hz.",
        ),
    ],
    execute_fn=_execute_compute_asymmetry,
    code_template=lambda p: (
        f"data = spectrum.get_data()\n"
        f"freqs = spectrum.freqs\n"
        f"mask = (freqs >= {p.get('fmin', 8.0)}) & (freqs <= {p.get('fmax', 13.0)})\n"
        f"left_idx = spectrum.ch_names.index('{p.get('left_channel', 'F3')}')\n"
        f"right_idx = spectrum.ch_names.index('{p.get('right_channel', 'F4')}')\n"
        f"left_power = data[left_idx, mask].mean()\n"
        f"right_power = data[right_idx, mask].mean()\n"
        f"asymmetry = np.log(right_power) - np.log(left_power)"
    ),
    methods_template=lambda p: (
        f"Frontal alpha asymmetry was calculated as ln(right) \u2212 ln(left) using "
        f"electrodes {p.get('right_channel', 'F4')} and {p.get('left_channel', 'F3')} "
        f"in the {p.get('fmin', 8.0)}\u2013{p.get('fmax', 13.0)} Hz alpha band "
        f"(MNE-Python; Gramfort et al., 2013)."
    ),
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html",
)


# ---------------------------------------------------------------------------
# Compute Band Ratio
# ---------------------------------------------------------------------------

def _execute_compute_band_ratio(
    spectrum: "mne.time_frequency.Spectrum",
    params: dict,
) -> dict:
    """
    Computes the ratio of mean PSD power in two frequency bands.

    Default computes theta/beta ratio — a clinical marker elevated in ADHD
    and attention disorders. Other useful ratios: gamma/alpha (cognitive load),
    delta/alpha (arousal), theta/alpha (memory encoding).

    When log_scale=True, computes log10(numerator) − log10(denominator),
    which is equivalent to log10(ratio) and avoids extreme values.
    """
    num_fmin = float(params["numerator_fmin"])
    num_fmax = float(params["numerator_fmax"])
    den_fmin = float(params["denominator_fmin"])
    den_fmax = float(params["denominator_fmax"])
    log_scale = bool(params.get("log_scale", True))

    freqs = spectrum.freqs
    data = spectrum.get_data()  # (n_ch, n_freqs_all)

    num_mask = (freqs >= num_fmin) & (freqs <= num_fmax)
    den_mask = (freqs >= den_fmin) & (freqs <= den_fmax)

    if not num_mask.any():
        raise ValueError(
            f"No frequency bins in numerator band {num_fmin}–{num_fmax} Hz."
        )
    if not den_mask.any():
        raise ValueError(
            f"No frequency bins in denominator band {den_fmin}–{den_fmax} Hz."
        )

    # Mean power across channels and across frequency bins in each band
    num_power = float(data[:, num_mask].mean())
    den_power = float(data[:, den_mask].mean())

    if log_scale:
        ratio = float(
            np.log10(num_power + 1e-30) - np.log10(den_power + 1e-30)
        )
    else:
        ratio = float(num_power / (den_power + 1e-30))

    return {
        "band_ratio": round(ratio, 4),
        "numerator_mean_power": float(np.log10(num_power + 1e-30)) if log_scale else round(num_power, 6),
        "denominator_mean_power": float(np.log10(den_power + 1e-30)) if log_scale else round(den_power, 6),
        "numerator_band_hz": f"{num_fmin}–{num_fmax}",
        "denominator_band_hz": f"{den_fmin}–{den_fmax}",
        "scale": "log10" if log_scale else "linear",
    }


COMPUTE_BAND_RATIO = NodeDescriptor(
    node_type="compute_band_ratio",
    display_name="Band Ratio",
    category="Clinical",
    description=(
        "Computes the ratio of mean spectral power in two user-defined frequency bands. "
        "The default (theta 4–8 Hz divided by beta 13–30 Hz) is the theta/beta ratio — "
        "a widely used clinical biomarker elevated in ADHD. "
        "Other clinical ratios: delta/alpha (arousal), gamma/alpha (cognitive load). "
        "Connect a Compute PSD node. Output is a metrics dict."
    ),
    tags=["band-ratio", "theta-beta", "tbr", "adhd", "clinical", "qeeg", "biomarker"],
    inputs=[
        HandleSchema(id="psd_in", type="psd", label="PSD"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Band Ratio Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="numerator_fmin",
            label="Numerator Band Min",
            type="float",
            default=4.0,
            min=0.5,
            max=100.0,
            step=0.5,
            unit="Hz",
            description=(
                "Lower edge of the numerator frequency band. "
                "Default 4 Hz: lower edge of the theta band."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="numerator_fmax",
            label="Numerator Band Max",
            type="float",
            default=8.0,
            min=1.0,
            max=100.0,
            step=0.5,
            unit="Hz",
            description=(
                "Upper edge of the numerator frequency band. "
                "Default 8 Hz: upper edge of the theta band."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="denominator_fmin",
            label="Denominator Band Min",
            type="float",
            default=13.0,
            min=0.5,
            max=100.0,
            step=0.5,
            unit="Hz",
            description=(
                "Lower edge of the denominator frequency band. "
                "Default 13 Hz: lower edge of the beta band."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="denominator_fmax",
            label="Denominator Band Max",
            type="float",
            default=30.0,
            min=1.0,
            max=200.0,
            step=0.5,
            unit="Hz",
            description=(
                "Upper edge of the denominator frequency band. "
                "Default 30 Hz: upper edge of the beta band."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="log_scale",
            label="Log Scale",
            type="bool",
            default=True,
            description=(
                "When enabled, computes log10(numerator) − log10(denominator) "
                "instead of a raw ratio. Recommended — log scale compresses extreme "
                "values and makes the result more interpretable."
            ),
        ),
    ],
    execute_fn=_execute_compute_band_ratio,
    code_template=lambda p: (
        f"data = spectrum.get_data()\n"
        f"freqs = spectrum.freqs\n"
        f"num_mask = (freqs >= {p.get('numerator_fmin', 4.0)}) & (freqs <= {p.get('numerator_fmax', 8.0)})\n"
        f"den_mask = (freqs >= {p.get('denominator_fmin', 13.0)}) & (freqs <= {p.get('denominator_fmax', 30.0)})\n"
        f"num_power = data[:, num_mask].mean()\n"
        f"den_power = data[:, den_mask].mean()\n"
        + (
            f"ratio = np.log10(num_power) - np.log10(den_power)"
            if p.get("log_scale", True)
            else f"ratio = num_power / den_power"
        )
    ),
    methods_template=lambda p: (
        f"The {'theta' if p.get('numerator_fmin', 4.0) == 4.0 else 'numerator'}"
        f"/{'beta' if p.get('denominator_fmin', 13.0) == 13.0 else 'denominator'} "
        f"band power ratio was computed from the {p.get('numerator_fmin', 4.0)}\u2013"
        f"{p.get('numerator_fmax', 8.0)} Hz and {p.get('denominator_fmin', 13.0)}\u2013"
        f"{p.get('denominator_fmax', 30.0)} Hz bands"
        + (f" on a log\u2081\u2080 scale" if p.get("log_scale", True) else "")
        + f" (MNE-Python; Gramfort et al., 2013)."
    ),
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html",
)


# ---------------------------------------------------------------------------
# Z-Score Normalize
# ---------------------------------------------------------------------------

def _execute_z_score_normalize(data: np.ndarray, params: dict) -> dict:
    """
    Z-score normalizes a 1D or 2D array against reference mean and std.

    When use_data_stats=True (default), computes mean/std from the input
    data itself (standard z-scoring). When use_data_stats=False, uses
    norm_mean and norm_std parameters — useful for normalizing against
    published normative values (e.g., comparison to age-matched norms).

    Input: numpy array (e.g., from Compute Band Power node).
    Output: metrics dict containing z-scored values and statistics used.
    """
    arr = np.asarray(data, dtype=float).flatten()
    use_data_stats = bool(params.get("use_data_stats", True))

    if use_data_stats:
        mean_used = float(arr.mean())
        std_used = float(arr.std())
    else:
        mean_used = float(params["norm_mean"])
        std_used = float(params["norm_std"])

    # Guard against division by zero
    if abs(std_used) < 1e-10:
        std_used = 1.0

    z = (arr - mean_used) / std_used

    return {
        "z_scores": [round(float(v), 4) for v in z],
        "mean_used": round(mean_used, 6),
        "std_used": round(std_used, 6),
        "n_values": int(len(arr)),
        "reference": "data-computed" if use_data_stats else "normative",
    }


Z_SCORE_NORMALIZE = NodeDescriptor(
    node_type="z_score_normalize",
    display_name="Z-Score Normalize",
    category="Clinical",
    description=(
        "Normalizes an array of values to z-scores: (x − mean) / std. "
        "Use after a Compute Band Power node to express channel band powers "
        "relative to a reference. "
        "When 'Use Data Stats' is on, normalizes to the data's own mean and std. "
        "When off, you provide normative mean and std (e.g., from a published database) "
        "to compare the patient against population norms."
    ),
    tags=["zscore", "normalize", "clinical", "norms", "statistics"],
    inputs=[
        HandleSchema(id="array_in", type="array", label="Array"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Z-Score Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="use_data_stats",
            label="Use Data Stats",
            type="bool",
            default=True,
            description=(
                "When enabled, computes mean and std from the input data itself. "
                "Disable to specify external normative reference values below."
            ),
        ),
        ParameterSchema(
            name="norm_mean",
            label="Normative Mean",
            type="float",
            default=0.0,
            description=(
                "Reference mean from a normative database. Used only when "
                "'Use Data Stats' is disabled. Units must match the input array."
            ),
        ),
        ParameterSchema(
            name="norm_std",
            label="Normative Std Dev",
            type="float",
            default=1.0,
            min=1e-6,
            description=(
                "Reference standard deviation from a normative database. "
                "Used only when 'Use Data Stats' is disabled. Must be > 0."
            ),
        ),
    ],
    execute_fn=_execute_z_score_normalize,
    code_template=lambda p: (
        f"arr = data.flatten()\n"
        + (
            f"mean_val = arr.mean()\n"
            f"std_val = arr.std()\n"
            if p.get("use_data_stats", True)
            else
            f"mean_val = {p.get('norm_mean', 0.0)}\n"
            f"std_val = {p.get('norm_std', 1.0)}\n"
        )
        + f"z_scores = (arr - mean_val) / std_val"
    ),
    methods_template=lambda p: (
        f"Channel values were z-score normalized "
        + (
            "against their own mean and standard deviation."
            if p.get("use_data_stats", True)
            else f"against normative reference values "
            f"(mean={p.get('norm_mean', 0.0)}, std={p.get('norm_std', 1.0)})."
        )
    ),
    docs_url="https://mne.tools/stable/index.html",
)


# ---------------------------------------------------------------------------
# Detect Spikes (Epileptic Spike Detection)
# ---------------------------------------------------------------------------

def _execute_detect_spikes(raw: mne.io.BaseRaw, params: dict) -> dict:
    """
    Detects candidate epileptic spike events using amplitude thresholding.

    Algorithm:
      1. Extract EEG channel data and convert to µV.
      2. Find all time points where |amplitude| exceeds the threshold on
         at least one channel.
      3. Group consecutive super-threshold samples into candidate events.
      4. Filter events by duration (min_duration_ms to max_duration_ms).
      5. For each event, identify the channel with the peak amplitude.

    Returns counts, onset times (seconds), and peak channels.
    Spike times are capped at 50 events for readability; full count is returned.

    Note: This is a threshold-based detector, not a morphology-based classifier.
    Clinical confirmation of detected events is required.
    """
    raw_copy = raw.copy()
    threshold_uv = float(params["threshold_uv"])
    min_duration_ms = float(params["min_duration_ms"])
    max_duration_ms = float(params["max_duration_ms"])

    sfreq = raw_copy.info["sfreq"]
    min_samples = max(1, int(min_duration_ms / 1000.0 * sfreq))
    max_samples = int(max_duration_ms / 1000.0 * sfreq)

    # Pick EEG channels; fall back to all channels if none typed as eeg
    eeg_picks = mne.pick_types(raw_copy.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        eeg_picks = np.arange(len(raw_copy.ch_names))

    data_uv = raw_copy.get_data(picks=eeg_picks) * 1e6  # (n_ch, n_times)
    ch_names = [raw_copy.ch_names[i] for i in eeg_picks]

    # Boolean mask: any channel exceeds threshold at each time point
    above = np.any(np.abs(data_uv) > threshold_uv, axis=0)  # (n_times,)

    # Group consecutive True samples into candidate events
    events: list[tuple[int, int]] = []
    in_event = False
    event_start = 0
    for i, val in enumerate(above):
        if val and not in_event:
            in_event = True
            event_start = i
        elif not val and in_event:
            in_event = False
            duration = i - event_start
            if min_samples <= duration <= max_samples:
                events.append((event_start, i))
    # Handle event still open at end of signal
    if in_event:
        duration = len(above) - event_start
        if min_samples <= duration <= max_samples:
            events.append((event_start, len(above)))

    # For each event: onset time and peak channel
    spike_times_s = [round(float(start / sfreq), 3) for start, _ in events]
    spike_channels: list[str] = []
    for start, end in events:
        seg = data_uv[:, start:end]
        peak_ch_flat = int(np.unravel_index(np.argmax(np.abs(seg)), seg.shape)[0])
        spike_channels.append(ch_names[peak_ch_flat])

    n_total = len(events)
    cap = 50  # display cap to avoid overwhelming the metrics panel

    return {
        "n_spikes": n_total,
        "spike_times_s": spike_times_s[:cap],
        "spike_channels": spike_channels[:cap],
        "threshold_uv": threshold_uv,
        "min_duration_ms": min_duration_ms,
        "max_duration_ms": max_duration_ms,
        "note": (
            f"Showing first {cap} of {n_total} events."
            if n_total > cap else f"All {n_total} events shown."
        ),
    }


DETECT_SPIKES = NodeDescriptor(
    node_type="detect_spikes",
    display_name="Detect Spikes",
    category="Clinical",
    description=(
        "Detects candidate epileptic spike events using amplitude thresholding. "
        "Finds time segments where at least one EEG channel exceeds the amplitude "
        "threshold, filtered by duration to match typical spike morphology (20–100 ms). "
        "Output is a metrics dict with spike count, onset times, and peak channels. "
        "Clinical confirmation of all detected events is required — this is a "
        "screening tool, not a validated diagnostic classifier."
    ),
    tags=["spike", "epilepsy", "seizure", "clinical", "threshold", "detection", "eeg"],
    inputs=[
        HandleSchema(id="eeg_in", type="raw_eeg", label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Spike Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="threshold_uv",
            label="Amplitude Threshold",
            type="float",
            default=150.0,
            min=10.0,
            max=2000.0,
            step=10.0,
            unit="µV",
            description=(
                "Amplitude threshold for spike detection. Any time point where a "
                "channel exceeds this value (in µV) is flagged. "
                "Typical clinical EEG spikes: 100–300 µV. "
                "Lower values increase sensitivity (more false positives); "
                "higher values increase specificity."
            ),
            exposed=True,
        ),
        ParameterSchema(
            name="min_duration_ms",
            label="Min Duration",
            type="float",
            default=20.0,
            min=1.0,
            max=500.0,
            step=5.0,
            unit="ms",
            description=(
                "Minimum duration of a super-threshold segment to be counted as "
                "a candidate spike. Events shorter than this are rejected as noise. "
                "20 ms corresponds to frequencies above ~50 Hz."
            ),
        ),
        ParameterSchema(
            name="max_duration_ms",
            label="Max Duration",
            type="float",
            default=100.0,
            min=10.0,
            max=1000.0,
            step=10.0,
            unit="ms",
            description=(
                "Maximum duration of a super-threshold segment to count as a spike. "
                "Events longer than this are excluded (they are more likely slow waves "
                "or muscle artifacts, not spikes). 100 ms is a standard upper bound."
            ),
        ),
    ],
    execute_fn=_execute_detect_spikes,
    code_template=lambda p: (
        f"raw_copy = raw.copy()\n"
        f"data_uv = raw_copy.get_data(picks='eeg') * 1e6\n"
        f"threshold = {p.get('threshold_uv', 150.0)}  # µV\n"
        f"sfreq = raw_copy.info['sfreq']\n"
        f"above = np.any(np.abs(data_uv) > threshold, axis=0)\n"
        f"min_samples = int({p.get('min_duration_ms', 20.0)} / 1000 * sfreq)\n"
        f"max_samples = int({p.get('max_duration_ms', 100.0)} / 1000 * sfreq)\n"
        f"# Group consecutive super-threshold samples and filter by duration"
    ),
    methods_template=lambda p: (
        f"Candidate epileptic spikes were detected using an amplitude threshold of "
        f"{p.get('threshold_uv', 150.0)} \u00b5V with duration constraints of "
        f"{p.get('min_duration_ms', 20.0)}\u2013{p.get('max_duration_ms', 100.0)} ms "
        f"(MNE-Python; Gramfort et al., 2013)."
    ),
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html",
)
