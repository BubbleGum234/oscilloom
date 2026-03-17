"""
backend/registry/nodes/analysis.py

Analysis node types: computation that produces derived data from EEG signals.

execute_fns here return MNE analysis objects (Spectrum, Epochs, Evoked, etc.)
rather than Raw objects. They do not produce visualizations directly —
pair them with a visualization node to render results.
"""

from __future__ import annotations

import numpy as np
import mne

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Compute PSD
# ---------------------------------------------------------------------------

def _execute_compute_psd(
    raw: mne.io.BaseRaw,
    params: dict,
) -> mne.time_frequency.Spectrum:
    """
    Computes Power Spectral Density using Welch or multitaper method.

    Returns an MNE Spectrum object. Does not mutate the input Raw object.
    n_fft and n_overlap are only passed for the Welch method; multitaper
    uses DPSS tapers and ignores these parameters.
    """
    method = str(params["method"])
    kwargs: dict = dict(
        method=method,
        fmin=float(params["fmin"]),
        fmax=float(params["fmax"]),
        verbose=False,
    )
    if method == "welch":
        n_fft = int(params["n_fft"])
        kwargs["n_fft"] = n_fft
        n_overlap = int(params["n_overlap"])
        if n_overlap >= n_fft:
            # Auto-clamp to 50% overlap instead of crashing — this commonly
            # happens when the user reduces n_fft but forgets n_overlap.
            n_overlap = n_fft // 2
        if n_overlap > 0:
            kwargs["n_overlap"] = n_overlap
    return raw.compute_psd(**kwargs)


COMPUTE_PSD = NodeDescriptor(
    node_type="compute_psd",
    display_name="Compute PSD",
    category="Analysis",
    description=(
        "Computes the Power Spectral Density (PSD) of the EEG signal. "
        "Welch's method (default) segments the signal, computes FFT on each segment, "
        "and averages — reducing noise compared to a single FFT. "
        "Multitaper (DPSS) provides better frequency concentration and is preferred "
        "for short recordings. Connect the output to a Plot PSD node to visualize."
    ),
    tags=["psd", "power", "spectrum", "frequency", "welch", "multitaper", "analysis", "fft"],
    inputs=[
        HandleSchema(id="raw_in", type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="eeg_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(
            id="psd_out",
            type="psd",
            label="PSD",
        ),
    ],
    parameters=[
        ParameterSchema(
            name="method",
            label="Method",
            type="select",
            default="welch",
            options=["welch", "multitaper"],
            description=(
                "Welch: segments the signal and averages FFT estimates. "
                "Good for long, stationary recordings. "
                "Multitaper (DPSS): uses multiple orthogonal tapers for better "
                "spectral concentration. Preferred for short or non-stationary segments."
            ),
        ),
        ParameterSchema(
            name="fmin",
            label="Min Frequency",
            type="float",
            default=0.5,
            min=0.0,
            max=500.0,
            step=0.5,
            unit="Hz",
            description=(
                "Minimum frequency included in the PSD. Frequencies below this "
                "are excluded from the output. Use 0.5 Hz for standard EEG "
                "analysis; lower values may be dominated by movement artifacts."
            ),
        ),
        ParameterSchema(
            name="fmax",
            label="Max Frequency",
            type="float",
            default=60.0,
            min=1.0,
            max=500.0,
            step=1.0,
            unit="Hz",
            description=(
                "Maximum frequency included in the PSD. Should not exceed the "
                "low-pass cutoff of any upstream filter — frequencies above the "
                "filter cutoff are noise and will distort the PSD."
            ),
        ),
        ParameterSchema(
            name="n_fft",
            label="FFT Length",
            type="int",
            default=2048,
            min=256,
            max=16384,
            step=256,
            unit="samples",
            description=(
                "Length of the FFT window in samples (Welch only). "
                "Frequency resolution = sfreq / n_fft. "
                "Larger values give finer resolution but require more memory. "
                "2048 is a good default for 256–1000 Hz recordings."
            ),
        ),
        ParameterSchema(
            name="n_overlap",
            label="Welch Overlap",
            type="int",
            default=1024,
            min=0,
            max=16384,
            step=256,
            unit="samples",
            description=(
                "Number of samples to overlap between Welch segments (Welch only). "
                "1024 = 50% of the default FFT length (2048 samples). "
                "Higher overlap reduces variance but increases computation."
            ),
        ),
    ],
    execute_fn=_execute_compute_psd,
    code_template=lambda p: (
        f'spectrum = raw.compute_psd(method="welch", fmin={p.get("fmin", 0.5)}, fmax={p.get("fmax", 60.0)}, n_fft={p.get("n_fft", 2048)}, n_overlap={min(int(p.get("n_overlap", 1024)), int(p.get("n_fft", 2048)) - 1)}, verbose=False)'
        if p.get("method", "welch") == "welch"
        else f'spectrum = raw.compute_psd(method="multitaper", fmin={p.get("fmin", 0.5)}, fmax={p.get("fmax", 60.0)}, verbose=False)'
    ),
    methods_template=lambda p: f'Power spectral density was estimated using the {"Welch" if p.get("method", "welch") == "welch" else "multitaper"} method ({p.get("fmin", 0.5)}–{p.get("fmax", 60.0)} Hz) with MNE-Python (Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.compute_psd",
)


# ---------------------------------------------------------------------------
# Compute Evoked (ERP)
# ---------------------------------------------------------------------------

def _execute_compute_evoked(epochs: mne.Epochs, params: dict) -> mne.Evoked:
    """
    Averages all epochs to produce an Event-Related Potential (ERP) waveform.

    mne.Epochs.average() returns an mne.Evoked object containing the trial-
    averaged signal. Does not modify the input epochs.
    """
    return epochs.average()


COMPUTE_EVOKED = NodeDescriptor(
    node_type="compute_evoked",
    display_name="Compute Evoked (ERP)",
    category="Analysis",
    description=(
        "Computes the Event-Related Potential (ERP) by averaging all epochs. "
        "Averaging cancels out random neural noise while preserving the "
        "time-locked response to the stimulus. The result is an mne.Evoked object "
        "showing the mean waveform across all trials. "
        "Connect to Plot Evoked to visualize the butterfly plot."
    ),
    tags=["erp", "evoked", "average", "epochs", "analysis", "erp"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="evoked_out", type="evoked", label="Evoked (ERP)"),
    ],
    parameters=[],
    execute_fn=_execute_compute_evoked,
    code_template=lambda p: 'evoked = epochs.average()',
    methods_template=lambda p: "Event-related potentials were computed by averaging all epochs using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.average",
)


# ---------------------------------------------------------------------------
# Time-Frequency (Morlet Wavelets)
# ---------------------------------------------------------------------------

def _execute_time_frequency_morlet(
    epochs: mne.Epochs,
    params: dict,
) -> "mne.time_frequency.AverageTFR":
    """
    Computes time-frequency representations using Morlet wavelets.

    Returns an AverageTFR object averaged across all epochs.
    freqs is a linear array from fmin to fmax with 1 Hz spacing.
    n_cycles controls wavelet width — higher values give better frequency
    resolution at the cost of temporal resolution.
    """
    fmin = float(params["fmin"])
    fmax = float(params["fmax"])
    n_cycles = float(params["n_cycles"])
    freq_step = float(params.get("freq_step", 1.0))
    freqs = np.arange(fmin, fmax + freq_step, freq_step)
    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        verbose=False,
    )
    return tfr


def _execute_compute_bandpower(
    spectrum: "mne.time_frequency.Spectrum",
    params: dict,
) -> "np.ndarray":
    """
    Computes mean PSD power in a specified frequency band per channel.

    spectrum.get_data(fmin, fmax) returns shape (n_channels, n_freqs).
    Averaging over the last axis gives per-channel mean power.
    When log_scale is True, converts to dB: 10 * log10(power).
    """
    data = spectrum.get_data(
        fmin=float(params["fmin"]),
        fmax=float(params["fmax"]),
    ).mean(axis=-1)   # shape (n_channels,)
    if bool(params.get("log_scale", True)):
        data = np.log10(data + 1e-30) * 10.0  # add epsilon to avoid log(0)
    return data


COMPUTE_BANDPOWER = NodeDescriptor(
    node_type="compute_bandpower",
    display_name="Compute Band Power",
    category="Analysis",
    description=(
        "Computes the mean PSD power in a specified frequency band for each channel. "
        "This is the core metric for resting-state EEG studies: individual alpha frequency, "
        "eyes-open vs eyes-closed alpha suppression, and sleep staging all rely on band power. "
        "Input must come from a Compute PSD node. "
        "Output is a per-channel power array, available for downstream analysis."
    ),
    tags=["bandpower", "alpha", "theta", "beta", "power", "analysis", "resting-state"],
    inputs=[
        HandleSchema(id="psd_in", type="psd", label="PSD"),
    ],
    outputs=[
        HandleSchema(id="array_out", type="array", label="Band Power (per channel)"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin",
            label="Band Min",
            type="float",
            default=8.0,
            min=0.0,
            max=500.0,
            step=0.5,
            unit="Hz",
            description=(
                "Lower edge of the frequency band. "
                "Common bands: delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45."
            ),
        ),
        ParameterSchema(
            name="fmax",
            label="Band Max",
            type="float",
            default=13.0,
            min=0.0,
            max=500.0,
            step=0.5,
            unit="Hz",
            description=(
                "Upper edge of the frequency band. "
                "Default 13 Hz: alpha band upper edge."
            ),
        ),
        ParameterSchema(
            name="log_scale",
            label="Log Scale (dB)",
            type="bool",
            default=True,
            description=(
                "When enabled, returns power in decibels (10 * log10(power)). "
                "Recommended for most analyses — dB scale is more interpretable "
                "and better suited for statistical tests."
            ),
        ),
    ],
    execute_fn=_execute_compute_bandpower,
    code_template=lambda p: f'band_power = spectrum.get_data(fmin={p.get("fmin", 8.0)}, fmax={p.get("fmax", 13.0)}).mean(axis=-1)',
    methods_template=lambda p: f'Mean band power was computed in the {p.get("fmin", 8.0)}–{p.get("fmax", 13.0)} Hz range for each channel.',
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html",
)


TIME_FREQUENCY_MORLET = NodeDescriptor(
    node_type="time_frequency_morlet",
    display_name="Time-Frequency (Morlet)",
    category="Analysis",
    description=(
        "Computes a time-frequency representation (TFR) of epoched EEG using "
        "Morlet wavelets. Shows how spectral power changes over time within each "
        "epoch — useful for studying oscillatory dynamics like alpha suppression, "
        "gamma bursts, or theta synchrony. "
        "The output is averaged across all epochs. Connect to a plot node to "
        "visualize the time-frequency spectrogram."
    ),
    tags=["time-frequency", "tfr", "morlet", "wavelet", "oscillation", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="tfr_out", type="tfr", label="TFR"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin",
            label="Min Frequency",
            type="float",
            default=4.0,
            min=1.0,
            max=200.0,
            step=1.0,
            unit="Hz",
            description=(
                "Lowest frequency to include in the analysis. "
                "4 Hz (theta band lower edge) is a common starting point."
            ),
        ),
        ParameterSchema(
            name="fmax",
            label="Max Frequency",
            type="float",
            default=40.0,
            min=1.0,
            max=200.0,
            step=1.0,
            unit="Hz",
            description=(
                "Highest frequency to include in the analysis. "
                "40 Hz covers theta, alpha, beta, and low gamma."
            ),
        ),
        ParameterSchema(
            name="n_cycles",
            label="Wavelet Cycles",
            type="float",
            default=7.0,
            min=1.0,
            max=20.0,
            step=0.5,
            description=(
                "Number of cycles in the Morlet wavelet. Controls the trade-off "
                "between temporal and frequency resolution. "
                "Higher values (7–10): better frequency resolution, worse time resolution. "
                "Lower values (3–5): better temporal precision, broader frequency spread."
            ),
        ),
        ParameterSchema(
            name="freq_step",
            label="Frequency Step",
            type="float",
            default=1.0,
            min=0.1,
            max=5.0,
            step=0.1,
            unit="Hz",
            description=(
                "Step size between frequency bins. 1.0 Hz is the default and "
                "sufficient for broad-band analysis. "
                "Reduce to 0.25–0.5 Hz for high-resolution analysis of narrow bands "
                "(e.g., theta 4–8 Hz yields only 4 bins at 1 Hz step, 17 bins at 0.25 Hz). "
                "Smaller steps increase computation time proportionally."
            ),
        ),
    ],
    execute_fn=_execute_time_frequency_morlet,
    code_template=lambda p: f'import numpy as np\nfreqs = np.arange({p.get("fmin", 4.0)}, {p.get("fmax", 40.0)} + {p.get("freq_step", 1.0)}, {p.get("freq_step", 1.0)})\ntfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles={p.get("n_cycles", 7.0)}, return_itc=False, verbose=False)',
    methods_template=lambda p: f'Time-frequency decomposition was performed using Morlet wavelets ({p.get("fmin", 4.0)}–{p.get("fmax", 40.0)} Hz, {p.get("n_cycles", 7.0)} cycles) with MNE-Python (Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.tfr_morlet.html",
)


# ---------------------------------------------------------------------------
# Summarize Annotations
# ---------------------------------------------------------------------------

def _execute_summarize_annotations(raw: mne.io.BaseRaw, params: dict) -> dict:
    """
    Returns a metrics dict with per-label counts, total durations, and
    percentage of recording marked as BAD.

    Read-only inspection node — does not modify the Raw object.
    """
    annotations = raw.annotations
    total_duration = raw.times[-1]

    if len(annotations) == 0:
        return {
            "total_bad_duration_s": 0.0,
            "total_bad_pct": 0.0,
            "total_annotations": 0,
            "recording_duration_s": round(total_duration, 2),
        }

    from collections import defaultdict
    label_stats = defaultdict(lambda: {"count": 0, "total_duration_s": 0.0})

    for ann in annotations:
        desc = ann["description"]
        label_stats[desc]["count"] += 1
        label_stats[desc]["total_duration_s"] += ann["duration"]

    metrics: dict = {}
    total_bad_duration = 0.0

    for label, stats in sorted(label_stats.items()):
        prefix = "bad" if label.startswith("BAD_") else "event"
        metrics[f"{prefix}_{label}_count"] = stats["count"]
        metrics[f"{prefix}_{label}_duration_s"] = round(stats["total_duration_s"], 2)
        if label.startswith("BAD_"):
            total_bad_duration += stats["total_duration_s"]

    metrics["total_bad_duration_s"] = round(total_bad_duration, 2)
    metrics["total_bad_pct"] = round((total_bad_duration / total_duration) * 100, 1) if total_duration > 0 else 0.0
    metrics["total_annotations"] = len(annotations)
    metrics["recording_duration_s"] = round(total_duration, 2)

    return metrics


SUMMARIZE_ANNOTATIONS = NodeDescriptor(
    node_type="summarize_annotations",
    display_name="Summarize Annotations",
    category="Analysis",
    description=(
        "Tabular summary of all annotations: per-label counts, durations, and "
        "percentage of recording marked as BAD."
    ),
    inputs=[HandleSchema(id="raw_in", type="filtered_eeg", label="Filtered EEG")],
    outputs=[HandleSchema(id="metrics_out", type="metrics", label="Annotation Summary")],
    parameters=[],
    execute_fn=_execute_summarize_annotations,
    tags=["annotation", "summary", "inspection", "quality"],
    code_template=lambda p: '# Summarize annotations: counts, durations, and BAD percentage\nmetrics = {}\nfor ann in raw.annotations:\n    # ... per-label aggregation',
    methods_template=lambda p: "Annotation statistics (per-label counts, durations, and percentage of recording marked as artifact) were computed for quality assessment.",
    docs_url="https://mne.tools/stable/generated/mne.Annotations.html",
)
