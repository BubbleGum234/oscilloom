"""
backend/registry/nodes/sleep.py

Sleep analysis node descriptors — Tier 6.

Pipeline flow:
  filtered_eeg → compute_sleep_stages        → metrics (hypnogram)
  metrics      → compute_sleep_architecture  → metrics (AASM stats)
  filtered_eeg → detect_spindles             → metrics
  filtered_eeg → detect_slow_oscillations    → metrics
  metrics      → plot_hypnogram              → plot

Soft dependency on YASA: server starts without it, sleep nodes raise
a clear error at execution time.

NumPy 2.0 compatibility: YASA 0.6.5 uses numpy.trapz which was removed
in NumPy 2.0. The monkey-patch below restores it.

Added in Tier 6.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np

# NumPy 2.0 compat: YASA 0.6.5 uses np.trapz (removed in NumPy 2.0)
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.registry.nodes._channel_utils import resolve_channel, resolve_channel_optional

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)

# ---------------------------------------------------------------------------
# Soft YASA import (same pattern as bci.py with sklearn)
# ---------------------------------------------------------------------------

try:
    import yasa
    _YASA_AVAILABLE = True
except ImportError:
    _YASA_AVAILABLE = False


def _require_yasa():
    """Raise a helpful error if YASA is not installed."""
    if not _YASA_AVAILABLE:
        raise ImportError(
            "YASA is required for sleep analysis nodes. "
            "Install it with: pip install yasa"
        )


# ---------------------------------------------------------------------------
# Node 1: Compute Sleep Stages
# ---------------------------------------------------------------------------

def _execute_compute_sleep_stages(input_data, params: dict) -> dict:
    """
    Automatic sleep staging using YASA's pre-trained LightGBM classifier.

    Takes continuous EEG (optionally EOG + EMG) and predicts sleep stages
    in 30-second epochs: W, N1, N2, N3, R.

    Returns a metrics dict with the full hypnogram (int + labels),
    stage counts, and metadata.
    """
    _require_yasa()
    import mne

    raw = input_data.copy()

    eeg_name = str(params.get("eeg_channel", "")).strip()
    eog_name = str(params.get("eog_channel", "")).strip() or None
    emg_name = str(params.get("emg_channel", "")).strip() or None

    # Auto-detect EEG channel if not specified
    if not eeg_name:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) == 0:
            raise ValueError(
                "No EEG channels found. Specify an EEG channel name in the "
                "'EEG Channel' parameter."
            )
        eeg_name = raw.ch_names[eeg_picks[0]]

    eeg_name = resolve_channel(eeg_name, raw.ch_names)

    # Silently skip optional channels if not found (with fuzzy matching)
    if eog_name:
        eog_name = resolve_channel_optional(eog_name, raw.ch_names)
    if emg_name:
        emg_name = resolve_channel_optional(emg_name, raw.ch_names)

    # Run YASA SleepStaging
    sls = yasa.SleepStaging(
        raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name
    )
    predicted = sls.predict()  # array of str: 'W', 'N1', 'N2', 'N3', 'R'

    label_to_int = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
    hypno_int = [label_to_int.get(s, -1) for s in predicted]

    # Count stages
    stage_counts = {}
    for s in ["W", "N1", "N2", "N3", "R"]:
        stage_counts[s] = int(sum(1 for x in predicted if x == s))

    return {
        "hypnogram": hypno_int,
        "hypnogram_labels": list(predicted),
        "stage_counts": stage_counts,
        "n_epochs": len(predicted),
        "epoch_duration_s": 30.0,
        "eeg_channel": eeg_name,
        "eog_channel": eog_name or "none",
        "emg_channel": emg_name or "none",
    }


COMPUTE_SLEEP_STAGES = NodeDescriptor(
    node_type="compute_sleep_stages",
    display_name="Sleep Staging (YASA)",
    category="Sleep",
    description=(
        "Automatic sleep staging using YASA's pre-trained classifier. "
        "Predicts sleep stages (Wake, N1, N2, N3, REM) in 30-second epochs "
        "from continuous EEG data. Adding EOG and EMG channels improves "
        "accuracy. Connect a filtered EEG signal (bandpass 0.3–35 Hz recommended). "
        "Output is a metrics dict containing the full hypnogram."
    ),
    tags=["sleep", "staging", "hypnogram", "yasa", "psg", "w", "n1", "n2", "n3", "rem"],
    inputs=[
        HandleSchema(id="eeg_in", type="raw_eeg", label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Sleep Stages"),
    ],
    parameters=[
        ParameterSchema(
            name="eeg_channel",
            label="EEG Channel",
            type="string",
            default="",
            description=(
                "Name of the EEG channel to use for staging. "
                "Leave empty to auto-select the first EEG channel. "
                "Recommended: C3 or C4 (central derivation)."
            ),
            exposed=True,
            channel_hint="single",
        ),
        ParameterSchema(
            name="eog_channel",
            label="EOG Channel",
            type="string",
            default="",
            description=(
                "Name of the EOG channel (optional). "
                "Adding an EOG channel improves REM detection accuracy. "
                "Leave empty to skip."
            ),
            exposed=True,
            channel_hint="single",
        ),
        ParameterSchema(
            name="emg_channel",
            label="EMG Channel",
            type="string",
            default="",
            description=(
                "Name of the EMG channel (optional). "
                "Adding an EMG channel improves REM vs N1 differentiation. "
                "Leave empty to skip."
            ),
            exposed=True,
            channel_hint="single",
        ),
    ],
    execute_fn=_execute_compute_sleep_stages,
    code_template=lambda p: (
        "sls = yasa.SleepStaging(raw"
        + (f', eeg_name="{p["eeg_channel"]}"' if p.get("eeg_channel") else "")
        + (f', eog_name="{p["eog_channel"]}"' if p.get("eog_channel") else "")
        + (f', emg_name="{p["emg_channel"]}"' if p.get("emg_channel") else "")
        + ")\nhypnogram = sls.predict()"
    ),
    methods_template=lambda p: (
        "Automatic sleep staging was performed using YASA's pre-trained "
        "LightGBM classifier (Vallat & Walker, 2021), which predicted "
        "sleep stages (W, N1, N2, N3, R) in 30-second epochs"
        + (f" from channel {p['eeg_channel']}" if p.get("eeg_channel") else "")
        + "."
    ),
    docs_url="https://raphaelvallat.com/yasa/build/html/generated/yasa.SleepStaging.html",
)


# ---------------------------------------------------------------------------
# Node 2: Compute Sleep Architecture
# ---------------------------------------------------------------------------

def _execute_compute_sleep_architecture(metrics: dict, params: dict) -> dict:
    """
    Computes AASM-standard sleep architecture statistics from a hypnogram.

    Reads the ``hypnogram`` list (int-coded stages) from the input metrics
    dict produced by compute_sleep_stages, creates a YASA Hypnogram, and
    calls sleep_statistics().

    Returns a metrics dict with TIB, SPT, TST, WASO, SOL, SE, SME,
    REM latency, stage durations, and percentages.
    """
    _require_yasa()

    hypno_int = metrics.get("hypnogram")
    if hypno_int is None:
        raise ValueError(
            "Input metrics must contain a 'hypnogram' key (list of int). "
            "Connect a Compute Sleep Stages node first."
        )

    epoch_duration_s = float(metrics.get("epoch_duration_s", 30.0))

    int_to_label = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    labels = [int_to_label.get(v, "W") for v in hypno_int]

    hyp = yasa.Hypnogram(labels, freq=f"{int(epoch_duration_s)}s")
    stats = hyp.sleep_statistics()

    # Convert numpy types to native Python for JSON serialization
    result = {}
    for k, v in stats.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = None if math.isnan(float(v)) else round(float(v), 4)
        else:
            result[k] = v

    result["n_epochs"] = len(hypno_int)
    result["epoch_duration_s"] = epoch_duration_s

    return result


COMPUTE_SLEEP_ARCHITECTURE = NodeDescriptor(
    node_type="compute_sleep_architecture",
    display_name="Sleep Architecture",
    category="Sleep",
    description=(
        "Computes AASM-standard sleep architecture statistics from a hypnogram. "
        "Outputs include Time in Bed (TIB), Sleep Period Time (SPT), Total Sleep "
        "Time (TST), Wake After Sleep Onset (WASO), Sleep Efficiency, Sleep Onset "
        "Latency, REM Latency, and time/percentage in each sleep stage. "
        "Connect a Compute Sleep Stages node to provide the hypnogram input."
    ),
    tags=["sleep", "architecture", "tst", "waso", "efficiency", "sol", "rem-latency", "aasm"],
    inputs=[
        HandleSchema(id="metrics_in", type="metrics", label="Sleep Stages"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Sleep Architecture"),
    ],
    parameters=[],
    execute_fn=_execute_compute_sleep_architecture,
    code_template=lambda p: (
        "int_to_label = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}\n"
        "labels = [int_to_label.get(v, 'W') for v in hypnogram]\n"
        "hyp = yasa.Hypnogram(labels, freq='30s')\n"
        "sleep_stats = hyp.sleep_statistics()"
    ),
    methods_template=lambda p: (
        "Sleep architecture statistics (TIB, SPT, TST, WASO, sleep efficiency, "
        "sleep onset latency, and stage-wise durations) were computed from the "
        "hypnogram using YASA's sleep_statistics function (Vallat & Walker, 2021)."
    ),
    docs_url="https://raphaelvallat.com/yasa/build/html/generated/yasa.sleep_statistics.html",
)


# ---------------------------------------------------------------------------
# Node 3: Detect Spindles
# ---------------------------------------------------------------------------

def _execute_detect_spindles(input_data, params: dict) -> dict:
    """
    Detects sleep spindles using YASA's spindles_detect.

    Spindles are bursts of sigma-band (12–15 Hz) oscillatory activity
    lasting 0.5–2 seconds, characteristic of N2 sleep. They play a
    critical role in memory consolidation.

    Returns a metrics dict with spindle count, density, mean duration,
    mean frequency, and individual event details (capped at 50).
    """
    _require_yasa()

    raw = input_data.copy()

    # Guard: EEG channels required
    ch_types = set(raw.get_channel_types())
    if "eeg" not in ch_types:
        raise ValueError(
            "Spindle detection requires EEG channels. "
            f"Your data has {', '.join(sorted(ch_types))} channels only. "
            "Check that channel types are correctly set "
            "(use Set Channel Types node if needed)."
        )

    freq_min = float(params.get("freq_min", 12.0))
    freq_max = float(params.get("freq_max", 15.0))
    dur_min = float(params.get("duration_min", 0.5))
    dur_max = float(params.get("duration_max", 2.0))
    thresh_rms = float(params.get("threshold_rms", 1.5))

    sp = yasa.spindles_detect(
        raw,
        freq_sp=(freq_min, freq_max),
        duration=(dur_min, dur_max),
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": thresh_rms},
        verbose=False,
    )

    if sp is None:
        return {
            "n_spindles": 0,
            "spindles": [],
            "density_per_min": 0.0,
            "mean_duration_s": 0.0,
            "mean_frequency_hz": 0.0,
            "freq_range_hz": f"{freq_min}-{freq_max}",
        }

    summary = sp.summary()
    n_spindles = len(summary)

    # Recording duration in minutes
    duration_min = float(raw.times[-1]) / 60.0 if len(raw.times) > 0 else 1.0

    cap = 50
    spindle_list = []
    for _, row in summary.head(cap).iterrows():
        entry = {
            "onset_s": round(float(row["Start"]), 3),
            "duration_s": round(float(row["Duration"]), 3),
            "channel": str(row["Channel"]),
            "frequency_hz": round(float(row["Frequency"]), 2),
            "amplitude_uv": round(float(row["Amplitude"]), 2),
        }
        spindle_list.append(entry)

    return {
        "n_spindles": int(n_spindles),
        "density_per_min": round(n_spindles / max(duration_min, 0.001), 2),
        "mean_duration_s": round(float(summary["Duration"].mean()), 3),
        "mean_frequency_hz": round(float(summary["Frequency"].mean()), 2),
        "freq_range_hz": f"{freq_min}-{freq_max}",
        "spindles": spindle_list,
        "note": (
            f"Showing first {cap} of {n_spindles} spindles."
            if n_spindles > cap
            else f"All {n_spindles} spindles shown."
        ),
    }


DETECT_SPINDLES = NodeDescriptor(
    node_type="detect_spindles",
    display_name="Detect Spindles",
    category="Sleep",
    description=(
        "Detects sleep spindles — bursts of sigma-band (12-15 Hz) oscillatory activity "
        "lasting 0.5-2 seconds. Spindles are a hallmark of N2 sleep and play a key role "
        "in memory consolidation. Uses YASA's automated spindle detection algorithm. "
        "Connect a filtered EEG signal. Output is a metrics dict with spindle count, "
        "density, and individual event details."
    ),
    tags=["sleep", "spindle", "sigma", "n2", "memory", "detection", "yasa"],
    inputs=[
        HandleSchema(id="eeg_in", type="raw_eeg", label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Spindle Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="freq_min",
            label="Spindle Band Min",
            type="float",
            default=12.0,
            min=8.0,
            max=18.0,
            step=0.5,
            unit="Hz",
            description="Lower frequency bound for spindle detection. Standard: 12 Hz.",
            exposed=True,
        ),
        ParameterSchema(
            name="freq_max",
            label="Spindle Band Max",
            type="float",
            default=15.0,
            min=10.0,
            max=20.0,
            step=0.5,
            unit="Hz",
            description="Upper frequency bound for spindle detection. Standard: 15 Hz.",
            exposed=True,
        ),
        ParameterSchema(
            name="duration_min",
            label="Min Duration",
            type="float",
            default=0.5,
            min=0.1,
            max=2.0,
            step=0.1,
            unit="s",
            description="Minimum spindle duration in seconds. Standard: 0.5 s.",
        ),
        ParameterSchema(
            name="duration_max",
            label="Max Duration",
            type="float",
            default=2.0,
            min=0.5,
            max=5.0,
            step=0.1,
            unit="s",
            description="Maximum spindle duration in seconds. Standard: 2.0 s.",
        ),
        ParameterSchema(
            name="threshold_rms",
            label="RMS Threshold",
            type="float",
            default=1.5,
            min=0.5,
            max=5.0,
            step=0.1,
            description=(
                "RMS amplitude threshold (in standard deviations above the mean). "
                "Lower values increase sensitivity (more detections). Standard: 1.5."
            ),
            exposed=True,
        ),
    ],
    execute_fn=_execute_detect_spindles,
    code_template=lambda p: (
        f"sp = yasa.spindles_detect(\n"
        f"    raw,\n"
        f"    freq_sp=({p.get('freq_min', 12.0)}, {p.get('freq_max', 15.0)}),\n"
        f"    duration=({p.get('duration_min', 0.5)}, {p.get('duration_max', 2.0)}),\n"
        f"    thresh={{'rel_pow': 0.2, 'corr': 0.65, 'rms': {p.get('threshold_rms', 1.5)}}},\n"
        f"    verbose=False,\n"
        f")\n"
        f"spindle_summary = sp.summary() if sp is not None else None"
    ),
    methods_template=lambda p: (
        f"Sleep spindles were automatically detected using YASA's spindles_detect "
        f"function (Vallat & Walker, 2021) with a sigma band of "
        f"{p.get('freq_min', 12.0)}-{p.get('freq_max', 15.0)} Hz, "
        f"duration limits of {p.get('duration_min', 0.5)}-{p.get('duration_max', 2.0)} s, "
        f"and an RMS threshold of {p.get('threshold_rms', 1.5)} standard deviations."
    ),
    docs_url="https://raphaelvallat.com/yasa/build/html/generated/yasa.spindles_detect.html",
)


# ---------------------------------------------------------------------------
# Node 4: Detect Slow Oscillations
# ---------------------------------------------------------------------------

def _execute_detect_slow_oscillations(input_data, params: dict) -> dict:
    """
    Detects slow oscillations (slow waves) using YASA's sw_detect.

    Slow oscillations are large-amplitude, low-frequency (0.3-1.5 Hz)
    events characteristic of N3 (deep/slow-wave) sleep. They play a
    key role in memory consolidation and synaptic homeostasis.

    Returns a metrics dict with event count, density, mean duration,
    mean peak-to-peak amplitude, and individual event details (capped at 50).
    """
    _require_yasa()

    raw = input_data.copy()

    # Guard: EEG channels required
    ch_types = set(raw.get_channel_types())
    if "eeg" not in ch_types:
        raise ValueError(
            "Slow oscillation detection requires EEG channels. "
            f"Your data has {', '.join(sorted(ch_types))} channels only. "
            "Check that channel types are correctly set "
            "(use Set Channel Types node if needed)."
        )

    freq_min = float(params.get("freq_min", 0.3))
    freq_max = float(params.get("freq_max", 1.5))
    amp_neg_min = float(params.get("amp_neg_min", 40.0))
    amp_neg_max = float(params.get("amp_neg_max", 200.0))
    amp_ptp_min = float(params.get("amp_ptp_min", 75.0))
    amp_ptp_max = float(params.get("amp_ptp_max", 350.0))

    sw = yasa.sw_detect(
        raw,
        freq_sw=(freq_min, freq_max),
        amp_neg=(amp_neg_min, amp_neg_max),
        amp_ptp=(amp_ptp_min, amp_ptp_max),
        verbose=False,
    )

    if sw is None:
        return {
            "n_slow_oscillations": 0,
            "slow_oscillations": [],
            "density_per_min": 0.0,
            "mean_duration_s": 0.0,
            "mean_ptp_uv": 0.0,
            "freq_range_hz": f"{freq_min}-{freq_max}",
        }

    summary = sw.summary()
    n_sw = len(summary)

    duration_min = float(raw.times[-1]) / 60.0 if len(raw.times) > 0 else 1.0

    cap = 50
    sw_list = []
    for _, row in summary.head(cap).iterrows():
        entry = {
            "onset_s": round(float(row["Start"]), 3),
            "duration_s": round(float(row["Duration"]), 3),
            "channel": str(row["Channel"]),
            "ptp_uv": round(float(row["PTP"]), 2),
            "neg_peak_uv": round(float(row["ValNegPeak"]), 2),
        }
        sw_list.append(entry)

    return {
        "n_slow_oscillations": int(n_sw),
        "density_per_min": round(n_sw / max(duration_min, 0.001), 2),
        "mean_duration_s": round(float(summary["Duration"].mean()), 3),
        "mean_ptp_uv": round(float(summary["PTP"].mean()), 2),
        "freq_range_hz": f"{freq_min}-{freq_max}",
        "slow_oscillations": sw_list,
        "note": (
            f"Showing first {cap} of {n_sw} events."
            if n_sw > cap
            else f"All {n_sw} events shown."
        ),
    }


DETECT_SLOW_OSCILLATIONS = NodeDescriptor(
    node_type="detect_slow_oscillations",
    display_name="Detect Slow Oscillations",
    category="Sleep",
    description=(
        "Detects slow oscillations (delta slow waves) — large-amplitude, low-frequency "
        "(0.3-1.5 Hz) events characteristic of N3 deep sleep. Slow oscillations play a "
        "key role in memory consolidation and synaptic homeostasis. "
        "Uses YASA's automated slow wave detection algorithm. "
        "Connect a filtered EEG signal. Output is a metrics dict with event count, "
        "density, and individual event details."
    ),
    tags=["sleep", "slow-oscillation", "slow-wave", "delta", "n3", "deep-sleep", "yasa"],
    inputs=[
        HandleSchema(id="eeg_in", type="raw_eeg", label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Slow Osc Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="freq_min",
            label="SW Band Min",
            type="float",
            default=0.3,
            min=0.1,
            max=1.0,
            step=0.1,
            unit="Hz",
            description="Lower frequency bound for slow wave detection. Standard: 0.3 Hz.",
        ),
        ParameterSchema(
            name="freq_max",
            label="SW Band Max",
            type="float",
            default=1.5,
            min=0.5,
            max=4.0,
            step=0.1,
            unit="Hz",
            description="Upper frequency bound for slow wave detection. Standard: 1.5 Hz.",
        ),
        ParameterSchema(
            name="amp_neg_min",
            label="Min Negative Amplitude",
            type="float",
            default=40.0,
            min=10.0,
            max=200.0,
            step=5.0,
            unit="\u00b5V",
            description="Minimum negative peak amplitude. Standard: 40 \u00b5V.",
        ),
        ParameterSchema(
            name="amp_neg_max",
            label="Max Negative Amplitude",
            type="float",
            default=200.0,
            min=50.0,
            max=500.0,
            step=10.0,
            unit="\u00b5V",
            description="Maximum negative peak amplitude. Standard: 200 \u00b5V.",
        ),
        ParameterSchema(
            name="amp_ptp_min",
            label="Min Peak-to-Peak",
            type="float",
            default=75.0,
            min=20.0,
            max=300.0,
            step=5.0,
            unit="\u00b5V",
            description="Minimum peak-to-peak amplitude. Standard: 75 \u00b5V.",
            exposed=True,
        ),
        ParameterSchema(
            name="amp_ptp_max",
            label="Max Peak-to-Peak",
            type="float",
            default=350.0,
            min=100.0,
            max=1000.0,
            step=10.0,
            unit="\u00b5V",
            description="Maximum peak-to-peak amplitude. Standard: 350 \u00b5V.",
        ),
    ],
    execute_fn=_execute_detect_slow_oscillations,
    code_template=lambda p: (
        f"sw = yasa.sw_detect(\n"
        f"    raw,\n"
        f"    freq_sw=({p.get('freq_min', 0.3)}, {p.get('freq_max', 1.5)}),\n"
        f"    amp_neg=({p.get('amp_neg_min', 40.0)}, {p.get('amp_neg_max', 200.0)}),\n"
        f"    amp_ptp=({p.get('amp_ptp_min', 75.0)}, {p.get('amp_ptp_max', 350.0)}),\n"
        f"    verbose=False,\n"
        f")\n"
        f"sw_summary = sw.summary() if sw is not None else None"
    ),
    methods_template=lambda p: (
        f"Slow oscillations were detected using YASA's sw_detect function "
        f"(Vallat & Walker, 2021) with a frequency band of "
        f"{p.get('freq_min', 0.3)}-{p.get('freq_max', 1.5)} Hz, "
        f"negative amplitude limits of {p.get('amp_neg_min', 40.0)}-{p.get('amp_neg_max', 200.0)} uV, "
        f"and peak-to-peak amplitude limits of {p.get('amp_ptp_min', 75.0)}-{p.get('amp_ptp_max', 350.0)} uV."
    ),
    docs_url="https://raphaelvallat.com/yasa/build/html/generated/yasa.sw_detect.html",
)


# ---------------------------------------------------------------------------
# Node 5: Plot Hypnogram
# ---------------------------------------------------------------------------

def _execute_plot_hypnogram(metrics: dict, params: dict) -> str:
    """
    Plots a hypnogram visualization from sleep staging results.

    Reads the ``hypnogram`` key (list of int) from the input metrics dict.
    Convention: W on top, REM next, then N1 → N3 going down.
    """
    hypno_int = metrics.get("hypnogram")
    if hypno_int is None:
        raise ValueError(
            "Input metrics must contain a 'hypnogram' key (list of int). "
            "Connect a Compute Sleep Stages node first."
        )

    epoch_dur = float(metrics.get("epoch_duration_s", 30.0))
    show_stats = bool(params.get("show_stats", True))

    n_epochs = len(hypno_int)
    times_min = np.arange(n_epochs) * epoch_dur / 60.0

    # Map int stages to y-axis positions: W=4(top), R=3, N1=2, N2=1, N3=0(bottom)
    stage_y_map = {0: 4, 1: 2, 2: 1, 3: 0, 4: 3}
    y_values = [stage_y_map.get(s, -1) for s in hypno_int]

    # Stage colors
    stage_colors = {
        0: "#e74c3c",  # Wake — red
        1: "#3498db",  # N1 — light blue
        2: "#2980b9",  # N2 — blue
        3: "#1a5276",  # N3 — dark blue
        4: "#27ae60",  # REM — green
    }

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    # Colored step plot
    for i in range(len(times_min) - 1):
        ax.fill_between(
            [times_min[i], times_min[i + 1]],
            [y_values[i], y_values[i]],
            alpha=0.7,
            color=stage_colors.get(hypno_int[i], "#999999"),
            step="post",
        )
    ax.step(
        times_min, y_values, where="post",
        color="white", linewidth=0.8, alpha=0.9,
    )

    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["N3", "N2", "N1", "REM", "Wake"], color="white", fontsize=10)
    ax.set_xlabel("Time (minutes)", color="white", fontsize=10)
    ax.set_title("Hypnogram", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, times_min[-1] if n_epochs > 1 else 1)
    ax.set_ylim(-0.5, 4.5)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="x", color="#334155", alpha=0.3)

    # Add stage distribution annotation
    if show_stats and "stage_counts" in metrics:
        counts = metrics["stage_counts"]
        total = sum(counts.values())
        if total > 0:
            parts = [
                f"{stage}: {count} ({count / total * 100:.0f}%)"
                for stage, count in counts.items()
                if count > 0
            ]
            ax.text(
                0.5, -0.25, " | ".join(parts),
                transform=ax.transAxes, ha="center", va="top",
                color="#94a3b8", fontsize=8,
            )

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=100, bbox_inches="tight",
        facecolor=fig.get_facecolor(), edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


PLOT_HYPNOGRAM = NodeDescriptor(
    node_type="plot_hypnogram",
    display_name="Plot Hypnogram",
    category="Sleep",
    description=(
        "Visualizes the sleep hypnogram — a timeline showing sleep stages over the "
        "recording. Wake is at the top, followed by REM, N1, N2, and N3 (deep sleep) "
        "at the bottom. Color-coded by stage for easy interpretation. "
        "Connect a Compute Sleep Stages node to provide the hypnogram data."
    ),
    tags=["sleep", "hypnogram", "plot", "visualization", "staging"],
    inputs=[
        HandleSchema(id="metrics_in", type="metrics", label="Sleep Stages"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Hypnogram Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="show_stats",
            label="Show Stage Stats",
            type="bool",
            default=True,
            description="Show stage distribution statistics below the hypnogram.",
        ),
    ],
    execute_fn=_execute_plot_hypnogram,
    code_template=lambda p: (
        "int_to_label = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}\n"
        "labels = [int_to_label.get(v, 'W') for v in hypnogram]\n"
        "hyp = yasa.Hypnogram(labels, freq='30s')\n"
        "hyp.plot_hypnogram()"
    ),
    methods_template=None,
    docs_url="https://raphaelvallat.com/yasa/build/html/generated/yasa.plot_hypnogram.html",
)
