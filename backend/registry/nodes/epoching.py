"""
backend/registry/nodes/epoching.py

Epoching node types: nodes that segment continuous EEG into trials (epochs).

These nodes consume filtered_eeg and produce epochs objects for downstream
ERP analysis, time-frequency analysis, or visualization.

execute_fn contract:
  - Always copy — never mutate input.
  - Use verbose=False on all MNE calls.
  - Return a new MNE object (Epochs).
"""

from __future__ import annotations

import logging

import mne

logger = logging.getLogger(__name__)

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Epoch by Events
# ---------------------------------------------------------------------------

def _execute_epoch_by_events(raw_in: mne.io.BaseRaw, params: dict) -> mne.Epochs:
    """
    Finds stimulus events in the recording and extracts fixed-length epochs
    around each event of the specified type.

    Supports two event encoding strategies:
      1. STI (stimulus) channel — older EDF files. mne.find_events() reads this.
      2. EDF+ annotations — used by PhysioNet, OpenNeuro, BIDS, and most modern
         EDF+ files. mne.events_from_annotations() reads these.

    Auto-detects: tries find_events first; if no STI channel is present (raises
    ValueError), falls back to events_from_annotations. Both paths produce an
    identical (n_events, 3) ndarray so downstream mne.Epochs() is unaffected.

    event_id can be:
      - An integer string like "1" or "2" (backward compatible)
      - An annotation label string like "T1" or "769" (EDF+ / BIDS files)
    When a label is given, the integer code is resolved via events_from_annotations.

    baseline is applied within MNE.Epochs to avoid an extra node for the
    most common case; the researcher can set tmin/tmax to (None, None) to
    skip baseline correction.
    """
    raw = raw_in.copy()  # copy-on-write contract
    event_id_param = str(params["event_id"]).strip()

    # Resolve events (STI channel or EDF+ annotations)
    try:
        events = mne.find_events(raw, verbose=False)
        events_map = None
    except ValueError:
        # No STI channel — use EDF+ annotations
        events, events_map = mne.events_from_annotations(raw, verbose=False)

    # Resolve event_id: integer string or annotation label
    try:
        event_id_value: int | dict = int(event_id_param)
    except ValueError:
        # It's an annotation label — look up the integer mapping
        if events_map is None:
            _, events_map = mne.events_from_annotations(raw, verbose=False)
        if event_id_param not in events_map:
            available = list(events_map.keys())
            raise ValueError(
                f"Event label '{event_id_param}' not found in this recording. "
                f"Available annotation labels: {available}. "
                f"Labels are shown in the session info panel after loading."
            )
        event_id_value = events_map[event_id_param]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_value,
        tmin=float(params["tmin"]),
        tmax=float(params["tmax"]),
        baseline=(float(params["baseline_tmin"]), float(params["baseline_tmax"])),
        preload=True,
        verbose=False,
    )
    n_dropped = len(epochs.drop_log) - len(epochs) if hasattr(epochs, 'drop_log') else 0
    if n_dropped > 0:
        logger.info("Epoch by events: %d/%d epochs dropped due to BAD_ annotations", n_dropped, len(epochs.drop_log))
    return epochs


EPOCH_BY_EVENTS = NodeDescriptor(
    node_type="epoch_by_events",
    display_name="Epoch by Events",
    category="Epoching",
    description=(
        "Segments the continuous EEG recording into fixed-length trials (epochs) "
        "locked to stimulus events. Automatically detects event encoding: reads from "
        "the STI (stimulus) channel for older EDF files, or from EDF+ annotations for "
        "modern files (PhysioNet, OpenNeuro, BIDS, and most clinical recordings). "
        "The Event ID field accepts either an integer code or an annotation label "
        "string (e.g. 'T1', '769') shown in the session info panel. "
        "Each epoch spans from tmin seconds before to tmax seconds after the event. "
        "Baseline correction is applied over the pre-stimulus interval by default. "
        "Connect the output to Compute Evoked for ERP analysis or to a "
        "time-frequency node for spectral analysis."
    ),
    tags=["epoch", "events", "erp", "segmentation", "stimulus", "epoching"],
    inputs=[
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="event_id",
            label="Event ID",
            type="string",
            default="1",
            description=(
                "Event to epoch around. Enter the annotation label exactly as shown "
                "in the session info panel (e.g. 'T1', '769', 'stimulus/standard') "
                "or an integer event code. The label is case-sensitive. "
                "Annotation labels are shown as coloured chips in the EDF Loader "
                "node after loading the file."
            ),
        ),
        ParameterSchema(
            name="tmin",
            label="Epoch Start",
            type="float",
            default=-0.2,
            min=-5.0,
            max=0.0,
            step=0.05,
            unit="s",
            description=(
                "Start time of each epoch relative to the event marker. "
                "-0.2 s (200 ms before the event) is the standard pre-stimulus "
                "baseline window for visual and auditory ERP paradigms."
            ),
        ),
        ParameterSchema(
            name="tmax",
            label="Epoch End",
            type="float",
            default=0.8,
            min=0.0,
            max=5.0,
            step=0.05,
            unit="s",
            description=(
                "End time of each epoch relative to the event marker. "
                "0.8 s captures the P300 component (300–600 ms post-stimulus) "
                "and late cognitive components up to 800 ms."
            ),
        ),
        ParameterSchema(
            name="baseline_tmin",
            label="Baseline Start",
            type="float",
            default=-0.2,
            min=-5.0,
            max=0.0,
            step=0.05,
            unit="s",
            description=(
                "Start of the baseline correction window. Should match or be "
                "within the epoch tmin. The baseline interval mean is subtracted "
                "from each channel to remove DC offset."
            ),
        ),
        ParameterSchema(
            name="baseline_tmax",
            label="Baseline End",
            type="float",
            default=0.0,
            min=-5.0,
            max=5.0,
            step=0.05,
            unit="s",
            description=(
                "End of the baseline correction window. "
                "Standard: 0.0 s (event onset). "
                "Post-stimulus baselines (>0) are used in some oddball and sensory paradigms."
            ),
        ),
    ],
    execute_fn=_execute_epoch_by_events,
    code_template=lambda p: f'events = mne.find_events(raw, verbose=False)\nepochs = mne.Epochs(raw, events, event_id={p.get("event_id", "1")}, tmin={p.get("tmin", -0.2)}, tmax={p.get("tmax", 0.8)}, baseline=({p.get("baseline_tmin", -0.2)}, {p.get("baseline_tmax", 0)}), preload=True, verbose=False)',
    methods_template=lambda p: f'Continuous EEG data were segmented into epochs from {p.get("tmin", -0.2)} to {p.get("tmax", 0.8)} s relative to stimulus onset, with baseline correction over [{p.get("baseline_tmin", -0.2)}, {p.get("baseline_tmax", 0)}] s (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html",
)


# ---------------------------------------------------------------------------
# Baseline Correction
# ---------------------------------------------------------------------------

def _execute_baseline_correction(epochs: mne.Epochs, params: dict) -> mne.Epochs:
    """
    Applies baseline correction to existing epochs by subtracting the mean
    over a specified pre-stimulus interval from each channel.

    Use this node when epochs were created without baseline correction
    (e.g., with baseline=(None, None)) and you want to apply it separately.
    Always operates on a copy.
    """
    return epochs.copy().apply_baseline(
        baseline=(float(params["tmin"]), float(params["tmax"])),
        verbose=False,
    )


BASELINE_CORRECTION = NodeDescriptor(
    node_type="baseline_correction",
    display_name="Baseline Correction",
    category="Epoching",
    description=(
        "Applies baseline correction to epoched data by subtracting the mean "
        "amplitude over a pre-stimulus time window from each channel. "
        "This removes DC offset and slow drifts, making the ERP waveform "
        "relative to the pre-stimulus baseline level. "
        "Use this node when you want to apply or re-apply baseline correction "
        "after epoching."
    ),
    tags=["baseline", "correction", "epochs", "erp", "epoching"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Baseline-corrected Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="tmin",
            label="Baseline Start",
            type="float",
            default=-0.2,
            min=-5.0,
            max=0.0,
            step=0.05,
            unit="s",
            description=(
                "Start of the baseline interval. Must be within the epoch time range."
            ),
        ),
        ParameterSchema(
            name="tmax",
            label="Baseline End",
            type="float",
            default=0.0,
            min=-5.0,
            max=5.0,
            step=0.05,
            unit="s",
            description=(
                "End of the baseline interval. 0.0 s (event onset) is standard. "
                "Post-stimulus baselines (>0) are used in some oddball and sensory paradigms. "
                "The mean over [tmin, tmax] is subtracted from every time point."
            ),
        ),
    ],
    execute_fn=_execute_baseline_correction,
    code_template=lambda p: f'epochs = epochs.copy().apply_baseline(baseline=({p.get("tmin", -0.2)}, {p.get("tmax", 0)}), verbose=False)',
    methods_template=lambda p: f'Baseline correction was applied using the interval [{p.get("tmin", -0.2)}, {p.get("tmax", 0)}] s.',
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.apply_baseline",
)


# ---------------------------------------------------------------------------
# Reject Epochs
# ---------------------------------------------------------------------------

def _execute_filter_epochs(epochs: mne.Epochs, params: dict) -> mne.Epochs:
    """
    Applies a bandpass filter to already-epoched data.

    Useful for gamma-band analyses where filtering after epoching avoids
    edge artifacts from long filter kernels on continuous data. Also used
    to apply a second low-pass filter for ERP waveform smoothing.
    Always operates on a copy.
    """
    return epochs.copy().filter(
        l_freq=float(params["low_cutoff_hz"]),
        h_freq=float(params["high_cutoff_hz"]),
        method=str(params["method"]),
        fir_window="hamming",
        verbose=False,
    )


FILTER_EPOCHS = NodeDescriptor(
    node_type="filter_epochs",
    display_name="Filter Epochs",
    category="Epoching",
    description=(
        "Applies a bandpass filter to already-epoched data. "
        "Filtering after epoching avoids edge artifacts from long filter kernels "
        "applied to continuous data — especially important for gamma-band analyses. "
        "Also used to apply a secondary low-pass filter for ERP waveform smoothing "
        "(e.g., 30 Hz low-pass to remove high-frequency noise from averaged ERPs)."
    ),
    tags=["filter", "bandpass", "epochs", "gamma", "epoching"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Filtered Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="low_cutoff_hz",
            label="Low Cutoff",
            type="float",
            default=1.0,
            min=0.0,
            max=500.0,
            step=0.5,
            unit="Hz",
            description="High-pass cutoff. Set to 0 or None to disable high-pass.",
        ),
        ParameterSchema(
            name="high_cutoff_hz",
            label="High Cutoff",
            type="float",
            default=40.0,
            min=1.0,
            max=500.0,
            step=1.0,
            unit="Hz",
            description="Low-pass cutoff. Frequencies above this are attenuated.",
        ),
        ParameterSchema(
            name="method",
            label="Filter Method",
            type="select",
            default="fir",
            options=["fir", "iir"],
            description=(
                "FIR: recommended for EEG — linear phase, no temporal distortion. "
                "IIR: faster but introduces phase distortion."
            ),
        ),
    ],
    execute_fn=_execute_filter_epochs,
    code_template=lambda p: f'epochs = epochs.copy().filter(l_freq={p.get("low_cutoff_hz", 1.0)}, h_freq={p.get("high_cutoff_hz", 40.0)}, method="{p.get("method", "fir")}", verbose=False)',
    methods_template=lambda p: f'Epochs were bandpass filtered between {p.get("low_cutoff_hz", 1.0)} and {p.get("high_cutoff_hz", 40.0)} Hz using a {p.get("method", "fir").upper()} filter (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.filter",
)


def _execute_apply_autoreject(epochs: mne.Epochs, params: dict) -> mne.Epochs:
    """
    Automatically determines epoch rejection thresholds using cross-validation.

    Uses the autoreject library (pip install autoreject). More principled than
    a fixed peak-to-peak threshold — learns optimal thresholds per channel.
    n_interpolate and consensus control the cross-validation parameters.
    """
    try:
        from autoreject import AutoReject  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'autoreject' package is required for this node. "
            "Install it with: pip install autoreject"
        )
    n_interpolate_str = str(params.get("n_interpolate", "1, 4, 32")).strip()
    n_interpolate = [int(x.strip()) for x in n_interpolate_str.split(",") if x.strip()]
    import numpy as np
    ar = AutoReject(
        n_interpolate=np.array(n_interpolate),
        consensus=float(params["consensus"]),
        random_state=42,
        verbose=False,
    )
    return ar.fit_transform(epochs)


APPLY_AUTOREJECT = NodeDescriptor(
    node_type="apply_autoreject",
    display_name="Auto Reject Epochs",
    category="Epoching",
    description=(
        "Automatically determines epoch rejection thresholds using cross-validation. "
        "More principled than a fixed peak-to-peak threshold — the autoreject algorithm "
        "learns optimal thresholds per channel from the data itself. "
        "Requires the 'autoreject' package (pip install autoreject). "
        "This is the recommended artifact rejection method for publication-quality analyses."
    ),
    tags=["autoreject", "artifact", "reject", "epochs", "cross-validation", "epoching"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Clean Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="n_interpolate",
            label="Interpolation Counts",
            type="string",
            default="1, 4, 32",
            description=(
                "Comma-separated list of candidate interpolation counts for "
                "cross-validation. More values = better but slower. "
                "Default '1, 4, 32' is a good balance for most studies."
            ),
        ),
        ParameterSchema(
            name="consensus",
            label="Consensus",
            type="float",
            default=0.1,
            min=0.0,
            max=1.0,
            step=0.05,
            description=(
                "Fraction of channels that must exceed the threshold for an epoch "
                "to be rejected. 0.1 = 10% of channels. Lower values are more "
                "lenient; higher values only reject epochs with widespread artifacts."
            ),
        ),
    ],
    execute_fn=_execute_apply_autoreject,
    code_template=lambda p: f'from autoreject import AutoReject\nar = AutoReject(consensus={p.get("consensus", 0.1)}, random_state=42, verbose=False)\nepochs = ar.fit_transform(epochs)',
    methods_template=lambda p: f'Automated epoch rejection was performed using AutoReject (Jas et al., 2017) with consensus threshold {p.get("consensus", 0.1)}.',
    docs_url="https://autoreject.github.io/stable/",
)


def _execute_reject_epochs(epochs: mne.Epochs, params: dict) -> mne.Epochs:
    """
    Drops epochs that contain amplitude excursions above a peak-to-peak
    threshold, removing trials contaminated by blinks, muscle bursts, or
    movement artifacts that ICA did not catch.

    Uses mne.Epochs.drop_bad() with an EEG reject dict. The threshold is
    expressed in µV for human readability and converted to Volts for MNE.
    Always operates on a copy.
    """
    threshold_v = float(params["peak_to_peak_uv"]) * 1e-6
    epochs_copy = epochs.copy()
    n_before = len(epochs_copy)
    epochs_copy.drop_bad(reject={"eeg": threshold_v}, verbose=False)
    n_after = len(epochs_copy)
    logger.info("Reject epochs: %d/%d epochs dropped (threshold=%.0f µV)", n_before - n_after, n_before, float(params["peak_to_peak_uv"]))
    return epochs_copy


REJECT_EPOCHS = NodeDescriptor(
    node_type="reject_epochs",
    display_name="Reject Epochs",
    category="Epoching",
    description=(
        "Drops epochs whose peak-to-peak EEG amplitude exceeds a threshold. "
        "This removes trials contaminated by eye blinks, muscle bursts, or body "
        "movement that were not removed by filtering or ICA. "
        "The standard threshold for EEG is 100–150 µV; clinical recordings with "
        "higher noise may require 200 µV. "
        "Apply before Compute Evoked to prevent artifact-contaminated averages."
    ),
    tags=["reject", "artifact", "threshold", "epochs", "epoching", "drop"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Clean Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="peak_to_peak_uv",
            label="Rejection Threshold",
            type="float",
            default=150.0,
            min=10.0,
            max=1000.0,
            step=10.0,
            unit="µV",
            description=(
                "Peak-to-peak amplitude threshold in microvolts. Epochs where any "
                "EEG channel exceeds this value are dropped. "
                "100–150 µV is standard for cognitive ERP paradigms. "
                "Use higher values (200–300 µV) for motor or clinical recordings "
                "with naturally larger amplitudes."
            ),
        ),
    ],
    execute_fn=_execute_reject_epochs,
    code_template=lambda p: f'epochs = epochs.copy().drop_bad(reject=dict(eeg={p.get("peak_to_peak_uv", 150.0)} * 1e-6), verbose=False)',
    methods_template=lambda p: f'Epochs with peak-to-peak amplitude exceeding {p.get("peak_to_peak_uv", 150.0)} µV were rejected.',
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.drop_bad",
)


# ---------------------------------------------------------------------------
# Epoch by Time (Fixed-Length)
# ---------------------------------------------------------------------------

def _execute_epoch_by_time(raw: mne.io.BaseRaw, params: dict) -> mne.Epochs:
    """
    Segments continuous EEG into fixed-length, non-overlapping (or overlapping)
    epochs without requiring stimulus events.

    Essential for:
      - Resting-state EEG analysis (qEEG, neurofeedback, connectivity)
      - Sleep analysis (30-second staging windows, spindle detection windows)
      - BCI offline training (fixed-length motor imagery windows)
      - Connectivity analysis (requires many short epochs to estimate coherence)

    Segments overlapping with BAD_ annotations are automatically dropped by MNE.
    """
    duration = float(params["duration_s"])
    overlap = float(params["overlap_s"])
    if overlap >= duration:
        raise ValueError(
            f"Overlap ({overlap} s) must be less than epoch duration ({duration} s)."
        )
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=overlap,
        preload=True,
        verbose=False,
    )
    n_dropped = len(epochs.drop_log) - len(epochs) if hasattr(epochs, 'drop_log') else 0
    if n_dropped > 0:
        logger.info("Epoch by time: %d/%d epochs dropped due to BAD_ annotations", n_dropped, len(epochs.drop_log))
    return epochs


EPOCH_BY_TIME = NodeDescriptor(
    node_type="epoch_by_time",
    display_name="Epoch by Time (Fixed-Length)",
    category="Epoching",
    description=(
        "Segments continuous EEG into fixed-length epochs without requiring stimulus events. "
        "Use this for resting-state EEG, sleep staging (30 s windows), BCI offline training, "
        "and connectivity analysis (which needs many short epochs). "
        "Unlike Epoch by Events, no event markers are needed — the recording is simply "
        "divided into equal windows. Segments overlapping BAD_ annotations are dropped."
    ),
    tags=["epoch", "fixed", "resting", "sleep", "bci", "connectivity", "time", "epoching"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Fixed-Length Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="duration_s",
            label="Epoch Duration",
            type="float",
            default=2.0,
            min=0.1,
            max=300.0,
            step=0.5,
            unit="s",
            description=(
                "Length of each epoch in seconds. "
                "2 s: resting-state EEG, connectivity. "
                "4–8 s: BCI motor imagery trials. "
                "30 s: sleep staging (AASM standard). "
                "Shorter epochs give more trials; longer epochs give better frequency resolution."
            ),
        ),
        ParameterSchema(
            name="overlap_s",
            label="Overlap",
            type="float",
            default=0.0,
            min=0.0,
            max=299.0,
            step=0.5,
            unit="s",
            description=(
                "Overlap between consecutive epochs in seconds. "
                "0.0 (default): non-overlapping windows — maximum independence between epochs. "
                "Positive overlap: sliding window — more epochs from the same data, "
                "useful when data is short. Must be less than epoch duration."
            ),
        ),
    ],
    execute_fn=_execute_epoch_by_time,
    code_template=lambda p: f'epochs = mne.make_fixed_length_epochs(raw, duration={p.get("duration_s", 2.0)}, overlap={p.get("overlap_s", 0.0)}, preload=True, verbose=False)',
    methods_template=lambda p: f'Continuous EEG data were segmented into fixed-length epochs of {p.get("duration_s", 2.0)} s with {p.get("overlap_s", 0.0)} s overlap using MNE-Python (Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.make_fixed_length_epochs.html",
)


# ---------------------------------------------------------------------------
# Equalize Event Counts
# ---------------------------------------------------------------------------

def _execute_equalize_event_counts(epochs: mne.Epochs, params: dict) -> mne.Epochs:
    """
    Balances the number of trials across all event types by dropping excess epochs.

    Unequal trial counts bias ERP averages — conditions with more trials produce
    lower-noise evoked responses, inflating apparent amplitude differences between
    conditions. Equalizing counts is required for valid within-subject comparisons.

    method="mincount" (default): drops excess epochs randomly until all conditions
    have the same count as the least-frequent condition.
    method="truncate": drops epochs from the end of each condition list.
    """
    method = str(params["method"])
    epochs_copy = epochs.copy()
    epochs_copy.equalize_event_counts(method=method)
    return epochs_copy


EQUALIZE_EVENT_COUNTS = NodeDescriptor(
    node_type="equalize_event_counts",
    display_name="Equalize Event Counts",
    category="Epoching",
    description=(
        "Balances the number of trials across all event types by randomly dropping "
        "excess epochs from over-represented conditions. Unequal trial counts bias "
        "ERP comparisons — conditions with more trials have lower noise, making "
        "amplitude differences appear larger than they are. Place this node before "
        "Compute Evoked when comparing two or more experimental conditions."
    ),
    tags=["equalize", "balance", "trials", "conditions", "erp", "epochs", "epoching"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="epochs_out", type="epochs", label="Balanced Epochs"),
    ],
    parameters=[
        ParameterSchema(
            name="method",
            label="Drop Method",
            type="select",
            default="mintime",
            options=["mintime", "truncate", "random"],
            description=(
                "mintime (recommended): drops excess epochs while minimising the "
                "temporal distance between remaining events — reduces time-varying "
                "noise biases. "
                "truncate: drops epochs from the end of each condition's list, "
                "preserving the earliest trials. "
                "random: randomly drops excess epochs (requires random_state "
                "for reproducibility)."
            ),
        ),
    ],
    execute_fn=_execute_equalize_event_counts,
    code_template=lambda p: f'epochs.equalize_event_counts(method="{p.get("method", "mintime")}")',
    methods_template=lambda p: f'Epoch counts were equalized across conditions using the "{p.get("method", "mintime")}" method to remove trial-count bias (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.equalize_event_counts",
)
