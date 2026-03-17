"""
backend/registry/nodes/preprocessing.py

Preprocessing node types: signal conditioning operations on raw EEG.

All execute_fns in this file follow the same contract:
  - Call raw.copy() before any operation — NEVER mutate input_data.
  - Use verbose=False on all MNE calls.
  - Return the processed copy.

To add a new preprocessing node:
  1. Write _execute_<name> following the patterns below.
  2. Define the NodeDescriptor constant.
  3. Import + register in backend/registry/__init__.py.
"""

from __future__ import annotations

import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Bandpass Filter
# ---------------------------------------------------------------------------

def _execute_bandpass_filter(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Applies a FIR or IIR bandpass filter to raw EEG.

    Always operates on a copy — the input Raw object is not modified.
    fir_window is hardcoded to "hamming" for MVP; expose as a parameter
    in a future version if researchers request it.

    Raises ValueError if high_cutoff_hz >= Nyquist frequency of the recording
    (PARAM-05 fix — prevents silent above-Nyquist filter values).
    """
    nyquist = raw.info["sfreq"] / 2.0
    low = float(params["low_cutoff_hz"])
    high = float(params["high_cutoff_hz"])
    if low >= high:
        raise ValueError(
            f"Low cutoff ({low} Hz) must be less than high cutoff ({high} Hz)."
        )
    if high >= nyquist:
        raise ValueError(
            f"High cutoff ({high} Hz) must be less than the Nyquist frequency "
            f"({nyquist:.1f} Hz) for this recording (sfreq={raw.info['sfreq']} Hz). "
            "Reduce the high cutoff frequency."
        )
    return raw.copy().filter(
        l_freq=low,
        h_freq=high,
        method=str(params["method"]),
        fir_window="hamming",
        verbose=False,
    )


BANDPASS_FILTER = NodeDescriptor(
    node_type="bandpass_filter",
    display_name="Bandpass Filter",
    category="Preprocessing",
    description=(
        "Applies a bandpass filter to remove frequencies outside a specified range. "
        "The low cutoff removes slow signal drifts (high-pass component); the high "
        "cutoff removes high-frequency noise (low-pass component). "
        "Typical EEG research range: 1–40 Hz. FIR method is recommended because "
        "it has linear phase and does not introduce temporal distortion. "
        "Typically connected after a Notch Filter — apply notch filtering first to "
        "remove power line interference, then use this node to define the frequency "
        "band of interest."
    ),
    tags=["filter", "bandpass", "highpass", "lowpass", "frequency", "preprocessing"],
    inputs=[
        HandleSchema(id="eeg_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Filtered EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="low_cutoff_hz",
            label="Low Cutoff",
            type="float",
            default=1.0,
            min=0.1,
            max=100.0,
            step=0.5,
            unit="Hz",
            description=(
                "High-pass cutoff frequency. Frequencies below this value are "
                "attenuated. Use 1.0 Hz for typical EEG; 0.1 Hz for slow "
                "cortical potentials (SCPs). Must be less than the high cutoff."
            ),
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
            description=(
                "Low-pass cutoff frequency. Frequencies above this value are "
                "attenuated. 40 Hz is standard for EEG; use 100 Hz if gamma-band "
                "activity is of interest. Cannot exceed half the sampling rate "
                "(Nyquist frequency)."
            ),
        ),
        ParameterSchema(
            name="method",
            label="Filter Method",
            type="select",
            default="fir",
            options=["fir", "iir"],
            description=(
                "FIR (Finite Impulse Response): recommended for EEG. Linear phase, "
                "no temporal distortion, slightly slower. "
                "IIR (Infinite Impulse Response): faster computation but introduces "
                "phase distortion. Use only if processing speed is critical."
            ),
        ),
    ],
    execute_fn=_execute_bandpass_filter,
    code_template=lambda p: f"raw = raw.copy().filter(l_freq={p['low_cutoff_hz']}, h_freq={p['high_cutoff_hz']}, method=\"{p['method']}\", fir_window=\"hamming\", verbose=False)",
    methods_template=lambda p: f"The EEG signal was bandpass filtered between {p['low_cutoff_hz']} and {p['high_cutoff_hz']} Hz using a {p['method'].upper()} filter (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter",
)


# ---------------------------------------------------------------------------
# Notch Filter
# ---------------------------------------------------------------------------

def _execute_notch_filter(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Removes power line interference at a specific frequency.

    MNE automatically notches the fundamental frequency and its harmonics
    (e.g., 60, 120, 180 Hz for a 60 Hz notch).
    Always operates on a copy.
    """
    return raw.copy().notch_filter(
        freqs=float(params["notch_freq_hz"]),
        verbose=False,
    )


NOTCH_FILTER = NodeDescriptor(
    node_type="notch_filter",
    display_name="Notch Filter",
    category="Preprocessing",
    description=(
        "Removes power line interference (electrical noise) at a specific frequency "
        "and its harmonics. Use 60 Hz for equipment in North America, 50 Hz for "
        "Europe and most of Asia. This node should typically be applied before "
        "bandpass filtering in the pipeline."
    ),
    tags=["filter", "notch", "power-line", "60hz", "50hz", "noise", "preprocessing"],
    inputs=[
        HandleSchema(id="eeg_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Filtered EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="notch_freq_hz",
            label="Notch Frequency",
            type="float",
            default=60.0,
            min=1.0,
            max=500.0,
            step=1.0,
            unit="Hz",
            description=(
                "The fundamental frequency to remove. Use 60 for North American "
                "power lines; 50 for European. MNE automatically removes harmonics "
                "(120, 180 Hz etc. for a 60 Hz notch) up to the Nyquist frequency."
            ),
        ),
    ],
    execute_fn=_execute_notch_filter,
    code_template=lambda p: f"raw = raw.copy().notch_filter(freqs={p['notch_freq_hz']}, verbose=False)",
    methods_template=lambda p: f"Power line noise was removed using a notch filter at {p['notch_freq_hz']} Hz and its harmonics (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.notch_filter",
)


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

def _execute_resample(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Resamples EEG data to a new sampling rate.

    MNE applies an anti-aliasing low-pass filter automatically before
    downsampling. Always operates on a copy.
    """
    target = float(params["target_sfreq"])
    if target <= 0:
        raise ValueError(
            f"Target sampling rate must be positive (got {target} Hz)."
        )
    return raw.copy().resample(
        sfreq=target,
        verbose=False,
    )


RESAMPLE = NodeDescriptor(
    node_type="resample",
    display_name="Resample",
    category="Preprocessing",
    description=(
        "Changes the sampling rate of the EEG data. Downsampling (e.g., from "
        "1000 Hz to 256 Hz) reduces data size and speeds up subsequent analysis. "
        "MNE applies an automatic anti-aliasing filter before downsampling. "
        "Apply a bandpass filter upstream before resampling to avoid aliasing."
    ),
    tags=["resample", "downsample", "upsample", "sampling-rate", "preprocessing"],
    inputs=[
        HandleSchema(id="eeg_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        # Output is filtered_eeg — resampled data is processed/conditioned data.
        # compute_psd only accepts filtered_eeg, enabling the common
        # Resample → Compute PSD chain. (BUG-01 fix)
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Resampled EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="target_sfreq",
            label="Target Sampling Rate",
            type="float",
            default=256.0,
            min=1.0,
            max=10000.0,
            step=1.0,
            unit="Hz",
            description=(
                "Target sampling frequency in Hz. Common values: 256 Hz (consumer "
                "EEG devices), 512 Hz (research-grade EEG), 1000 Hz (clinical EEG). "
                "Must be at least twice the highest frequency of interest "
                "(Nyquist criterion)."
            ),
        ),
    ],
    execute_fn=_execute_resample,
    code_template=lambda p: f"raw = raw.copy().resample(sfreq={p['target_sfreq']}, verbose=False)",
    methods_template=lambda p: f"The continuous EEG data were resampled to {p['target_sfreq']} Hz (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample",
)


# ---------------------------------------------------------------------------
# Set EEG Reference
# ---------------------------------------------------------------------------

def _execute_set_eeg_reference(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Re-references EEG data to the chosen reference electrode(s).

    Average reference is the most common choice for high-density EEG.
    set_eeg_reference returns (raw, projs); we take index [0].
    Always operates on a copy.
    """
    ref = str(params["reference"]).strip()
    if ref == "average":
        ref_channels: list[str] | str = "average"
    else:
        # Support comma-separated channel names for linked references
        # e.g. "TP9, TP10" → ["TP9", "TP10"] for linked mastoids.
        parts = [r.strip() for r in ref.split(",") if r.strip()]
        ref_channels = parts  # Single element list also valid for MNE
    raw_copy = raw.copy()
    raw_copy, _ = mne.set_eeg_reference(raw_copy, ref_channels=ref_channels, verbose=False)
    return raw_copy


SET_EEG_REFERENCE = NodeDescriptor(
    node_type="set_eeg_reference",
    display_name="Set EEG Reference",
    category="Preprocessing",
    description=(
        "Re-references the EEG signal to a new reference electrode. "
        "Average reference (recommended for most analyses) subtracts the mean of all "
        "electrodes from each channel, removing common-mode noise. "
        "Linked mastoids (TP9/TP10) is the traditional reference for clinical EEG. "
        "Apply before filtering for best results."
    ),
    tags=["reference", "re-reference", "average", "mastoid", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Referenced EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="reference",
            label="Reference",
            type="string",
            default="average",
            description=(
                "Reference scheme to apply. Type one of the following: "
                "'average' — subtracts the mean of all EEG channels (recommended for "
                "high-density caps). "
                "A single electrode name (e.g. 'Cz', 'Pz', 'Fz') — common for "
                "auditory and visual ERP paradigms. "
                "Two comma-separated names (e.g. 'TP9, TP10') — linked mastoids "
                "reference, the traditional clinical EEG standard."
            ),
        ),
    ],
    execute_fn=_execute_set_eeg_reference,
    code_template=lambda p: f"raw, _ = mne.set_eeg_reference(raw.copy(), ref_channels='{p['reference']}', verbose=False)",
    methods_template=lambda p: f"The EEG data were re-referenced to {p['reference']} (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_eeg_reference",
)


# ---------------------------------------------------------------------------
# ICA Decomposition
# ---------------------------------------------------------------------------

def _execute_ica_decomposition(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Fits an ICA model and automatically removes the most prominent artifact
    components (eye movements, muscle artifacts) using MNE's EOG/ECG detection.

    In MVP, automatic component rejection is not applied — ICA is fitted and
    applied with no components excluded, providing a clean ICA-decomposed signal
    that the researcher can inspect. Manual component selection is post-MVP.

    Uses random_state=42 for reproducibility across runs.
    Always operates on a copy.
    """
    raw_copy = raw.copy()
    ica = mne.preprocessing.ICA(
        n_components=int(params["n_components"]),
        method=str(params["method"]),
        random_state=42,
        verbose=False,
    )
    ica.fit(raw_copy, verbose=False)
    # Parse optional comma-separated component exclusion list.
    excl_str = str(params.get("exclude_components", "")).strip()
    exclude = (
        [int(i.strip()) for i in excl_str.split(",") if i.strip()]
        if excl_str else []
    )
    ica.apply(raw_copy, exclude=exclude, verbose=False)
    return raw_copy


ICA_DECOMPOSITION = NodeDescriptor(
    node_type="ica_decomposition",
    display_name="ICA Decomposition",
    category="Preprocessing",
    description=(
        "Fits an Independent Component Analysis (ICA) model to separate EEG signals "
        "into statistically independent components. "
        "ICA is the standard method for removing eye blink, eye movement, and "
        "muscle artifacts. The current node fits and applies ICA without removing "
        "components — use the output to inspect the ICA-decomposed signal. "
        "Requires bandpass-filtered data (0.5–40 Hz recommended) as input."
    ),
    tags=["ica", "artifact", "independent-component", "eye-blink", "preprocessing"],
    inputs=[
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="ICA-processed EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="n_components",
            label="Components",
            type="int",
            default=20,
            min=2,
            max=64,
            step=1,
            description=(
                "Number of ICA components to compute. "
                "Typical values: 15–25 for most research EEG systems. "
                "Cannot exceed the number of channels. "
                "Set lower for faster computation and fewer artifact components to inspect."
            ),
        ),
        ParameterSchema(
            name="method",
            label="Algorithm",
            type="select",
            default="fastica",
            options=["fastica", "infomax", "picard"],
            description=(
                "FastICA: fast, widely used, recommended for most EEG labs. "
                "Infomax: the algorithm used in EEGLAB — useful for cross-lab "
                "reproducibility. "
                "Picard: newer algorithm, more reliable convergence for many datasets."
            ),
        ),
        ParameterSchema(
            name="exclude_components",
            label="Exclude Components",
            type="string",
            default="",
            description=(
                "Comma-separated zero-based indices of ICA components to remove "
                "(e.g. '0, 1'). Leave empty to apply ICA without removing any "
                "components (inspect-only mode). "
                "Component 0 is typically the strongest eye-blink artifact "
                "(large frontal topography); component 1 is often the heartbeat "
                "(frontocentral topography). Identify components by running the "
                "pipeline once without exclusions, then re-run with components listed."
            ),
        ),
    ],
    execute_fn=_execute_ica_decomposition,
    code_template=lambda p: (
        "raw = raw.copy()\n"
        f"ica = mne.preprocessing.ICA(n_components={p['n_components']}, method='{p['method']}', random_state=42, verbose=False)\n"
        "ica.fit(raw, verbose=False)\n"
        "ica.apply(raw, exclude=[{}], verbose=False)".format(
            ", ".join(
                i.strip() for i in str(p.get("exclude_components", "")).split(",") if i.strip()
            )
        )
    ),
    methods_template=lambda p: f"Independent component analysis ({p['method']}, {p['n_components']} components) was performed to identify and remove artifactual sources (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.preprocessing.ICA.html",
)


# ---------------------------------------------------------------------------
# Mark Bad Channels
# ---------------------------------------------------------------------------

def _execute_mark_bad_channels(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Marks specified channels as bad and interpolates them using spherical splines.

    Parses a comma-separated list of channel names (e.g., "Fp1, Fp2").
    Channels not found in the recording are silently skipped.
    Always operates on a copy.
    """
    raw_copy = raw.copy()
    bad_str = str(params.get("bad_channels", "")).strip()
    if bad_str:
        bads = [ch.strip() for ch in bad_str.split(",") if ch.strip()]
        # Case-insensitive lookup: maps lowercase channel name → actual name.
        # Resolves "FP1" → "Fp1", "CZ" → "Cz", etc.
        ch_map = {ch.lower(): ch for ch in raw_copy.ch_names}
        bads_not_found = [b for b in bads if b.lower() not in ch_map]
        if bads_not_found:
            raise ValueError(
                f"Channel(s) not found in this recording: {bads_not_found}. "
                f"Check the channel names in the session info panel. "
                f"Channel names are case-insensitive."
            )
        valid_bads = [ch_map[b.lower()] for b in bads]
        raw_copy.info["bads"] = valid_bads
        if valid_bads:
            raw_copy.interpolate_bads(verbose=False)
    return raw_copy


MARK_BAD_CHANNELS = NodeDescriptor(
    node_type="mark_bad_channels",
    display_name="Mark Bad Channels",
    category="Preprocessing",
    description=(
        "Marks specified channels as bad and interpolates them using spherical spline "
        "interpolation. Bad channels are electrodes with poor contact, bridging, or "
        "persistent noise that cannot be removed by filtering. "
        "Enter the channel names to reject as a comma-separated list (e.g., Fp1, Fp2). "
        "Only channels that exist in the recording are affected."
    ),
    tags=["bad-channels", "interpolate", "artifact", "electrode", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Interpolated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="bad_channels",
            label="Bad Channels",
            type="string",
            default="",
            description=(
                "Comma-separated list of channel names to mark as bad and interpolate. "
                "Example: Fp1, Fp2, AF7. "
                "Channels not found in the recording are ignored."
            ),
            channel_hint="multi",
        ),
    ],
    execute_fn=_execute_mark_bad_channels,
    code_template=lambda p: f"raw = raw.copy()\nraw.info['bads'] = [ch.strip() for ch in '{p['bad_channels']}'.split(',') if ch.strip()]\nraw.interpolate_bads(verbose=False)",
    methods_template=lambda p: f"Channels {p['bad_channels']} were marked as bad and reconstructed using spherical spline interpolation (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html",
)


# ---------------------------------------------------------------------------
# Crop Recording
# ---------------------------------------------------------------------------

def _execute_crop(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Extracts a time segment from the recording.

    tmax is clamped to the recording's last sample time to prevent an
    MNE error when the requested end time exceeds the recording length.
    Always operates on a copy.
    """
    tmin = float(params["tmin"])
    tmax = float(params["tmax"])
    tmax = min(tmax, raw.times[-1])  # Clamp to recording length
    return raw.copy().crop(tmin=tmin, tmax=tmax, verbose=False)


CROP = NodeDescriptor(
    node_type="crop",
    display_name="Crop Recording",
    category="Preprocessing",
    description=(
        "Extracts a time segment from the recording, discarding the rest. "
        "Use this to focus on a specific task block or rest period within a longer "
        "recording. Cropping early in the pipeline speeds up all downstream "
        "computations. "
        "If tmax exceeds the recording duration, the recording is kept to the end."
    ),
    tags=["crop", "trim", "segment", "time", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out",      type="raw_eeg",      label="Raw EEG (cropped)"),
        HandleSchema(id="filtered_out", type="filtered_eeg", label="Filtered EEG (cropped)"),
    ],
    parameters=[
        ParameterSchema(
            name="tmin",
            label="Start Time",
            type="float",
            default=0.0,
            min=0.0,
            max=86400.0,
            step=1.0,
            unit="s",
            description=(
                "Start time of the segment to keep, in seconds from recording onset. "
                "0.0 keeps from the beginning."
            ),
        ),
        ParameterSchema(
            name="tmax",
            label="End Time",
            type="float",
            default=60.0,
            min=0.0,
            max=86400.0,
            step=1.0,
            unit="s",
            description=(
                "End time of the segment to keep, in seconds from recording onset. "
                "If larger than the recording duration, the full remaining recording is kept."
            ),
        ),
    ],
    execute_fn=_execute_crop,
    code_template=lambda p: f"raw = raw.copy().crop(tmin={p['tmin']}, tmax={p['tmax']}, verbose=False)",
    methods_template=lambda p: f"The recording was cropped to the time window from {p['tmin']} to {p['tmax']} seconds (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.crop",
)


# ---------------------------------------------------------------------------
# Pick Channels
# ---------------------------------------------------------------------------

def _execute_pick_channels(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Selects a subset of channels by type.

    Returns a copy containing only the specified channel type.
    Useful to isolate EEG channels when the file also contains EOG, ECG,
    or other sensor types that should not pass to downstream analysis nodes.
    """
    return raw.copy().pick(picks=str(params["channel_type"]), verbose=False)


PICK_CHANNELS = NodeDescriptor(
    node_type="pick_channels",
    display_name="Pick Channels",
    category="Preprocessing",
    description=(
        "Selects a subset of channels by type, dropping all others. "
        "Use this node when your EEG file contains multiple sensor types "
        "(e.g., EEG + EOG + ECG) and downstream nodes should only receive EEG data. "
        "The output contains only the selected channel type."
    ),
    tags=["pick", "select", "channels", "eeg", "eog", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Selected Channels"),
    ],
    parameters=[
        ParameterSchema(
            name="channel_type",
            label="Channel Type",
            type="select",
            default="eeg",
            options=["eeg", "eog", "grad", "mag"],
            description=(
                "EEG: standard scalp electrodes — select for most analyses. "
                "EOG: electro-oculogram channels (eye movement monitoring). "
                "Grad/Mag: MEG gradiometers and magnetometers (MEG systems only)."
            ),
        ),
    ],
    execute_fn=_execute_pick_channels,
    code_template=lambda p: f"raw = raw.copy().pick(picks='{p['channel_type']}', verbose=False)",
    methods_template=lambda p: f"Only {p['channel_type'].upper()} channels were retained for subsequent analysis (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.pick_channels",
)


# ---------------------------------------------------------------------------
# Set Channel Types
# ---------------------------------------------------------------------------

def _execute_set_channel_types(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Reassign channel types for specified channels.

    Use this when your file has mislabeled channels (e.g., EOG channels
    marked as EEG, or biosignal channels marked as misc). Correct channel
    types ensure downstream nodes like ICA and Pick Channels work properly.
    """
    mapping_str = str(params.get("mapping", "")).strip()
    if not mapping_str:
        return raw.copy()  # No changes requested — passthrough

    VALID_TYPES = {
        "eeg", "eog", "ecg", "emg", "misc", "stim", "bio", "resp",
        "seeg", "dbs", "ecog", "hbo", "hbr", "fnirs_cw_amplitude",
        "fnirs_od", "meg", "grad", "mag", "ref_meg",
    }

    mapping: dict[str, str] = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Invalid mapping '{pair}'. Use format: CHANNEL_NAME=type. "
                f"Example: EOG1=eog, EMG1=emg"
            )
        ch, ch_type = pair.split("=", 1)
        ch, ch_type = ch.strip(), ch_type.strip().lower()
        if ch_type not in VALID_TYPES:
            raise ValueError(
                f"Unknown channel type '{ch_type}' for channel '{ch}'. "
                f"Valid types: {', '.join(sorted(VALID_TYPES))}"
            )
        mapping[ch] = ch_type

    raw = raw.copy()

    # Validate channel names exist
    missing = [ch for ch in mapping if ch not in raw.ch_names]
    if missing:
        available = ", ".join(raw.ch_names[:20])
        suffix = "..." if len(raw.ch_names) > 20 else ""
        raise ValueError(
            f"Channel(s) not found in data: {', '.join(missing)}. "
            f"Available channels: {available}{suffix}"
        )

    raw.set_channel_types(mapping, verbose=False)
    return raw


SET_CHANNEL_TYPES = NodeDescriptor(
    node_type="set_channel_types",
    display_name="Set Channel Types",
    category="Preprocessing",
    description=(
        "Reassign channel types for specific channels. Use this when your file "
        "has mislabeled channels (e.g., EOG channels marked as EEG, or biosignal "
        "channels marked as misc). Correct types ensure downstream nodes like "
        "ICA and Pick Channels work properly. Check the channel type table in "
        "the right panel after uploading a file to see current assignments."
    ),
    tags=["channel", "type", "eog", "ecg", "emg", "misc", "retype", "fix", "set"],
    inputs=[
        HandleSchema(id="raw_in", type="raw_eeg", label="Raw EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out", type="filtered_eeg", label="Retyped EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="mapping",
            label="Channel Type Mapping",
            type="string",
            default="",
            description=(
                "Comma-separated CHANNEL=TYPE pairs. "
                "Example: EOG1=eog, EOG2=eog, EMG1=emg, EXG1=ecg. "
                "Valid types: eeg, eog, ecg, emg, misc, stim, bio, resp."
            ),
        ),
    ],
    execute_fn=_execute_set_channel_types,
    code_template=lambda p: (
        "raw = raw.copy()\n"
        + (
            "raw.set_channel_types({"
            + ", ".join(
                f'"{k.strip()}": "{v.strip()}"'
                for pair in p.get("mapping", "").split(",")
                if "=" in pair
                for k, v in [pair.split("=", 1)]
            )
            + "}, verbose=False)"
            if p.get("mapping", "").strip()
            else "# No channel type changes"
        )
    ),
    methods_template=lambda p: (
        "Channel types were reassigned prior to processing "
        "(MNE-Python; Gramfort et al., 2013)."
    ),
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_channel_types",
)


# ---------------------------------------------------------------------------
# Set Montage
# ---------------------------------------------------------------------------

def _execute_set_montage(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Assigns 3-D electrode positions from a standard montage to the recording.

    Without a montage, MNE cannot render topographic maps (topomaps) — all
    scalp distribution plots will fail. This node should be placed immediately
    after the file loader for any pipeline that includes topomap or source
    analysis nodes.

    Before montage assignment, trailing dots in channel names are stripped
    (common in PhysioNet EDF files like eegmmidb, where "Fc5." → "FC5").

    Uses on_missing="warn" so that recordings with channels not included in
    the standard montage (e.g., EOG, EMG, reference electrodes) do not crash.
    Those channels simply have no position assigned.
    """
    raw = raw.copy()

    # Strip trailing dots from channel names (PhysioNet EDF convention).
    # "Fc5." → "Fc5", "C3.." → "C3". match_case=False handles the rest.
    rename_map = {
        name: name.rstrip(".")
        for name in raw.ch_names
        if name.rstrip(".") and name.rstrip(".") != name
    }
    # Guard against duplicates: only rename if all targets are unique
    if rename_map and len(set(rename_map.values())) == len(rename_map):
        existing = set(raw.ch_names) - set(rename_map.keys())
        if not existing.intersection(rename_map.values()):
            raw.rename_channels(rename_map)

    montage_name = str(params["montage"])
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="warn", match_case=False, verbose=False)
    return raw


SET_MONTAGE = NodeDescriptor(
    node_type="set_montage",
    display_name="Set Montage",
    category="Preprocessing",
    description=(
        "Assigns standard 3-D electrode positions (a montage) to the recording. "
        "Required before any topomap, source analysis, or interpolation node — without "
        "electrode positions, MNE cannot render scalp distributions. Choose the montage "
        "that matches the cap used during recording. Channels not in the montage "
        "(e.g., EOG, EMG) are silently skipped."
    ),
    tags=["montage", "electrodes", "positions", "10-20", "topomap", "layout", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="EEG with Montage"),
    ],
    parameters=[
        ParameterSchema(
            name="montage",
            label="Montage",
            type="select",
            default="standard_1020",
            options=[
                "standard_1020",
                "standard_1005",
                "standard_1010",
                "biosemi64",
                "biosemi128",
                "GSN-HydroCel-129",
                "easycap-M1",
            ],
            description=(
                "Standard electrode layout to assign. "
                "standard_1020: classic 10-20 system (19-21 electrodes). "
                "standard_1005: extended 10-5 system (up to 345 electrodes). "
                "biosemi64/128: BioSemi ActiveTwo standard layouts. "
                "GSN-HydroCel-129: EGI/Geodesic 128-channel cap. "
                "easycap-M1: EasyCap 32-channel layout."
            ),
        ),
    ],
    execute_fn=_execute_set_montage,
    code_template=lambda p: f"montage = mne.channels.make_standard_montage('{p['montage']}')\nraw = raw.copy()\nraw.set_montage(montage, on_missing='warn', match_case=False, verbose=False)",
    methods_template=lambda p: f"Electrode positions were assigned using the {p['montage']} standard montage (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_montage",
)


# ---------------------------------------------------------------------------
# Interpolate Bad Channels
# ---------------------------------------------------------------------------

def _execute_interpolate_bad_channels(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Reconstructs bad channels using spherical spline interpolation.

    Requires:
      1. A montage must be set (use Set Montage node upstream).
      2. Bad channels must be marked (from file header or Mark Bad Channels node).

    If no bad channels are marked, returns a copy unchanged — this is safe to
    include in any pipeline without conditional logic.

    reset_bads=True (default) clears the bads list after interpolation so that
    downstream nodes treat all channels as good.
    """
    raw = raw.copy()
    if not raw.info["bads"]:
        return raw  # nothing to interpolate — passthrough
    reset_bads = bool(params["reset_bads"])
    raw.interpolate_bads(reset_bads=reset_bads, verbose=False)
    return raw


INTERPOLATE_BAD_CHANNELS = NodeDescriptor(
    node_type="interpolate_bad_channels",
    display_name="Interpolate Bad Channels",
    category="Preprocessing",
    description=(
        "Reconstructs channels marked as bad using spherical spline interpolation. "
        "Requires a montage to be set upstream (electrode positions are needed for "
        "interpolation). If no channels are marked bad, the node passes data through "
        "unchanged. Place after Mark Bad Channels and Set Montage in your pipeline."
    ),
    tags=["interpolate", "bad", "channels", "repair", "spherical", "spline", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Interpolated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="reset_bads",
            label="Clear Bads List After",
            type="bool",
            default=True,
            description=(
                "When True, clears the list of bad channels after interpolation so "
                "downstream nodes treat all channels as good. Set to False to keep "
                "the bad channel labels for reference."
            ),
        ),
    ],
    execute_fn=_execute_interpolate_bad_channels,
    code_template=lambda p: f"raw = raw.copy()\nraw.interpolate_bads(reset_bads={p['reset_bads']}, verbose=False)",
    methods_template=lambda p: f"Bad channels were reconstructed using spherical spline interpolation (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads",
)


# ---------------------------------------------------------------------------
# Annotate Artifacts
# ---------------------------------------------------------------------------

def _execute_annotate_artifacts(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Adds time-segment annotations marking specific intervals as artifacts.

    Annotations added here will cause epoching nodes to reject any epoch that
    overlaps with a BAD_ annotation (MNE's standard behavior). This is the
    programmatic equivalent of manually marking segments in MNE's raw browser.

    onsets_s and durations_s are comma-separated lists of the same length.
    If durations_s contains a single value, it is applied to all onsets.

    Example: onsets_s="10.5,25.0,60.0", durations_s="2.0", description="BAD_muscle"
    → marks three 2-second muscle artifact windows at 10.5 s, 25 s, and 60 s.
    """
    raw = raw.copy()

    onsets_str = str(params.get("onsets_s", "")).strip()
    if not onsets_str:
        return raw  # no annotations requested — passthrough

    onsets = [float(t.strip()) for t in onsets_str.split(",") if t.strip()]
    durations_str = str(params.get("durations_s", "1.0")).strip()
    durations_raw = [float(d.strip()) for d in durations_str.split(",") if d.strip()]

    # If single duration given, broadcast to all onsets
    if len(durations_raw) == 1:
        durations = durations_raw * len(onsets)
    elif len(durations_raw) == len(onsets):
        durations = durations_raw
    else:
        raise ValueError(
            f"onsets_s has {len(onsets)} values but durations_s has {len(durations_raw)}. "
            "Either provide one duration (applied to all onsets) or one per onset."
        )

    description = str(params.get("description", "BAD_artifact"))
    if not description.startswith("BAD_"):
        description = f"BAD_{description}"
        logger.info("Auto-prepended BAD_ prefix → %r (required for MNE epoch rejection)", description)
    new_annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=[description] * len(onsets),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + new_annotations)
    return raw


ANNOTATE_ARTIFACTS = NodeDescriptor(
    node_type="annotate_artifacts",
    display_name="Annotate Artifacts",
    category="Preprocessing",
    description=(
        "Marks specific time segments as artifact annotations. Annotated segments "
        "starting with 'BAD_' are automatically excluded by MNE's epoching functions, "
        "so this is the cleanest way to reject bad time windows before epoching. "
        "Provide comma-separated onset times (in seconds) and duration(s). "
        "A single duration value applies to all onsets."
    ),
    tags=["annotate", "artifact", "bad", "segment", "reject", "mark", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Annotated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="onsets_s",
            label="Onset Times (s)",
            type="string",
            default="",
            description=(
                "Comma-separated start times in seconds for each artifact segment. "
                "Example: '10.5, 25.0, 60.0' marks three windows. "
                "Leave blank to skip annotation."
            ),
        ),
        ParameterSchema(
            name="durations_s",
            label="Duration(s) (s)",
            type="string",
            default="1.0",
            description=(
                "Duration(s) in seconds. A single value applies to all onsets. "
                "Or provide one duration per onset: '2.0, 3.5, 2.0'."
            ),
        ),
        ParameterSchema(
            name="description",
            label="Annotation Label",
            type="string",
            default="BAD_artifact",
            description=(
                "Label for the annotation. 'BAD_' prefix is auto-prepended if "
                "missing (required for MNE epoch rejection). "
                "Examples: 'BAD_muscle', 'BAD_movement', 'BAD_electrode_pop'."
            ),
        ),
    ],
    execute_fn=_execute_annotate_artifacts,
    code_template=lambda p: f"annotations = mne.Annotations(onset=[{p['onsets_s']}], duration=[{p['durations_s']}], description=['{p['description']}'])\nraw = raw.copy()\nraw.set_annotations(raw.annotations + annotations)",
    methods_template=lambda p: f"Time segments were manually annotated as {p['description']} for subsequent epoch rejection (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.Annotations.html",
)


# ---------------------------------------------------------------------------
# Rename Channels
# ---------------------------------------------------------------------------

import re as _re


def _execute_rename_channels(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Renames EEG channels to match standard 10-20 naming conventions.

    Supports three modes:
      - strip_prefix: Removes a prefix from all channel names.
        E.g., prefix="EEG " turns "EEG F3-Ref" → "F3-Ref".
      - strip_suffix: Removes a suffix from all channel names.
        E.g., suffix="-Ref" turns "F3-Ref" → "F3".
      - regex: Applies a regex substitution to every channel name.
        E.g., pattern="^EEG\\s+", replacement="" strips "EEG " prefix.

    Modes can be combined: prefix stripping runs first, then suffix, then regex.
    Always operates on a copy.
    """
    raw = raw.copy()
    prefix = str(params.get("strip_prefix", "")).strip()
    suffix = str(params.get("strip_suffix", "")).strip()
    pattern = str(params.get("regex_pattern", "")).strip()
    replacement = str(params.get("regex_replacement", ""))

    mapping: dict[str, str] = {}
    for ch in raw.ch_names:
        new_name = ch
        # Case-insensitive prefix/suffix stripping
        if prefix and new_name.lower().startswith(prefix.lower()):
            new_name = new_name[len(prefix):]
        if suffix and new_name.lower().endswith(suffix.lower()):
            new_name = new_name[: -len(suffix)]
        if pattern:
            new_name = _re.sub(pattern, replacement, new_name)
        new_name = new_name.strip()
        if new_name and new_name != ch:
            mapping[ch] = new_name

    # Check for duplicate target names before renaming
    if mapping:
        targets = list(mapping.values())
        dupes = [n for n in targets if targets.count(n) > 1]
        if dupes:
            raise ValueError(
                f"Renaming would produce duplicate channel names: "
                f"{sorted(set(dupes))}. Adjust prefix/suffix/regex parameters."
            )
        raw.rename_channels(mapping)
    return raw


RENAME_CHANNELS = NodeDescriptor(
    node_type="rename_channels",
    display_name="Rename Channels",
    category="Preprocessing",
    description=(
        "Renames channels to match standard naming conventions. "
        "Many EDF files use prefixed names like 'EEG F3-Ref' or 'EEG Cz' that "
        "prevent downstream nodes from recognizing standard electrode names. "
        "This node strips prefixes, suffixes, or applies regex substitution to "
        "normalize channel names. Place immediately after the file loader."
    ),
    tags=["rename", "channels", "prefix", "suffix", "normalize", "10-20", "preprocessing"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="Renamed EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="strip_prefix",
            label="Strip Prefix",
            type="string",
            default="EEG ",
            description=(
                "Remove this prefix from all channel names. "
                "Common prefixes: 'EEG ' (with trailing space), 'eeg', 'EMG '. "
                "Leave blank to skip prefix stripping."
            ),
        ),
        ParameterSchema(
            name="strip_suffix",
            label="Strip Suffix",
            type="string",
            default="",
            description=(
                "Remove this suffix from all channel names. "
                "Common suffixes: '-Ref', '-REF', '-LE' (linked ear). "
                "Leave blank to skip suffix stripping."
            ),
        ),
        ParameterSchema(
            name="regex_pattern",
            label="Regex Pattern",
            type="string",
            default="",
            description=(
                "Regular expression pattern for advanced renaming. "
                "Applied after prefix/suffix stripping. "
                "Example: '\\.' removes trailing dots from PhysioNet names."
            ),
        ),
        ParameterSchema(
            name="regex_replacement",
            label="Regex Replacement",
            type="string",
            default="",
            description=(
                "Replacement string for the regex pattern. "
                "Leave blank to delete matches. Supports backreferences (\\1, \\2)."
            ),
        ),
    ],
    execute_fn=_execute_rename_channels,
    code_template=lambda p: f"raw = raw.copy()\nraw.rename_channels(lambda ch: ch.removeprefix('{p['strip_prefix']}').removesuffix('{p['strip_suffix']}').strip())",
    methods_template=lambda p: f"Channel names were standardized by removing the prefix '{p['strip_prefix']}' and suffix '{p['strip_suffix']}' to conform to the 10-20 naming convention (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.rename_channels",
)


# ---------------------------------------------------------------------------
# Detect Bad Segments (Amplitude Threshold)
# ---------------------------------------------------------------------------

def _execute_detect_bad_segments(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Scans continuous data with a sliding window and marks segments where
    peak-to-peak amplitude exceeds a threshold as BAD_amplitude.

    Overlapping bad windows are merged into a single annotation span.
    Always operates on a copy.
    """
    raw = raw.copy()
    threshold_uv = float(params.get("threshold_uv", 150))
    threshold_v = threshold_uv * 1e-6
    window_s = float(params.get("window_s", 1.0))
    step_s = float(params.get("step_s", 0.5))

    sfreq = raw.info["sfreq"]
    window_samples = int(window_s * sfreq)
    step_samples = int(step_s * sfreq)

    data = raw.get_data(picks="eeg")
    n_samples = data.shape[1]

    bad_onsets: list[float] = []
    bad_durations: list[float] = []

    start = 0
    while start + window_samples <= n_samples:
        window_data = data[:, start:start + window_samples]
        ptp = window_data.max(axis=1) - window_data.min(axis=1)
        if ptp.max() > threshold_v:
            onset_time = start / sfreq
            if bad_onsets and (onset_time <= bad_onsets[-1] + bad_durations[-1]):
                new_end = onset_time + window_s
                bad_durations[-1] = new_end - bad_onsets[-1]
            else:
                bad_onsets.append(onset_time)
                bad_durations.append(window_s)
        start += step_samples

    if bad_onsets:
        new_annot = mne.Annotations(
            onset=bad_onsets,
            duration=bad_durations,
            description=["BAD_amplitude"] * len(bad_onsets),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + new_annot)
        logger.info(
            "Detected %d BAD_amplitude segments (threshold=%.0f µV)",
            len(bad_onsets), threshold_uv,
        )

    return raw


DETECT_BAD_SEGMENTS = NodeDescriptor(
    node_type="detect_bad_segments",
    display_name="Detect Bad Segments",
    category="Preprocessing",
    description=(
        "Scan continuous data with a sliding window and mark segments where "
        "peak-to-peak amplitude exceeds a threshold as BAD_amplitude. "
        "Overlapping bad windows are automatically merged. Use after filtering "
        "to detect residual high-amplitude artifacts before epoching."
    ),
    tags=["artifact", "detection", "amplitude"],
    inputs=[
        HandleSchema(id="raw_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out", type="filtered_eeg", label="Annotated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="threshold_uv",
            label="Threshold",
            type="float",
            default=150.0,
            min=10.0,
            max=1000.0,
            step=10.0,
            unit="µV",
            description=(
                "Peak-to-peak amplitude threshold in µV. Any sliding window "
                "where any channel exceeds this value is marked BAD_amplitude. "
                "Typical values: 100–200 µV for scalp EEG."
            ),
        ),
        ParameterSchema(
            name="window_s",
            label="Window Length",
            type="float",
            default=1.0,
            min=0.1,
            max=30.0,
            step=0.1,
            unit="s",
            description=(
                "Length of the sliding window in seconds. Shorter windows "
                "detect brief transients; longer windows catch sustained artifacts."
            ),
        ),
        ParameterSchema(
            name="step_s",
            label="Step Size",
            type="float",
            default=0.5,
            min=0.01,
            max=10.0,
            step=0.1,
            unit="s",
            description=(
                "Step size for the sliding window in seconds. Smaller steps "
                "give finer-grained detection but take longer to compute."
            ),
        ),
    ],
    execute_fn=_execute_detect_bad_segments,
    code_template=lambda p: f"# Sliding-window peak-to-peak artifact detection (threshold={p['threshold_uv']} uV, window={p['window_s']} s, step={p['step_s']} s)\n# Segments exceeding threshold are annotated as BAD_amplitude",
    methods_template=lambda p: f"Automated artifact detection was performed using a sliding window of {p['window_s']} s with a peak-to-peak amplitude threshold of {p['threshold_uv']} uV; segments exceeding this threshold were annotated for rejection (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.Annotations.html",
)


# ---------------------------------------------------------------------------
# Detect Flatline
# ---------------------------------------------------------------------------

def _execute_detect_flatline(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Detects flat (near-zero variance) channels and/or time segments.

    Flat channels are marked as bad in raw.info['bads'].
    Flat time segments are annotated as BAD_flatline.
    Always operates on a copy.
    """
    raw = raw.copy()
    min_std_uv = float(params.get("min_std_uv", 0.5))
    min_std_v = min_std_uv * 1e-6
    window_s = float(params.get("window_s", 5.0))
    mark_channels = bool(params.get("mark_channels", True))
    mark_segments = bool(params.get("mark_segments", True))

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_names = [raw.ch_names[i] for i in eeg_picks]
    data = raw.get_data(picks="eeg")
    sfreq = raw.info["sfreq"]

    if mark_channels:
        ch_stds = data.std(axis=1)
        flat_channels = [eeg_names[i] for i, s in enumerate(ch_stds) if s < min_std_v]
        if flat_channels:
            raw.info["bads"] = list(set(raw.info["bads"]) | set(flat_channels))
            logger.info(
                "Marked %d flat channels as bad: %s",
                len(flat_channels), flat_channels,
            )

    if mark_segments:
        window_samples = int(window_s * sfreq)
        n_samples = data.shape[1]
        bad_onsets: list[float] = []
        bad_durations: list[float] = []

        start = 0
        while start + window_samples <= n_samples:
            window_data = data[:, start:start + window_samples]
            window_stds = window_data.std(axis=1)
            if window_stds.min() < min_std_v:
                onset_time = start / sfreq
                if bad_onsets and (onset_time <= bad_onsets[-1] + bad_durations[-1]):
                    bad_durations[-1] = onset_time + window_s - bad_onsets[-1]
                else:
                    bad_onsets.append(onset_time)
                    bad_durations.append(window_s)
            start += window_samples

        if bad_onsets:
            new_annot = mne.Annotations(
                onset=bad_onsets,
                duration=bad_durations,
                description=["BAD_flatline"] * len(bad_onsets),
                orig_time=raw.annotations.orig_time,
            )
            raw.set_annotations(raw.annotations + new_annot)
            logger.info("Detected %d BAD_flatline segments", len(bad_onsets))

    return raw


DETECT_FLATLINE = NodeDescriptor(
    node_type="detect_flatline",
    display_name="Detect Flatline",
    category="Preprocessing",
    description=(
        "Detect flat (near-zero variance) channels and/or time segments. "
        "Flat channels are marked as bad; flat segments are annotated as BAD_flatline. "
        "A channel or segment is considered flat when its standard deviation falls "
        "below the minimum threshold. Place after filtering to catch dead electrodes."
    ),
    tags=["artifact", "detection", "flatline"],
    inputs=[
        HandleSchema(id="raw_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out", type="filtered_eeg", label="Annotated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="min_std_uv",
            label="Minimum Std Dev",
            type="float",
            default=0.5,
            min=0.01,
            max=50.0,
            step=0.1,
            unit="µV",
            description=(
                "Minimum standard deviation in µV. Channels or windows with "
                "std below this threshold are flagged as flat."
            ),
        ),
        ParameterSchema(
            name="window_s",
            label="Window Length",
            type="float",
            default=5.0,
            min=0.5,
            max=60.0,
            step=0.5,
            unit="s",
            description=(
                "Length of the sliding window for segment-level flatline detection. "
                "Longer windows reduce false positives from brief pauses."
            ),
        ),
        ParameterSchema(
            name="mark_channels",
            label="Mark Flat Channels",
            type="bool",
            default=True,
            description=(
                "When True, channels whose overall standard deviation is below the "
                "threshold are added to the bad channels list."
            ),
        ),
        ParameterSchema(
            name="mark_segments",
            label="Mark Flat Segments",
            type="bool",
            default=True,
            description=(
                "When True, time segments where any channel falls below the "
                "threshold are annotated as BAD_flatline."
            ),
        ),
    ],
    execute_fn=_execute_detect_flatline,
    code_template=lambda p: f"# Flatline detection (min_std={p['min_std_uv']} uV, window={p['window_s']} s)\n# Flat channels marked as bad; flat segments annotated as BAD_flatline",
    methods_template=lambda p: f"Flatline detection was applied with a minimum standard deviation threshold of {p['min_std_uv']} uV over {p['window_s']}-second windows; flat channels were marked as bad and flat segments were annotated for rejection (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html",
)


# ---------------------------------------------------------------------------
# Detect Bad Gradient (Sudden Jumps)
# ---------------------------------------------------------------------------

def _execute_detect_bad_gradient(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Detects sudden amplitude jumps by computing the sample-to-sample derivative.

    Segments where the maximum gradient exceeds the threshold are annotated
    as BAD_jump. Always operates on a copy.
    """
    raw = raw.copy()
    max_grad = float(params.get("max_gradient_uv_per_ms", 10.0))
    max_grad_v_per_s = max_grad * 1e-6 * 1000  # µV/ms → V/s
    window_s = float(params.get("window_s", 0.5))

    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks="eeg")
    gradient = np.abs(np.diff(data, axis=1)) * sfreq  # V/s

    window_samples = int(window_s * sfreq)
    n_samples = gradient.shape[1]
    bad_onsets: list[float] = []
    bad_durations: list[float] = []

    start = 0
    while start + window_samples <= n_samples:
        window_grad = gradient[:, start:start + window_samples]
        if window_grad.max() > max_grad_v_per_s:
            onset_time = start / sfreq
            if bad_onsets and (onset_time <= bad_onsets[-1] + bad_durations[-1]):
                bad_durations[-1] = onset_time + window_s - bad_onsets[-1]
            else:
                bad_onsets.append(onset_time)
                bad_durations.append(window_s)
        start += window_samples

    if bad_onsets:
        new_annot = mne.Annotations(
            onset=bad_onsets,
            duration=bad_durations,
            description=["BAD_jump"] * len(bad_onsets),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + new_annot)
        logger.info(
            "Detected %d BAD_jump segments (threshold=%.1f µV/ms)",
            len(bad_onsets), max_grad,
        )

    return raw


DETECT_BAD_GRADIENT = NodeDescriptor(
    node_type="detect_bad_gradient",
    display_name="Detect Bad Gradient",
    category="Preprocessing",
    description=(
        "Detect sudden amplitude jumps by checking the sample-to-sample derivative. "
        "Segments exceeding the gradient threshold are marked as BAD_jump. "
        "Useful for catching electrode pops, cable movements, and other transient "
        "artifacts that produce sharp voltage changes."
    ),
    tags=["artifact", "detection", "gradient", "jump"],
    inputs=[
        HandleSchema(id="raw_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out", type="filtered_eeg", label="Annotated EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="max_gradient_uv_per_ms",
            label="Max Gradient",
            type="float",
            default=10.0,
            min=0.1,
            max=500.0,
            step=1.0,
            unit="µV/ms",
            description=(
                "Maximum allowed gradient in µV/ms. Sample-to-sample voltage "
                "changes exceeding this value trigger a BAD_jump annotation. "
                "Lower values catch subtler jumps but may over-flag."
            ),
        ),
        ParameterSchema(
            name="window_s",
            label="Window Length",
            type="float",
            default=0.5,
            min=0.1,
            max=10.0,
            step=0.1,
            unit="s",
            description=(
                "Length of the analysis window in seconds. Adjacent bad windows "
                "are merged into a single annotation."
            ),
        ),
    ],
    execute_fn=_execute_detect_bad_gradient,
    code_template=lambda p: f"# Gradient-based artifact detection (max_gradient={p['max_gradient_uv_per_ms']} uV/ms, window={p['window_s']} s)\n# Segments with sudden jumps are annotated as BAD_jump",
    methods_template=lambda p: f"Gradient-based artifact detection was performed with a maximum allowable gradient of {p['max_gradient_uv_per_ms']} uV/ms over {p['window_s']}-second windows; segments exceeding this threshold were annotated as BAD_jump (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html",
)


# ---------------------------------------------------------------------------
# Filter / Remove Annotations
# ---------------------------------------------------------------------------

def _execute_filter_annotations(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Removes or keeps annotations matching a description pattern.

    Two modes:
      - remove_pattern: Remove annotations whose description matches the regex.
      - keep_pattern: Keep ONLY annotations whose description matches the regex
        (remove everything else). If both are provided, remove_pattern runs first.

    Always operates on a copy.
    """
    import re
    raw = raw.copy()
    remove_pattern = str(params.get("remove_pattern", "")).strip()
    keep_pattern = str(params.get("keep_pattern", "")).strip()

    if not remove_pattern and not keep_pattern:
        return raw  # nothing to do

    annotations = raw.annotations
    if len(annotations) == 0:
        return raw

    # Collect annotations to keep
    kept_onsets = []
    kept_durations = []
    kept_descriptions = []

    for ann in annotations:
        desc = str(ann["description"])
        onset = float(ann["onset"])
        duration = float(ann["duration"])

        # Step 1: Check remove_pattern
        if remove_pattern and re.search(remove_pattern, desc):
            continue  # skip this annotation

        # Step 2: Check keep_pattern (if set, only keep matching)
        if keep_pattern and not re.search(keep_pattern, desc):
            continue  # skip — doesn't match keep pattern

        kept_onsets.append(onset)
        kept_durations.append(duration)
        kept_descriptions.append(desc)

    n_removed = len(annotations) - len(kept_onsets)

    if n_removed > 0:
        if kept_onsets:
            new_annotations = mne.Annotations(
                onset=kept_onsets,
                duration=kept_durations,
                description=kept_descriptions,
            )
            raw.set_annotations(new_annotations)
        else:
            raw.set_annotations(mne.Annotations([], [], []))
        logger.info("Filtered annotations: removed %d, kept %d", n_removed, len(kept_onsets))

    return raw


FILTER_ANNOTATIONS = NodeDescriptor(
    node_type="filter_annotations",
    display_name="Filter Annotations",
    category="Preprocessing",
    description=(
        "Remove or keep annotations by description pattern. Use to undo overly "
        "aggressive auto-detection (e.g., remove all BAD_flatline) or to keep only "
        "specific annotation types before epoching."
    ),
    tags=["annotation", "filter", "remove", "cleanup"],
    inputs=[
        HandleSchema(id="raw_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="raw_out", type="filtered_eeg", label="Filtered EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="remove_pattern",
            label="Remove Pattern",
            type="string",
            default="",
            description=(
                "Regex pattern — annotations whose description matches are REMOVED. "
                "Examples: 'BAD_flatline' (exact), 'BAD_.*' (all BAD_ annotations), "
                "'BAD_(flatline|jump)' (specific types). Leave blank to skip."
            ),
        ),
        ParameterSchema(
            name="keep_pattern",
            label="Keep Pattern",
            type="string",
            default="",
            description=(
                "Regex pattern — only annotations whose description matches are KEPT. "
                "All others are removed. Applied after remove_pattern. "
                "Leave blank to skip (keeps everything not already removed)."
            ),
        ),
    ],
    execute_fn=_execute_filter_annotations,
    code_template=lambda p: f"raw = raw.copy()\n# Filter annotations: remove_pattern='{p['remove_pattern']}', keep_pattern='{p['keep_pattern']}'\nraw.set_annotations(filtered_annotations)",
    methods_template=lambda p: f"Annotations were filtered by removing entries matching '{p['remove_pattern']}' and retaining entries matching '{p['keep_pattern']}' prior to epoching (MNE-Python; Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_annotations",
)
