"""
backend/registry/nodes/visualization.py

Visualization node types: nodes that render plots as base64-encoded PNG images.

CRITICAL SETUP:
  matplotlib.use("Agg") must be called before any pyplot import.
  "Agg" is a non-interactive backend — it renders to a memory buffer instead
  of opening a GUI window. This is required in a FastAPI server context.
  On headless systems (Linux CI, remote servers) pyplot will crash without it.

All execute_fns in this file return a base64-encoded PNG data URI string:
  "data:image/png;base64,<base64_encoded_bytes>"

This string is sent to the frontend and rendered as <img src={...}> inside
the React Flow node body.

MEMORY MANAGEMENT:
  Always call plt.close(fig) after converting to PNG. Matplotlib holds
  figures in memory until explicitly closed. Over many pipeline executions,
  unclosed figures will accumulate and eventually exhaust RAM.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend. Must precede pyplot import.
import matplotlib.pyplot as plt
import mne

import numpy as np

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _figure_to_base64_png(fig: plt.Figure) -> str:
    """
    Converts a Matplotlib Figure to a base64-encoded PNG data URI.

    The figure is closed after encoding to prevent memory leaks.
    Returns a string suitable for use as an <img> src attribute.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)  # Always close — prevents memory accumulation
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Plot PSD
# ---------------------------------------------------------------------------

def _execute_plot_psd(
    spectrum: mne.time_frequency.Spectrum,
    params: dict,
) -> str:
    """
    Renders the PSD as a Matplotlib figure and returns a base64 PNG.

    `spectrum` is an mne.time_frequency.Spectrum object from a Compute PSD node.
    Returns a base64 PNG data URI string (not an mne or numpy object).
    """
    fig = spectrum.plot(
        dB=bool(params["dB"]),
        average=bool(params["show_average"]),
        show=False,  # Never show a GUI window in server context
    )

    # spectrum.plot() may return a list of figures for multi-axes layouts.
    # Normalise to a single Figure object.
    if isinstance(fig, list):
        fig = fig[0]

    return _figure_to_base64_png(fig)


# ---------------------------------------------------------------------------
# Plot Raw Signal
# ---------------------------------------------------------------------------

def _execute_plot_raw(
    raw: mne.io.BaseRaw,
    params: dict,
) -> str:
    """
    Renders M EEG channels over one or more consecutive time windows.

    n_panels renders N consecutive windows of duration_s seconds stacked
    vertically, giving a scrolling view without re-running the pipeline.

    scale_uv sets a fixed µV-per-division for all channels when > 0;
    when 0 the spacing is auto-computed from peak-to-peak amplitude.

    Uses matplotlib directly (raw.plot() returns an interactive viewer
    that cannot be captured with savefig()).
    """
    if raw.n_times == 0:
        raise ValueError(
            "Cannot plot an empty recording (0 samples). "
            "Check that upstream nodes did not discard all data."
        )
    n_ch = min(int(params["n_channels"]), len(raw.ch_names))
    start_s = float(params.get("start_time_s", 0.0))
    start_s = max(0.0, min(start_s, raw.times[-1]))
    duration = min(float(params["duration_s"]), raw.times[-1] - start_s)
    n_panels = max(1, min(int(params.get("n_panels", 1)), 5))
    scale_uv = float(params.get("scale_uv", 0.0))

    panel_height = max(3.0, n_ch * 0.55)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, panel_height * n_panels))
    if n_panels == 1:
        axes = [axes]

    for panel_idx, ax in enumerate(axes):
        panel_start = start_s + panel_idx * duration
        if panel_start >= raw.times[-1]:
            ax.set_visible(False)
            continue
        panel_dur = min(duration, raw.times[-1] - panel_start)
        start_sample = int(raw.info["sfreq"] * panel_start)
        n_samples = int(raw.info["sfreq"] * panel_dur)

        data, times = raw[:n_ch, start_sample:start_sample + n_samples]
        data_uv = data * 1e6  # Volts → µV

        if scale_uv > 0.0:
            spacing = scale_uv
        else:
            ptp = float((data_uv.max(axis=1) - data_uv.min(axis=1)).mean()) if data_uv.size > 0 else 100.0
            spacing = max(ptp * 1.5, 10.0)

        for i, (ch_data, ch_name) in enumerate(zip(data_uv, raw.ch_names[:n_ch])):
            offset = (n_ch - i - 1) * spacing
            ax.plot(times, ch_data + offset, color="#94a3b8", linewidth=0.6)
            ax.text(
                times[0] - (times[-1] - times[0]) * 0.01,
                offset,
                ch_name,
                ha="right",
                va="center",
                fontsize=7,
                color="#e2e8f0",
            )

        ax.set_xlabel("Time (s)", fontsize=9, color="#94a3b8")
        ax.set_title(
            f"EEG Signal — {panel_start:.1f}–{panel_start + panel_dur:.1f} s, {n_ch} channels",
            fontsize=10,
            color="#e2e8f0",
        )
        ax.set_xlim(times[0], times[-1])
        ax.set_yticks([])
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # ------------------------------------------------------------------
    # Annotation overlays: BAD_ segments (red bands) + events (green lines)
    # ------------------------------------------------------------------
    show_annotations = bool(params.get("show_annotations", True))
    if show_annotations and raw.annotations is not None and len(raw.annotations) > 0:
        for panel_idx, ax in enumerate(axes):
            if not ax.get_visible():
                continue
            panel_start = start_s + panel_idx * duration
            panel_end = panel_start + min(duration, raw.times[-1] - panel_start)
            for ann in raw.annotations:
                onset = float(ann["onset"])
                ann_dur = float(ann["duration"])
                desc = str(ann["description"])
                ann_end = onset + ann_dur
                # Skip annotations outside this panel's visible window
                if ann_end < panel_start or onset > panel_end:
                    continue
                if desc.startswith("BAD_"):
                    ax.axvspan(
                        max(onset, panel_start),
                        min(ann_end, panel_end),
                        alpha=0.15, color="red", zorder=0,
                    )
                    # Label at the top of the panel (first visible edge)
                    label_x = max(onset, panel_start)
                    ax.text(
                        label_x, ax.get_ylim()[1], desc,
                        fontsize=6, color="red", alpha=0.7,
                        va="bottom", ha="left",
                    )
                else:
                    # Stimulus / other event — green dashed vertical line
                    if panel_start <= onset <= panel_end:
                        ax.axvline(
                            onset, color="green", linestyle="--",
                            alpha=0.4, linewidth=0.8, zorder=0,
                        )
                        ax.text(
                            onset, ax.get_ylim()[1], desc,
                            fontsize=6, color="green", alpha=0.7,
                            va="bottom", ha="left",
                        )

    plt.tight_layout()
    return _figure_to_base64_png(fig)


PLOT_RAW = NodeDescriptor(
    node_type="plot_raw",
    display_name="Plot Raw Signal",
    category="Visualization",
    description=(
        "Renders the EEG signal as a stacked time-domain waveform. Each channel is "
        "plotted as a separate trace offset vertically for readability. "
        "Useful for inspecting signal quality, identifying artefacts, and comparing "
        "channels before and after filtering. "
        "Accepts both unfiltered (raw_eeg) and filtered (filtered_eeg) signals."
    ),
    tags=["plot", "raw", "waveform", "time-domain", "eeg", "visualization"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG"),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="n_channels",
            label="Channels to Show",
            type="int",
            default=10,
            min=1,
            max=256,
            step=1,
            description=(
                "Number of EEG channels to display, counted from the first channel "
                "in the recording. Capped at the total number of channels available. "
                "Supports up to 256 channels for high-density EEG systems."
            ),
        ),
        ParameterSchema(
            name="start_time_s",
            label="Start Time",
            type="float",
            default=0.0,
            min=0.0,
            max=3600.0,
            step=1.0,
            unit="s",
            description=(
                "Time offset to begin the display window. "
                "Use to browse through different segments of the recording."
            ),
        ),
        ParameterSchema(
            name="duration_s",
            label="Duration",
            type="float",
            default=10.0,
            min=1.0,
            max=60.0,
            step=1.0,
            unit="s",
            description=(
                "Length of each time window in seconds. "
                "Capped at the remaining recording duration after the start time."
            ),
        ),
        ParameterSchema(
            name="n_panels",
            label="Panels",
            type="int",
            default=1,
            min=1,
            max=5,
            step=1,
            description=(
                "Number of consecutive time windows to render, stacked vertically. "
                "1 = single window (default). 3 = three windows of duration_s each, "
                "starting at start_time_s, start_time_s + duration_s, etc. "
                "Gives a scrolling view without re-running the pipeline."
            ),
        ),
        ParameterSchema(
            name="scale_uv",
            label="Fixed Scale",
            type="float",
            default=0.0,
            min=0.0,
            max=5000.0,
            step=10.0,
            unit="µV",
            description=(
                "Fixed amplitude scale in µV per channel division. "
                "0 = auto-scale (default, each run adapts to the data). "
                "Set to e.g. 100 µV to make all channels use the same scale, "
                "enabling meaningful cross-channel amplitude comparisons."
            ),
        ),
        ParameterSchema(
            name="show_annotations",
            label="Show Annotations",
            type="bool",
            default=True,
            description=(
                "Overlay BAD_ annotations (red bands) and stimulus events "
                "(green lines) on the plot. BAD_ segments appear as shaded "
                "red regions; other annotations appear as green dashed vertical "
                "lines. Small text labels are drawn at the top of each panel."
            ),
        ),
    ],
    execute_fn=_execute_plot_raw,
    code_template=lambda p: (
        f"fig = raw.plot(n_channels={p['n_channels']}, "
        f"start={p['start_time_s']}, duration={p['duration_s']}, show=False)\n"
        f'fig.savefig("raw_signal.png", dpi=150, bbox_inches="tight")\n'
        f"plt.close(fig)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot",
)


# ---------------------------------------------------------------------------
# Channel name normalisation for montage matching
# ---------------------------------------------------------------------------

def _build_montage_rename_map(ch_names: list[str]) -> dict[str, str]:
    """
    Build a rename mapping that strips trailing dots from channel names.

    PhysioNet EDF files (e.g., eegmmidb) pad channel names with trailing
    dots ("Fc5.", "C3.."). The standard 10-20 montage uses clean names
    ("FC5", "C3"). MNE's ``match_case=False`` handles case differences
    but not trailing characters, so all channels silently fail to match.

    Returns a dict mapping ``old_name → cleaned_name`` only for channels
    that actually need renaming. Channels that would produce duplicates
    or empty names after stripping are left unchanged.
    """
    mapping: dict[str, str] = {}
    seen: set[str] = set()
    for name in ch_names:
        cleaned = name.rstrip(".")
        if cleaned and cleaned != name and cleaned not in seen:
            mapping[name] = cleaned
            seen.add(cleaned)
        else:
            seen.add(name)
    return mapping


# ---------------------------------------------------------------------------
# Plot Topomap
# ---------------------------------------------------------------------------

_BAND_RANGES: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _execute_plot_topomap(
    spectrum: mne.time_frequency.Spectrum,
    params: dict,
) -> str:
    """
    Renders a scalp topography map of PSD power in a selected frequency band.

    Attempts to assign standard 10-20 electrode positions automatically.
    Before montage assignment, trailing dots in channel names are stripped
    (common in PhysioNet EDF files like eegmmidb, where "Fc5." → "FC5").
    Channels whose names are not in the 10-20 standard after normalisation
    are dropped and the count is reported in the plot title.
    """
    band = str(params.get("bands", "alpha"))
    fmin, fmax = _BAND_RANGES.get(band, (8.0, 13.0))

    # Work on a copy so upstream data is never mutated
    spectrum = spectrum.copy()

    # Strip trailing dots from channel names (PhysioNet EDF convention)
    rename_map = _build_montage_rename_map(list(spectrum.ch_names))
    if rename_map:
        mne.rename_channels(spectrum.info, rename_map)

    n_before = len(spectrum.ch_names)

    # Auto-assign standard 10-20 positions; report how many channels were dropped
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        # match_case=False handles files where channel names use mixed case
        # (e.g. "Fc5", "Cpz") while the montage uses "FC5", "CPz".
        spectrum.info.set_montage(
            montage, on_missing="ignore", match_case=False, verbose=False
        )
    except Exception:
        pass  # If montage fails entirely, let plot_topomap raise a clear MNE error

    # Count channels that received a valid 3D position (non-zero location vector)
    n_positioned = sum(
        1 for ch in spectrum.info["chs"]
        if not np.allclose(ch["loc"][:3], 0.0)
    )
    n_dropped = n_before - n_positioned
    drop_note = (
        f" ({n_dropped} of {n_before} channels dropped — not in 10-20 standard)"
        if n_dropped > 0 else ""
    )

    fig = spectrum.plot_topomap(
        bands={band: (fmin, fmax)},
        show=False,
    )

    if isinstance(fig, list):
        fig = fig[0]

    if drop_note:
        fig.suptitle(drop_note, fontsize=7, color="#f87171", y=0.02)

    return _figure_to_base64_png(fig)


PLOT_TOPOMAP = NodeDescriptor(
    node_type="plot_topomap",
    display_name="Plot Topomap",
    category="Visualization",
    description=(
        "Renders a scalp topography (topomap) showing the spatial distribution of "
        "EEG power across electrodes for a selected frequency band. "
        "Warmer colours indicate higher power; cooler colours indicate lower power. "
        "Requires standard 10-20 electrode names (e.g. Fp1, Fz, Cz, Pz, Oz). "
        "Non-standard channel names are excluded from the map. "
        "This is a terminal node — its output cannot connect to other nodes."
    ),
    tags=["topomap", "topography", "spatial", "scalp", "eeg", "visualization", "power"],
    inputs=[
        HandleSchema(id="psd_in", type="psd", label="PSD"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="bands",
            label="Frequency Band",
            type="select",
            default="alpha",
            options=["delta", "theta", "alpha", "beta", "gamma"],
            description=(
                "The frequency band to visualise on the topomap. "
                "Delta (0.5–4 Hz): deep sleep, pathology. "
                "Theta (4–8 Hz): drowsiness, memory. "
                "Alpha (8–13 Hz): relaxed wakefulness, eyes closed. "
                "Beta (13–30 Hz): active thinking, focus. "
                "Gamma (30–45 Hz): high-level cognition."
            ),
        ),
    ],
    execute_fn=_execute_plot_topomap,
    code_template=lambda p: (
        "# Strip trailing dots from channel names (PhysioNet EDF convention)\n"
        "rename_map = {name: name.rstrip(\".\") for name in spectrum.ch_names "
        "if name != name.rstrip(\".\") and name.rstrip(\".\")}\n"
        "if rename_map:\n"
        "    mne.rename_channels(spectrum.info, rename_map)\n"
        'montage = mne.channels.make_standard_montage("standard_1020")\n'
        'spectrum.info.set_montage(montage, on_missing="ignore", match_case=False, verbose=False)\n'
        f"fig = spectrum.plot_topomap(bands=\"{p.get('bands', 'alpha')}\", show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html#mne.time_frequency.Spectrum.plot_topomap",
)


PLOT_PSD = NodeDescriptor(
    node_type="plot_psd",
    display_name="Plot PSD",
    category="Visualization",
    description=(
        "Visualizes the Power Spectral Density as a line plot. The x-axis shows "
        "frequency in Hz; the y-axis shows power in dB (or µV²/Hz if dB is off). "
        "Each EEG channel is plotted as a separate line. Enabling 'Show Average' "
        "overlays a bold line showing the mean across all channels. "
        "This is a terminal node — its output cannot connect to other nodes."
    ),
    tags=["plot", "psd", "visualization", "frequency", "power", "spectrum"],
    inputs=[
        HandleSchema(
            id="psd_in",
            type="psd",
            label="PSD",
        ),
    ],
    outputs=[
        # "plot" type is terminal — the frontend prevents connecting downstream.
        HandleSchema(
            id="plot_out",
            type="plot",
            label="Plot Image",
        ),
    ],
    parameters=[
        ParameterSchema(
            name="dB",
            label="Show in dB",
            type="bool",
            default=True,
            description=(
                "When enabled, power is displayed in decibels (10 * log10(power)). "
                "This is the standard for EEG PSD plots as it compresses the dynamic "
                "range and makes band structure more visible. Disable to show raw "
                "power in µV²/Hz."
            ),
        ),
        ParameterSchema(
            name="show_average",
            label="Show Average",
            type="bool",
            default=True,
            description=(
                "When enabled, overlays a bold line showing the mean PSD across all "
                "channels. Useful for quickly assessing overall signal quality and "
                "identifying dominant frequency bands."
            ),
        ),
    ],
    execute_fn=_execute_plot_psd,
    code_template=lambda p: f'fig = spectrum.plot(dB={p.get("dB", True)}, average={p.get("show_average", True)}, show=False)\nif isinstance(fig, list):\n    fig = fig[0]\nfig.savefig("psd_output.png", dpi=150, bbox_inches="tight")\nplt.close(fig)',
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html#mne.time_frequency.Spectrum.plot",
)


# ---------------------------------------------------------------------------
# Plot Evoked (Butterfly)
# ---------------------------------------------------------------------------

def _execute_plot_evoked(evoked: mne.Evoked, params: dict) -> str:
    """
    Renders an ERP butterfly plot — all channels overlaid on a single axis.

    Uses evoked.plot() with show=False. spatial_colors assigns a unique colour
    to each channel based on its scalp location, making it easier to see which
    regions dominate the ERP.
    """
    spatial_colors = bool(params["spatial_colors"])
    try:
        fig = evoked.plot(
            spatial_colors=spatial_colors,
            time_unit=str(params["time_unit"]),
            show=False,
        )
    except RuntimeError:
        # spatial_colors=True requires electrode positions (a montage).
        # Fall back gracefully for files with custom channel names that
        # are not in the standard 10-20 set.
        fig = evoked.plot(
            spatial_colors=False,
            time_unit=str(params["time_unit"]),
            show=False,
        )
    if isinstance(fig, list):
        fig = fig[0]
    return _figure_to_base64_png(fig)


PLOT_EVOKED = NodeDescriptor(
    node_type="plot_evoked",
    display_name="Plot Evoked (ERP)",
    category="Visualization",
    description=(
        "Renders an ERP butterfly plot showing all channels overlaid on a single "
        "time axis. The x-axis shows time relative to the stimulus; the y-axis shows "
        "amplitude in µV. Spatial colours (enabled by default) map each channel to a "
        "colour based on its scalp location — frontal channels are blue, occipital "
        "channels are red, etc. This is the standard way to inspect ERP components."
    ),
    tags=["plot", "erp", "evoked", "butterfly", "eeg", "visualization"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked (ERP)"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="time_unit",
            label="Time Unit",
            type="select",
            default="ms",
            options=["ms", "s"],
            description=(
                "Unit for the x-axis. 'ms' (milliseconds) is standard for ERP "
                "plots — component latencies (P100, N200, P300) are described in ms."
            ),
        ),
        ParameterSchema(
            name="spatial_colors",
            label="Spatial Colors",
            type="bool",
            default=True,
            description=(
                "When enabled, channels are coloured by scalp location: "
                "frontal = blue, central = green, parietal = orange, occipital = red. "
                "Requires standard 10-20 electrode names (e.g. Fp1, Cz, Oz). "
                "For files with custom channel names, leave disabled — the node will "
                "fall back to uniform colouring automatically."
            ),
        ),
    ],
    execute_fn=_execute_plot_evoked,
    code_template=lambda p: (
        f"fig = evoked.plot(spatial_colors={p['spatial_colors']}, "
        f"time_unit='{p['time_unit']}', show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.plot",
)


# ---------------------------------------------------------------------------
# Plot Epochs Image
# ---------------------------------------------------------------------------

def _execute_plot_epochs_image(epochs: mne.Epochs, params: dict) -> str:
    """
    Renders an epochs image for a single channel — each row is one trial,
    colour represents amplitude. Also shows the ERP average below.

    epochs.plot_image() returns a list of figures (one per channel requested).
    We request a single channel and take fig[0].
    """
    # Prefer channel_name (string) if provided; fall back to channel_index (int).
    channel_name = str(params.get("channel_name", "")).strip()
    if channel_name:
        picks = channel_name
    else:
        picks = [int(params["channel_index"])]
    figs = epochs.plot_image(
        picks=picks,
        show=False,
        combine=None,
    )
    fig = figs[0] if isinstance(figs, list) else figs
    return _figure_to_base64_png(fig)


PLOT_EPOCHS_IMAGE = NodeDescriptor(
    node_type="plot_epochs_image",
    display_name="Plot Epochs Image",
    category="Visualization",
    description=(
        "Renders an epochs image (also called a raster plot or single-trial image) "
        "for one EEG channel. Each row represents one trial; colour encodes amplitude. "
        "Below the image, the ERP average is shown. "
        "This visualisation reveals trial-by-trial variability and response consistency. "
        "Use channel index 0 for the first channel, 1 for the second, and so on."
    ),
    tags=["plot", "epochs", "image", "raster", "single-trial", "visualization"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="channel_name",
            label="Channel Name",
            type="string",
            default="",
            description=(
                "Name of the channel to visualize (e.g. 'Cz', 'Pz', 'Oz'). "
                "Takes precedence over Channel Index when non-empty. "
                "Use this instead of the index — channel names are visible in the "
                "EDF Loader session info panel and are stable across recordings."
            ),
            channel_hint="single",
        ),
        ParameterSchema(
            name="channel_index",
            label="Channel Index",
            type="int",
            default=0,
            min=0,
            max=255,
            step=1,
            description=(
                "Zero-based index of the channel to visualize (used when Channel "
                "Name is empty). 0 = first channel, 1 = second, etc. "
                "Prefer Channel Name above for clarity."
            ),
        ),
    ],
    execute_fn=_execute_plot_epochs_image,
    code_template=lambda p: (
        f"figs = epochs.plot_image(picks='{p['channel_name']}', show=False)"
        if p.get("channel_name", "").strip()
        else f"figs = epochs.plot_image(picks=[{p['channel_index']}], show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.plot_image",
)


# ---------------------------------------------------------------------------
# Plot TFR (Time-Frequency Representation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plot Evoked Topomap
# ---------------------------------------------------------------------------

def _execute_plot_evoked_topomap(evoked: "mne.Evoked", params: dict) -> str:
    """
    Renders scalp topography maps at specified ERP latencies.

    evoked.plot_topomap() produces one topomap panel per time point.
    The average window smooths each topomap over a short interval to reduce
    noise — this is the standard presentation in ERP papers.
    """
    times = [float(t.strip()) for t in str(params["times"]).split(",") if t.strip()]
    average = float(params["average_window_ms"]) / 1000.0
    fig = evoked.plot_topomap(times=times, average=average, show=False)
    if isinstance(fig, list):
        fig = fig[0]
    return _figure_to_base64_png(fig)


PLOT_EVOKED_TOPOMAP = NodeDescriptor(
    node_type="plot_evoked_topomap",
    display_name="Plot ERP Topomap",
    category="Visualization",
    description=(
        "Renders scalp topography maps at specific ERP latencies, showing the "
        "spatial distribution of EEG activity at each time point. "
        "Each time in the 'times' field produces one topomap panel. "
        "This is required in every ERP paper — the P300 (350 ms) and N200 (200 ms) "
        "topomaps identify the scalp generators of the ERP components. "
        "Requires standard 10-20 electrode names. Connects from a Compute Evoked node."
    ),
    tags=["plot", "erp", "topomap", "topography", "evoked", "visualization", "spatial"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked (ERP)"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="times",
            label="Latencies (s)",
            type="string",
            default="0.1, 0.2, 0.3",
            description=(
                "Comma-separated latencies in seconds at which to plot topomaps. "
                "Each time value produces one topomap panel. "
                "Example: '0.1, 0.2, 0.3' shows three maps at 100 ms, 200 ms, 300 ms. "
                "Use 0.3 for the P300, 0.17 for the N170, 0.1 for early visual responses."
            ),
        ),
        ParameterSchema(
            name="average_window_ms",
            label="Average Window",
            type="float",
            default=50.0,
            min=0.0,
            max=200.0,
            step=5.0,
            unit="ms",
            description=(
                "Averaging window centred on each latency. Power is averaged over "
                "[t - window/2, t + window/2]. Larger windows reduce noise but blur "
                "temporal precision. 50 ms is a standard choice for ERP topomaps."
            ),
        ),
    ],
    execute_fn=_execute_plot_evoked_topomap,
    code_template=lambda p: (
        f"fig = evoked.plot_topomap(times=[{p['times']}], "
        f"average={float(p['average_window_ms']) / 1000.0}, show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.plot_topomap",
)


# ---------------------------------------------------------------------------
# Inspect ICA Components
# ---------------------------------------------------------------------------

def _execute_plot_ica_components(raw: "mne.io.BaseRaw", params: dict) -> str:
    """
    Fits ICA on the input signal and renders a grid of component topographies.

    This node re-fits ICA (same params, same random_state=42 for reproducibility)
    to produce a visual inspection grid. The researcher identifies artifact
    component indices from the topographies, then enters them in the ICA
    Decomposition node's Exclude Components field and re-runs the pipeline.

    ica.plot_components() returns a list of figures (one per page of components).
    We take the first figure which contains the most prominent components.
    """
    import numpy as np
    n_components = int(params["n_components"])
    method = str(params["method"])
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=42,
        verbose=False,
    )
    ica.fit(raw, verbose=False)
    figs = ica.plot_components(show=False)
    fig = figs[0] if isinstance(figs, list) else figs
    return _figure_to_base64_png(fig)


PLOT_ICA_COMPONENTS = NodeDescriptor(
    node_type="plot_ica_components",
    display_name="Inspect ICA Components",
    category="Visualization",
    description=(
        "Fits ICA on the input signal and renders a grid of component topographies. "
        "Use this node to identify which component indices correspond to eye blinks "
        "(large frontal topography at Fp1/Fp2) and heartbeat artifacts (frontocentral). "
        "Once identified, enter those indices in the ICA Decomposition node's "
        "'Exclude Components' field and re-run the pipeline to remove them. "
        "Use the same n_components and method as your ICA Decomposition node."
    ),
    tags=["ica", "components", "artifact", "inspection", "topography", "visualization"],
    inputs=[
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Component Grid"),
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
                "Number of ICA components to compute and visualise. "
                "Must match the value used in your ICA Decomposition node. "
                "Typical values: 15–25 for most research EEG systems."
            ),
        ),
        ParameterSchema(
            name="method",
            label="Algorithm",
            type="select",
            default="fastica",
            options=["fastica", "infomax", "picard"],
            description=(
                "ICA algorithm. Must match the value used in your ICA Decomposition node. "
                "FastICA is fast and widely used. Infomax matches EEGLAB. "
                "Picard offers more reliable convergence on some datasets."
            ),
        ),
    ],
    execute_fn=_execute_plot_ica_components,
    code_template=lambda p: (
        f"ica = mne.preprocessing.ICA(n_components={p['n_components']}, "
        f"method='{p['method']}', random_state=42)\n"
        f"ica.fit(raw, verbose=False)\n"
        f"figs = ica.plot_components(show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_components",
)


def _execute_plot_evoked_joint(evoked: "mne.Evoked", params: dict) -> str:
    """
    Renders a joint plot: ERP butterfly waveform with topomap insets.

    When times='peaks', MNE auto-detects peak latencies.
    The researcher can override with comma-separated latency values (in seconds).
    """
    times_param = str(params.get("times", "peaks")).strip()
    if times_param == "peaks":
        times: str | list = "peaks"
    else:
        times = [float(t.strip()) for t in times_param.split(",") if t.strip()]
    try:
        fig = evoked.plot_joint(times=times, show=False)
    except RuntimeError as exc:
        # EDF files without electrode positions cannot render topomaps.
        # Fall back to a butterfly-only plot so the node never hard-fails.
        if "digitization" in str(exc).lower() or "layout" in str(exc).lower():
            fig = evoked.plot(show=False)
        else:
            raise
    if isinstance(fig, list):
        fig = fig[0]
    return _figure_to_base64_png(fig)


PLOT_EVOKED_JOINT = NodeDescriptor(
    node_type="plot_evoked_joint",
    display_name="Plot ERP Joint",
    category="Visualization",
    description=(
        "Renders a joint ERP plot — the butterfly waveform with topomap insets at "
        "the automatically detected peak latencies. "
        "This is the most publication-ready single-figure ERP summary. "
        "Set 'times' to 'peaks' to let MNE detect peaks automatically, or enter "
        "specific latencies (e.g. '0.1, 0.3, 0.5') to highlight components of interest. "
        "Connects from a Compute Evoked node."
    ),
    tags=["plot", "erp", "evoked", "joint", "butterfly", "topomap", "visualization", "publication"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked (ERP)"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="times",
            label="Peak Times",
            type="string",
            default="peaks",
            description=(
                "Latencies for the topomap insets. "
                "'peaks' (default): MNE automatically selects the three largest peaks. "
                "Custom: comma-separated values in seconds, e.g. '0.1, 0.3, 0.5'."
            ),
        ),
    ],
    execute_fn=_execute_plot_evoked_joint,
    code_template=lambda p: (
        f"fig = evoked.plot_joint(times='{p['times']}', show=False)"
        if str(p.get("times", "peaks")).strip() == "peaks"
        else f"fig = evoked.plot_joint(times=[{p['times']}], show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.plot_joint",
)


def _execute_plot_tfr(
    tfr: "mne.time_frequency.AverageTFR",
    params: dict,
) -> str:
    """
    Renders a time-frequency spectrogram from an AverageTFR object.

    tfr.plot() returns a list of figures (one per channel/pick). We request
    a single aggregated plot by passing picks as a string type (e.g. "eeg")
    which collapses across channels, or a single channel name.

    Baseline correction is applied in the plot call for display normalisation.
    When apply_baseline is False, baseline=None disables it.
    """
    picks = str(params.get("picks", "eeg")).strip() or "eeg"
    mode = str(params.get("mode", "logratio"))
    apply_baseline = bool(params.get("apply_baseline", True))

    baseline: tuple | None = None
    if apply_baseline:
        baseline = (
            float(params.get("baseline_tmin", -0.5)),
            float(params.get("baseline_tmax", 0.0)),
        )

    figs = tfr.plot(
        picks=picks,
        baseline=baseline,
        mode=mode,
        show=False,
    )

    # tfr.plot() returns a list of Figure objects.
    fig = figs[0] if isinstance(figs, list) else figs
    return _figure_to_base64_png(fig)


PLOT_TFR = NodeDescriptor(
    node_type="plot_tfr",
    display_name="Plot TFR",
    category="Visualization",
    description=(
        "Renders a time-frequency spectrogram showing how spectral power changes "
        "over time within epochs. The x-axis shows time relative to the stimulus; "
        "the y-axis shows frequency. Colour encodes power (or normalised power). "
        "Log-ratio baseline normalisation (default) divides by the pre-stimulus "
        "power and takes the log — this is the standard for event-related oscillation "
        "analyses. "
        "Connect from a Time-Frequency (Morlet) node."
    ),
    tags=["plot", "tfr", "time-frequency", "spectrogram", "morlet", "visualization"],
    inputs=[
        HandleSchema(id="tfr_in", type="tfr", label="TFR"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Plot Image"),
    ],
    parameters=[
        ParameterSchema(
            name="picks",
            label="Channel / Type",
            type="string",
            default="eeg",
            description=(
                "Channel selection for the plot. "
                "Type 'eeg' to average across all EEG channels (recommended for a "
                "global view). "
                "Type a single channel name (e.g. 'Cz', 'Pz') to plot one electrode."
            ),
        ),
        ParameterSchema(
            name="mode",
            label="Normalisation Mode",
            type="select",
            default="logratio",
            options=["logratio", "ratio", "mean", "percent"],
            description=(
                "How to normalise power relative to the baseline. "
                "logratio (default): 10 * log10(power / baseline_mean) — standard for "
                "EEG oscillation studies. "
                "ratio: power / baseline_mean — linear scale. "
                "mean: power - baseline_mean — absolute change. "
                "percent: (power - baseline_mean) / baseline_mean * 100."
            ),
        ),
        ParameterSchema(
            name="apply_baseline",
            label="Apply Baseline",
            type="bool",
            default=True,
            description=(
                "When enabled, normalises power relative to the baseline interval "
                "defined below. Disable to plot raw (unnormalised) power values."
            ),
        ),
        ParameterSchema(
            name="baseline_tmin",
            label="Baseline Start",
            type="float",
            default=-0.5,
            min=-5.0,
            max=0.0,
            step=0.05,
            unit="s",
            description=(
                "Start of the baseline interval for normalisation. "
                "Typically the pre-stimulus window (e.g. -0.5 s)."
            ),
        ),
        ParameterSchema(
            name="baseline_tmax",
            label="Baseline End",
            type="float",
            default=0.0,
            min=-5.0,
            max=0.0,
            step=0.05,
            unit="s",
            description=(
                "End of the baseline interval. 0.0 s (event onset) is standard — "
                "the pre-stimulus period defines the baseline power level."
            ),
        ),
    ],
    execute_fn=_execute_plot_tfr,
    code_template=lambda p: (
        f"figs = tfr.plot(picks='{p['picks']}', "
        + (f"baseline=({p['baseline_tmin']}, {p['baseline_tmax']}), mode='{p['mode']}', "
           if p.get('apply_baseline', True) else "baseline=None, ")
        + "show=False)"
    ),
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.time_frequency.AverageTFR.html#mne.time_frequency.AverageTFR.plot",
)
