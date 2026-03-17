"""
backend/registry/nodes/erp.py

ERP analysis node types: nodes for event-related potential analysis.

These nodes operate on mne.Evoked or mne.Epochs objects and produce ERP-specific
metrics, derived waveforms, and publication-quality comparison figures.

All execute_fns follow the standard contract:
  - Never mutate input_data — always work on copies or new objects.
  - Use verbose=False on all MNE calls.
  - Visualization nodes: use matplotlib.use("Agg") and return base64 PNG data URI.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from backend.registry.nodes._channel_utils import resolve_channel

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Compute GFP (Global Field Power)
# ---------------------------------------------------------------------------

def _execute_compute_gfp(evoked: mne.Evoked, params: dict) -> np.ndarray:
    """
    Computes the Global Field Power (GFP) from an evoked response.

    GFP = standard deviation across channels at each time point:
        GFP(t) = std(V_1(t), V_2(t), ..., V_N(t))

    GFP is reference-independent (unlike individual channel traces) and
    provides a single summary measure of the overall ERP amplitude at each
    time point. GFP peaks correspond to moments of maximal global activation
    and are used to define ERP component latencies without channel selection bias.

    Returns a 1-D numpy array of shape (n_times,).
    """
    return evoked.data.std(axis=0)


COMPUTE_GFP = NodeDescriptor(
    node_type="compute_gfp",
    display_name="Compute GFP",
    category="Analysis",
    description=(
        "Computes Global Field Power (GFP) from an evoked ERP response. "
        "GFP is the standard deviation across all channels at each time point — "
        "a reference-independent measure of overall brain activation. "
        "GFP peaks mark moments of maximal coherent neural activity and are used "
        "to objectively identify ERP component latencies (P300, N200, N400) without "
        "needing to choose a specific electrode. Output is a 1-D array (n_times,)."
    ),
    tags=["gfp", "global", "field", "power", "erp", "amplitude", "analysis"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked ERP"),
    ],
    outputs=[
        HandleSchema(id="gfp_out", type="array", label="GFP (n_times,)"),
    ],
    parameters=[],
    execute_fn=_execute_compute_gfp,
    code_template=lambda p: 'gfp = evoked.data.std(axis=0)  # Global Field Power',
    methods_template=lambda p: "Global Field Power (GFP) was computed as the standard deviation across all channels at each time point (Lehmann & Skrandies, 1980).",
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html",
)


# ---------------------------------------------------------------------------
# Plot GFP
# ---------------------------------------------------------------------------

def _execute_plot_gfp(evoked: mne.Evoked, params: dict) -> str:
    """
    Plots the Global Field Power timecourse with optional peak annotation.

    Computes GFP internally (std across channels) and renders a time-series
    plot with the time axis in milliseconds. If highlight_peaks=True, marks
    the maximum GFP peak with a vertical dashed line and text label.
    """
    gfp = evoked.data.std(axis=0)
    times_ms = evoked.times * 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_ms, gfp * 1e6, color="#a855f7", linewidth=1.8, label="GFP")
    ax.fill_between(times_ms, 0, gfp * 1e6, alpha=0.15, color="#a855f7")

    if bool(params.get("highlight_peaks", True)):
        peak_idx = int(np.argmax(gfp))
        peak_t = times_ms[peak_idx]
        peak_v = gfp[peak_idx] * 1e6
        ax.axvline(x=peak_t, color="#ec4899", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            peak_t + 5, peak_v * 0.95,
            f"{peak_t:.0f} ms\n{peak_v:.2f} µV",
            color="#ec4899", fontsize=8, va="top",
        )

    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("GFP (µV)")
    ax.set_title("Global Field Power")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


PLOT_GFP = NodeDescriptor(
    node_type="plot_gfp",
    display_name="Plot GFP",
    category="Visualization",
    description=(
        "Plots the Global Field Power (GFP) timecourse from an evoked ERP response. "
        "GFP is reference-independent and summarizes overall brain activation at each "
        "time point with a single value. Peak GFP latencies correspond to major ERP "
        "components (P300, N200, N400) and are marked on the plot when highlight_peaks is on."
    ),
    tags=["gfp", "global", "field", "power", "erp", "plot", "visualization"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked ERP"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="GFP Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="highlight_peaks",
            label="Highlight Peaks",
            type="bool",
            default=True,
            description=(
                "When True, marks the maximum GFP peak with a vertical dashed line "
                "and annotation showing latency (ms) and amplitude (µV)."
            ),
        ),
    ],
    execute_fn=_execute_plot_gfp,
    code_template=lambda p: 'gfp = evoked.data.std(axis=0)\nfig, ax = plt.subplots()\nax.plot(evoked.times * 1000, gfp * 1e6)\nax.set_xlabel("Time (ms)")\nax.set_ylabel("GFP (µV)")',
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html",
)


# ---------------------------------------------------------------------------
# Detect ERP Peak
# ---------------------------------------------------------------------------

def _execute_detect_erp_peak(evoked: mne.Evoked, params: dict) -> dict[str, Any]:
    """
    Detects the peak of an ERP component in a specified time window and channel.

    Returns a metrics dict with peak latency (ms), peak amplitude (µV),
    and context about the search. This is the standard approach for quantifying
    ERP components: P300 (250-500 ms, Pz, positive), N200 (150-300 ms, FCz,
    negative), N400 (300-500 ms, Cz, negative).

    polarity:
      "positive" — finds the maximum (most positive) value
      "negative" — finds the minimum (most negative) value
      "absolute" — finds the value with the largest absolute amplitude
    """
    channel = str(params["channel"]).strip()
    tmin_ms = float(params["tmin_ms"])
    tmax_ms = float(params["tmax_ms"])
    polarity = str(params["polarity"])

    channel = resolve_channel(channel, evoked.ch_names)

    ch_idx = evoked.ch_names.index(channel)
    times = evoked.times
    tmin_s = tmin_ms / 1000.0
    tmax_s = tmax_ms / 1000.0
    mask = (times >= tmin_s) & (times <= tmax_s)

    if not np.any(mask):
        raise ValueError(
            f"No time points found in window {tmin_ms:.0f}–{tmax_ms:.0f} ms. "
            "Check that the evoked response covers this time range."
        )

    data_window = evoked.data[ch_idx, mask]
    times_window = times[mask]

    if polarity == "negative":
        peak_idx = int(np.argmin(data_window))
    elif polarity == "absolute":
        peak_idx = int(np.argmax(np.abs(data_window)))
    else:  # "positive"
        peak_idx = int(np.argmax(data_window))

    peak_latency_ms = float(times_window[peak_idx] * 1000.0)
    peak_amplitude_uv = float(data_window[peak_idx] * 1e6)

    return {
        "peak_latency_ms": round(peak_latency_ms, 2),
        "peak_amplitude_uv": round(peak_amplitude_uv, 3),
        "channel": channel,
        "search_window_ms": f"{tmin_ms:.0f}–{tmax_ms:.0f}",
        "polarity": polarity,
    }


DETECT_ERP_PEAK = NodeDescriptor(
    node_type="detect_erp_peak",
    display_name="Detect ERP Peak",
    category="Analysis",
    description=(
        "Detects the peak of an ERP component in a specified time window and electrode. "
        "Returns peak latency (ms) and amplitude (µV). Standard component windows: "
        "P300 → 250–500 ms, Pz, positive polarity. "
        "N200 → 150–300 ms, FCz, negative polarity. "
        "N400 → 300–500 ms, Cz, negative polarity. "
        "Output is a metrics dict for use in batch processing CSV exports."
    ),
    tags=["erp", "peak", "latency", "amplitude", "p300", "n200", "n400", "component", "analysis"],
    inputs=[
        HandleSchema(id="evoked_in", type="evoked", label="Evoked ERP"),
    ],
    outputs=[
        HandleSchema(id="metrics_out", type="metrics", label="Peak Metrics"),
    ],
    parameters=[
        ParameterSchema(
            name="channel",
            label="Channel",
            type="string",
            default="Cz",
            description=(
                "Electrode name to detect the peak on. "
                "P300: Pz. N200/N400: Cz or FCz. Use GFP-based analysis for "
                "reference-independent peak detection."
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
            description="Start of the component search window in milliseconds post-stimulus.",
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
            description="End of the component search window in milliseconds post-stimulus.",
        ),
        ParameterSchema(
            name="polarity",
            label="Peak Polarity",
            type="select",
            default="positive",
            options=["positive", "negative", "absolute"],
            description=(
                "positive: find the maximum value (P300, P100, P600). "
                "negative: find the minimum value (N200, N400, N170). "
                "absolute: find the value with the largest magnitude regardless of sign."
            ),
        ),
    ],
    execute_fn=_execute_detect_erp_peak,
    code_template=lambda p: f'ch_idx = evoked.ch_names.index("{p.get("channel", "Cz")}")\ntmin_s, tmax_s = {p.get("tmin_ms", 250)} / 1000, {p.get("tmax_ms", 500)} / 1000\nmask = (evoked.times >= tmin_s) & (evoked.times <= tmax_s)\npeak_idx = evoked.data[ch_idx, mask].argmax()  # or argmin for negative polarity',
    methods_template=lambda p: f'ERP peak amplitude and latency were detected at electrode {p.get("channel", "Cz")} in the {p.get("tmin_ms", 250)}–{p.get("tmax_ms", 500)} ms window ({p.get("polarity", "positive")} polarity).',
    docs_url="https://mne.tools/stable/generated/mne.Evoked.html#mne.Evoked.get_peak",
)


# ---------------------------------------------------------------------------
# Compute Difference Wave
# ---------------------------------------------------------------------------

def _execute_compute_difference_wave(epochs: mne.Epochs, params: dict) -> mne.Evoked:
    """
    Computes an ERP difference wave: condition_a minus condition_b.

    The difference wave isolates the neural activity specific to one condition
    relative to another. Classic examples:
      - Mismatch Negativity (MMN): standard minus deviant
      - N400: incongruent minus congruent
      - LRP: ipsilateral minus contralateral motor activity

    Takes epochs as input (which can contain multiple event types) rather than
    a pre-averaged evoked — this allows the node to average each condition
    independently before subtracting, which is more accurate than subtracting
    pre-averaged responses.

    Uses mne.combine_evoked([a, b], weights=[1, -1]) for the subtraction.
    """
    cond_a = str(params["condition_a"]).strip()
    cond_b = str(params["condition_b"]).strip()

    # Validate conditions exist in epochs
    available = list(epochs.event_id.keys())
    if cond_a not in available:
        raise ValueError(
            f"Condition '{cond_a}' not found in epochs. "
            f"Available event IDs: {available}"
        )
    if cond_b not in available:
        raise ValueError(
            f"Condition '{cond_b}' not found in epochs. "
            f"Available event IDs: {available}"
        )

    evoked_a = epochs[cond_a].average()
    evoked_b = epochs[cond_b].average()
    diff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
    diff.comment = f"{cond_a} − {cond_b}"
    return diff


COMPUTE_DIFFERENCE_WAVE = NodeDescriptor(
    node_type="compute_difference_wave",
    display_name="Compute Difference Wave",
    category="Analysis",
    description=(
        "Computes an ERP difference wave by subtracting condition B from condition A. "
        "The difference wave isolates neural activity specific to one condition relative "
        "to another. Classic uses: MMN (standard − deviant), N400 (incongruent − congruent), "
        "LRP (ipsilateral − contralateral). "
        "Accepts epochs with multiple event types and averages each condition internally."
    ),
    tags=["difference", "wave", "erp", "contrast", "mmn", "n400", "lrp", "subtraction", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs (multi-condition)"),
    ],
    outputs=[
        HandleSchema(id="evoked_out", type="evoked", label="Difference Wave"),
    ],
    parameters=[
        ParameterSchema(
            name="condition_a",
            label="Condition A",
            type="string",
            default="1",
            description=(
                "Event ID or label for condition A (minuend). "
                "Must match an event type in the epochs. "
                "Example: '769' for left hand, 'T1' for target stimulus."
            ),
        ),
        ParameterSchema(
            name="condition_b",
            label="Condition B",
            type="string",
            default="2",
            description=(
                "Event ID or label for condition B (subtrahend). "
                "Result = A − B. "
                "Example: '770' for right hand, 'T2' for standard stimulus."
            ),
        ),
    ],
    execute_fn=_execute_compute_difference_wave,
    code_template=lambda p: f'evoked_a = epochs["{p.get("condition_a", "1")}"].average()\nevoked_b = epochs["{p.get("condition_b", "2")}"].average()\ndiff = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])',
    methods_template=lambda p: f'A difference wave was computed by subtracting the "{p.get("condition_b", "2")}" condition from "{p.get("condition_a", "1")}" using MNE-Python (Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.combine_evoked.html",
)


# ---------------------------------------------------------------------------
# Plot Comparison Evoked
# ---------------------------------------------------------------------------

def _execute_plot_comparison_evoked(epochs: mne.Epochs, params: dict) -> str:
    """
    Plots multiple ERP conditions on the same axes for visual comparison.

    Uses mne.viz.plot_compare_evokeds which handles condition color coding,
    confidence intervals (from trial-to-trial variability), and legend rendering.

    Accepts a comma-separated list of condition names and a channel to plot.
    Computes evoked per condition internally from the epochs object.
    """
    conditions_str = str(params["conditions"]).strip()
    conditions = [c.strip() for c in conditions_str.split(",") if c.strip()]
    channel = str(params["channel"]).strip()

    available = list(epochs.event_id.keys())
    missing = [c for c in conditions if c not in available]
    if missing:
        raise ValueError(
            f"Conditions not found in epochs: {missing}. "
            f"Available: {available}"
        )

    evokeds = {cond: epochs[cond].average() for cond in conditions}

    figs = mne.viz.plot_compare_evokeds(
        evokeds,
        picks=channel,
        show=False,
        title=f"ERP Comparison — {channel}",
    )
    # plot_compare_evokeds returns a list of figures
    fig = figs[0] if isinstance(figs, list) else figs

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


PLOT_COMPARISON_EVOKED = NodeDescriptor(
    node_type="plot_comparison_evoked",
    display_name="Plot ERP Comparison",
    category="Visualization",
    description=(
        "Overlays multiple ERP conditions on a single plot for visual comparison. "
        "Automatically computes the evoked response per condition from the epochs object. "
        "Specify a comma-separated list of condition names (matching event IDs in the epochs) "
        "and the electrode to plot. Shaded regions show within-condition variability. "
        "Essential for publication figures showing condition effects."
    ),
    tags=["erp", "comparison", "conditions", "overlay", "plot", "visualization", "evoked"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs (multi-condition)"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="ERP Comparison Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="conditions",
            label="Conditions",
            type="string",
            default="1,2",
            description=(
                "Comma-separated list of event IDs or condition labels to compare. "
                "Must match the event_id keys in the epochs object. "
                "Example: '769,770' or 'target,standard' or 'T1,T2'."
            ),
        ),
        ParameterSchema(
            name="channel",
            label="Channel",
            type="string",
            default="Cz",
            description=(
                "Electrode to plot. Use a channel where the component of interest "
                "is maximal: Pz for P300, Cz/FCz for N200/N400, Oz for visual ERPs."
            ),
        ),
    ],
    execute_fn=_execute_plot_comparison_evoked,
    code_template=lambda p: f'evokeds = {{cond: epochs[cond].average() for cond in "{p.get("conditions", "1,2")}".split(",")}}\nmne.viz.plot_compare_evokeds(evokeds, picks="{p.get("channel", "Cz")}", show=False)',
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.viz.plot_compare_evokeds.html",
)
