# Lightweight preview image generator for pipeline node outputs.

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import numpy as np


_BG_COLOR = "#1e1e2e"
_LINE_COLOR = "#cdd6f4"
_LINE_COLORS = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5"]
_FIGSIZE = (6, 2.0)
_DPI = 100
_LINEWIDTH = 0.8


def generate_preview(output: Any) -> str | None:
    """Return a tiny base64 PNG preview of the given pipeline output, or None."""

    # Visualization nodes already produce their own thumbnails.
    if isinstance(output, str) and output.startswith("data:image"):
        return None

    fig = None
    try:
        if isinstance(output, mne.io.BaseRaw):
            fig = _preview_raw(output)
        elif isinstance(output, mne.time_frequency.Spectrum):
            fig = _preview_spectrum(output)
        elif isinstance(output, mne.Epochs):
            fig = _preview_epochs(output)
        elif isinstance(output, mne.Evoked):
            fig = _preview_evoked(output)
        elif isinstance(output, np.ndarray):
            fig = _preview_ndarray(output)
        else:
            return None

        if fig is None:
            return None

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight",
                    facecolor=_BG_COLOR)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    except Exception:
        return None

    finally:
        if fig is not None:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Per-type preview renderers
# ---------------------------------------------------------------------------

def _make_fig() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.tick_params(colors="#7f849c", labelsize=6, length=2)
    ax.grid(True, alpha=0.15, color="#7f849c", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_color("#45475a")
        spine.set_linewidth(0.5)
    return fig, ax


def _preview_raw(raw: mne.io.BaseRaw) -> plt.Figure:
    fig, ax = _make_fig()
    n_ch = min(5, len(raw.ch_names))
    sfreq = raw.info["sfreq"]
    n_samples = min(int(2.0 * sfreq), raw.n_times)
    data, _ = raw[:n_ch, :n_samples]
    times = np.arange(n_samples) / sfreq

    # Offset channels vertically for readability
    for i in range(n_ch):
        trace = data[i]
        std = np.std(trace) if np.std(trace) > 0 else 1.0
        offset = i * std * 3
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(times, trace + offset, color=color, linewidth=_LINEWIDTH,
                label=raw.ch_names[i])

    ax.set_xlabel("Time (s)", fontsize=6, color="#7f849c")
    ax.legend(fontsize=5, loc="upper right", frameon=False, labelcolor="#cdd6f4")
    fig.tight_layout(pad=0.3)
    return fig


def _preview_spectrum(spectrum: mne.time_frequency.Spectrum) -> plt.Figure:
    fig, ax = _make_fig()
    psd_data = spectrum.get_data()
    freqs = spectrum.freqs
    # Plot individual channels faintly + bold mean
    n_ch = min(5, psd_data.shape[0])
    for i in range(n_ch):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.semilogy(freqs, psd_data[i], color=color, linewidth=0.3, alpha=0.4)
    mean_psd = psd_data.mean(axis=0)
    ax.semilogy(freqs, mean_psd, color="#f5c2e7", linewidth=1.2, label="Mean")
    ax.set_xlabel("Frequency (Hz)", fontsize=6, color="#7f849c")
    ax.set_ylabel("PSD", fontsize=6, color="#7f849c")
    ax.legend(fontsize=5, loc="upper right", frameon=False, labelcolor="#cdd6f4")
    fig.tight_layout(pad=0.3)
    return fig


def _preview_epochs(epochs: mne.Epochs) -> plt.Figure:
    fig, ax = _make_fig()
    data = epochs.get_data(verbose=False)  # (n_epochs, n_channels, n_times)
    n_ch = min(5, data.shape[1])
    avg = data.mean(axis=0)  # (n_channels, n_times)
    times = epochs.times
    for i in range(n_ch):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(times, avg[i], color=color, linewidth=_LINEWIDTH,
                label=epochs.ch_names[i])
    ax.set_xlabel("Time (s)", fontsize=6, color="#7f849c")
    ax.axvline(0, color="#f38ba8", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.legend(fontsize=5, loc="upper right", frameon=False, labelcolor="#cdd6f4")
    fig.tight_layout(pad=0.3)
    return fig


def _preview_evoked(evoked: mne.Evoked) -> plt.Figure:
    fig, ax = _make_fig()
    data = evoked.data  # (n_channels, n_times)
    n_ch = min(8, data.shape[0])
    for i in range(n_ch):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        ax.plot(evoked.times, data[i], color=color, linewidth=_LINEWIDTH,
                alpha=0.7)
    # Bold mean across channels
    ax.plot(evoked.times, data.mean(axis=0), color="#f5c2e7", linewidth=1.2,
            label="Mean")
    ax.set_xlabel("Time (s)", fontsize=6, color="#7f849c")
    ax.axvline(0, color="#f38ba8", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.legend(fontsize=5, loc="upper right", frameon=False, labelcolor="#cdd6f4")
    fig.tight_layout(pad=0.3)
    return fig


def _preview_ndarray(arr: np.ndarray) -> plt.Figure:
    fig, ax = _make_fig()
    if arr.ndim == 1:
        ax.plot(arr, color=_LINE_COLORS[0], linewidth=_LINEWIDTH)
    elif arr.ndim >= 2:
        n_rows = min(5, arr.shape[0])
        for i in range(n_rows):
            color = _LINE_COLORS[i % len(_LINE_COLORS)]
            ax.plot(arr[i], color=color, linewidth=_LINEWIDTH)
    else:
        return None
    fig.tight_layout(pad=0.3)
    return fig
