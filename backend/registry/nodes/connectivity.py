"""
backend/registry/nodes/connectivity.py

Connectivity analysis node types (Tier 2).

Nodes in this file compute and visualize functional connectivity between EEG
channels using the mne-connectivity package (pip install mne-connectivity).

All connectivity measures are frequency-domain and require segmented Epochs
(not continuous raw data) to estimate cross-spectral densities reliably.
A minimum of ~20–30 epochs is recommended for stable connectivity estimates.

execute_fn contract:
  - Never mutate input — always work on copies or new objects.
  - Use verbose=False on all MNE calls.
  - Visualization nodes: set matplotlib.use("Agg") and return base64 PNG.

New handle type added in Tier 2:
  "connectivity" — mne_connectivity.SpectralConnectivity object.
  The full frequency-resolved object is passed between nodes so that
  visualization nodes can display any frequency band without recomputing.
"""

from __future__ import annotations

import base64
import dataclasses
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)

# Soft dependency — mne-connectivity is required for spectral connectivity nodes.
# If not installed, the import error surfaces at node execution time with a
# clear message rather than crashing the server at startup.
try:
    from mne_connectivity import spectral_connectivity_epochs, SpectralConnectivity
    from mne_connectivity.viz import plot_connectivity_circle as _mne_plot_circle
    _MNE_CONNECTIVITY_AVAILABLE = True
except ImportError:
    _MNE_CONNECTIVITY_AVAILABLE = False


def _require_mne_connectivity() -> None:
    if not _MNE_CONNECTIVITY_AVAILABLE:
        raise ImportError(
            "The 'mne-connectivity' package is required for connectivity nodes. "
            "Install it with: pip install mne-connectivity"
        )


# ---------------------------------------------------------------------------
# ConnectivityMatrix — lightweight wrapper for non-spectral connectivity
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ConnectivityMatrix:
    """
    Lightweight wrapper for broadband (non-spectral) connectivity results
    such as Amplitude Envelope Correlation (AEC).

    Provides the same interface used by visualization nodes so that
    plot_connectivity_circle and plot_connectivity_matrix can accept
    both SpectralConnectivity objects and ConnectivityMatrix objects
    without branching.
    """

    matrix: np.ndarray   # shape (n_ch, n_ch), values in [-1, 1]
    names: list           # list[str] — channel names
    method: str           # e.g. "aec"
    freqs: list = dataclasses.field(default_factory=list)  # empty for broadband

    def get_data(self, output: str = "compact") -> np.ndarray:
        """Return matrix with a dummy frequency axis when output="dense"."""
        if output == "dense":
            # Shape (n_ch, n_ch, 1) — single "frequency bin" for broadband
            return self.matrix[:, :, np.newaxis]
        return self.matrix


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_connectivity(
    epochs: mne.Epochs,
    method: str,
    fmin_hz: float,
    fmax_hz: float,
) -> "SpectralConnectivity":
    """
    Core wrapper around spectral_connectivity_epochs.

    Returns a SpectralConnectivity object containing the frequency-resolved
    connectivity matrix. Downstream nodes call get_data(output="dense") to
    get an (n_ch, n_ch, n_freqs) array and average over the desired band.
    """
    _require_mne_connectivity()
    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        fmin=fmin_hz,
        fmax=fmax_hz,
        faverage=False,  # keep full frequency resolution for flexible downstream use
        verbose=False,
    )
    return con


def _connectivity_to_matrix(con: Any) -> np.ndarray:
    """
    Convert a connectivity object to a 2-D (n_ch × n_ch) matrix.

    Accepts both SpectralConnectivity (spectral nodes) and ConnectivityMatrix
    (broadband AEC). For SpectralConnectivity, values are averaged over the
    retained frequency axis. Returns a symmetric matrix.
    """
    if isinstance(con, ConnectivityMatrix):
        mat = con.matrix.copy()
        mat = (mat + mat.T) / 2.0
        return mat
    # SpectralConnectivity path — dense shape: (n_ch, n_ch, n_freqs)
    dense = con.get_data(output="dense")
    # Average across frequency axis → (n_ch, n_ch)
    mat = np.mean(dense, axis=-1)
    # Symmetrize: spectral_connectivity_epochs fills only upper or lower triangle
    mat = (mat + mat.T) / 2.0
    return mat


# ---------------------------------------------------------------------------
# Compute Coherence
# ---------------------------------------------------------------------------

def _execute_compute_coherence(
    epochs: mne.Epochs, params: dict
) -> "SpectralConnectivity":
    """
    Computes magnitude-squared coherence (COH) between all EEG channel pairs.

    Coherence is a frequency-domain measure of linear synchrony between two
    signals, ranging from 0 (no linear coupling) to 1 (perfect linear coupling).
    It captures amplitude and phase coupling jointly.

    CAUTION: coherence is sensitive to volume conduction — a signal from a deep
    source appears coherent at all surface electrodes without any true neural
    coupling.  Use PLI (Phase Lag Index) for a volume-conduction-robust estimate.
    """
    return _compute_connectivity(
        epochs,
        method="coh",
        fmin_hz=float(params["fmin_hz"]),
        fmax_hz=float(params["fmax_hz"]),
    )


COMPUTE_COHERENCE = NodeDescriptor(
    node_type="compute_coherence",
    display_name="Compute Coherence",
    category="Analysis",
    description=(
        "Computes magnitude-squared coherence (COH) between all pairs of EEG channels. "
        "Coherence is a frequency-domain measure of linear synchrony ranging from 0 to 1. "
        "Values above ~0.5 suggest coupling in the specified band. "
        "Requires segmented Epochs — at least 20 epochs recommended for stable estimates. "
        "Output is a SpectralConnectivity object; connect to Plot Connectivity Circle or "
        "Plot Connectivity Matrix for visualization. "
        "Note: coherence is affected by volume conduction. Use PLI for a robust alternative."
    ),
    tags=["coherence", "coh", "connectivity", "synchrony", "eeg", "frequency", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="conn_out", type="connectivity", label="Coherence"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin_hz",
            label="Min Frequency",
            type="float",
            default=4.0,
            min=0.5,
            max=250.0,
            step=1.0,
            unit="Hz",
            description=(
                "Lower bound of the frequency band to compute coherence in. "
                "Common bands: delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, "
                "beta 13–30 Hz, gamma 30–80 Hz."
            ),
        ),
        ParameterSchema(
            name="fmax_hz",
            label="Max Frequency",
            type="float",
            default=40.0,
            min=1.0,
            max=250.0,
            step=1.0,
            unit="Hz",
            description=(
                "Upper bound of the frequency band. Must be less than the Nyquist "
                "frequency (sfreq / 2). Connectivity is averaged across all "
                "frequencies between fmin and fmax."
            ),
        ),
    ],
    execute_fn=_execute_compute_coherence,
    code_template=lambda p: f'from mne_connectivity import spectral_connectivity_epochs\ncon = spectral_connectivity_epochs(epochs, method="coh", fmin={p.get("fmin_hz", 4.0)}, fmax={p.get("fmax_hz", 40.0)}, verbose=False)',
    methods_template=lambda p: f'Magnitude-squared coherence was computed between all EEG channel pairs in the {p.get("fmin_hz", 4.0)}–{p.get("fmax_hz", 40.0)} Hz range using mne-connectivity (Gramfort et al., 2013).',
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html",
)


# ---------------------------------------------------------------------------
# Compute PLV (Phase Locking Value)
# ---------------------------------------------------------------------------

def _execute_compute_plv(
    epochs: mne.Epochs, params: dict
) -> "SpectralConnectivity":
    """
    Computes Phase Locking Value (PLV) between all channel pairs.

    PLV measures the consistency of the phase difference between two signals
    across trials, ranging from 0 (random phase relationship) to 1 (perfectly
    locked phase). Unlike coherence, PLV is amplitude-independent — it captures
    purely phase-based coupling.

    PLV is still susceptible to volume conduction because a common source
    creates zero-lag phase coupling. Use PLI for volume-conduction-robust analysis.
    """
    return _compute_connectivity(
        epochs,
        method="plv",
        fmin_hz=float(params["fmin_hz"]),
        fmax_hz=float(params["fmax_hz"]),
    )


COMPUTE_PLV = NodeDescriptor(
    node_type="compute_plv",
    display_name="Compute PLV",
    category="Analysis",
    description=(
        "Computes Phase Locking Value (PLV) between all EEG channel pairs. "
        "PLV measures phase synchrony across trials, independent of amplitude. "
        "Values range from 0 (no phase coupling) to 1 (perfect phase locking). "
        "Requires segmented Epochs — the phase relationship is estimated across trials. "
        "PLV is affected by volume conduction (zero-lag coupling from shared sources). "
        "Use PLI for a volume-conduction-robust alternative."
    ),
    tags=["plv", "phase", "locking", "value", "synchrony", "connectivity", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="conn_out", type="connectivity", label="PLV"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin_hz",
            label="Min Frequency",
            type="float",
            default=4.0,
            min=0.5,
            max=250.0,
            step=1.0,
            unit="Hz",
            description="Lower bound of the frequency band for PLV estimation.",
        ),
        ParameterSchema(
            name="fmax_hz",
            label="Max Frequency",
            type="float",
            default=40.0,
            min=1.0,
            max=250.0,
            step=1.0,
            unit="Hz",
            description="Upper bound of the frequency band for PLV estimation.",
        ),
    ],
    execute_fn=_execute_compute_plv,
    code_template=lambda p: f'from mne_connectivity import spectral_connectivity_epochs\ncon = spectral_connectivity_epochs(epochs, method="plv", fmin={p.get("fmin_hz", 4.0)}, fmax={p.get("fmax_hz", 40.0)}, verbose=False)',
    methods_template=lambda p: f'Phase Locking Value (PLV) was computed between all EEG channel pairs in the {p.get("fmin_hz", 4.0)}–{p.get("fmax_hz", 40.0)} Hz range using mne-connectivity (Lachaux et al., 1999).',
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html",
)


# ---------------------------------------------------------------------------
# Compute PLI (Phase Lag Index)
# ---------------------------------------------------------------------------

def _execute_compute_pli(
    epochs: mne.Epochs, params: dict
) -> "SpectralConnectivity":
    """
    Computes Phase Lag Index (PLI) between all channel pairs.

    PLI measures the asymmetry of the distribution of phase differences: if a
    phase lead or lag is consistently present across trials, PLI is high. If
    the coupling is zero-lag (as with volume conduction), PLI is zero.

    PLI is therefore robust to volume conduction and common reference effects,
    making it the preferred measure for scalp EEG connectivity analyses in
    clinical and resting-state research. Values range from 0 to 1.
    """
    return _compute_connectivity(
        epochs,
        method="pli",
        fmin_hz=float(params["fmin_hz"]),
        fmax_hz=float(params["fmax_hz"]),
    )


COMPUTE_PLI = NodeDescriptor(
    node_type="compute_pli",
    display_name="Compute PLI",
    category="Analysis",
    description=(
        "Computes Phase Lag Index (PLI) between all EEG channel pairs. "
        "PLI measures phase synchrony that is NOT zero-lag — it is insensitive to "
        "volume conduction and common reference artifacts, making it the recommended "
        "measure for scalp EEG connectivity. Values range from 0 to 1. "
        "Use PLI when comparing groups or conditions where volume conduction differences "
        "could confound coherence- or PLV-based results."
    ),
    tags=["pli", "phase", "lag", "index", "connectivity", "volume-conduction", "robust", "analysis"],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="conn_out", type="connectivity", label="PLI"),
    ],
    parameters=[
        ParameterSchema(
            name="fmin_hz",
            label="Min Frequency",
            type="float",
            default=4.0,
            min=0.5,
            max=250.0,
            step=1.0,
            unit="Hz",
            description="Lower bound of the frequency band for PLI estimation.",
        ),
        ParameterSchema(
            name="fmax_hz",
            label="Max Frequency",
            type="float",
            default=40.0,
            min=1.0,
            max=250.0,
            step=1.0,
            unit="Hz",
            description="Upper bound of the frequency band for PLI estimation.",
        ),
    ],
    execute_fn=_execute_compute_pli,
    code_template=lambda p: f'from mne_connectivity import spectral_connectivity_epochs\ncon = spectral_connectivity_epochs(epochs, method="pli", fmin={p.get("fmin_hz", 4.0)}, fmax={p.get("fmax_hz", 40.0)}, verbose=False)',
    methods_template=lambda p: f'Phase Lag Index (PLI) was computed between all EEG channel pairs in the {p.get("fmin_hz", 4.0)}–{p.get("fmax_hz", 40.0)} Hz range using mne-connectivity (Stam et al., 2007).',
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html",
)


# ---------------------------------------------------------------------------
# Plot Connectivity Circle
# ---------------------------------------------------------------------------

def _execute_plot_connectivity_circle(con: Any, params: dict) -> str:
    """
    Renders a circular graph (Circos-style) showing the strongest connections
    between EEG channels.

    The input SpectralConnectivity is averaged across its retained frequency
    band to produce a 2-D connectivity matrix, then the n_lines strongest
    connections are drawn as colored arcs.

    This is the canonical visualization for functional connectivity in EEG
    publications — it clearly shows which channel pairs are most strongly coupled
    and in what spatial pattern.
    """
    _require_mne_connectivity()

    con_matrix = _connectivity_to_matrix(con)
    node_names = list(con.names)

    n_lines = int(params.get("n_lines", 10))
    cmap = str(params.get("colormap", "hot"))
    method_label = str(con.method) if con.method else "Connectivity"
    freq_label = (
        f"{con.freqs[0]:.1f}–{con.freqs[-1]:.1f} Hz"
        if len(con.freqs) > 0
        else ""
    )
    title = f"{method_label.upper()} — {freq_label}"

    fig, ax = plt.subplots(
        figsize=(8, 8),
        facecolor="black",
        subplot_kw=dict(polar=True),
    )

    _mne_plot_circle(
        con_matrix,
        node_names=node_names,
        n_lines=n_lines,
        colormap=cmap,
        title=title,
        ax=ax,
        show=False,
        interactive=False,
        facecolor="black",
        textcolor="white",
        node_edgecolor="white",
        colorbar=True,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


PLOT_CONNECTIVITY_CIRCLE = NodeDescriptor(
    node_type="plot_connectivity_circle",
    display_name="Plot Connectivity Circle",
    category="Visualization",
    description=(
        "Renders a circular (Circos-style) connectivity graph showing the strongest "
        "EEG channel connections. Each node on the circle represents an electrode; "
        "arcs between nodes show connection strength (color and thickness). "
        "The n_lines parameter controls how many connections to display — use a low "
        "value (10–20) to highlight only the dominant connections. "
        "Accepts output from Compute Coherence, PLV, or PLI."
    ),
    tags=["connectivity", "circle", "circos", "graph", "visualization", "plot", "eeg"],
    inputs=[
        HandleSchema(id="conn_in", type="connectivity", label="Connectivity"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Circle Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="n_lines",
            label="Top N Connections",
            type="int",
            default=10,
            min=1,
            max=500,
            step=1,
            description=(
                "Number of strongest connections to draw. Showing all connections "
                "on dense EEG arrays produces an unreadable plot — 10–30 is a "
                "good range for publication figures."
            ),
        ),
        ParameterSchema(
            name="colormap",
            label="Colormap",
            type="select",
            default="hot",
            options=["hot", "RdYlBu_r", "viridis", "plasma", "Reds", "Blues"],
            description=(
                "Colormap for the connection strength arcs. "
                "'hot': black→red→yellow (classic for connectivity). "
                "'RdYlBu_r': diverging, good for difference matrices. "
                "'viridis': perceptually uniform, colorblind-safe."
            ),
        ),
    ],
    execute_fn=_execute_plot_connectivity_circle,
    code_template=lambda p: f'from mne_connectivity.viz import plot_connectivity_circle\nplot_connectivity_circle(con_matrix, node_names=ch_names, n_lines={p.get("n_lines", 10)}, colormap="{p.get("colormap", "hot")}")',
    methods_template=None,
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.viz.plot_connectivity_circle.html",
)


# ---------------------------------------------------------------------------
# Plot Connectivity Matrix
# ---------------------------------------------------------------------------

def _execute_plot_connectivity_matrix(con: Any, params: dict) -> str:
    """
    Renders the pairwise connectivity matrix as an N×N heatmap.

    Rows and columns correspond to EEG channels; cell color encodes connection
    strength. This gives a complete view of all pairwise connections (not just
    the top N), making it useful for identifying spatial patterns across the
    entire scalp.

    The matrix is symmetric (upper and lower triangles are averaged) and the
    diagonal is set to NaN to hide self-connections (which are always 1).

    Accepts SpectralConnectivity (from compute_coherence/plv/pli) or
    ConnectivityMatrix (from compute_envelope_correlation) — no mne-connectivity
    required for rendering since only matplotlib is used here.
    """

    con_matrix = _connectivity_to_matrix(con)
    node_names = list(con.names)
    n_ch = len(node_names)

    # Hide diagonal (self-connections = trivially 1)
    np.fill_diagonal(con_matrix, np.nan)

    cmap = str(params.get("colormap", "RdYlBu_r"))
    method_label = str(con.method) if con.method else "Connectivity"
    freq_label = (
        f"{con.freqs[0]:.1f}–{con.freqs[-1]:.1f} Hz"
        if len(con.freqs) > 0
        else ""
    )

    fig, ax = plt.subplots(figsize=(max(6, n_ch * 0.5), max(5, n_ch * 0.5)))
    im = ax.imshow(
        con_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Connectivity strength")

    ax.set_xticks(range(n_ch))
    ax.set_yticks(range(n_ch))
    ax.set_xticklabels(node_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(node_names, fontsize=7)
    ax.set_title(f"{method_label.upper()} Matrix — {freq_label}", fontsize=11)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#334155")
    # Fix colorbar text colors
    cbar = im.axes.figure.axes[-1]
    cbar.yaxis.label.set_color("white")
    cbar.tick_params(colors="white")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


PLOT_CONNECTIVITY_MATRIX = NodeDescriptor(
    node_type="plot_connectivity_matrix",
    display_name="Plot Connectivity Matrix",
    category="Visualization",
    description=(
        "Renders a pairwise EEG connectivity matrix as an N×N heatmap. "
        "Each cell shows the connection strength between two electrodes — "
        "bright colors indicate strong coupling, dark colors indicate weak coupling. "
        "Diagonal cells (self-connections) are hidden. "
        "Unlike the circle plot, this view shows ALL pairwise connections simultaneously, "
        "making it easy to spot clusters of strongly connected regions. "
        "Accepts output from Compute Coherence, PLV, or PLI."
    ),
    tags=["connectivity", "matrix", "heatmap", "grid", "visualization", "plot", "eeg"],
    inputs=[
        HandleSchema(id="conn_in", type="connectivity", label="Connectivity"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="Matrix Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="colormap",
            label="Colormap",
            type="select",
            default="RdYlBu_r",
            options=["RdYlBu_r", "hot", "viridis", "plasma", "Reds", "Blues", "YlOrRd"],
            description=(
                "Color scale for the matrix cells. "
                "'RdYlBu_r': diverging (blue=low, red=high) — highlights strong vs weak links. "
                "'hot': classic grayscale→red→yellow for connectivity publication figures. "
                "'viridis': perceptually uniform, colorblind-safe."
            ),
        ),
    ],
    execute_fn=_execute_plot_connectivity_matrix,
    code_template=lambda p: f'fig, ax = plt.subplots()\nim = ax.imshow(con_matrix, cmap="{p.get("colormap", "RdYlBu_r")}", vmin=0, vmax=1)\nplt.colorbar(im, ax=ax)',
    methods_template=None,
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html",
)


# ---------------------------------------------------------------------------
# Compute Envelope Correlation (AEC)
# ---------------------------------------------------------------------------

def _execute_compute_envelope_correlation(
    epochs: mne.Epochs, params: dict
) -> ConnectivityMatrix:
    """
    Computes Amplitude Envelope Correlation (AEC) between all EEG channel pairs.

    AEC measures broadband amplitude coupling by:
      1. Applying the Hilbert transform to each channel's signal to get the
         analytic signal (carrier + envelope).
      2. Taking the absolute value of the analytic signal to extract the
         amplitude envelope (slow modulation of power over time).
      3. Computing Pearson correlation between all channel-pair envelopes,
         concatenating all epochs before correlating.

    AEC captures co-modulation of amplitude envelopes across time — a measure
    complementary to phase-based connectivity (PLV, PLI). It is widely used
    in MEG resting-state analyses and is sensitive to long-range amplitude
    coupling at frequencies in the alpha/beta band.

    Unlike spectral coherence or PLV, AEC does NOT require a frequency
    resolution from many cycles — it works on broadband data filtered to
    the band of interest before epoching.

    Returns a ConnectivityMatrix (not SpectralConnectivity) compatible with
    Plot Connectivity Circle and Plot Connectivity Matrix.
    """
    from scipy.signal import hilbert

    ch_names = list(epochs.ch_names)
    n_ch = len(ch_names)

    # Get raw epoch data: (n_epochs, n_ch, n_times)
    data = epochs.get_data()

    # Amplitude envelope via Hilbert transform along the time axis
    # envelopes shape: (n_epochs, n_ch, n_times)
    envelopes = np.abs(hilbert(data, axis=-1))

    # Reshape to (n_ch, n_epochs * n_times) so that np.corrcoef gives an
    # (n_ch, n_ch) Pearson correlation matrix across all time points and epochs
    envelopes_flat = envelopes.transpose(1, 0, 2).reshape(n_ch, -1)

    # Pearson correlation — symmetric (n_ch, n_ch) matrix, values in [-1, 1]
    corr_matrix = np.corrcoef(envelopes_flat)

    # Guard against NaN rows (e.g., zero-variance channel in synthetic data)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    return ConnectivityMatrix(
        matrix=corr_matrix,
        names=ch_names,
        method="aec",
        freqs=[],  # broadband — no single frequency
    )


COMPUTE_ENVELOPE_CORRELATION = NodeDescriptor(
    node_type="compute_envelope_correlation",
    display_name="Compute Envelope Correlation",
    category="Analysis",
    description=(
        "Computes Amplitude Envelope Correlation (AEC) between all EEG channel pairs. "
        "AEC measures broadband amplitude coupling: for each channel, the Hilbert transform "
        "extracts the amplitude envelope (slow power modulations), and Pearson correlations "
        "between all envelope pairs are computed across all time points and epochs. "
        "AEC is complementary to phase-based measures (PLV, PLI) — it captures slow "
        "amplitude co-modulation rather than instantaneous phase locking. "
        "Best results when input epochs are pre-filtered to the band of interest "
        "using Bandpass Filter before epoching. "
        "Connect to Plot Connectivity Circle or Plot Connectivity Matrix to visualize results."
    ),
    tags=[
        "aec", "envelope", "correlation", "amplitude", "coupling",
        "broadband", "connectivity", "resting-state", "analysis",
    ],
    inputs=[
        HandleSchema(id="epochs_in", type="epochs", label="Epochs"),
    ],
    outputs=[
        HandleSchema(id="conn_out", type="connectivity", label="AEC"),
    ],
    parameters=[],  # broadband: no frequency parameters needed
    execute_fn=_execute_compute_envelope_correlation,
    code_template=lambda p: 'from scipy.signal import hilbert\nenvelopes = np.abs(hilbert(epochs.get_data(), axis=-1))\ncorr_matrix = np.corrcoef(envelopes.reshape(n_ch, -1))',
    methods_template=lambda p: "Amplitude Envelope Correlation (AEC) was computed by correlating Hilbert-transformed amplitude envelopes across all channel pairs and epochs.",
    docs_url="https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.envelope_correlation.html",
)
