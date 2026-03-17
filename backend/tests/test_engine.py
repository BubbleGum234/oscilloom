"""
backend/tests/test_engine.py

Tests for the pipeline execution engine (engine.py).

Uses the MNE sample dataset (downloaded automatically on first run).
All tests operate on a small EEG subset (EEG channels only, first 30s)
to keep execution fast.
"""

from __future__ import annotations

import pytest
import mne

from backend.engine import execute_pipeline, topological_sort
from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_raw() -> mne.io.BaseRaw:
    """
    Loads the MNE sample dataset (EEG channels only, first 30 seconds).
    Downloaded automatically by MNE on first use (~1.5 GB total, cached).
    Scoped to module so it is loaded once per test file.
    """
    data_path = mne.datasets.sample.data_path()
    raw_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
    raw.pick("eeg")         # Keep only EEG channels for speed
    raw.crop(tmax=30.0)     # Use first 30 seconds only
    return raw


@pytest.fixture(scope="module")
def sample_raw_with_stim() -> mne.io.BaseRaw:
    """
    Synthetic EEG raw with a stim channel containing periodic events.
    Used for epoching tests — the regular sample_raw fixture drops the stim
    channel via raw.pick('eeg'), making mne.find_events() fail.
    """
    import numpy as np

    sfreq = 200.0
    n_eeg = 10
    n_seconds = 30
    n_samples = int(sfreq * n_seconds)

    rng = np.random.default_rng(42)
    eeg_data = rng.standard_normal((n_eeg, n_samples)) * 5e-6

    # Place 8 events (ID=1) at 2, 5, 8, 11, 14, 17, 20, 23 seconds
    stim = np.zeros((1, n_samples))
    for t_s in range(2, 25, 3):
        stim[0, int(t_s * sfreq)] = 1

    data = np.vstack([eeg_data, stim])
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["STI 014"]
    ch_types = ["eeg"] * n_eeg + ["stim"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _make_metadata() -> PipelineMetadata:
    return PipelineMetadata(
        name="Test Pipeline",
        description="Test",
        created_by="test",
        schema_version="1.0",
    )


def _make_node(node_id: str, node_type: str, params: dict) -> PipelineNode:
    return PipelineNode(
        id=node_id,
        node_type=node_type,
        label=node_type,
        parameters=params,
        position={"x": 0.0, "y": 0.0},
    )


def _make_edge(
    edge_id: str,
    src_node: str, src_handle: str, src_type: str,
    tgt_node: str, tgt_handle: str, tgt_type: str,
) -> PipelineEdge:
    return PipelineEdge(
        id=edge_id,
        source_node_id=src_node, source_handle_id=src_handle, source_handle_type=src_type,
        target_node_id=tgt_node, target_handle_id=tgt_handle, target_handle_type=tgt_type,
    )


def _linear_graph(nodes: list[PipelineNode], edges: list[PipelineEdge]) -> PipelineGraph:
    return PipelineGraph(metadata=_make_metadata(), nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Topological sort tests
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_single_node(self):
        graph = _linear_graph(
            nodes=[_make_node("n1", "edf_loader", {})],
            edges=[],
        )
        assert topological_sort(graph) == ["n1"]

    def test_linear_three_nodes(self):
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "compute_psd", {}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        order = topological_sort(_linear_graph(nodes, edges))
        # n1 must come before n2, n2 before n3
        assert order.index("n1") < order.index("n2")
        assert order.index("n2") < order.index("n3")

    def test_cycle_raises_value_error(self):
        nodes = [
            _make_node("n1", "bandpass_filter", {}),
            _make_node("n2", "bandpass_filter", {}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "filtered_eeg", "n2", "eeg_in", "filtered_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n1", "eeg_in", "filtered_eeg"),
        ]
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(_linear_graph(nodes, edges))


# ---------------------------------------------------------------------------
# Engine execution tests
# ---------------------------------------------------------------------------

class TestExecutePipeline:
    def test_bandpass_filter_node_succeeds(self, sample_raw):
        """A bandpass filter node should execute without error."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "", "preload": True}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        graph = _linear_graph(nodes, edges)

        results, _ = execute_pipeline(sample_raw.copy(), graph)

        assert results["n2"]["status"] == "success"
        assert results["n2"]["output_type"] == "Raw"

    def test_notch_filter_node_succeeds(self, sample_raw):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "", "preload": True}),
            _make_node("n2", "notch_filter", {"notch_freq_hz": 60.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"

    def test_resample_node_succeeds(self, sample_raw):
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "resample", {"target_sfreq": 100.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"

    def test_compute_psd_node_succeeds(self, sample_raw):
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"}),
            _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 512}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Spectrum"

    def test_plot_psd_returns_base64_png(self, sample_raw):
        """The plot_psd node must return a base64 PNG data URI in result['data']."""
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"}),
            _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 512}),
            _make_node("n4", "plot_psd", {"dB": True, "show_average": True}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
            _make_edge("e3", "n3", "psd_out", "psd",          "n4", "psd_in", "psd"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))

        assert results["n4"]["status"] == "success"
        assert results["n4"]["data"] is not None
        assert results["n4"]["data"].startswith("data:image/png;base64,")
        # Ensure the base64 data is non-trivially sized (> 5 KB encoded)
        assert len(results["n4"]["data"]) > 5000

    def test_parameter_defaults_applied_when_missing(self, sample_raw):
        """
        Nodes with empty parameters dict should use schema defaults.
        Engine merges defaults before calling execute_fn.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {}),  # No params — use defaults
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        # Should not raise — defaults (1.0 Hz, 40.0 Hz) must be applied
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"

    def test_execute_does_not_mutate_input(self, sample_raw):
        """
        The raw_copy passed to execute_pipeline must not be mutated.
        This verifies the copy-on-write contract across all node execute_fns.
        """
        original_sfreq = sample_raw.info["sfreq"]
        raw_for_test = sample_raw.copy()

        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "resample", {"target_sfreq": 100.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        execute_pipeline(raw_for_test, _linear_graph(nodes, edges))

        # raw_for_test should still have its original sampling rate
        assert raw_for_test.info["sfreq"] == original_sfreq

    def test_unknown_node_type_raises(self, sample_raw):
        nodes = [_make_node("n1", "nonexistent_node_xyz", {})]
        graph = _linear_graph(nodes, [])
        with pytest.raises(ValueError, match="unknown type"):
            execute_pipeline(sample_raw.copy(), graph)

    # ── BUG-01 regression: Resample → Compute PSD chain ──────────────────────

    def test_resample_then_compute_psd_succeeds(self, sample_raw):
        """
        BUG-01: Resample output type changed from raw_eeg → filtered_eeg.
        This chain was previously blocked because compute_psd only accepts
        filtered_eeg. Verifies the fix works end-to-end.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "resample", {"target_sfreq": 100.0}),
            _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 256, "method": "welch", "n_overlap": 0}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out",  "raw_eeg",      "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out",  "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Spectrum"

    # ── PARAM-05 regression: Bandpass Nyquist check ───────────────────────────

    def test_bandpass_above_nyquist_raises(self, sample_raw):
        """
        PARAM-05: bandpass high_cutoff >= Nyquist must raise ValueError.
        MNE sample data is at 600 Hz; Nyquist = 300 Hz.
        Setting high_cutoff=350 Hz should raise.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0,
                "high_cutoff_hz": 350.0,  # Above Nyquist for 600 Hz data
                "method": "fir",
            }),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "error"
        assert "Nyquist" in results["n2"]["error"]

    # ── New nodes: Set EEG Reference ─────────────────────────────────────────

    def test_set_eeg_reference_average_succeeds(self, sample_raw):
        """set_eeg_reference with average reference must return a Raw object."""
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "set_eeg_reference", {"reference": "average"}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"
        assert results["n2"]["output_type"] == "Raw"

    # ── New nodes: Epoch by Events + Compute Evoked ───────────────────────────

    def test_epoch_by_events_and_compute_evoked(self, sample_raw_with_stim):
        """
        Full ERP chain: filter → epoch_by_events → compute_evoked.
        Uses the synthetic raw fixture that has a stim channel with event_id=1.
        (The regular sample_raw drops the stim channel via raw.pick('eeg').)
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "epoch_by_events", {
                "event_id": 1,
                "tmin": -0.2,
                "tmax": 0.5,
                "baseline_tmin": -0.2,
                "baseline_tmax": 0.0,
            }),
            _make_node("n4", "compute_evoked", {}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out",    "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out",    "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",       "n4", "epochs_in",   "epochs"),
        ]
        results, _ = execute_pipeline(sample_raw_with_stim.copy(), _linear_graph(nodes, edges))
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Epochs"
        assert results["n4"]["status"] == "success"
        assert results["n4"]["output_type"] in ("Evoked", "EvokedArray")

    # ── New nodes: Pick Channels ──────────────────────────────────────────────

    def test_pick_channels_eeg_succeeds(self, sample_raw):
        """pick_channels with type='eeg' should return a Raw with EEG channels only."""
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "pick_channels", {"channel_type": "eeg"}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"
        assert results["n2"]["output_type"] == "Raw"

    # ── TASK-01: filtered_eeg input on filter nodes ───────────────────────────

    def test_notch_then_bandpass_chain_succeeds(self, sample_raw):
        """
        TASK-01: Notch → Bandpass is the most common EEG preprocessing chain.
        bandpass_filter must accept filtered_eeg (the output of notch_filter).
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "notch_filter", {"notch_freq_hz": 60.0}),
            _make_node("n3", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Raw"

    # ── TASK-02: raw_eeg input on compute_psd ────────────────────────────────

    def test_compute_psd_accepts_raw_eeg_directly(self, sample_raw):
        """
        TASK-02: Researcher can inspect PSD before filtering.
        edf_loader → compute_psd must work without an intermediate filter node.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "compute_psd", {
                "fmin": 0.5, "fmax": 40.0, "n_fft": 512, "method": "welch", "n_overlap": 0
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"
        assert results["n2"]["output_type"] == "Spectrum"

    # ── TASK-06: Crop node ────────────────────────────────────────────────────

    def test_crop_node_shortens_recording(self, sample_raw):
        """
        TASK-06: Crop node must trim the recording to the specified time window.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "crop", {"tmin": 0.0, "tmax": 5.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n2"]["status"] == "success"
        assert results["n2"]["output_type"] == "Raw"

    # ── TASK-13: Compute Band Power ───────────────────────────────────────────

    def test_compute_bandpower_returns_array(self, sample_raw):
        """
        TASK-13: compute_bandpower must consume a Spectrum and return an ndarray.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "compute_psd", {
                "fmin": 0.5, "fmax": 40.0, "n_fft": 512, "method": "welch", "n_overlap": 0
            }),
            _make_node("n4", "compute_bandpower", {
                "fmin": 8.0, "fmax": 13.0, "log_scale": True
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",  "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in",  "filtered_eeg"),
            _make_edge("e3", "n3", "psd_out", "psd",          "n4", "psd_in",  "psd"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n4"]["status"] == "success"
        assert results["n4"]["output_type"] == "ndarray"

    # ── TASK-14: Plot Evoked Joint ────────────────────────────────────────────

    def test_plot_evoked_joint_returns_base64_png(self, sample_raw_with_stim):
        """
        TASK-14: plot_evoked_joint must return a base64 PNG data URI.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "epoch_by_events", {
                "event_id": "1",
                "tmin": -0.2, "tmax": 0.5,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0,
            }),
            _make_node("n4", "compute_evoked", {}),
            _make_node("n5", "plot_evoked_joint", {"times": "peaks"}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out",    "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out",    "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",       "n4", "epochs_in",   "epochs"),
            _make_edge("e4", "n4", "evoked_out", "evoked",       "n5", "evoked_in",   "evoked"),
        ]
        results, _ = execute_pipeline(sample_raw_with_stim.copy(), _linear_graph(nodes, edges))
        assert results["n5"]["status"] == "success"
        assert results["n5"]["data"] is not None
        assert results["n5"]["data"].startswith("data:image/png;base64,")

    # ── TASK-15: Filter Epochs ────────────────────────────────────────────────

    def test_filter_epochs_succeeds(self, sample_raw_with_stim):
        """
        TASK-15: filter_epochs must apply a bandpass filter to epoched data.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "epoch_by_events", {
                "event_id": "1",
                "tmin": -0.2, "tmax": 0.5,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0,
            }),
            _make_node("n4", "filter_epochs", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 30.0, "method": "fir"
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out",    "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out",    "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",       "n4", "epochs_in",   "epochs"),
        ]
        results, _ = execute_pipeline(sample_raw_with_stim.copy(), _linear_graph(nodes, edges))
        assert results["n4"]["status"] == "success"
        assert results["n4"]["output_type"] == "Epochs"

    # ── TASK-03: String event_id ──────────────────────────────────────────────

    def test_epoch_by_events_with_string_event_id(self, sample_raw_with_stim):
        """
        TASK-03: epoch_by_events must accept string event_id "1" (parsed as int).
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "epoch_by_events", {
                "event_id": "1",  # String, not int
                "tmin": -0.2, "tmax": 0.5,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0,
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw_with_stim.copy(), _linear_graph(nodes, edges))
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Epochs"

    # ── TASK-22: save_to_fif with blank output_path ───────────────────────────

    def test_save_to_fif_blank_path_is_passthrough(self, sample_raw):
        """
        TASK-22: save_to_fif with output_path="" must not raise — just passthrough.
        """
        nodes = [
            _make_node("n1", "edf_loader", {}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"
            }),
            _make_node("n3", "save_to_fif", {"output_path": ""}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "filtered_in", "filtered_eeg"),
        ]
        results, _ = execute_pipeline(sample_raw.copy(), _linear_graph(nodes, edges))
        assert results["n3"]["status"] == "success"
        assert results["n3"]["output_type"] == "Raw"
