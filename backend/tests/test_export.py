"""
backend/tests/test_export.py

Tests for the Python script export feature (script_exporter.py).

Tests verify:
  - Linear pipeline exports produce valid Python (ast.parse-able)
  - The script contains expected MNE API calls for each node type
  - Audit log entries appear as comments in the generated script
  - Default parameters are rendered explicitly (no hidden defaults)
  - Branching pipelines raise ValueError
  - Single-node (source only) pipelines export correctly
  - All 6 MVP node types appear in the exported script when used
"""

from __future__ import annotations

import ast
import re

import pytest

from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)
from backend.script_exporter import export


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(name: str = "Test Pipeline") -> PipelineMetadata:
    return PipelineMetadata(
        name=name,
        description="Exported test pipeline",
        created_by="test",
        schema_version="1.0",
    )


def _make_node(node_id: str, node_type: str, params: dict | None = None) -> PipelineNode:
    return PipelineNode(
        id=node_id,
        node_type=node_type,
        label=node_type,
        parameters=params or {},
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


def _graph(nodes: list[PipelineNode], edges: list[PipelineEdge]) -> PipelineGraph:
    return PipelineGraph(metadata=_make_metadata(), nodes=nodes, edges=edges)


def _bandpass_filter_graph() -> PipelineGraph:
    """Minimal valid pipeline: edf_loader → bandpass_filter."""
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
        _make_node("n2", "bandpass_filter", {"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"}),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
    ]
    return _graph(nodes, edges)


def _full_pipeline_graph() -> PipelineGraph:
    """Full 4-node pipeline: edf_loader → bandpass_filter → compute_psd → plot_psd."""
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
        _make_node("n2", "bandpass_filter", {"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"}),
        _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 512}),
        _make_node("n4", "plot_psd", {"dB": True, "show_average": True}),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
        _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        _make_edge("e3", "n3", "psd_out", "psd", "n4", "psd_in", "psd"),
    ]
    return _graph(nodes, edges)


# ---------------------------------------------------------------------------
# Syntactic validity
# ---------------------------------------------------------------------------

class TestScriptSyntax:
    def test_exported_script_is_valid_python(self):
        """The exported script must parse cleanly with ast.parse()."""
        script = export(_bandpass_filter_graph(), audit_log=[])
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Exported script has a syntax error at line {e.lineno}: {e.msg}\n{script}")

    def test_full_pipeline_script_is_valid_python(self):
        script = export(_full_pipeline_graph(), audit_log=[])
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Full pipeline script has syntax error: {e.msg}")

    def test_export_returns_string(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert isinstance(script, str)
        assert len(script) > 0


# ---------------------------------------------------------------------------
# MNE API call presence
# ---------------------------------------------------------------------------

class TestMneApiCalls:
    def test_edf_loader_generates_read_raw_edf_call(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "mne.io.read_raw_edf" in script

    def test_edf_loader_file_path_in_script(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "/data/test.edf" in script

    def test_bandpass_filter_generates_filter_call(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert ".filter(" in script

    def test_bandpass_filter_params_explicit(self):
        """Low cutoff, high cutoff, and method must appear explicitly."""
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "l_freq=1.0" in script
        assert "h_freq=40.0" in script
        assert 'method="fir"' in script

    def test_notch_filter_generates_notch_filter_call(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "notch_filter", {"notch_freq_hz": 60.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        script = export(_graph(nodes, edges), audit_log=[])
        assert ".notch_filter(" in script
        assert "60.0" in script

    def test_resample_generates_resample_call(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "resample", {"target_sfreq": 250.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg")]
        script = export(_graph(nodes, edges), audit_log=[])
        assert ".resample(" in script
        assert "250.0" in script

    def test_compute_psd_generates_compute_psd_call(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 512}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert ".compute_psd(" in script
        assert "fmin=0.5" in script
        assert "fmax=40.0" in script
        assert "n_fft=512" in script

    def test_plot_psd_generates_spectrum_plot_call(self):
        script = export(_full_pipeline_graph(), audit_log=[])
        assert "spectrum.plot(" in script

    def test_plot_psd_saves_figure(self):
        """The plot node must save the figure to a file."""
        script = export(_full_pipeline_graph(), audit_log=[])
        assert "fig.savefig(" in script
        assert "plt.close(fig)" in script

    def test_imports_present(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "import mne" in script

    def test_matplotlib_agg_backend_set(self):
        """The non-GUI backend must be set for headless execution."""
        script = export(_full_pipeline_graph(), audit_log=[])
        assert 'matplotlib.use("Agg")' in script


# ---------------------------------------------------------------------------
# Audit log rendering
# ---------------------------------------------------------------------------

class TestAuditLogRendering:
    def test_audit_log_empty_no_audit_section(self):
        """When audit_log is empty, no AUDIT LOG header should appear."""
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "AUDIT LOG" not in script

    def test_audit_log_entries_appear_as_comments(self):
        audit_log = [
            {
                "timestamp": "2026-02-20T10:00:00",
                "nodeId": "n2",
                "nodeDisplayName": "Bandpass Filter",
                "paramLabel": "High Cutoff",
                "oldValue": "40.0",
                "newValue": "45.0",
                "unit": "Hz",
            }
        ]
        script = export(_bandpass_filter_graph(), audit_log=audit_log)
        assert "AUDIT LOG" in script
        # All comment lines start with #
        audit_section_lines = [
            line for line in script.splitlines()
            if "AUDIT LOG" in line or "Bandpass Filter" in line
        ]
        for line in audit_section_lines:
            stripped = line.strip()
            assert stripped.startswith("#"), (
                f"Audit log line is not a comment: {line!r}"
            )

    def test_audit_log_shows_param_change(self):
        audit_log = [
            {
                "timestamp": "2026-02-20T10:00:00",
                "nodeId": "n2",
                "nodeDisplayName": "Bandpass Filter",
                "paramLabel": "High Cutoff",
                "oldValue": "40.0",
                "newValue": "45.0",
                "unit": "Hz",
            }
        ]
        script = export(_bandpass_filter_graph(), audit_log=audit_log)
        assert "High Cutoff" in script
        assert "40.0" in script
        assert "45.0" in script

    def test_multiple_audit_entries_all_present(self):
        audit_log = [
            {
                "timestamp": "2026-02-20T10:00:00",
                "nodeId": "n2",
                "nodeDisplayName": "Bandpass Filter",
                "paramLabel": "Low Cutoff",
                "oldValue": "1.0",
                "newValue": "2.0",
                "unit": "Hz",
            },
            {
                "timestamp": "2026-02-20T10:01:00",
                "nodeId": "n2",
                "nodeDisplayName": "Bandpass Filter",
                "paramLabel": "High Cutoff",
                "oldValue": "40.0",
                "newValue": "50.0",
                "unit": "Hz",
            },
        ]
        script = export(_bandpass_filter_graph(), audit_log=audit_log)
        assert "Low Cutoff" in script
        assert "High Cutoff" in script


# ---------------------------------------------------------------------------
# Default parameter rendering
# ---------------------------------------------------------------------------

class TestDefaultParameters:
    def test_defaults_applied_when_no_params_given(self):
        """
        Nodes with empty parameters dict should use schema defaults.
        The defaults must appear explicitly in the exported script.
        """
        nodes = [
            _make_node("n1", "edf_loader"),            # empty params
            _make_node("n2", "bandpass_filter"),        # empty params → uses schema defaults
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        # Schema defaults for bandpass_filter are 1.0 Hz and 40.0 Hz
        assert "l_freq=1.0" in script
        assert "h_freq=40.0" in script


# ---------------------------------------------------------------------------
# Branching pipeline rejection
# ---------------------------------------------------------------------------

class TestBranchingPipelineExport:
    def test_branching_pipeline_exports_valid_python(self):
        """Branching pipelines (one node → multiple targets) now export correctly."""
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
            _make_node("n3", "notch_filter"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n1", "eeg_out", "raw_eeg", "n3", "eeg_in", "raw_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        ast.parse(script)
        assert ".filter(" in script
        assert ".notch_filter(" in script

    def test_merging_pipeline_exports_valid_python(self):
        """Merging pipelines (multiple sources → one node) now export correctly."""
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
            _make_node("n3", "notch_filter"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n3", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "raw_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        ast.parse(script)


# ---------------------------------------------------------------------------
# Metadata in script header
# ---------------------------------------------------------------------------

class TestScriptHeader:
    def test_pipeline_name_in_header(self):
        graph = PipelineGraph(
            metadata=_make_metadata("My EEG Pipeline"),
            nodes=[_make_node("n1", "edf_loader")],
            edges=[],
        )
        script = export(graph, audit_log=[])
        assert "My EEG Pipeline" in script

    def test_timestamp_in_header(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        # Should contain a date in YYYY-MM-DD format
        assert re.search(r"\d{4}-\d{2}-\d{2}", script)

    def test_oscilloom_branding_in_header(self):
        script = export(_bandpass_filter_graph(), audit_log=[])
        assert "Oscilloom" in script


# ---------------------------------------------------------------------------
# BUG-02 regression: plot_raw and plot_topomap export templates
# ---------------------------------------------------------------------------

class TestPlotNodeExportTemplates:
    def test_plot_raw_export_is_valid_python(self):
        """BUG-02: plot_raw must have an export template (was NotImplementedError)."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "plot_raw", {
                "n_channels": 10, "start_time_s": 0.0, "duration_s": 10.0
            }),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "NotImplementedError" not in script
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"plot_raw export has syntax error: {e.msg}")

    def test_plot_raw_export_contains_matplotlib(self):
        """plot_raw export must use matplotlib (not raw.plot())."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "plot_raw", {"n_channels": 10, "start_time_s": 0.0, "duration_s": 10.0}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "raw_signal.png" in script
        assert "savefig" in script

    def test_plot_topomap_export_is_valid_python(self):
        """BUG-02: plot_topomap must have an export template (was NotImplementedError)."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "compute_psd", {}),
            _make_node("n4", "plot_topomap", {"bands": "alpha"}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",  "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in",  "filtered_eeg"),
            _make_edge("e3", "n3", "psd_out", "psd",          "n4", "psd_in",  "psd"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "NotImplementedError" not in script
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"plot_topomap export has syntax error: {e.msg}")

    def test_plot_topomap_export_contains_montage(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "compute_psd", {}),
            _make_node("n4", "plot_topomap", {"bands": "alpha"}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
            _make_edge("e3", "n3", "psd_out", "psd",          "n4", "psd_in", "psd"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "standard_1020" in script
        assert "plot_topomap" in script


# ---------------------------------------------------------------------------
# New node export template tests
# ---------------------------------------------------------------------------

class TestNewNodeExportTemplates:
    def test_set_eeg_reference_export_valid_python(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "set_eeg_reference", {"reference": "average"}),
        ]
        edges = [_make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg")]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "NotImplementedError" not in script
        assert "set_eeg_reference" in script
        ast.parse(script)

    def test_epoch_by_events_export_valid_python(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "epoch_by_events", {
                "event_id": 1, "tmin": -0.2, "tmax": 0.8,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",     "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "filtered_in","filtered_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "NotImplementedError" not in script
        assert "mne.Epochs" in script
        assert "find_events" in script
        ast.parse(script)

    def test_compute_evoked_export_valid_python(self):
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "epoch_by_events", {
                "event_id": 1, "tmin": -0.2, "tmax": 0.8,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0,
            }),
            _make_node("n4", "compute_evoked", {}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out",    "raw_eeg",      "n2", "eeg_in",     "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out",    "filtered_eeg", "n3", "filtered_in","filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",       "n4", "epochs_in",  "epochs"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert "NotImplementedError" not in script
        assert "epochs.average()" in script
        ast.parse(script)

    def test_compute_psd_multitaper_export_valid_python(self):
        """PARAM-04: multitaper method must produce valid export without n_fft."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {}),
            _make_node("n3", "compute_psd", {
                "method": "multitaper", "fmin": 0.5, "fmax": 40.0,
                "n_fft": 2048, "n_overlap": 0,
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        assert 'method="multitaper"' in script
        assert "n_fft" not in script.split('method="multitaper"')[1].split("print")[0]
        ast.parse(script)


# ---------------------------------------------------------------------------
# code_template-based export system tests
# ---------------------------------------------------------------------------

from unittest.mock import patch
import dataclasses

from backend.registry import NODE_REGISTRY


class TestCodeTemplateExport:
    """Tests for the generic code_template-based export system."""

    # 1. All nodes produce valid Python when exported as single-node pipelines
    @pytest.mark.parametrize("node_type", list(NODE_REGISTRY.keys()))
    def test_single_node_export_produces_valid_python(self, node_type):
        """Every node type must produce ast.parse-able Python when exported alone."""
        descriptor = NODE_REGISTRY[node_type]
        default_params = {p.name: p.default for p in descriptor.parameters}
        node = _make_node("n1", node_type, default_params)
        graph = _graph([node], [])
        try:
            script = export(graph, audit_log=[])
        except ValueError:
            # Nodes that fail the linear pipeline check or other validation
            # are acceptable to skip here — the test is about syntax, not topology.
            pytest.skip(f"Node '{node_type}' cannot be exported as a single-node pipeline")
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(
                f"Node '{node_type}' export has syntax error at line {e.lineno}: {e.msg}"
            )

    # 2. code_template coverage — every node in NODE_REGISTRY has a non-None code_template
    def test_all_nodes_have_code_template(self):
        """Every registered node must have a non-None code_template callable."""
        missing = [
            nt for nt, desc in NODE_REGISTRY.items()
            if desc.code_template is None and not nt.startswith("c_")
        ]
        assert missing == [], (
            f"The following {len(missing)} node(s) are missing code_template: {missing}"
        )

    # 3. No "SKIPPED" message for any node with code_template
    @pytest.mark.parametrize("node_type", list(NODE_REGISTRY.keys()))
    def test_no_skipped_message_for_nodes_with_code_template(self, node_type):
        """Nodes with code_template must NOT produce a SKIPPED message in export."""
        descriptor = NODE_REGISTRY[node_type]
        if descriptor.code_template is None:
            pytest.skip(f"Node '{node_type}' has no code_template")
        default_params = {p.name: p.default for p in descriptor.parameters}
        node = _make_node("n1", node_type, default_params)
        graph = _graph([node], [])
        try:
            script = export(graph, audit_log=[])
        except ValueError:
            pytest.skip(f"Node '{node_type}' cannot be exported as a single-node pipeline")
        assert "SKIPPED" not in script, (
            f"Node '{node_type}' has code_template but export contains SKIPPED"
        )

    # 4. Nodes without code_template get SKIPPED message (mock test)
    def test_node_without_code_template_gets_skipped_message(self):
        """A hypothetical node with code_template=None should produce a SKIPPED message."""
        # Pick any existing node and temporarily nullify its code_template
        sample_type = "edf_loader"
        descriptor = NODE_REGISTRY[sample_type]
        patched = dataclasses.replace(descriptor, code_template=None)

        with patch.dict(NODE_REGISTRY, {sample_type: patched}):
            node = _make_node("n1", sample_type, {"file_path": "/data/test.edf"})
            graph = _graph([node], [])
            script = export(graph, audit_log=[])
            assert "SKIPPED" in script
            assert sample_type in script
            # Still must be valid Python
            ast.parse(script)

    # 5. Multi-node pipeline with newer nodes exports cleanly
    def test_multi_node_pipeline_with_newer_nodes_exports_cleanly(self):
        """edf_loader -> bandpass_filter -> epoch_by_events -> compute_evoked exports valid Python."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 0.1, "high_cutoff_hz": 30.0, "method": "fir",
            }),
            _make_node("n3", "epoch_by_events", {
                "event_id": 1, "tmin": -0.2, "tmax": 0.8,
                "baseline_tmin": -0.2, "baseline_tmax": 0.0,
            }),
            _make_node("n4", "compute_evoked", {}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg",  "n3", "filtered_in", "filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",     "n4", "epochs_in",   "epochs"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Multi-node pipeline has syntax error: {e.msg}")
        # Key strings for each node stage must be present
        assert "read_raw_edf" in script
        assert ".filter(" in script
        assert "mne.Epochs" in script or "Epochs" in script
        assert "average()" in script
        assert "SKIPPED" not in script

    # 6. Connectivity pipeline exports valid Python
    def test_connectivity_pipeline_exports_valid_python(self):
        """edf_loader -> bandpass_filter -> epoch_by_time -> compute_coherence exports cleanly."""
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
            }),
            _make_node("n3", "epoch_by_time", {
                "epoch_length_s": 2.0, "overlap_s": 0.0,
            }),
            _make_node("n4", "compute_coherence", {
                "fmin_hz": 4.0, "fmax_hz": 40.0,
            }),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg",      "n2", "eeg_in",      "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg",  "n3", "filtered_in", "filtered_eeg"),
            _make_edge("e3", "n3", "epochs_out", "epochs",     "n4", "epochs_in",   "epochs"),
        ]
        script = export(_graph(nodes, edges), audit_log=[])
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(f"Connectivity pipeline has syntax error: {e.msg}")
        assert "spectral_connectivity" in script
        assert "SKIPPED" not in script

    # 7. methods_template smoke test
    def test_methods_template_returns_nonempty_string_for_non_viz_nodes(self):
        """Non-visualization nodes with methods_template should return a non-empty string."""
        tested = 0
        for node_type, descriptor in NODE_REGISTRY.items():
            if descriptor.methods_template is None:
                continue
            if descriptor.category == "Visualization":
                continue
            default_params = {p.name: p.default for p in descriptor.parameters}
            result = descriptor.methods_template(default_params)
            assert isinstance(result, str), (
                f"Node '{node_type}' methods_template returned {type(result)}, expected str"
            )
            assert len(result.strip()) > 0, (
                f"Node '{node_type}' methods_template returned an empty string"
            )
            tested += 1
        assert tested > 0, "No non-visualization nodes with methods_template were found"
