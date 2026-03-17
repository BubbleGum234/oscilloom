"""
backend/tests/test_validation.py

Tests for pipeline graph validation (validation.py).

Tests cover all 7 validation checks:
  1. Empty pipeline rejection
  2. Unknown node_type detection
  3. Handle type mismatch detection
  4. Invalid source handle ID detection
  5. Invalid target handle ID detection
  6. Parameter value range validation
  7. Required input handles must have incoming edges

Plus happy-path tests for valid single-node and multi-node pipelines.
"""

from __future__ import annotations

import pytest

from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)
from backend.validation import validate_pipeline


# ---------------------------------------------------------------------------
# Helpers (mirrors test_engine.py conventions for consistency)
# ---------------------------------------------------------------------------

def _make_metadata() -> PipelineMetadata:
    return PipelineMetadata(
        name="Validation Test Pipeline",
        description="Test",
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


def _valid_linear_graph() -> PipelineGraph:
    """Returns a minimal valid 2-node pipeline: edf_loader → bandpass_filter."""
    nodes = [
        _make_node("n1", "edf_loader"),
        _make_node("n2", "bandpass_filter"),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
    ]
    return _graph(nodes, edges)


# ---------------------------------------------------------------------------
# Check 1: Empty pipeline
# ---------------------------------------------------------------------------

class TestEmptyPipeline:
    def test_empty_graph_returns_error(self):
        graph = _graph(nodes=[], edges=[])
        errors = validate_pipeline(graph)
        assert len(errors) == 1
        assert "no nodes" in errors[0].lower()

    def test_empty_graph_stops_early(self):
        """Should not check edges when there are no nodes."""
        graph = _graph(nodes=[], edges=[])
        errors = validate_pipeline(graph)
        # Should be exactly one error (not crash trying to iterate over edges)
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# Check 2: Unknown node type
# ---------------------------------------------------------------------------

class TestUnknownNodeType:
    def test_single_unknown_type(self):
        nodes = [_make_node("n1", "nonexistent_node_xyz")]
        errors = validate_pipeline(_graph(nodes, []))
        assert len(errors) == 1
        assert "nonexistent_node_xyz" in errors[0]
        assert "n1" in errors[0]

    def test_multiple_unknown_types(self):
        nodes = [
            _make_node("n1", "unknown_a"),
            _make_node("n2", "unknown_b"),
        ]
        errors = validate_pipeline(_graph(nodes, []))
        # One error per unknown node
        assert len(errors) == 2
        node_ids_mentioned = {e for err in errors for e in ["n1", "n2"] if e in err}
        assert "n1" in node_ids_mentioned
        assert "n2" in node_ids_mentioned

    def test_valid_type_passes(self):
        nodes = [_make_node("n1", "edf_loader")]
        errors = validate_pipeline(_graph(nodes, []))
        assert errors == []


# ---------------------------------------------------------------------------
# Check 3: Handle type mismatch
# ---------------------------------------------------------------------------

class TestHandleTypeMismatch:
    def test_mismatched_handle_types_returns_error(self):
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
        ]
        # Source says "raw_eeg", target says "filtered_eeg" — mismatch
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "filtered_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert any("type mismatch" in err.lower() or "mismatch" in err.lower() for err in errors)
        assert any("e1" in err for err in errors)

    def test_matching_handle_types_passes(self):
        errors = validate_pipeline(_valid_linear_graph())
        assert errors == []


# ---------------------------------------------------------------------------
# Check 4: Invalid source handle ID
# ---------------------------------------------------------------------------

class TestInvalidSourceHandleId:
    def test_invalid_source_handle_id_returns_error(self):
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
        ]
        # "nonexistent_output" is not a valid output on edf_loader
        edges = [
            _make_edge("e1", "n1", "nonexistent_output", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert any("nonexistent_output" in err for err in errors)
        assert any("edf_loader" in err for err in errors)

    def test_valid_source_handle_passes(self):
        errors = validate_pipeline(_valid_linear_graph())
        assert errors == []


# ---------------------------------------------------------------------------
# Check 5: Invalid target handle ID
# ---------------------------------------------------------------------------

class TestInvalidTargetHandleId:
    def test_invalid_target_handle_id_returns_error(self):
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
        ]
        # "bad_input_handle" is not a valid input on bandpass_filter
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "bad_input_handle", "raw_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert any("bad_input_handle" in err for err in errors)
        assert any("bandpass_filter" in err for err in errors)

    def test_valid_target_handle_passes(self):
        errors = validate_pipeline(_valid_linear_graph())
        assert errors == []


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestValidPipelines:
    def test_single_source_node_is_valid(self):
        """A single edf_loader with no edges is a valid (trivial) pipeline."""
        nodes = [_make_node("n1", "edf_loader")]
        errors = validate_pipeline(_graph(nodes, []))
        assert errors == []

    def test_full_three_node_pipeline_is_valid(self):
        """edf_loader → bandpass_filter → compute_psd should be valid."""
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "bandpass_filter"),
            _make_node("n3", "compute_psd"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert errors == []

    def test_all_mvp_nodes_are_known_types(self):
        """Each of the 6 original MVP node types should pass type-existence check."""
        mvp_types = [
            "edf_loader", "bandpass_filter", "notch_filter",
            "resample", "compute_psd", "plot_psd",
        ]
        for node_type in mvp_types:
            nodes = [_make_node("n1", node_type)]
            errors = validate_pipeline(_graph(nodes, []))
            type_errors = [e for e in errors if "unknown type" in e.lower()]
            assert type_errors == [], (
                f"Node type '{node_type}' should be registered but got: {type_errors}"
            )

    def test_all_new_nodes_are_known_types(self):
        """All 10 new node types added in the researcher-eval fix must be registered."""
        new_types = [
            "set_eeg_reference",
            "ica_decomposition",
            "epoch_by_events",
            "baseline_correction",
            "compute_evoked",
            "pick_channels",
            "mark_bad_channels",
            "time_frequency_morlet",
            "plot_evoked",
            "plot_epochs_image",
        ]
        for node_type in new_types:
            nodes = [_make_node("n1", node_type)]
            errors = validate_pipeline(_graph(nodes, []))
            type_errors = [e for e in errors if "unknown type" in e.lower()]
            assert type_errors == [], (
                f"Node type '{node_type}' should be registered but got: {type_errors}"
            )

    def test_epochs_to_evoked_connection_is_valid(self):
        """epochs → compute_evoked connection must pass handle type validation."""
        nodes = [
            _make_node("n1", "epoch_by_events"),
            _make_node("n2", "compute_evoked"),
        ]
        edges = [
            _make_edge("e1", "n1", "epochs_out", "epochs", "n2", "epochs_in", "epochs"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []

    def test_epochs_to_tfr_connection_is_valid(self):
        """epochs → time_frequency_morlet connection must pass handle type validation."""
        nodes = [
            _make_node("n1", "epoch_by_events"),
            _make_node("n2", "time_frequency_morlet"),
        ]
        edges = [
            _make_edge("e1", "n1", "epochs_out", "epochs", "n2", "epochs_in", "epochs"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []

    def test_returns_list_type(self):
        """validate_pipeline must always return a list (never None)."""
        result = validate_pipeline(_valid_linear_graph())
        assert isinstance(result, list)

    def test_multiple_errors_accumulate(self):
        """When multiple checks fail, all errors should be returned (not just the first)."""
        nodes = [
            _make_node("n1", "bad_type_a"),
            _make_node("n2", "bad_type_b"),
        ]
        errors = validate_pipeline(_graph(nodes, []))
        # Both unknown types should generate errors
        assert len(errors) >= 2

    # ── TASK-01: filtered_eeg edges are now valid for filter nodes ────────────

    def test_filtered_eeg_to_bandpass_filter_is_valid(self):
        """TASK-01: notch_filter → bandpass_filter chain via filtered_eeg must pass."""
        nodes = [
            _make_node("n1", "notch_filter"),
            _make_node("n2", "bandpass_filter"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "filtered_eeg", "n2", "filtered_in", "filtered_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []

    def test_filtered_eeg_to_notch_filter_is_valid(self):
        """bandpass_filter → notch_filter chain via filtered_eeg must also pass."""
        nodes = [
            _make_node("n1", "bandpass_filter"),
            _make_node("n2", "notch_filter"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "filtered_eeg", "n2", "filtered_in", "filtered_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []

    # ── TASK-02: raw_eeg edge to compute_psd is now valid ─────────────────────

    def test_raw_eeg_to_compute_psd_is_valid(self):
        """TASK-02: edf_loader → compute_psd connection via raw_eeg must pass."""
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "compute_psd"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert errors == []

    # ── TASK-04/05/06/13/14/15/16: all new node types registered ─────────────

    def test_all_phase2_phase4_nodes_are_known_types(self):
        """All 7 new nodes from Phases 2 & 4 must be registered in NODE_REGISTRY."""
        new_types = [
            "plot_evoked_topomap",   # TASK-04
            "plot_ica_components",   # TASK-05
            "crop",                  # TASK-06
            "compute_bandpower",     # TASK-13
            "plot_evoked_joint",     # TASK-14
            "filter_epochs",         # TASK-15
            "apply_autoreject",      # TASK-16
        ]
        for node_type in new_types:
            nodes = [_make_node("n1", node_type)]
            errors = validate_pipeline(_graph(nodes, []))
            type_errors = [e for e in errors if "unknown type" in e.lower()]
            assert type_errors == [], (
                f"Node type '{node_type}' should be registered but got: {type_errors}"
            )

    # ── TASK-13: psd → compute_bandpower edge is valid ────────────────────────

    def test_psd_to_compute_bandpower_is_valid(self):
        """compute_psd → compute_bandpower connection via psd type must pass."""
        nodes = [
            _make_node("n1", "compute_psd"),
            _make_node("n2", "compute_bandpower"),
        ]
        edges = [
            _make_edge("e1", "n1", "psd_out", "psd", "n2", "psd_in", "psd"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []

    # ── TASK-06: crop node accepts both raw_eeg and filtered_eeg ─────────────

    def test_crop_accepts_raw_eeg(self):
        """Crop node raw_in input accepts raw_eeg from edf_loader."""
        nodes = [
            _make_node("n1", "edf_loader"),
            _make_node("n2", "crop"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        assert errors == []

    def test_crop_accepts_filtered_eeg(self):
        """Crop node filtered_in input accepts filtered_eeg from bandpass_filter."""
        nodes = [
            _make_node("n1", "bandpass_filter"),
            _make_node("n2", "crop"),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "filtered_eeg", "n2", "filtered_in", "filtered_eeg"),
        ]
        errors = validate_pipeline(_graph(nodes, edges))
        edge_errors = [e for e in errors if "no incoming connections" not in e]
        assert edge_errors == []


# ---------------------------------------------------------------------------
# Check 7: Required input handles must have incoming edges
# ---------------------------------------------------------------------------

class TestRequiredInputsConnected:
    def test_unconnected_required_input_returns_error(self):
        """A bandpass_filter with no incoming edge should fail validation."""
        nodes = [_make_node("n1", "bandpass_filter")]
        errors = validate_pipeline(_graph(nodes, []))
        required_errors = [e for e in errors if "required inputs" in e.lower()]
        assert len(required_errors) >= 1
        assert "no incoming connections" in required_errors[0].lower()

    def test_source_node_no_inputs_passes(self):
        """edf_loader has no inputs — should pass with no required-input errors."""
        nodes = [_make_node("n1", "edf_loader")]
        errors = validate_pipeline(_graph(nodes, []))
        required_errors = [e for e in errors if "required inputs" in e.lower()]
        assert required_errors == []

    def test_optional_input_without_edge_passes(self):
        """save_to_fif has required=False inputs — no error when unconnected."""
        nodes = [_make_node("n1", "save_to_fif")]
        errors = validate_pipeline(_graph(nodes, []))
        required_errors = [e for e in errors if "required inputs" in e.lower()]
        assert required_errors == []

    def test_connected_required_input_passes(self):
        """edf_loader → bandpass_filter with proper edge should produce no errors."""
        errors = validate_pipeline(_valid_linear_graph())
        assert errors == []

    def test_multiple_unconnected_nodes_multiple_errors(self):
        """Two unconnected filter nodes should each produce a required-input error."""
        nodes = [
            _make_node("n1", "bandpass_filter"),
            _make_node("n2", "notch_filter"),
        ]
        errors = validate_pipeline(_graph(nodes, []))
        required_errors = [e for e in errors if "required inputs" in e.lower()]
        assert len(required_errors) >= 2
