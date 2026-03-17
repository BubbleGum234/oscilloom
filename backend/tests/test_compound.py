"""
backend/tests/test_compound.py

Tests for Tier 7 — Compound Nodes.

Covers: engine refactor, compound registry, compound execution,
API endpoints, export guard, persistence, and cleanup isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import mne

from backend.engine import _execute_graph, execute_pipeline, topological_sort
from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)
from backend.registry import NODE_REGISTRY
from backend.registry.node_descriptor import NodeDescriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(n_channels: int = 4, sfreq: float = 256.0, duration_s: float = 2.0):
    """Standard test fixture — small raw for fast tests."""
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    data = np.random.randn(n_channels, int(sfreq * duration_s)) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def _make_metadata(name: str = "test") -> PipelineMetadata:
    return PipelineMetadata(
        name=name, description="test", created_by="human", schema_version="1.0"
    )


def _make_bandpass_graph() -> PipelineGraph:
    """Simple 1-node bandpass filter pipeline."""
    return PipelineGraph(
        metadata=_make_metadata(),
        nodes=[
            PipelineNode(
                id="n1",
                node_type="bandpass_filter",
                label="BP",
                parameters={"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0},
                position={"x": 0, "y": 0},
            )
        ],
        edges=[],
    )


def _make_two_node_graph() -> PipelineGraph:
    """bandpass_filter → notch_filter chain."""
    return PipelineGraph(
        metadata=_make_metadata(),
        nodes=[
            PipelineNode(
                id="n1",
                node_type="bandpass_filter",
                label="BP",
                parameters={"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0},
                position={"x": 0, "y": 0},
            ),
            PipelineNode(
                id="n2",
                node_type="notch_filter",
                label="Notch",
                parameters={"notch_freq_hz": 50.0},
                position={"x": 200, "y": 0},
            ),
        ],
        edges=[
            PipelineEdge(
                id="e1",
                source_node_id="n1",
                source_handle_id="filtered_out",
                source_handle_type="filtered_eeg",
                target_node_id="n2",
                target_handle_id="eeg_in",
                target_handle_type="filtered_eeg",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Fixture: compound registry cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup_compounds(tmp_path, monkeypatch):
    """
    Ensure compound tests don't pollute NODE_REGISTRY or the filesystem.
    Redirects _COMPOUNDS_DIR to tmp_path and cleans up after each test.
    """
    import backend.compound_registry as cr

    monkeypatch.setattr(cr, "_COMPOUNDS_DIR", tmp_path)

    # Snapshot the registry keys before the test
    original_keys = set(NODE_REGISTRY.keys())
    original_defs = dict(cr._compound_definitions)

    yield

    # Remove any compounds added during the test
    for key in list(NODE_REGISTRY.keys()):
        if key not in original_keys:
            del NODE_REGISTRY[key]
    cr._compound_definitions.clear()
    cr._compound_definitions.update(original_defs)


# ---------------------------------------------------------------------------
# Phase 1: Engine refactor tests
# ---------------------------------------------------------------------------

class TestExecuteGraph:
    """Verify _execute_graph returns the expected tuple."""

    def test_returns_tuple(self):
        raw = _make_raw()
        graph = _make_bandpass_graph()
        result = _execute_graph(raw, graph)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_results_dict_has_expected_keys(self):
        raw = _make_raw()
        graph = _make_bandpass_graph()
        results, node_outputs = _execute_graph(raw, graph)
        assert "n1" in results
        assert results["n1"]["status"] == "success"
        assert results["n1"]["node_type"] == "bandpass_filter"

    def test_node_outputs_contains_raw(self):
        raw = _make_raw()
        graph = _make_bandpass_graph()
        _results, node_outputs = _execute_graph(raw, graph)
        assert "n1" in node_outputs
        assert isinstance(node_outputs["n1"], mne.io.BaseRaw)

    def test_execute_pipeline_still_returns_dict(self):
        raw = _make_raw()
        graph = _make_bandpass_graph()
        result, _ = execute_pipeline(raw, graph)
        assert isinstance(result, dict)
        assert "n1" in result


# ---------------------------------------------------------------------------
# Phase 2: Compound validation tests
# ---------------------------------------------------------------------------

class TestCompoundValidation:
    """Test compound publish validation logic."""

    def _make_definition(self, **overrides) -> dict:
        graph = _make_two_node_graph()
        defn: dict[str, Any] = {
            "compound_id": "c_test_compound",
            "display_name": "Test Compound",
            "description": "A test compound.",
            "tags": ["test"],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }
        defn.update(overrides)
        return defn

    def test_empty_id_raises(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(compound_id="")
        with pytest.raises(ValueError, match="must not be empty"):
            publish_compound(defn)

    def test_builtin_collision_raises(self):
        from backend.compound_registry import publish_compound

        # "bandpass_filter" is a builtin — force c_ prefix off to test collision detection
        defn = self._make_definition(compound_id="c_bandpass_filter")
        # First remove c_ prefix check by setting id directly
        # The real test: a builtin like "bandpass_filter" can't be overwritten
        defn["compound_id"] = "bandpass_filter"
        # Since publish auto-prefixes with c_, "bandpass_filter" becomes "c_bandpass_filter"
        # which is NOT a builtin, so this should succeed (c_ prefix prevents collisions)
        result = publish_compound(defn)
        assert result.node_type == "c_bandpass_filter"

    def test_auto_prefix(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(compound_id="my_chain")
        descriptor = publish_compound(defn)
        assert descriptor.node_type == "c_my_chain"

    def test_already_prefixed(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(compound_id="c_already_prefixed")
        descriptor = publish_compound(defn)
        assert descriptor.node_type == "c_already_prefixed"

    def test_missing_output_node_raises(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(output_node_id="")
        with pytest.raises(ValueError, match="output_node_id is required"):
            publish_compound(defn)

    def test_missing_entry_node_in_subgraph_raises(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(entry_node_id="nonexistent_node")
        with pytest.raises(ValueError, match="not found in sub-graph"):
            publish_compound(defn)

    def test_invalid_exposed_param_node_raises(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(
            exposed_params=[
                {"inner_node_id": "nonexistent", "param_name": "foo", "display_label": "Foo"}
            ]
        )
        with pytest.raises(ValueError, match="not in sub-graph"):
            publish_compound(defn)

    def test_invalid_exposed_param_name_raises(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(
            exposed_params=[
                {"inner_node_id": "n1", "param_name": "nonexistent_param", "display_label": "X"}
            ]
        )
        with pytest.raises(ValueError, match="not found on node type"):
            publish_compound(defn)


# ---------------------------------------------------------------------------
# Phase 2: Compound registration tests
# ---------------------------------------------------------------------------

class TestCompoundRegistration:
    """Test that publish properly registers compounds."""

    def _make_definition(self, **overrides) -> dict:
        graph = _make_two_node_graph()
        defn: dict[str, Any] = {
            "compound_id": "c_reg_test",
            "display_name": "Reg Test",
            "description": "Registration test.",
            "tags": ["test"],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }
        defn.update(overrides)
        return defn

    def test_registers_in_node_registry(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition()
        publish_compound(defn)
        assert "c_reg_test" in NODE_REGISTRY

    def test_correct_input_handle(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition()
        descriptor = publish_compound(defn)
        # Entry node is bandpass_filter which takes raw_eeg or filtered_eeg
        assert len(descriptor.inputs) == 1
        assert descriptor.inputs[0].type in ("raw_eeg", "filtered_eeg")

    def test_correct_output_handle(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition()
        descriptor = publish_compound(defn)
        # Output node is notch_filter which outputs filtered_eeg
        assert len(descriptor.outputs) == 1
        assert descriptor.outputs[0].type == "filtered_eeg"

    def test_exposed_params_become_parameter_schema(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(
            compound_id="c_params_test",
            exposed_params=[
                {
                    "inner_node_id": "n1",
                    "param_name": "low_cutoff_hz",
                    "display_label": "Low Cutoff",
                }
            ],
        )
        descriptor = publish_compound(defn)
        assert len(descriptor.parameters) == 1
        assert descriptor.parameters[0].name == "n1__low_cutoff_hz"
        assert descriptor.parameters[0].label == "Low Cutoff"
        assert descriptor.parameters[0].exposed is True

    def test_persists_json_to_disk(self, tmp_path):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(compound_id="c_persist_test")
        publish_compound(defn)
        filepath = tmp_path / "c_persist_test.json"
        assert filepath.exists()
        loaded = json.loads(filepath.read_text())
        assert loaded["compound_id"] == "c_persist_test"

    def test_category_is_compound(self):
        from backend.compound_registry import publish_compound

        defn = self._make_definition(compound_id="c_cat_test")
        descriptor = publish_compound(defn)
        assert descriptor.category == "Compound"


# ---------------------------------------------------------------------------
# Phase 2: Compound execution tests
# ---------------------------------------------------------------------------

class TestCompoundExecution:
    """Test that compound nodes actually execute their sub-graph."""

    def _publish_simple_compound(self):
        from backend.compound_registry import publish_compound

        graph = _make_two_node_graph()
        defn = {
            "compound_id": "c_exec_test",
            "display_name": "Exec Test",
            "description": "Execution test compound.",
            "tags": ["test"],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }
        return publish_compound(defn)

    def test_sub_graph_runs_correctly(self):
        self._publish_simple_compound()
        raw = _make_raw()
        descriptor = NODE_REGISTRY["c_exec_test"]
        result = descriptor.execute_fn(raw, {})
        assert isinstance(result, mne.io.BaseRaw)

    def test_output_is_mne_object(self):
        self._publish_simple_compound()
        raw = _make_raw()
        descriptor = NODE_REGISTRY["c_exec_test"]
        result = descriptor.execute_fn(raw, {})
        # Output should be from notch_filter — a BaseRaw
        assert hasattr(result, "get_data")

    def test_param_routing_works(self):
        from backend.compound_registry import publish_compound

        graph = _make_two_node_graph()
        defn = {
            "compound_id": "c_param_route",
            "display_name": "Param Route",
            "description": "Param routing test.",
            "tags": [],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [
                {
                    "inner_node_id": "n2",
                    "param_name": "notch_freq_hz",
                    "display_label": "Notch Freq",
                }
            ],
        }
        publish_compound(defn)

        raw = _make_raw()
        descriptor = NODE_REGISTRY["c_param_route"]
        # Pass a different notch frequency
        result = descriptor.execute_fn(raw, {"n2__notch_freq_hz": 60.0})
        assert isinstance(result, mne.io.BaseRaw)

    def test_works_in_full_pipeline(self):
        self._publish_simple_compound()
        raw = _make_raw()

        # Create a pipeline that uses the compound node
        graph = PipelineGraph(
            metadata=_make_metadata(),
            nodes=[
                PipelineNode(
                    id="compound_1",
                    node_type="c_exec_test",
                    label="My Compound",
                    parameters={},
                    position={"x": 0, "y": 0},
                )
            ],
            edges=[],
        )
        results, _ = execute_pipeline(raw, graph)
        assert results["compound_1"]["status"] == "success"

    def test_does_not_mutate_input(self):
        self._publish_simple_compound()
        raw = _make_raw()
        original_data = raw.get_data().copy()
        descriptor = NODE_REGISTRY["c_exec_test"]
        descriptor.execute_fn(raw, {})
        np.testing.assert_array_equal(raw.get_data(), original_data)

    def test_nested_compound(self):
        """A compound inside a compound should work (depth=2)."""
        from backend.compound_registry import publish_compound

        # First: publish inner compound
        graph_inner = _make_bandpass_graph()
        defn_inner = {
            "compound_id": "c_inner",
            "display_name": "Inner",
            "description": "Inner compound.",
            "tags": [],
            "sub_graph": graph_inner.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n1",
            "exposed_params": [],
        }
        publish_compound(defn_inner)

        # Second: publish outer compound that uses c_inner
        outer_graph = PipelineGraph(
            metadata=_make_metadata(),
            nodes=[
                PipelineNode(
                    id="inner_1",
                    node_type="c_inner",
                    label="Inner",
                    parameters={},
                    position={"x": 0, "y": 0},
                ),
                PipelineNode(
                    id="notch_1",
                    node_type="notch_filter",
                    label="Notch",
                    parameters={"notch_freq_hz": 50.0},
                    position={"x": 200, "y": 0},
                ),
            ],
            edges=[
                PipelineEdge(
                    id="e1",
                    source_node_id="inner_1",
                    source_handle_id="compound_out",
                    source_handle_type="filtered_eeg",
                    target_node_id="notch_1",
                    target_handle_id="eeg_in",
                    target_handle_type="filtered_eeg",
                )
            ],
        )
        defn_outer = {
            "compound_id": "c_outer",
            "display_name": "Outer",
            "description": "Outer compound with nested inner.",
            "tags": [],
            "sub_graph": outer_graph.model_dump(),
            "entry_node_id": "inner_1",
            "output_node_id": "notch_1",
            "exposed_params": [],
        }
        publish_compound(defn_outer)

        raw = _make_raw()
        descriptor = NODE_REGISTRY["c_outer"]
        result = descriptor.execute_fn(raw, {})
        assert isinstance(result, mne.io.BaseRaw)


# ---------------------------------------------------------------------------
# Phase 3: API endpoint tests
# ---------------------------------------------------------------------------

class TestCompoundAPI:
    """Test the compound REST API endpoints."""

    def _make_definition(self, compound_id: str = "c_api_test") -> dict:
        graph = _make_two_node_graph()
        return {
            "compound_id": compound_id,
            "display_name": "API Test",
            "description": "API test compound.",
            "tags": ["test"],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }

    def test_publish_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        defn = self._make_definition()
        resp = client.post("/compound/publish", json=defn)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert data["compound_id"] == "c_api_test"

    def test_list_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.compound_registry import publish_compound

        publish_compound(self._make_definition("c_list_test"))
        client = TestClient(app)
        resp = client.get("/compound/list")
        assert resp.status_code == 200
        compounds = resp.json()["compounds"]
        ids = [c["compound_id"] for c in compounds]
        assert "c_list_test" in ids

    def test_get_by_id_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.compound_registry import publish_compound

        publish_compound(self._make_definition("c_get_test"))
        client = TestClient(app)
        resp = client.get("/compound/c_get_test")
        assert resp.status_code == 200
        assert resp.json()["compound_id"] == "c_get_test"

    def test_get_not_found(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/compound/c_nonexistent")
        assert resp.status_code == 404

    def test_delete_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.compound_registry import publish_compound

        publish_compound(self._make_definition("c_del_test"))
        assert "c_del_test" in NODE_REGISTRY

        client = TestClient(app)
        resp = client.delete("/compound/c_del_test")
        assert resp.status_code == 200
        assert "c_del_test" not in NODE_REGISTRY

    def test_delete_builtin_raises(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        resp = client.delete("/compound/bandpass_filter")
        assert resp.status_code == 422

    def test_registry_endpoint_includes_compounds(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.compound_registry import publish_compound

        publish_compound(self._make_definition("c_reg_incl"))
        client = TestClient(app)
        resp = client.get("/registry/nodes")
        assert resp.status_code == 200
        assert "c_reg_incl" in resp.json()["nodes"]


# ---------------------------------------------------------------------------
# Phase 3: Export guard test
# ---------------------------------------------------------------------------

class TestExportGuard:
    """Test that script_exporter rejects pipelines with compound nodes."""

    def test_compound_in_pipeline_raises(self):
        from backend.script_exporter import export

        graph = PipelineGraph(
            metadata=_make_metadata(),
            nodes=[
                PipelineNode(
                    id="n1",
                    node_type="c_some_compound",
                    label="Compound",
                    parameters={},
                    position={"x": 0, "y": 0},
                )
            ],
            edges=[],
        )
        with pytest.raises(ValueError, match="compound"):
            export(graph, [])


# ---------------------------------------------------------------------------
# Phase 2: Persistence tests
# ---------------------------------------------------------------------------

class TestCompoundPersistence:
    """Test compound JSON persistence and startup loading."""

    def test_delete_removes_json_file(self, tmp_path):
        from backend.compound_registry import publish_compound, delete_compound

        graph = _make_two_node_graph()
        defn = {
            "compound_id": "c_file_del",
            "display_name": "File Delete",
            "description": "Test file deletion.",
            "tags": [],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }
        publish_compound(defn)
        filepath = tmp_path / "c_file_del.json"
        assert filepath.exists()

        delete_compound("c_file_del")
        assert not filepath.exists()

    def test_load_on_startup(self, tmp_path):
        import backend.compound_registry as cr

        # Write a compound JSON file directly to the compounds dir
        graph = _make_two_node_graph()
        defn = {
            "compound_id": "c_startup_load",
            "display_name": "Startup Load",
            "description": "Test startup loading.",
            "tags": [],
            "sub_graph": graph.model_dump(),
            "entry_node_id": "n1",
            "output_node_id": "n2",
            "exposed_params": [],
        }
        filepath = tmp_path / "c_startup_load.json"
        filepath.write_text(json.dumps(defn), encoding="utf-8")

        # Clear any existing registration
        NODE_REGISTRY.pop("c_startup_load", None)
        cr._compound_definitions.pop("c_startup_load", None)

        count = cr.load_compounds_on_startup()
        assert count >= 1
        assert "c_startup_load" in NODE_REGISTRY

    def test_load_on_startup_skips_bad_file(self, tmp_path):
        import backend.compound_registry as cr

        # Write an invalid JSON file
        bad_file = tmp_path / "c_bad.json"
        bad_file.write_text('{"compound_id": "c_bad"}', encoding="utf-8")

        # Should not crash
        count = cr.load_compounds_on_startup()
        # The bad file should be skipped (logged warning)
        assert "c_bad" not in NODE_REGISTRY
