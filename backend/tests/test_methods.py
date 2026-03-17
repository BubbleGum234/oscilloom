"""
backend/tests/test_methods.py

Tests for the Methods section generator and reproducibility package endpoints.

Covers:
  - POST /pipeline/generate-methods  — generates academic Methods prose from a pipeline
  - POST /pipeline/export-package    — downloads a zip with pipeline.py, pipeline.json,
                                       requirements.txt, and README.md
"""
from __future__ import annotations

import ast
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)

client = TestClient(app)


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


def _three_node_pipeline() -> PipelineGraph:
    """edf_loader -> bandpass_filter -> compute_psd."""
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
        _make_node("n2", "bandpass_filter", {
            "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
        }),
        _make_node("n3", "compute_psd", {"fmin": 0.5, "fmax": 40.0, "n_fft": 512}),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
        _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
    ]
    return _graph(nodes, edges)


def _single_node_pipeline() -> PipelineGraph:
    """Just edf_loader alone — no edges."""
    nodes = [_make_node("n1", "edf_loader", {"file_path": "/data/test.edf"})]
    return _graph(nodes, [])


def _visualization_pipeline() -> PipelineGraph:
    """edf_loader -> plot_raw — visualization node has methods_template=None."""
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
        _make_node("n2", "plot_raw", {
            "n_channels": 10, "start_time_s": 0.0, "duration_s": 10.0,
        }),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "raw_in", "raw_eeg"),
    ]
    return _graph(nodes, edges)


def _bandpass_filter_graph() -> PipelineGraph:
    """Minimal valid pipeline: edf_loader -> bandpass_filter."""
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "/data/test.edf"}),
        _make_node("n2", "bandpass_filter", {
            "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
        }),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
    ]
    return _graph(nodes, edges)


# ---------------------------------------------------------------------------
# Tests: POST /pipeline/generate-methods
# ---------------------------------------------------------------------------

class TestGenerateMethods:
    """Tests for the generate-methods endpoint that produces academic prose."""

    def test_methods_returns_nonempty_prose(self):
        """POST /pipeline/generate-methods returns 200 with non-empty methods_section."""
        graph = _three_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "methods_section" in data
        assert isinstance(data["methods_section"], str)
        assert len(data["methods_section"].strip()) > 0, "methods_section is empty"
        assert "word_count" in data
        assert data["word_count"] > 0

    def test_methods_contains_mne_reference(self):
        """The generated methods section must reference MNE-Python."""
        graph = _three_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "MNE-Python" in data["methods_section"], (
            "Methods section should reference MNE-Python"
        )

    def test_methods_contains_gramfort_citation(self):
        """The methods section or citations list must include the Gramfort citation."""
        graph = _three_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        data = resp.json()
        # Check either the methods_section text or a separate citations field
        methods_text = data["methods_section"]
        citations = data.get("citations", [])
        found = "Gramfort" in methods_text or any(
            "Gramfort" in c for c in citations
        )
        assert found, "Gramfort citation not found in methods_section or citations"

    def test_methods_word_count_matches(self):
        """word_count must equal the actual word count of methods_section."""
        graph = _three_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        data = resp.json()
        actual_count = len(data["methods_section"].split())
        assert data["word_count"] == actual_count, (
            f"word_count={data['word_count']} but actual count={actual_count}"
        )

    def test_methods_includes_filter_description(self):
        """The methods text should describe the bandpass filter or its frequency values."""
        graph = _three_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        text = resp.json()["methods_section"].lower()
        has_filter_mention = (
            "bandpass" in text
            or "band-pass" in text
            or "filter" in text
            or "1.0" in text
            or "40.0" in text
            or "1" in text and "40" in text
        )
        assert has_filter_mention, (
            "Methods section does not mention bandpass filtering or frequency values"
        )

    def test_methods_single_node_pipeline(self):
        """A pipeline with only edf_loader should still return a valid response."""
        graph = _single_node_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "methods_section" in data
        assert isinstance(data["methods_section"], str)
        assert data["word_count"] >= 0

    def test_methods_visualization_only_pipeline(self):
        """Visualization nodes (methods_template=None) should not break generation."""
        graph = _visualization_pipeline()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
        }
        resp = client.post("/pipeline/generate-methods", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "methods_section" in data
        assert isinstance(data["methods_section"], str)


# ---------------------------------------------------------------------------
# Tests: POST /pipeline/export-package
# ---------------------------------------------------------------------------

class TestExportPackage:
    """Tests for the reproducibility package zip export endpoint."""

    def _post_export_package(self, graph: PipelineGraph | None = None):
        """Helper: POST to /pipeline/export-package and return the response."""
        if graph is None:
            graph = _bandpass_filter_graph()
        body = {
            "pipeline": graph.model_dump(),
            "session_id": "test-session",
            "audit_log": [],
        }
        return client.post("/pipeline/export-package", json=body)

    def test_export_package_returns_zip(self):
        """POST /pipeline/export-package returns 200 with application/zip content."""
        resp = self._post_export_package()
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        assert "application/zip" in resp.headers.get("content-type", ""), (
            f"Expected application/zip, got {resp.headers.get('content-type')}"
        )

    def test_zip_contains_four_files(self):
        """The zip must contain exactly: pipeline.py, pipeline.json, requirements.txt, README.md."""
        resp = self._post_export_package()
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = sorted(zf.namelist())
        expected = sorted(["pipeline.py", "pipeline.json", "requirements.txt", "README.md"])
        assert names == expected, f"Expected {expected}, got {names}"

    def test_pipeline_py_is_valid_python(self):
        """pipeline.py in the zip must parse cleanly with ast.parse()."""
        resp = self._post_export_package()
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        script = zf.read("pipeline.py").decode("utf-8")
        try:
            ast.parse(script)
        except SyntaxError as e:
            pytest.fail(
                f"pipeline.py has syntax error at line {e.lineno}: {e.msg}"
            )

    def test_pipeline_json_is_valid_json(self):
        """pipeline.json in the zip must be valid JSON with nodes and edges keys."""
        resp = self._post_export_package()
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        raw_json = zf.read("pipeline.json").decode("utf-8")
        data = json.loads(raw_json)
        assert "nodes" in data, "pipeline.json missing 'nodes' key"
        assert "edges" in data, "pipeline.json missing 'edges' key"

    def test_requirements_contains_mne(self):
        """requirements.txt must pin MNE-Python with a version specifier."""
        resp = self._post_export_package()
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        reqs = zf.read("requirements.txt").decode("utf-8")
        assert "mne==" in reqs, (
            f"requirements.txt should contain 'mne==' but got:\n{reqs}"
        )

    def test_readme_contains_pipeline_name(self):
        """README.md must include the pipeline's metadata name."""
        graph = PipelineGraph(
            metadata=_make_metadata("My Custom Pipeline"),
            nodes=[_make_node("n1", "edf_loader", {"file_path": "/data/test.edf"})],
            edges=[],
        )
        resp = self._post_export_package(graph)
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        readme = zf.read("README.md").decode("utf-8")
        assert "My Custom Pipeline" in readme, (
            "README.md should contain the pipeline name"
        )

    def test_export_package_filename_in_header(self):
        """Content-Disposition header must contain a .zip filename."""
        resp = self._post_export_package()
        assert resp.status_code == 200
        disposition = resp.headers.get("content-disposition", "")
        assert ".zip" in disposition, (
            f"Content-Disposition should contain .zip filename, got: {disposition}"
        )
