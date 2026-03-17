"""
backend/tests/test_bids.py

Tests for the BIDS export node (execute_fn) and the /pipeline/export-bids endpoint.

The BIDS export node uses mne-bids, which is an optional dependency.
All tests in this file are skipped gracefully if mne-bids is not installed.
"""

from __future__ import annotations

import io
import zipfile

import pytest
import mne
import numpy as np

mne_bids = pytest.importorskip("mne_bids")

from backend.registry.nodes.io import _execute_bids_export
from backend.engine import execute_pipeline
from backend.models import (
    BidsExportRequest,
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)
from backend import session_store
from backend.execution_cache import ExecutionCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw() -> mne.io.BaseRaw:
    """
    Synthetic 256 Hz, 10-channel, 10-second MNE Raw object.
    Matches the convention used by conftest.py and other test files.
    """
    sfreq = 256.0
    n_channels = 10
    n_seconds = 10
    n_samples = int(sfreq * n_seconds)

    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_samples)) * 5e-6

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# Helpers (same patterns as test_engine.py / test_export.py)
# ---------------------------------------------------------------------------

def _make_metadata() -> PipelineMetadata:
    return PipelineMetadata(
        name="BIDS Test Pipeline",
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
# Node tests (using execute_fn directly)
# ---------------------------------------------------------------------------

class TestBidsExportNode:

    def test_bids_export_passthrough_when_no_output_dir(self, raw):
        """
        When output_dir is empty, the node must return raw data unchanged
        (same n_channels, sfreq, n_times) without writing any files.
        """
        result = _execute_bids_export(raw.copy(), {"output_dir": ""})

        assert isinstance(result, mne.io.BaseRaw)
        assert result.info["nchan"] == raw.info["nchan"]
        assert result.info["sfreq"] == raw.info["sfreq"]
        assert result.n_times == raw.n_times

    def test_bids_export_writes_bids_structure(self, raw, tmp_path):
        """
        With a valid output_dir, the node must create BIDS directory structure
        including sub-XX/ directory, data files, and dataset_description.json.
        Returns raw unchanged.
        """
        output_dir = str(tmp_path / "bids_out")
        params = {
            "output_dir": output_dir,
            "subject_id": "01",
            "task": "rest",
            "format": "BrainVision",
        }

        result = _execute_bids_export(raw.copy(), params)

        # Assert raw returned unchanged
        assert isinstance(result, mne.io.BaseRaw)
        assert result.info["nchan"] == raw.info["nchan"]
        assert result.info["sfreq"] == raw.info["sfreq"]

        # Assert BIDS directory structure was created
        import pathlib
        root = pathlib.Path(output_dir)
        assert root.exists()

        # Check for sub-01/ directory
        sub_dirs = list(root.glob("sub-01"))
        assert len(sub_dirs) == 1, f"Expected sub-01/ directory, found: {list(root.iterdir())}"

        # Check for dataset_description.json
        desc_file = root / "dataset_description.json"
        assert desc_file.exists(), "dataset_description.json not found in BIDS root"

        # Check for a data file (.vhdr for BrainVision)
        vhdr_files = list(root.rglob("*.vhdr"))
        assert len(vhdr_files) > 0, f"No .vhdr files found in BIDS output: {list(root.rglob('*'))}"

    def test_bids_export_with_edf_format(self, raw, tmp_path):
        """
        With format='EDF', the node must create BIDS structure with .edf files.
        """
        output_dir = str(tmp_path / "bids_edf")
        params = {
            "output_dir": output_dir,
            "subject_id": "01",
            "task": "rest",
            "format": "EDF",
        }

        result = _execute_bids_export(raw.copy(), params)

        assert isinstance(result, mne.io.BaseRaw)

        import pathlib
        root = pathlib.Path(output_dir)
        edf_files = list(root.rglob("*.edf"))
        assert len(edf_files) > 0, f"No .edf files found in BIDS output: {list(root.rglob('*'))}"

    def test_bids_export_custom_metadata(self, raw, tmp_path):
        """
        Custom subject/session/task/run values must be reflected in the
        BIDS directory structure (sub-42/ses-02/eeg/).
        """
        output_dir = str(tmp_path / "bids_custom")
        params = {
            "output_dir": output_dir,
            "subject_id": "42",
            "session_id": "02",
            "task": "oddball",
            "run": "03",
            "format": "BrainVision",
        }

        result = _execute_bids_export(raw.copy(), params)

        assert isinstance(result, mne.io.BaseRaw)

        import pathlib
        root = pathlib.Path(output_dir)

        # Check for sub-42/ses-02/eeg/ directory structure
        eeg_dir = root / "sub-42" / "ses-02" / "eeg"
        assert eeg_dir.exists(), (
            f"Expected sub-42/ses-02/eeg/ directory, "
            f"found: {list(root.rglob('*'))}"
        )


# ---------------------------------------------------------------------------
# Endpoint tests (using TestClient)
# ---------------------------------------------------------------------------

class TestBidsExportEndpoint:

    def test_bids_export_endpoint_returns_zip(self, client, raw):
        """
        POST /pipeline/export-bids with a valid pipeline must return a zip
        archive containing BIDS-formatted files.
        """
        # Store a session so the endpoint can find the Raw
        session_id = "bids-test-session-zip"
        session_store._sessions[session_id] = raw.copy()
        session_store._session_last_access[session_id] = __import__("time").time()
        session_store._execution_caches[session_id] = ExecutionCache()

        try:
            # Build a minimal pipeline: edf_loader → bandpass_filter
            pipeline = _linear_graph(
                nodes=[
                    _make_node("n1", "edf_loader", {"file_path": ""}),
                    _make_node("n2", "bandpass_filter", {
                        "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
                    }),
                ],
                edges=[
                    _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
                ],
            )

            body = BidsExportRequest(
                session_id=session_id,
                pipeline=pipeline,
                subject_id="01",
                task="rest",
                run="01",
                format="BrainVision",
            )

            resp = client.post(
                "/pipeline/export-bids",
                json=body.model_dump(),
            )

            assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
            assert "application/zip" in resp.headers.get("content-type", "")

            # Inspect the zip contents
            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                assert len(names) > 0, "Zip archive is empty"

                # Should contain dataset_description.json
                has_description = any("dataset_description.json" in n for n in names)
                assert has_description, (
                    f"dataset_description.json not found in zip. Contents: {names}"
                )

                # Should contain a sub-01 directory entry
                has_subject = any("sub-01" in n for n in names)
                assert has_subject, (
                    f"sub-01 directory not found in zip. Contents: {names}"
                )

        finally:
            # Clean up session
            session_store._sessions.pop(session_id, None)
            session_store._session_last_access.pop(session_id, None)
            session_store._execution_caches.pop(session_id, None)

    def test_bids_export_endpoint_invalid_session(self, client):
        """
        POST /pipeline/export-bids with a non-existent session_id must
        return a 404 error.
        """
        pipeline = _linear_graph(
            nodes=[
                _make_node("n1", "edf_loader", {"file_path": ""}),
                _make_node("n2", "bandpass_filter", {
                    "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
                }),
            ],
            edges=[
                _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            ],
        )

        body = BidsExportRequest(
            session_id="nonexistent-session-id-xyz",
            pipeline=pipeline,
            subject_id="01",
            task="rest",
        )

        resp = client.post(
            "/pipeline/export-bids",
            json=body.model_dump(),
        )

        assert resp.status_code == 404, (
            f"Expected 404 for invalid session, got {resp.status_code}: {resp.text}"
        )
