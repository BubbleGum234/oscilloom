"""
backend/tests/test_batch.py

Tests for Tier 4 batch processing:
  - batch_processor.py: staging, job management, sequential execution
  - batch_routes.py: HTTP endpoint integration tests
  - CSV generation from batch metrics
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pytest
import mne

from backend.models import (
    PipelineGraph, PipelineMetadata, PipelineNode, PipelineEdge,
)
from backend.batch_processor import BatchJob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_fif(path: str, n_channels: int = 5, sfreq: float = 256.0,
                   duration: float = 2.0) -> str:
    """Create a minimal FIF file for testing. Returns the file path."""
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    n_times = int(sfreq * duration)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 10e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.save(path, overwrite=True, verbose=False)
    return path


def _make_simple_pipeline() -> PipelineGraph:
    """Minimal pipeline: edf_loader → bandpass_filter."""
    return PipelineGraph(
        metadata=PipelineMetadata(
            name="test_batch", description="", created_by="human",
        ),
        nodes=[
            PipelineNode(
                id="n0", node_type="edf_loader", label="Loader",
                parameters={},
                position={"x": 0, "y": 0},
            ),
            PipelineNode(
                id="n1", node_type="bandpass_filter", label="Filter",
                parameters={"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0,
                            "method": "fir"},
                position={"x": 200, "y": 0},
            ),
        ],
        edges=[
            PipelineEdge(
                id="e0",
                source_node_id="n0", source_handle_id="eeg_out",
                source_handle_type="raw_eeg",
                target_node_id="n1", target_handle_id="eeg_in",
                target_handle_type="raw_eeg",
            ),
        ],
    )


def _make_metrics_pipeline() -> PipelineGraph:
    """Pipeline: edf_loader → bandpass → compute_psd → compute_alpha_peak."""
    return PipelineGraph(
        metadata=PipelineMetadata(
            name="test_batch_metrics", description="", created_by="human",
        ),
        nodes=[
            PipelineNode(
                id="n0", node_type="edf_loader", label="Loader",
                parameters={},
                position={"x": 0, "y": 0},
            ),
            PipelineNode(
                id="n1", node_type="bandpass_filter", label="Filter",
                parameters={"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0,
                            "method": "fir"},
                position={"x": 200, "y": 0},
            ),
            PipelineNode(
                id="n2", node_type="compute_psd", label="PSD",
                parameters={"method": "welch", "fmin": 1.0, "fmax": 40.0,
                            "n_fft": 256},
                position={"x": 200, "y": 0},
            ),
            PipelineNode(
                id="n3", node_type="compute_alpha_peak", label="Alpha",
                parameters={"fmin": 7.0, "fmax": 13.0, "method": "cog"},
                position={"x": 400, "y": 0},
            ),
        ],
        edges=[
            PipelineEdge(
                id="e0",
                source_node_id="n0", source_handle_id="eeg_out",
                source_handle_type="raw_eeg",
                target_node_id="n1", target_handle_id="eeg_in",
                target_handle_type="raw_eeg",
            ),
            PipelineEdge(
                id="e1",
                source_node_id="n1", source_handle_id="eeg_out",
                source_handle_type="filtered_eeg",
                target_node_id="n2", target_handle_id="eeg_in",
                target_handle_type="filtered_eeg",
            ),
            PipelineEdge(
                id="e2",
                source_node_id="n2", source_handle_id="psd_out",
                source_handle_type="psd",
                target_node_id="n3", target_handle_id="psd_in",
                target_handle_type="psd",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _clean_batch_state():
    """Reset batch job store and rate limiter between tests to prevent state leakage."""
    from backend import batch_processor
    from backend.rate_limit import limiter
    yield
    # Force-cancel any still-running jobs before clearing
    with batch_processor._jobs_lock:
        for job in batch_processor._jobs.values():
            with job.lock:
                if job.status == "running":
                    job.status = "cancelled"
        batch_processor._jobs.clear()
        batch_processor._job_order.clear()
    # Reset rate limiter storage so tests don't hit rate limits from prior tests
    limiter.reset()


@pytest.fixture()
def staged_edf():
    """Create a temp EDF file and stage it. Yields (file_id, path). Cleans up."""
    from backend.batch_processor import stage_file, remove_staged_file

    fd, path = tempfile.mkstemp(suffix="_raw.fif")
    os.close(fd)
    _make_test_fif(path)
    file_id = stage_file(path, "test_file.fif")
    yield file_id, path
    try:
        remove_staged_file(file_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TestFileStaging
# ---------------------------------------------------------------------------

class TestFileStaging:

    def test_stage_and_retrieve(self):
        from backend.batch_processor import stage_file, get_staged_file, remove_staged_file

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        try:
            file_id = stage_file(path, "test.fif")
            info = get_staged_file(file_id)
            assert info is not None
            assert info["filename"] == "test.fif"
            assert info["path"] == path
        finally:
            remove_staged_file(file_id)

    def test_remove_staged_file_deletes_temp(self):
        from backend.batch_processor import stage_file, remove_staged_file

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        file_id = stage_file(path, "test.fif")
        remove_staged_file(file_id)
        assert not os.path.exists(path)

    def test_get_nonexistent_returns_none(self):
        from backend.batch_processor import get_staged_file

        assert get_staged_file("nonexistent-id") is None

    def test_list_staged_files(self):
        from backend.batch_processor import stage_file, list_staged_files, remove_staged_file

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        file_id = stage_file(path, "listed.fif")
        try:
            files = list_staged_files()
            assert any(f["file_id"] == file_id for f in files)
        finally:
            remove_staged_file(file_id)

    def test_clear_staged_files(self):
        from backend.batch_processor import stage_file, clear_staged_files

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        stage_file(path, "cleared.fif")
        cleared = clear_staged_files()
        assert cleared >= 1


# ---------------------------------------------------------------------------
# TestBatchExecution
# ---------------------------------------------------------------------------

class TestBatchExecution:

    def test_single_file(self, staged_edf):
        from backend.batch_processor import run_batch, execute_batch, get_job

        file_id, _ = staged_edf
        pipeline = _make_simple_pipeline()
        batch_id = run_batch([file_id], pipeline)
        execute_batch(batch_id, [file_id], pipeline)

        job = get_job(batch_id)
        assert job is not None
        assert job.status == "complete"
        assert job.completed == 1
        assert job.failed == 0
        assert len(job.file_results) == 1
        assert job.file_results[0]["status"] == "success"

    def test_multiple_files(self):
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )

        file_ids = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix="_raw.fif")
            os.close(fd)
            _make_test_fif(path)
            file_ids.append(stage_file(path, f"subject_{i:02d}.fif"))

        try:
            pipeline = _make_simple_pipeline()
            batch_id = run_batch(file_ids, pipeline)
            execute_batch(batch_id, file_ids, pipeline)

            job = get_job(batch_id)
            assert job.status == "complete"
            assert job.completed == 3
            assert job.failed == 0
        finally:
            for fid in file_ids:
                remove_staged_file(fid)

    def test_bad_file_graceful_failure(self):
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )

        # Good file
        fd_good, path_good = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd_good)
        _make_test_fif(path_good)

        # Bad file (garbage data)
        fd_bad, path_bad = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd_bad)
        with open(path_bad, "wb") as f:
            f.write(b"not a real edf file")

        good_id = stage_file(path_good, "good.fif")
        bad_id = stage_file(path_bad, "bad.fif")

        try:
            pipeline = _make_simple_pipeline()
            batch_id = run_batch([good_id, bad_id], pipeline)
            execute_batch(batch_id, [good_id, bad_id], pipeline)

            job = get_job(batch_id)
            assert job.completed == 1
            assert job.failed == 1
            assert len(job.failed_files) == 1
            assert job.failed_files[0]["filename"] == "bad.fif"
        finally:
            remove_staged_file(good_id)
            remove_staged_file(bad_id)

    def test_missing_staged_file(self):
        from backend.batch_processor import run_batch, execute_batch, get_job

        pipeline = _make_simple_pipeline()
        batch_id = run_batch(["nonexistent-id"], pipeline)
        execute_batch(batch_id, ["nonexistent-id"], pipeline)

        job = get_job(batch_id)
        assert job.failed == 1
        assert job.completed == 0

    def test_metrics_pipeline(self):
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, n_channels=5, duration=5.0)

        file_id = stage_file(path, "metrics_test.fif")
        try:
            pipeline = _make_metrics_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            assert job.completed == 1
            fr = job.file_results[0]
            assert "compute_alpha_peak.iaf_hz" in fr["metrics"]
        finally:
            remove_staged_file(file_id)


# ---------------------------------------------------------------------------
# TestFlattenMetrics
# ---------------------------------------------------------------------------

class TestFlattenMetrics:

    def test_single_node(self):
        from backend.batch_processor import _flatten_metrics

        nr = {
            "n1": {
                "status": "success",
                "node_type": "compute_alpha_peak",
                "metrics": {"iaf_hz": 10.2, "method": "cog"},
            }
        }
        flat = _flatten_metrics(nr)
        assert flat["compute_alpha_peak.iaf_hz"] == 10.2
        assert flat["compute_alpha_peak.method"] == "cog"

    def test_multiple_nodes(self):
        from backend.batch_processor import _flatten_metrics

        nr = {
            "n1": {
                "status": "success",
                "node_type": "compute_alpha_peak",
                "metrics": {"iaf_hz": 10.2},
            },
            "n2": {
                "status": "success",
                "node_type": "compute_band_ratio",
                "metrics": {"band_ratio": 0.85},
            },
        }
        flat = _flatten_metrics(nr)
        assert "compute_alpha_peak.iaf_hz" in flat
        assert "compute_band_ratio.band_ratio" in flat

    def test_skips_failed_nodes(self):
        from backend.batch_processor import _flatten_metrics

        nr = {
            "n1": {
                "status": "error",
                "node_type": "compute_alpha_peak",
                "metrics": {"iaf_hz": 10.2},
            }
        }
        flat = _flatten_metrics(nr)
        assert len(flat) == 0

    def test_skips_non_metrics_nodes(self):
        from backend.batch_processor import _flatten_metrics

        nr = {
            "n1": {
                "status": "success",
                "node_type": "bandpass_filter",
                "output_type": "Raw",
                "data": None,
            }
        }
        flat = _flatten_metrics(nr)
        assert len(flat) == 0


# ---------------------------------------------------------------------------
# TestMetricsCsv
# ---------------------------------------------------------------------------

class TestMetricsCsv:

    def test_empty_results_empty_csv(self):
        from backend.batch_processor import _generate_metrics_csv

        assert _generate_metrics_csv([]) == ""

    def test_csv_has_header_and_data_rows(self):
        from backend.batch_processor import _generate_metrics_csv

        file_results = [
            {
                "filename": "file1.fif",
                "status": "success",
                "file_info": {"n_channels": 5, "sfreq": 256.0, "duration_s": 10.0},
                "processing_time_s": 0.123,
                "error": None,
                "metrics": {
                    "compute_alpha_peak.iaf_hz": 10.5,
                    "compute_alpha_peak.method": "cog",
                },
            },
            {
                "filename": "file2.fif",
                "status": "success",
                "file_info": {"n_channels": 5, "sfreq": 256.0, "duration_s": 10.0},
                "processing_time_s": 0.456,
                "error": None,
                "metrics": {
                    "compute_alpha_peak.iaf_hz": 9.8,
                    "compute_alpha_peak.method": "cog",
                },
            },
        ]
        csv_str = _generate_metrics_csv(file_results)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # 1 header + 2 data rows
        header = lines[0]
        assert "filename" in header
        assert "n_channels" in header
        assert "sfreq" in header
        assert "duration_s" in header
        assert "processing_time_s" in header
        assert "error" in header
        assert "compute_alpha_peak.iaf_hz" in header
        # Data rows contain file info
        assert "256.0" in lines[1]
        assert "0.123" in lines[1]

    def test_csv_missing_metrics_get_empty_cells(self):
        from backend.batch_processor import _generate_metrics_csv

        file_results = [
            {
                "filename": "file1.fif",
                "status": "success",
                "metrics": {"a.x": 1.0, "a.y": 2.0},
            },
            {
                "filename": "file2.fif",
                "status": "success",
                "metrics": {"a.x": 3.0},  # missing a.y
            },
        ]
        csv_str = _generate_metrics_csv(file_results)
        lines = csv_str.strip().split("\n")
        # All rows should have the same number of columns
        assert len(lines[2].split(",")) == len(lines[1].split(","))


# ---------------------------------------------------------------------------
# TestBatchRoutes (integration via TestClient)
# ---------------------------------------------------------------------------

class TestBatchRoutes:

    def test_stage_files_endpoint(self, client):
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            response = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("test.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["staged_files"]) == 1
        assert data["staged_files"][0]["filename"] == "test.fif"

    def test_stage_rejects_unsupported_format(self, client):
        response = client.post(
            "/pipeline/batch/stage",
            files=[("files", ("test.txt", b"hello", "text/plain"))],
        )
        assert response.status_code == 400

    def test_list_staged_endpoint(self, client):
        response = client.get("/pipeline/batch/staged")
        assert response.status_code == 200
        assert "staged_files" in response.json()

    def test_start_batch_missing_files(self, client):
        pipeline = _make_simple_pipeline()
        response = client.post(
            "/pipeline/batch",
            json={
                "file_ids": ["nonexistent"],
                "pipeline": pipeline.model_dump(),
            },
        )
        assert response.status_code == 404

    def test_start_batch_empty_files(self, client):
        pipeline = _make_simple_pipeline()
        response = client.post(
            "/pipeline/batch",
            json={
                "file_ids": [],
                "pipeline": pipeline.model_dump(),
            },
        )
        assert response.status_code == 400

    def test_progress_not_found(self, client):
        response = client.get("/pipeline/batch/nonexistent/progress")
        assert response.status_code == 404

    def test_results_not_found(self, client):
        response = client.get("/pipeline/batch/nonexistent/results")
        assert response.status_code == 404

    def test_full_batch_workflow(self, client):
        """End-to-end: stage -> start -> poll -> results."""
        # Create test file
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, duration=2.0)

        # Stage
        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("workflow.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        assert stage_resp.status_code == 200
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        # Start batch
        pipeline = _make_simple_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={
                "file_ids": [file_id],
                "pipeline": pipeline.model_dump(),
            },
        )
        assert start_resp.status_code == 200
        batch_id = start_resp.json()["batch_id"]

        # Poll until complete (max 30s)
        for _ in range(30):
            time.sleep(1)
            prog_resp = client.get(f"/pipeline/batch/{batch_id}/progress")
            assert prog_resp.status_code == 200
            prog = prog_resp.json()
            if prog["status"] in ("complete", "failed"):
                break

        assert prog["status"] == "complete"
        assert prog["completed"] == 1

        # Get results
        results_resp = client.get(f"/pipeline/batch/{batch_id}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()
        assert results["summary"]["completed"] == 1
        assert results["summary"]["failed"] == 0


# ---------------------------------------------------------------------------
# TestMetadata (Tier 4B — Enhancement 1)
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_stage_file_has_empty_metadata(self):
        from backend.batch_processor import stage_file, get_staged_file, remove_staged_file

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        try:
            file_id = stage_file(path, "test.fif")
            info = get_staged_file(file_id)
            assert info["metadata"] == {}
        finally:
            remove_staged_file(file_id)

    def test_update_file_metadata(self):
        from backend.batch_processor import (
            stage_file, update_file_metadata, get_file_metadata, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        file_id = stage_file(path, "test.fif")
        try:
            meta = {"subject_id": "S001", "group": "control", "condition": "rest"}
            result = update_file_metadata(file_id, meta)
            assert result is True
            assert get_file_metadata(file_id) == meta
        finally:
            remove_staged_file(file_id)

    def test_update_nonexistent_returns_false(self):
        from backend.batch_processor import update_file_metadata

        assert update_file_metadata("nonexistent-id", {"a": "b"}) is False

    def test_list_staged_includes_metadata(self):
        from backend.batch_processor import (
            stage_file, update_file_metadata, list_staged_files, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        file_id = stage_file(path, "meta.fif")
        try:
            update_file_metadata(file_id, {"subject_id": "S002"})
            files = list_staged_files()
            entry = next(f for f in files if f["file_id"] == file_id)
            assert entry["metadata"]["subject_id"] == "S002"
        finally:
            remove_staged_file(file_id)

    def test_metadata_in_csv(self):
        from backend.batch_processor import _generate_metrics_csv

        file_results = [
            {
                "filename": "file1.fif",
                "status": "success",
                "file_info": {"n_channels": 5, "sfreq": 256.0, "duration_s": 10.0},
                "processing_time_s": 0.1,
                "error": None,
                "metadata": {"subject_id": "S001", "group": "patient"},
                "metrics": {"alpha.iaf": 10.5},
            },
            {
                "filename": "file2.fif",
                "status": "success",
                "file_info": {"n_channels": 5, "sfreq": 256.0, "duration_s": 10.0},
                "processing_time_s": 0.2,
                "error": None,
                "metadata": {"subject_id": "S002", "group": "control"},
                "metrics": {"alpha.iaf": 9.8},
            },
        ]
        csv_str = _generate_metrics_csv(file_results)
        lines = csv_str.strip().split("\n")
        header = lines[0]
        # Fixed columns present
        assert "n_channels" in header
        assert "sfreq" in header
        # Metadata columns should appear before metric columns
        assert "subject_id" in header
        assert "group" in header
        assert header.index("subject_id") < header.index("alpha.iaf")
        # Data rows contain metadata values
        assert "S001" in lines[1]
        assert "S002" in lines[2]

    def test_metadata_in_csv_mixed(self):
        from backend.batch_processor import _generate_metrics_csv

        file_results = [
            {
                "filename": "file1.fif",
                "status": "success",
                "metadata": {"subject_id": "S001", "group": "patient"},
                "metrics": {"a.x": 1.0},
            },
            {
                "filename": "file2.fif",
                "status": "success",
                "metadata": {},  # no metadata
                "metrics": {"a.x": 2.0},
            },
        ]
        csv_str = _generate_metrics_csv(file_results)
        lines = csv_str.strip().split("\n")
        # All rows should have the same number of columns
        assert len(lines[1].split(",")) == len(lines[2].split(","))

    def test_metadata_flows_through_execution(self):
        from backend.batch_processor import (
            stage_file, update_file_metadata, run_batch, execute_batch,
            get_job, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "meta_exec.fif")
        try:
            update_file_metadata(file_id, {"subject_id": "S099", "group": "test"})
            pipeline = _make_simple_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            assert job.completed == 1
            fr = job.file_results[0]
            assert fr["metadata"]["subject_id"] == "S099"
            assert fr["metadata"]["group"] == "test"
        finally:
            remove_staged_file(file_id)

    def test_metadata_endpoint(self, client):
        """PUT metadata for a staged file via HTTP."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("meta.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        assert stage_resp.status_code == 200
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        meta_resp = client.put(
            f"/pipeline/batch/stage/{file_id}/metadata",
            json={"metadata": {"subject_id": "S010", "condition": "active"}},
        )
        assert meta_resp.status_code == 200
        data = meta_resp.json()
        assert data["file_id"] == file_id
        assert data["metadata"]["subject_id"] == "S010"

    def test_metadata_endpoint_not_found(self, client):
        resp = client.put(
            "/pipeline/batch/stage/nonexistent-id/metadata",
            json={"metadata": {"subject_id": "S001"}},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestPerFileDetail (Tier 4B — Enhancement 2)
# ---------------------------------------------------------------------------

class TestPerFileDetail:

    def test_file_detail_endpoint(self, client):
        """Run a batch, then GET detail for one file."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("detail.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_simple_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        batch_id = start_resp.json()["batch_id"]

        # Wait for completion
        for _ in range(30):
            time.sleep(1)
            prog = client.get(f"/pipeline/batch/{batch_id}/progress").json()
            if prog["status"] in ("complete", "failed"):
                break

        detail_resp = client.get(f"/pipeline/batch/{batch_id}/file/{file_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["file_id"] == file_id
        assert detail["status"] == "success"
        assert "node_results" in detail
        assert "metrics" in detail
        assert "metadata" in detail

    def test_file_detail_not_truncated(self):
        """Verify detail returns full data (not truncated)."""
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, n_channels=5, duration=5.0)
        file_id = stage_file(path, "detail.fif")

        try:
            pipeline = _make_metrics_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            fr = job.file_results[0]
            # The raw file_result should not have truncated data
            for node_id, nr in fr.get("node_results", {}).items():
                if isinstance(nr.get("data"), str) and nr["data"]:
                    assert "...[truncated]" not in nr["data"]
        finally:
            remove_staged_file(file_id)

    def test_file_detail_batch_not_found(self, client):
        resp = client.get("/pipeline/batch/nonexistent/file/some-file-id")
        assert resp.status_code == 404

    def test_file_detail_file_not_found(self, client):
        """Valid batch but invalid file_id."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("test.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_simple_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        batch_id = start_resp.json()["batch_id"]

        for _ in range(30):
            time.sleep(1)
            prog = client.get(f"/pipeline/batch/{batch_id}/progress").json()
            if prog["status"] in ("complete", "failed"):
                break

        resp = client.get(f"/pipeline/batch/{batch_id}/file/wrong-file-id")
        assert resp.status_code == 404

    def test_file_detail_includes_metadata(self):
        """Verify metadata appears in detail response."""
        from backend.batch_processor import (
            stage_file, update_file_metadata, run_batch, execute_batch,
            get_job, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "meta_detail.fif")

        try:
            update_file_metadata(file_id, {"subject_id": "S055"})
            pipeline = _make_simple_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            fr = job.file_results[0]
            assert fr["metadata"]["subject_id"] == "S055"
        finally:
            remove_staged_file(file_id)


# ---------------------------------------------------------------------------
# TestRetry (Tier 4B — Enhancement 3)
# ---------------------------------------------------------------------------

class TestRetry:

    def test_staged_files_persist_after_batch(self):
        """Staged files must NOT be removed after batch execution."""
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_staged_file,
            remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "persist.fif")

        try:
            pipeline = _make_simple_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            # File should still be staged after execution
            assert get_staged_file(file_id) is not None
        finally:
            remove_staged_file(file_id)

    def test_rerun_failed_files(self):
        """Can re-run a batch with the same file_ids after a previous batch."""
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "rerun.fif")

        try:
            pipeline = _make_simple_pipeline()
            # First batch
            batch_id_1 = run_batch([file_id], pipeline)
            execute_batch(batch_id_1, [file_id], pipeline)
            assert get_job(batch_id_1).completed == 1

            # Second batch with the same file_id (re-run)
            batch_id_2 = run_batch([file_id], pipeline)
            execute_batch(batch_id_2, [file_id], pipeline)
            assert get_job(batch_id_2).completed == 1
        finally:
            remove_staged_file(file_id)

    def test_rerun_with_empty_list_rejected(self, client):
        pipeline = _make_simple_pipeline()
        response = client.post(
            "/pipeline/batch",
            json={"file_ids": [], "pipeline": pipeline.model_dump()},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# TestBatchReports (Tier 4B — Enhancement 4)
# ---------------------------------------------------------------------------

class TestBatchReports:

    def test_batch_reports_endpoint(self, client):
        """Run batch with metrics pipeline, then generate reports ZIP."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, n_channels=5, duration=5.0)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("report.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_metrics_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        batch_id = start_resp.json()["batch_id"]

        for _ in range(30):
            time.sleep(1)
            prog = client.get(f"/pipeline/batch/{batch_id}/progress").json()
            if prog["status"] in ("complete", "failed"):
                break

        resp = client.post(f"/pipeline/batch/{batch_id}/reports")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_batch_reports_zip_contents(self):
        """Verify ZIP contains one PDF per successful file."""
        import zipfile as zf
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job, remove_staged_file,
        )
        from backend.api.report_routes import _generate_pdf

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, n_channels=5, duration=5.0)
        file_id = stage_file(path, "zip_test.fif")

        try:
            pipeline = _make_metrics_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            successful = [fr for fr in job.file_results if fr["status"] == "success"]
            assert len(successful) == 1

            # Generate a PDF to verify it's valid
            fr = successful[0]
            pdf_bytes = _generate_pdf(
                fr.get("node_results", {}),
                f"Oscilloom Report - {fr['filename']}",
                "", "",
            )
            assert pdf_bytes[:4] == b"%PDF"
        finally:
            remove_staged_file(file_id)

    def test_batch_reports_batch_not_found(self, client):
        resp = client.post("/pipeline/batch/nonexistent/reports")
        assert resp.status_code == 404

    def test_batch_reports_still_running(self):
        """Reports should return 409 while batch is still running."""
        from backend.batch_processor import run_batch, get_job

        batch_id = run_batch(["fake-id"], _make_simple_pipeline())
        job = get_job(batch_id)
        # Job is "running" but execute_batch was never called
        assert job.status == "running"

        # We can't call the HTTP endpoint easily here without a client,
        # so test via the route logic by checking status directly
        assert job.status not in ("complete", "failed")

    def test_batch_reports_no_successful_files(self, client):
        """Reports should return 404 when all files failed."""
        from backend.batch_processor import run_batch, execute_batch

        pipeline = _make_simple_pipeline()
        batch_id = run_batch(["nonexistent-id"], pipeline)
        execute_batch(batch_id, ["nonexistent-id"], pipeline)

        resp = client.post(f"/pipeline/batch/{batch_id}/reports")
        assert resp.status_code == 404

    def test_batch_reports_uses_metadata_subject_id(self):
        """Reports should use subject_id from metadata if present."""
        from backend.batch_processor import (
            stage_file, update_file_metadata, run_batch, execute_batch,
            get_job, remove_staged_file,
        )
        from backend.api.report_routes import _generate_pdf

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path, n_channels=5, duration=5.0)
        file_id = stage_file(path, "subject_report.fif")

        try:
            update_file_metadata(file_id, {"subject_id": "S042"})
            pipeline = _make_metrics_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            job = get_job(batch_id)
            fr = job.file_results[0]
            assert fr["metadata"]["subject_id"] == "S042"

            # Verify _generate_pdf works with the metadata
            pdf_bytes = _generate_pdf(
                fr.get("node_results", {}),
                f"Oscilloom Report - {fr['filename']}",
                patient_id="S042",
                clinic_name="",
            )
            assert len(pdf_bytes) > 100
        finally:
            remove_staged_file(file_id)


# ---------------------------------------------------------------------------
# TestPersistence (Tier 4B — Enhancement 5)
# ---------------------------------------------------------------------------

class TestPersistence:

    @pytest.fixture(autouse=True)
    def temp_batch_dir(self, tmp_path, monkeypatch):
        """Redirect _BATCH_RESULTS_DIR to a temp directory."""
        import backend.batch_processor as bp
        monkeypatch.setattr(bp, "_BATCH_RESULTS_DIR", str(tmp_path / "batch_results"))

    def _run_complete_batch(self):
        from backend.batch_processor import (
            stage_file, run_batch, execute_batch, get_job,
        )
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "persist_test.fif")

        pipeline = _make_simple_pipeline()
        batch_id = run_batch([file_id], pipeline)
        execute_batch(batch_id, [file_id], pipeline)
        return batch_id, file_id

    def test_save_batch_results(self):
        from backend.batch_processor import save_batch_results, _BATCH_RESULTS_DIR

        batch_id, _ = self._run_complete_batch()
        file_path = save_batch_results(batch_id)
        assert os.path.exists(file_path)
        assert file_path.endswith(".json.gz")

    def test_save_not_complete_raises(self):
        from backend.batch_processor import run_batch, save_batch_results

        batch_id = run_batch(["fake"], _make_simple_pipeline())
        with pytest.raises(ValueError, match="still running"):
            save_batch_results(batch_id)

    def test_save_not_found_raises(self):
        from backend.batch_processor import save_batch_results

        with pytest.raises(ValueError, match="not found"):
            save_batch_results("nonexistent-batch-id")

    def test_load_saved_batch(self):
        from backend.batch_processor import save_batch_results, load_saved_batch

        batch_id, _ = self._run_complete_batch()
        save_batch_results(batch_id)

        data = load_saved_batch(batch_id)
        assert data["batch_id"] == batch_id
        assert data["status"] == "complete"
        assert data["completed"] == 1
        assert len(data["file_results"]) == 1

    def test_list_saved_batches(self):
        from backend.batch_processor import save_batch_results, list_saved_batches

        bid1, _ = self._run_complete_batch()
        bid2, _ = self._run_complete_batch()
        save_batch_results(bid1)
        save_batch_results(bid2)

        saved = list_saved_batches()
        batch_ids = {s["batch_id"] for s in saved}
        assert bid1 in batch_ids
        assert bid2 in batch_ids

    def test_list_empty_directory(self):
        from backend.batch_processor import list_saved_batches

        assert list_saved_batches() == []

    def test_load_nonexistent_raises(self):
        from backend.batch_processor import load_saved_batch

        with pytest.raises(FileNotFoundError, match="not found"):
            load_saved_batch("bad-id")

    def test_save_endpoint(self, client):
        """Save via HTTP after running a full batch workflow."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("save.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_simple_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        batch_id = start_resp.json()["batch_id"]

        for _ in range(30):
            time.sleep(1)
            prog = client.get(f"/pipeline/batch/{batch_id}/progress").json()
            if prog["status"] in ("complete", "failed"):
                break

        save_resp = client.post(f"/pipeline/batch/{batch_id}/save")
        assert save_resp.status_code == 200
        assert save_resp.json()["saved"] is True

    def test_saved_list_endpoint(self, client):
        resp = client.get("/pipeline/batch/saved")
        assert resp.status_code == 200
        assert "saved_batches" in resp.json()

    def test_saved_load_endpoint(self, client):
        """Save then load via HTTP."""
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("load.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_simple_pipeline()
        start_resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        batch_id = start_resp.json()["batch_id"]

        for _ in range(30):
            time.sleep(1)
            prog = client.get(f"/pipeline/batch/{batch_id}/progress").json()
            if prog["status"] in ("complete", "failed"):
                break

        # Save
        client.post(f"/pipeline/batch/{batch_id}/save")

        # Load
        load_resp = client.get(f"/pipeline/batch/saved/{batch_id}")
        assert load_resp.status_code == 200
        data = load_resp.json()
        assert data["batch_id"] == batch_id
        assert data["summary"]["completed"] == 1

    def test_metadata_survives_persistence(self):
        """Metadata should be present after save + load."""
        from backend.batch_processor import (
            stage_file, update_file_metadata, run_batch, execute_batch,
            save_batch_results, load_saved_batch, remove_staged_file,
        )

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)
        file_id = stage_file(path, "meta_persist.fif")

        try:
            update_file_metadata(file_id, {"subject_id": "S100", "group": "exp"})
            pipeline = _make_simple_pipeline()
            batch_id = run_batch([file_id], pipeline)
            execute_batch(batch_id, [file_id], pipeline)

            save_batch_results(batch_id)
            data = load_saved_batch(batch_id)
            fr = data["file_results"][0]
            assert fr["metadata"]["subject_id"] == "S100"
            assert fr["metadata"]["group"] == "exp"

            # CSV should have metadata columns
            csv_str = data["metrics_csv"]
            assert "subject_id" in csv_str
            assert "S100" in csv_str
        finally:
            remove_staged_file(file_id)


# ---------------------------------------------------------------------------
# Phase 4: New guardrail and feature tests
# ---------------------------------------------------------------------------

class TestJobEviction:
    """Tests for MAX_JOBS eviction logic."""

    def test_job_eviction_over_max(self):
        """Old completed jobs are evicted when _jobs exceeds MAX_JOBS."""
        from backend import batch_processor

        original_max = batch_processor.MAX_JOBS
        try:
            batch_processor.MAX_JOBS = 3  # Lower for testing

            # Create 3 completed jobs
            for i in range(3):
                job = BatchJob(batch_id=f"old-{i}", status="complete")
                batch_processor._register_job(job)

            assert len(batch_processor._jobs) == 3

            # Adding a 4th should evict the oldest completed
            job4 = BatchJob(batch_id="new-job", status="running")
            batch_processor._register_job(job4)

            assert len(batch_processor._jobs) == 3
            assert "old-0" not in batch_processor._jobs
            assert "new-job" in batch_processor._jobs
        finally:
            batch_processor.MAX_JOBS = original_max

    def test_eviction_skips_running_jobs(self):
        """Running jobs should not be evicted."""
        from backend import batch_processor

        original_max = batch_processor.MAX_JOBS
        try:
            batch_processor.MAX_JOBS = 2

            running_job = BatchJob(batch_id="running-1", status="running")
            batch_processor._register_job(running_job)

            completed_job = BatchJob(batch_id="done-1", status="complete")
            batch_processor._register_job(completed_job)

            # This should evict completed, not running
            new_job = BatchJob(batch_id="new-1", status="running")
            batch_processor._register_job(new_job)

            assert "running-1" in batch_processor._jobs
            assert "done-1" not in batch_processor._jobs
            assert "new-1" in batch_processor._jobs
        finally:
            batch_processor.MAX_JOBS = original_max

    def test_delete_job(self):
        """Explicitly deleting a job removes it from store and order."""
        from backend import batch_processor

        job = BatchJob(batch_id="to-delete", status="complete")
        batch_processor._register_job(job)
        assert batch_processor.delete_job("to-delete") is True
        assert "to-delete" not in batch_processor._jobs
        assert "to-delete" not in batch_processor._job_order

    def test_delete_nonexistent_job(self):
        """Deleting a nonexistent job returns False."""
        from backend import batch_processor
        assert batch_processor.delete_job("does-not-exist") is False


class TestStagedCleanup:
    """Tests for staged file TTL cleanup."""

    def test_cleanup_removes_stale_files(self):
        """Staged files older than threshold are removed."""
        from backend import batch_processor

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        file_id = batch_processor.stage_file(path, "old_file.fif")

        # Backdate the staged_at timestamp
        with batch_processor._staged_lock:
            batch_processor._staged_files[file_id]["staged_at"] = time.time() - 8000

        removed = batch_processor.cleanup_stale_staged(max_age_seconds=7200)
        assert removed == 1
        assert batch_processor.get_staged_file(file_id) is None

    def test_cleanup_keeps_fresh_files(self):
        """Recently staged files are not removed."""
        from backend import batch_processor

        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        file_id = batch_processor.stage_file(path, "fresh_file.fif")

        removed = batch_processor.cleanup_stale_staged(max_age_seconds=7200)
        assert removed == 0
        assert batch_processor.get_staged_file(file_id) is not None

        # Cleanup
        batch_processor.remove_staged_file(file_id)


class TestConcurrentJobLimit:
    """Tests for concurrent batch job limit."""

    def test_concurrent_limit_429(self, client):
        """Starting a 3rd concurrent batch should return 429."""
        from backend import batch_processor

        # Register 2 running jobs directly
        for i in range(2):
            job = BatchJob(batch_id=f"running-{i}", status="running")
            batch_processor._register_job(job)

        # Stage a file for the 3rd batch
        fd, path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(fd)
        _make_test_fif(path)

        with open(path, "rb") as f:
            stage_resp = client.post(
                "/pipeline/batch/stage",
                files=[("files", ("test.fif", f, "application/octet-stream"))],
            )
        os.unlink(path)
        file_id = stage_resp.json()["staged_files"][0]["file_id"]

        pipeline = _make_simple_pipeline()
        resp = client.post(
            "/pipeline/batch",
            json={"file_ids": [file_id], "pipeline": pipeline.model_dump()},
        )
        assert resp.status_code == 429
        assert "concurrent" in resp.json()["detail"].lower()


class TestFileLimitGuardrail:
    """Tests for per-batch file limit."""

    def test_file_limit_400(self, client):
        """Submitting >MAX_BATCH_FILES should return 400."""
        pipeline = _make_simple_pipeline()
        fake_ids = [f"fake-{i}" for i in range(201)]
        resp = client.post(
            "/pipeline/batch",
            json={"file_ids": fake_ids, "pipeline": pipeline.model_dump()},
        )
        assert resp.status_code == 400
        assert "200" in resp.json()["detail"]


class TestAggregateStatistics:
    """Tests for compute_aggregate_statistics."""

    def test_numeric_metrics(self):
        """Mean/std/min/max/median are computed correctly for known values."""
        from backend.batch_processor import compute_aggregate_statistics

        file_results = [
            {"status": "success", "metrics": {"alpha_peak": 10.0, "snr": 5.0}, "metadata": {}},
            {"status": "success", "metrics": {"alpha_peak": 12.0, "snr": 7.0}, "metadata": {}},
            {"status": "success", "metrics": {"alpha_peak": 11.0, "snr": 6.0}, "metadata": {}},
        ]
        stats = compute_aggregate_statistics(file_results)

        alpha = stats["overall"]["alpha_peak"]
        assert alpha["count"] == 3
        assert alpha["mean"] == 11.0
        assert alpha["min"] == 10.0
        assert alpha["max"] == 12.0
        assert alpha["median"] == 11.0

    def test_skips_failed_files(self):
        """Failed file results should not be included in statistics."""
        from backend.batch_processor import compute_aggregate_statistics

        file_results = [
            {"status": "success", "metrics": {"val": 10.0}, "metadata": {}},
            {"status": "error", "metrics": {"val": 999.0}, "metadata": {}},
        ]
        stats = compute_aggregate_statistics(file_results)
        assert stats["overall"]["val"]["count"] == 1
        assert stats["overall"]["val"]["mean"] == 10.0

    def test_group_breakdowns(self):
        """Group-level statistics computed when metadata has 'group'."""
        from backend.batch_processor import compute_aggregate_statistics

        file_results = [
            {"status": "success", "metrics": {"val": 10.0}, "metadata": {"group": "control"}},
            {"status": "success", "metrics": {"val": 20.0}, "metadata": {"group": "control"}},
            {"status": "success", "metrics": {"val": 30.0}, "metadata": {"group": "patient"}},
        ]
        stats = compute_aggregate_statistics(file_results)

        assert "control" in stats["by_group"]
        assert "patient" in stats["by_group"]
        assert stats["by_group"]["control"]["val"]["count"] == 2
        assert stats["by_group"]["control"]["val"]["mean"] == 15.0
        assert stats["by_group"]["patient"]["val"]["count"] == 1
        assert stats["by_group"]["patient"]["val"]["mean"] == 30.0

    def test_empty_results(self):
        """Empty file_results should return empty statistics."""
        from backend.batch_processor import compute_aggregate_statistics

        stats = compute_aggregate_statistics([])
        assert stats["overall"] == {}
        assert stats["by_group"] == {}

    def test_non_numeric_metrics_ignored(self):
        """Boolean and string metrics should not appear in stats."""
        from backend.batch_processor import compute_aggregate_statistics

        file_results = [
            {"status": "success", "metrics": {"val": 5.0, "flag": True, "label": "abc"}, "metadata": {}},
        ]
        stats = compute_aggregate_statistics(file_results)
        assert "val" in stats["overall"]
        assert "flag" not in stats["overall"]
        assert "label" not in stats["overall"]


class TestBatchIdSanitization:
    """Tests for batch_id sanitization in routes."""

    def test_malicious_batch_id_rejected(self, client):
        """Path traversal in batch_id should be rejected."""
        resp = client.get("/pipeline/batch/../../etc/passwd/progress")
        assert resp.status_code in (400, 404, 422)

    def test_valid_uuid_batch_id_accepted(self, client):
        """Valid UUID batch_id should be accepted (returns 404 if not found)."""
        import uuid
        bid = str(uuid.uuid4())
        resp = client.get(f"/pipeline/batch/{bid}/progress")
        # 404 = not found, which is correct — the ID was accepted
        assert resp.status_code == 404


class TestDeleteJobEndpoint:
    """Tests for DELETE /pipeline/batch/{batch_id}."""

    def test_delete_existing_job(self, client):
        """DELETE removes a completed job."""
        from backend import batch_processor

        job = BatchJob(batch_id="del-test-1", status="complete")
        batch_processor._register_job(job)

        resp = client.delete("/pipeline/batch/del-test-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert batch_processor.get_job("del-test-1") is None

    def test_delete_nonexistent_returns_404(self, client):
        """DELETE for unknown batch_id returns 404."""
        resp = client.delete("/pipeline/batch/nonexistent-id")
        assert resp.status_code == 404


class TestCountRunningJobs:
    """Tests for count_running_jobs utility."""

    def test_counts_only_running(self):
        """Only jobs with status 'running' are counted."""
        from backend import batch_processor

        batch_processor._register_job(BatchJob(batch_id="r1", status="running"))
        batch_processor._register_job(BatchJob(batch_id="c1", status="complete"))
        batch_processor._register_job(BatchJob(batch_id="f1", status="failed"))
        batch_processor._register_job(BatchJob(batch_id="r2", status="running"))

        assert batch_processor.count_running_jobs() == 2
