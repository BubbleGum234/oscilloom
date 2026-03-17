"""
backend/api/batch_routes.py

Batch processing endpoints for Tier 4 + Tier 4B.

Endpoints:
  POST   /pipeline/batch/stage                  -- Upload files for batch processing
  GET    /pipeline/batch/staged                 -- List staged files
  DELETE /pipeline/batch/staged                 -- Clear all staged files
  PUT    /pipeline/batch/stage/{id}/metadata    -- Set per-file metadata (4B)
  POST   /pipeline/batch                        -- Start a batch job
  GET    /pipeline/batch/{id}/progress          -- Poll job progress
  GET    /pipeline/batch/{id}/results           -- Get final results (truncated plots)
  GET    /pipeline/batch/{id}/file/{file_id}    -- Get one file detail (full plots, 4B)
  POST   /pipeline/batch/{id}/reports           -- Generate PDF reports ZIP (4B)
  POST   /pipeline/batch/{id}/cancel            -- Cancel a running job
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
import zipfile
from typing import List

from pydantic import BaseModel
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from backend import batch_processor, validation
from backend.models import BatchRequest, ReportSections
from backend.path_security import sanitize_id, sanitize_filename
from backend.rate_limit import limiter


class FileMetadataUpdate(BaseModel):
    metadata: dict[str, str]


class BatchReportConfig(BaseModel):
    """Configuration for batch report generation."""
    clinic_name: str = ""
    notes: str = ""
    pipeline_config: list[dict] | None = None
    sections: ReportSections = ReportSections()

router = APIRouter(prefix="/pipeline/batch", tags=["Batch Processing"])

# Supported extensions (same as session_routes)
SUPPORTED_EXTENSIONS = {".edf", ".fif", ".bdf", ".set", ".vhdr", ".cnt"}

MAX_BATCH_FILES = 200


# ---------------------------------------------------------------------------
# File staging
# ---------------------------------------------------------------------------

@router.post("/stage", summary="Upload files for batch processing")
@limiter.limit("10/minute")
async def stage_files(request: Request, files: List[UploadFile] = File(...)) -> dict:
    """
    Accepts one or more EEG file uploads and stages them on the server.

    Returns a list of {file_id, filename} objects. Use the file_ids in
    the POST /pipeline/batch request body.
    """
    # Clean up stale staged files from previous sessions
    batch_processor.cleanup_stale_staged()

    staged = []

    for upload_file in files:
        filename = upload_file.filename or "unknown"
        _, ext = os.path.splitext(filename.lower())

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '{ext}' for file '{filename}'. "
                       f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        # Write to temp file (enforce upload size limit)
        MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
                content = await upload_file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{filename}' exceeds maximum upload size of 500 MB.",
                    )
                tmp.write(content)

            file_id = batch_processor.stage_file(tmp_path, filename)
        except HTTPException:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        staged.append({"file_id": file_id, "filename": filename})

    return {"staged_files": staged, "count": len(staged)}


@router.get("/staged", summary="List all staged files")
def list_staged() -> dict:
    """Returns all files currently staged for batch processing."""
    files = batch_processor.list_staged_files()
    return {"staged_files": files, "count": len(files)}


@router.delete("/staged", summary="Clear all staged files")
def clear_staged() -> dict:
    """Remove all staged files and free disk space."""
    count = batch_processor.clear_staged_files()
    return {"cleared": count}


@router.put("/stage/{file_id}/metadata", summary="Set metadata for a staged file")
def update_metadata(file_id: str, body: FileMetadataUpdate) -> dict:
    """Update subject metadata (subject_id, group, condition) for a staged file."""
    if batch_processor.get_staged_file(file_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Staged file '{file_id}' not found."
        )
    batch_processor.update_file_metadata(file_id, body.metadata)
    return {"file_id": file_id, "metadata": body.metadata}


# ---------------------------------------------------------------------------
# Saved batch results (Tier 4B — must be BEFORE /{batch_id}/... routes)
# ---------------------------------------------------------------------------

@router.get("/saved", summary="List saved batch results")
def list_saved() -> dict:
    """Returns summaries of all saved batch results (no full file_results)."""
    batches = batch_processor.list_saved_batches()
    return {"saved_batches": batches, "count": len(batches)}


@router.get("/saved/{saved_batch_id}", summary="Load saved batch results")
def load_saved(saved_batch_id: str) -> dict:
    """Load a previously saved batch result from disk."""
    saved_batch_id = sanitize_id(saved_batch_id)
    try:
        data = batch_processor.load_saved_batch(saved_batch_id)

        # Apply the same plot truncation as get_results()
        slim_results = []
        for fr in data.get("file_results", []):
            slim_nr = {}
            for node_id, nr in fr.get("node_results", {}).items():
                slim_entry = {**nr}
                if (
                    isinstance(slim_entry.get("data"), str)
                    and len(slim_entry["data"]) > 200
                ):
                    slim_entry["data"] = (
                        slim_entry["data"][:100] + "...[truncated]"
                    )
                slim_nr[node_id] = slim_entry
            slim_results.append({**fr, "node_results": slim_nr})

        runtime_s = (
            round(data.get("finished_at", 0) - data.get("started_at", 0), 2)
            if data.get("finished_at", 0) > 0
            else 0
        )

        statistics = batch_processor.compute_aggregate_statistics(
            data.get("file_results", [])
        )

        return {
            "batch_id": data["batch_id"],
            "status": data["status"],
            "file_results": slim_results,
            "failed_files": data.get("failed_files", []),
            "metrics_csv": data.get("metrics_csv", ""),
            "statistics": statistics,
            "summary": {
                "total": data["total"],
                "completed": data["completed"],
                "failed": data["failed"],
                "runtime_s": runtime_s,
            },
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Saved batch '{saved_batch_id}' not found.",
        )


# ---------------------------------------------------------------------------
# Job deletion
# ---------------------------------------------------------------------------

@router.delete("/{batch_id}", summary="Delete a batch job")
def delete_batch_job(batch_id: str) -> dict:
    batch_id = sanitize_id(batch_id)
    if not batch_processor.delete_job(batch_id):
        raise HTTPException(status_code=404, detail=f"Batch job '{batch_id}' not found.")
    return {"batch_id": batch_id, "deleted": True}


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

@router.post("", summary="Start a batch processing job")
@limiter.limit("5/minute")
async def start_batch(request: Request, body: BatchRequest = ...) -> dict:
    """
    Starts a batch job to process multiple files through a pipeline.

    Returns immediately with a batch_id. Poll GET /progress for status.
    The pipeline is validated before starting.
    """
    # Validate the pipeline first
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pipeline: {'; '.join(errors)}",
        )

    if not body.file_ids:
        raise HTTPException(status_code=400, detail="No files specified.")

    if len(body.file_ids) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(body.file_ids)}). Maximum is {MAX_BATCH_FILES}.",
        )

    # Enforce concurrent job limit
    if batch_processor.count_running_jobs() >= 2:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent batch jobs. Wait for a running job to finish.",
        )

    # Verify all file_ids are valid
    for file_id in body.file_ids:
        if batch_processor.get_staged_file(file_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Staged file '{file_id}' not found. "
                       "Stage files first via POST /pipeline/batch/stage.",
            )

    # Create the job
    batch_id = batch_processor.run_batch(
        file_ids=body.file_ids,
        graph=body.pipeline,
    )

    # Start processing in background (fire-and-forget)
    executor = request.app.state.executor
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        executor,
        batch_processor.execute_batch,
        batch_id,
        body.file_ids,
        body.pipeline,
    )

    return {"batch_id": batch_id, "total_files": len(body.file_ids)}


# ---------------------------------------------------------------------------
# Progress & results
# ---------------------------------------------------------------------------

@router.get("/{batch_id}/progress", summary="Get batch job progress")
def get_progress(batch_id: str) -> dict:
    """Poll this endpoint to update the progress bar in the frontend."""
    batch_id = sanitize_id(batch_id)
    job = batch_processor.get_job(batch_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Batch job '{batch_id}' not found."
        )

    with job.lock:
        return {
            "batch_id": job.batch_id,
            "status": job.status,
            "total": job.total,
            "completed": job.completed,
            "failed": job.failed,
            "current_file": job.current_file,
        }


@router.get("/{batch_id}/results", summary="Get batch job results")
def get_results(batch_id: str) -> dict:
    """
    Returns full results for a completed (or partially completed) batch.

    Plot data in node_results is truncated to keep the response small.
    """
    batch_id = sanitize_id(batch_id)
    job = batch_processor.get_job(batch_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Batch job '{batch_id}' not found."
        )

    with job.lock:
        # Strip base64 plot data to keep response manageable
        slim_results = []
        for fr in job.file_results:
            slim_nr = {}
            for node_id, nr in fr.get("node_results", {}).items():
                slim_entry = {**nr}
                if (
                    isinstance(slim_entry.get("data"), str)
                    and len(slim_entry["data"]) > 200
                ):
                    slim_entry["data"] = (
                        slim_entry["data"][:100] + "...[truncated]"
                    )
                slim_nr[node_id] = slim_entry
            slim_results.append({**fr, "node_results": slim_nr})

        metrics_csv = batch_processor._generate_metrics_csv(job.file_results)
        statistics = batch_processor.compute_aggregate_statistics(
            job.file_results
        )

        runtime_s = (
            round(job.finished_at - job.started_at, 2)
            if job.finished_at > 0
            else round(time.time() - job.started_at, 2)
        )

        return {
            "batch_id": job.batch_id,
            "status": job.status,
            "file_results": slim_results,
            "failed_files": list(job.failed_files),
            "metrics_csv": metrics_csv,
            "statistics": statistics,
            "summary": {
                "total": job.total,
                "completed": job.completed,
                "failed": job.failed,
                "runtime_s": runtime_s,
            },
        }


@router.get("/{batch_id}/file/{file_id}", summary="Get detailed results for one file")
def get_file_detail(batch_id: str, file_id: str) -> dict:
    """
    Returns the un-truncated node_results for a single file in a batch job.
    Plot base64 data is NOT truncated (unlike GET /results).
    """
    batch_id = sanitize_id(batch_id)
    job = batch_processor.get_job(batch_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Batch job '{batch_id}' not found."
        )

    with job.lock:
        for fr in job.file_results:
            if fr.get("file_id") == file_id:
                return {
                    "file_id": fr["file_id"],
                    "filename": fr["filename"],
                    "status": fr["status"],
                    "node_results": fr.get("node_results", {}),
                    "metrics": fr.get("metrics", {}),
                    "metadata": fr.get("metadata", {}),
                    "error": fr.get("error"),
                }

    raise HTTPException(
        status_code=404,
        detail=f"File '{file_id}' not found in batch '{batch_id}'.",
    )


@router.post("/{batch_id}/reports", summary="Generate PDF reports for all successful files")
@limiter.limit("3/minute")
def generate_batch_reports(
    batch_id: str,
    request: Request,
    config: BatchReportConfig | None = None,
) -> Response:
    """
    Generates a PDF report per successful file and returns them as a ZIP.
    Reuses _generate_pdf() from report_routes with all enhanced sections.

    Accepts an optional BatchReportConfig body with clinic_name, notes,
    pipeline_config, and section toggles. If no body is provided, uses
    defaults (all sections enabled, no notes).
    """
    batch_id = sanitize_id(batch_id)
    try:
        from backend.api.report_routes import _generate_pdf, _FPDF_AVAILABLE
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PDF generation requires fpdf2. Install with: pip install 'fpdf2>=2.7.0'",
        )

    if not _FPDF_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PDF generation requires fpdf2. Install with: pip install 'fpdf2>=2.7.0'",
        )

    if config is None:
        config = BatchReportConfig()

    job = batch_processor.get_job(batch_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Batch job '{batch_id}' not found."
        )

    with job.lock:
        if job.status not in ("complete", "failed"):
            raise HTTPException(
                status_code=409,
                detail=f"Batch job is still {job.status}. Wait for completion.",
            )
        file_results = list(job.file_results)

    successful = [fr for fr in file_results if fr.get("status") == "success"]
    if not successful:
        raise HTTPException(
            status_code=404,
            detail="No successful file results to generate reports from.",
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fr in successful:
            filename = fr.get("filename", "unknown")
            metadata = fr.get("metadata", {})
            subject_id = metadata.get("subject_id", "")
            title = f"Oscilloom Report - {filename}"

            pdf_bytes = _generate_pdf(
                node_results=fr.get("node_results", {}),
                title=title,
                patient_id=subject_id,
                clinic_name=config.clinic_name,
                pipeline_config=config.pipeline_config,
                notes=config.notes,
                sections=config.sections,
            )

            base_name = filename.rsplit(".", 1)[0]
            safe_name = sanitize_filename(base_name) + "_report.pdf"
            zf.writestr(safe_name, pdf_bytes)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="batch_reports_{batch_id[:8]}.zip"',
        },
    )


@router.post("/{batch_id}/save", summary="Save batch results to disk")
@limiter.limit("10/minute")
def save_results(batch_id: str, request: Request) -> dict:
    """Persist batch results as gzipped JSON for later retrieval."""
    batch_id = sanitize_id(batch_id)
    try:
        batch_processor.save_batch_results(batch_id)
        return {"batch_id": batch_id, "saved": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{batch_id}/cancel", summary="Cancel a running batch job")
def cancel_batch(batch_id: str) -> dict:
    """Cancel a running batch job. Already-processed files keep their results."""
    batch_id = sanitize_id(batch_id)
    job = batch_processor.get_job(batch_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Batch job '{batch_id}' not found."
        )

    with job.lock:
        if job.status == "running":
            job.status = "cancelled"

    return {"batch_id": batch_id, "status": "cancelled"}
