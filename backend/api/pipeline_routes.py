"""
backend/api/pipeline_routes.py

Routes for pipeline validation, execution, and Python script export.

All compute-heavy operations (MNE filtering, PSD computation) run in a
ThreadPoolExecutor thread via asyncio.run_in_executor so they do not block
the FastAPI event loop. The executor is attached to app.state in main.py.

Error handling strategy:
  - Validation errors → returned as structured {"valid": false, "errors": [...]}
  - Session not found → 404
  - MNE execution errors → returned as {"status": "error", "error": "<message>"}
    (not a 500) so the frontend can display them meaningfully per-node
  - Unexpected server errors → 500 with traceback logged server-side
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

import matplotlib
import mne
import numpy

from backend import engine, script_exporter, session_store, validation
from backend.engine import topological_sort
from backend.models import BidsExportRequest, ExecuteRequest, ExecuteResponse, ExportRequest, ExportResponse
from backend.rate_limit import limiter
from backend.registry import NODE_REGISTRY
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/validate", summary="Validate a pipeline graph")
def validate_pipeline_route(request: ExecuteRequest) -> dict:
    """
    Validates a PipelineGraph without executing it.

    Returns:
        {"valid": true, "errors": []}          on success
        {"valid": false, "errors": ["...", ...]}  on failure
    """
    errors = validation.validate_pipeline(request.pipeline)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


@router.post("/execute", response_model=ExecuteResponse, summary="Execute a pipeline")
@limiter.limit("20/minute")
async def execute_pipeline(body: ExecuteRequest, request: Request) -> ExecuteResponse:
    """
    Validates and executes a pipeline against a loaded EEG session.

    Execution runs in a ThreadPoolExecutor thread (see main.py) to avoid
    blocking FastAPI's event loop during MNE computation.

    Returns node results for each node in the pipeline. Visualization nodes
    include a base64-encoded PNG in the "data" field.

    Error responses use status="error" (not HTTP 4xx/5xx) so the frontend
    can display per-node error states on the canvas.
    """
    # Step 1: Validate the graph structure (fast, synchronous)
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        return ExecuteResponse(
            status="error",
            node_results={},
            error=f"Pipeline validation failed: {'; '.join(errors)}",
        )

    # Step 2: Get a copy of the session Raw object (raises KeyError if missing)
    try:
        raw_copy = session_store.get_raw_copy(body.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Step 3: Execute in thread pool (MNE computation, non-blocking)
    executor = request.app.state.executor
    loop = asyncio.get_event_loop()
    cache = session_store.get_execution_cache(body.session_id)

    try:
        node_results, node_outputs = await loop.run_in_executor(
            executor,
            lambda: engine.execute_pipeline(raw_copy, body.pipeline, cache=cache),
        )
        # Cache node outputs for inspector, export, and re-run features
        session_store.cache_node_outputs(body.session_id, node_outputs)
        return ExecuteResponse(status="success", node_results=node_results)

    except ValueError as e:
        # Cycle detection, unknown node type, etc. — controlled messages, safe to return.
        return ExecuteResponse(status="error", node_results={}, error=str(e))

    except Exception:
        # MNE errors (bad filter params, etc.) — log full traceback server-side,
        # return generic message to avoid leaking internal details.
        traceback.print_exc()
        return ExecuteResponse(
            status="error",
            node_results={},
            error="An internal error occurred during pipeline execution.",
        )


@router.post("/download_fif", summary="Download processed EEG data as .fif")
async def download_processed_fif(body: ExecuteRequest, request: Request) -> Response:
    """
    Re-runs the pipeline and returns the last Raw output as a binary .fif file.

    The returned file contains the most-downstream processed Raw object —
    i.e., the output of the last filter or resample node before any analysis
    or visualization nodes. If the pipeline has no Raw-producing nodes,
    returns 422.

    The file can be loaded back into MNE with:
        raw = mne.io.read_raw_fif("processed.fif", preload=True)
    """
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pipeline: {'; '.join(errors)}",
        )

    try:
        raw_copy = session_store.get_raw_copy(body.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    executor = request.app.state.executor
    loop = asyncio.get_event_loop()

    # Try cached outputs first (instant, no re-execution)
    cached = session_store.get_all_cached_outputs(body.session_id)
    last_raw = None

    if cached:
        try:
            order = engine.topological_sort(body.pipeline)
            for nid in order:
                output = cached.get(nid)
                if isinstance(output, mne.io.BaseRaw):
                    last_raw = output
        except ValueError as sort_err:
            # Topology sort failed on cached outputs — fall through to re-execution
            import logging
            logging.getLogger(__name__).warning("Cached topology sort failed: %s", sort_err)

    if last_raw is None:
        try:
            last_raw = await loop.run_in_executor(
                executor,
                engine.execute_pipeline_return_last_raw,
                raw_copy,
                body.pipeline,
            )
        except Exception:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while generating the .fif file.",
            )

    if last_raw is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No Raw data output found in the pipeline. "
                "Add a filter (Bandpass, Notch) or Resample node to produce "
                "processable data that can be exported as .fif."
            ),
        )

    # Save to temp file, read bytes, clean up
    fd, tmp_path = tempfile.mkstemp(suffix=".fif")
    os.close(fd)
    try:
        last_raw.save(tmp_path, overwrite=True, verbose=False)
        with open(tmp_path, "rb") as f:
            fif_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return Response(
        content=fif_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=processed.fif"},
    )


@router.post("/export", response_model=ExportResponse, summary="Export pipeline as Python script")
async def export_pipeline(body: ExportRequest, request: Request) -> ExportResponse:
    """
    Generates a standalone Python script that reproduces the pipeline.

    The script includes:
      - All MNE API calls with explicit parameters (no hidden defaults)
      - The session audit log as a comment block at the top
      - Print statements for progress feedback
      - Is validated with ast.parse() before being returned

    MVP scope: linear pipelines only. Branching pipelines return a 422 error.
    """
    # Validate before attempting export
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot export invalid pipeline: {'; '.join(errors)}",
        )

    executor = request.app.state.executor
    loop = asyncio.get_event_loop()

    try:
        script = await loop.run_in_executor(
            executor,
            script_exporter.export,
            body.pipeline,
            body.audit_log,
        )
    except ValueError as e:
        # Branching pipeline or template bug
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during script export.",
        )

    # Sanitize pipeline name for use as a filename
    from backend.path_security import sanitize_filename
    safe_name = sanitize_filename(body.pipeline.metadata.name) or "pipeline"

    return ExportResponse(
        script=script,
        filename=f"{safe_name}.py",
    )


@router.post("/export-package", summary="Download reproducibility package")
async def export_package(body: ExportRequest, request: Request) -> StreamingResponse:
    """
    Generates a zip file containing everything needed to reproduce a pipeline.

    The package includes:
      - pipeline.py — standalone Python script
      - pipeline.json — pipeline graph (re-importable into Oscilloom)
      - requirements.txt — pinned Python dependencies
      - README.md — reproduction instructions
    """
    # Validate before attempting export
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot export invalid pipeline: {'; '.join(errors)}",
        )

    executor = request.app.state.executor
    loop = asyncio.get_event_loop()

    try:
        script = await loop.run_in_executor(
            executor,
            script_exporter.export,
            body.pipeline,
            body.audit_log,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during script export.",
        )

    # Build package contents
    pipeline_json = body.pipeline.model_dump_json(indent=2)

    mne_version = mne.__version__
    matplotlib_version = matplotlib.__version__
    numpy_version = numpy.__version__

    requirements = (
        f"mne=={mne_version}\n"
        f"matplotlib=={matplotlib_version}\n"
        f"numpy=={numpy_version}\n"
    )

    pipeline_name = body.pipeline.metadata.name
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    readme = (
        f"# {pipeline_name} — Oscilloom Export\n"
        f"\n"
        f"## Quick Start\n"
        f"pip install -r requirements.txt\n"
        f"python pipeline.py\n"
        f"\n"
        f"## Contents\n"
        f"- `pipeline.py` — Standalone Python script reproducing this pipeline\n"
        f"- `pipeline.json` — Pipeline graph (re-importable into Oscilloom)\n"
        f"- `requirements.txt` — Pinned Python dependencies\n"
        f"\n"
        f"## Generated\n"
        f"- Date: {timestamp}\n"
        f"- Oscilloom version: 0.1.0-beta\n"
        f"- MNE-Python version: {mne_version}\n"
    )

    # Create zip in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pipeline.py", script)
        zf.writestr("pipeline.json", pipeline_json)
        zf.writestr("requirements.txt", requirements)
        zf.writestr("README.md", readme)
    buf.seek(0)

    from backend.path_security import sanitize_filename
    safe_name = sanitize_filename(pipeline_name) or "pipeline"
    filename = safe_name.replace(" ", "_").lower() + "_package.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/export-bids", summary="Download processed EEG in BIDS format")
async def export_bids(body: BidsExportRequest, request: Request) -> StreamingResponse:
    """
    Executes the pipeline and exports the last Raw output as a BIDS-formatted
    zip archive using mne-bids.

    The zip contains a full BIDS directory structure (sub-XX/, dataset_description.json,
    etc.) that can be used directly with BIDS-compatible tools.

    Requires mne-bids to be installed. Returns 422 if the pipeline has no
    Raw-producing nodes, 500 if mne-bids is not available.
    """
    # Validate the pipeline
    errors = validation.validate_pipeline(body.pipeline)
    if errors:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pipeline: {'; '.join(errors)}",
        )

    # Get session Raw copy
    try:
        raw_copy = session_store.get_raw_copy(body.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Execute pipeline to get last Raw
    executor = request.app.state.executor
    loop = asyncio.get_event_loop()

    # Try cached outputs first
    cached = session_store.get_all_cached_outputs(body.session_id)
    last_raw = None

    if cached:
        try:
            order = engine.topological_sort(body.pipeline)
            for nid in order:
                output = cached.get(nid)
                if isinstance(output, mne.io.BaseRaw):
                    last_raw = output
        except ValueError:
            pass

    if last_raw is None:
        try:
            last_raw = await loop.run_in_executor(
                executor,
                engine.execute_pipeline_return_last_raw,
                raw_copy,
                body.pipeline,
            )
        except Exception:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while executing the pipeline.",
            )

    if last_raw is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No Raw data output found in the pipeline. "
                "Add a filter (Bandpass, Notch) or Resample node to produce "
                "processable data that can be exported in BIDS format."
            ),
        )

    # Lazy import mne_bids to avoid hard dependency
    try:
        import mne_bids
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail=(
                "mne-bids is not installed. Install it with: "
                "pip install mne-bids"
            ),
        )

    # Write BIDS structure to a temp directory and zip it
    tmp_dir = tempfile.mkdtemp(prefix="oscilloom_bids_")
    try:
        bids_path = mne_bids.BIDSPath(
            subject=body.subject_id,
            session=body.session if body.session else None,
            task=body.task,
            run=body.run,
            root=tmp_dir,
        )

        await loop.run_in_executor(
            executor,
            lambda: mne_bids.write_raw_bids(
                last_raw,
                bids_path,
                format=body.format,
                overwrite=True,
                allow_preload=True,
                verbose=False,
            ),
        )

        # Zip the BIDS directory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirnames, filenames in os.walk(tmp_dir):
                for fname in filenames:
                    abs_path = os.path.join(dirpath, fname)
                    arcname = os.path.relpath(abs_path, tmp_dir)
                    zf.write(abs_path, arcname)
        buf.seek(0)

        zip_filename = f"bids_sub-{body.subject_id}_task-{body.task}.zip"

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        )

    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the BIDS export.",
        )
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/generate-methods", summary="Generate Methods section")
def generate_methods(request: ExecuteRequest) -> dict:
    """
    Generates academic Methods-section prose from a pipeline graph.

    Walks the pipeline in topological order, collects methods_template
    sentences from each node descriptor, and wraps them in a standard
    opening and closing sentence with citations.

    Does not execute the pipeline — only inspects node types and parameters.
    The session_id field in the request body is ignored.
    """
    pipeline = request.pipeline

    # Get execution order
    try:
        execution_order = topological_sort(pipeline)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    node_by_id = {n.id: n for n in pipeline.nodes}

    # Collect methods sentences from each node in order
    sentences: list[str] = []
    citations: set[str] = {"Gramfort et al., 2013"}

    for node_id in execution_order:
        node = node_by_id[node_id]
        descriptor = NODE_REGISTRY.get(node.node_type)
        if descriptor is None:
            continue

        # Merge schema defaults with pipeline-provided values
        merged_params: dict[str, Any] = {
            p.name: p.default for p in descriptor.parameters
        }
        merged_params.update(node.parameters)

        # Generate methods sentence if the descriptor provides a template
        if descriptor.methods_template is not None:
            try:
                sentence = descriptor.methods_template(merged_params)
                if sentence:
                    sentences.append(sentence)
            except Exception:
                # Skip nodes whose methods_template fails with given params
                pass

    mne_version = mne.__version__
    opening = (
        f"EEG data were processed using Oscilloom (v0.1.0-beta), a visual pipeline "
        f"builder based on MNE-Python {mne_version} (Gramfort et al., 2013)."
    )
    closing = (
        "All processing parameters are reported explicitly above; the complete "
        "pipeline is available as a reproducible Python script."
    )

    parts = [opening] + sentences + [closing]
    full_text = " ".join(parts)
    word_count = len(full_text.split())

    return {
        "methods_section": full_text,
        "word_count": word_count,
        "citations": sorted(citations),
    }
