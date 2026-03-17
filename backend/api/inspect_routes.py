"""
backend/api/inspect_routes.py

Routes for node inspection, interactive MNE browser, and per-node re-execution.

These routes depend on the node output cache in session_store — the pipeline
must be executed at least once before any inspect operation can be performed.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import tempfile
import threading
import traceback
from typing import Any

import mne
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend import engine, session_store
from backend.models import ExecuteResponse, PipelineGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline/inspect", tags=["Inspect"])

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BrowserRequest(BaseModel):
    session_id: str
    target_node_id: str
    node_label: str | None = None


class SyncAnnotationsRequest(BaseModel):
    session_id: str
    target_node_id: str


class ExecuteFromRequest(BaseModel):
    session_id: str
    pipeline: PipelineGraph
    from_node_id: str


# ---------------------------------------------------------------------------
# MNE Browser — Feature 2 + Interactive Annotation (P3-11)
# ---------------------------------------------------------------------------

_active_browsers: dict[str, multiprocessing.Process] = {}
_browser_lock = threading.Lock()

# Maps browser key → temp .fif file path used for annotation exchange
_browser_temp_files: dict[str, str] = {}
_temp_files_lock = threading.Lock()


def _run_mne_browser(temp_fif_path: str, title: str) -> None:
    """Run MNE browser in a separate process (GUI needs its own event loop).

    The Raw object is loaded from a temp .fif file. After the user closes the
    browser window, the Raw (including any annotations the user added
    interactively) is saved back to the same temp file so the main process
    can read the annotations back.
    """
    import sys
    import matplotlib
    if sys.platform == "darwin":
        matplotlib.use("MacOSX")
    else:
        matplotlib.use("TkAgg")

    raw = mne.io.read_raw_fif(temp_fif_path, preload=True, verbose=False)
    fig = raw.plot(
        block=True,
        title=f"{title}  —  Press 'a' to annotate, then click+drag",
    )
    # After the browser window closes, save annotations back to the temp file
    raw.save(temp_fif_path, overwrite=True, verbose=False)


@router.post("/browser", summary="Open MNE interactive raw data browser")
async def open_mne_browser(request: BrowserRequest) -> dict:
    """
    Opens MNE's interactive raw data browser for a specific node's output.

    Requires the pipeline to have been executed at least once (uses cached
    node outputs). The browser opens in a separate process to avoid blocking
    the FastAPI event loop.

    The Raw data is saved to a temp .fif file before opening. After the user
    closes the browser, annotations can be synced back via the
    /browser/sync-annotations endpoint.
    """
    key = f"{request.session_id}:{request.target_node_id}"

    with _browser_lock:
        proc = _active_browsers.get(key)
        if proc and proc.is_alive():
            raise HTTPException(400, "Browser already open for this node.")

    try:
        output = session_store.get_cached_output(
            request.session_id, request.target_node_id
        )
    except KeyError as e:
        raise HTTPException(404, str(e))

    if not isinstance(output, mne.io.BaseRaw):
        raise HTTPException(
            422,
            f"Node output is {type(output).__name__}, not Raw. "
            "The MNE browser only works with Raw/filtered EEG data."
        )

    # Save Raw to a temp .fif file for IPC with the browser subprocess
    temp_fd, temp_path = tempfile.mkstemp(suffix="_raw.fif")
    os.close(temp_fd)
    try:
        output.save(temp_path, overwrite=True, verbose=False)
    except Exception:
        os.unlink(temp_path)
        raise

    title = f"Oscilloom — {request.node_label or request.target_node_id}"

    proc = multiprocessing.Process(
        target=_run_mne_browser,
        args=(temp_path, title),
        daemon=True,
    )
    proc.start()

    with _browser_lock:
        _active_browsers[key] = proc
    with _temp_files_lock:
        _browser_temp_files[key] = temp_path

    return {"status": "opened", "node_id": request.target_node_id}


@router.get("/browser/status", summary="Check if MNE browser is still open")
async def browser_status(session_id: str, node_id: str) -> dict:
    """Check whether an MNE browser window is still open for a node."""
    key = f"{session_id}:{node_id}"
    with _browser_lock:
        proc = _active_browsers.get(key)
        if proc and proc.is_alive():
            return {"open": True}
        _active_browsers.pop(key, None)
        return {"open": False}


@router.post("/browser/sync-annotations",
             summary="Sync annotations from MNE browser back into session")
async def sync_annotations(request: SyncAnnotationsRequest) -> dict:
    """
    After the MNE browser window closes, read the temp .fif file that was
    modified by the browser subprocess. Extract any annotations the user
    added interactively and update the session store's cached Raw object.

    Returns the list of annotations that were synced.
    """
    key = f"{request.session_id}:{request.target_node_id}"

    # Ensure the browser is no longer running
    with _browser_lock:
        proc = _active_browsers.get(key)
        if proc and proc.is_alive():
            raise HTTPException(
                400,
                "Browser is still open. Close the MNE browser window first."
            )
        _active_browsers.pop(key, None)

    # Get the temp file path
    with _temp_files_lock:
        temp_path = _browser_temp_files.get(key)

    if not temp_path or not os.path.exists(temp_path):
        raise HTTPException(
            404,
            "Annotations were already synced, or no browser session exists. "
            "Open the MNE browser again to add more annotations."
        )

    try:
        # Read the modified Raw from the temp file
        modified_raw = mne.io.read_raw_fif(temp_path, preload=True, verbose=False)
        modified_raw._filenames = [None]
    except Exception as e:
        logger.error("Failed to read temp .fif file '%s': %s", temp_path, e)
        raise HTTPException(500, "Failed to read annotations from browser session.")

    # Extract annotations
    annotations = modified_raw.annotations
    annotation_list = []
    for idx in range(len(annotations)):
        annotation_list.append({
            "onset": float(annotations.onset[idx]),
            "duration": float(annotations.duration[idx]),
            "description": str(annotations.description[idx]),
        })

    # 1. Persist annotations to the SESSION STORE so they survive full re-runs.
    #    This is an intentional user edit, not a pipeline mutation.
    try:
        session_store.update_session_annotations(request.session_id, annotations)
    except KeyError as e:
        raise HTTPException(404, str(e))

    # 2. Also update the node cache so "re-run from here" picks them up.
    try:
        cached_output = session_store.get_cached_output(
            request.session_id, request.target_node_id
        )
        if isinstance(cached_output, mne.io.BaseRaw):
            cached_output.set_annotations(annotations)
    except KeyError:
        pass  # No cache yet — fine, next full run will include annotations

    logger.info(
        "Synced %d annotations for session=%s node=%s",
        len(annotations), request.session_id[:8], request.target_node_id
    )

    # Clean up the temp file
    try:
        os.unlink(temp_path)
    except OSError:
        pass
    with _temp_files_lock:
        _browser_temp_files.pop(key, None)

    # 3. Return updated session info so frontend can refresh annotation chips.
    try:
        updated_info = session_store.get_info(request.session_id)
    except KeyError:
        updated_info = None

    return {
        "status": "synced",
        "node_id": request.target_node_id,
        "n_annotations": len(annotation_list),
        "annotations": annotation_list,
        "session_info": updated_info,
    }


# ---------------------------------------------------------------------------
# Per-Node Re-Run — Feature 4
# ---------------------------------------------------------------------------

@router.post("/execute_from", response_model=ExecuteResponse,
             summary="Re-execute pipeline from a specific node")
async def execute_from_node(request: ExecuteFromRequest, req: Request) -> ExecuteResponse:
    """
    Partial re-execution: reuses cached outputs for nodes upstream of
    from_node_id, re-executes from_node_id and all its descendants.

    Requires at least one prior full pipeline execution (for cache).
    """
    cached = session_store.get_all_cached_outputs(request.session_id)
    if not cached:
        raise HTTPException(
            400,
            "No cached outputs. Run the full pipeline first."
        )

    try:
        raw_copy = session_store.get_raw_copy(request.session_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    executor = req.app.state.executor
    loop = asyncio.get_event_loop()

    try:
        results, updated_outputs = await loop.run_in_executor(
            executor,
            engine.execute_from_node,
            raw_copy,
            request.pipeline,
            request.from_node_id,
            cached,
        )
        # Update cache with new outputs
        session_store.cache_node_outputs(request.session_id, updated_outputs)
        return ExecuteResponse(status="success", node_results=results)

    except ValueError as e:
        return ExecuteResponse(status="error", node_results={}, error=str(e))

    except Exception:
        traceback.print_exc()
        return ExecuteResponse(
            status="error", node_results={},
            error="An internal error occurred during re-execution.",
        )
