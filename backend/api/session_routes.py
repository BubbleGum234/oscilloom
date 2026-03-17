"""
backend/api/session_routes.py

Routes for EEG file session management.

Session lifecycle:
  1. POST /session/load   — upload EDF/FIF file → get session_id
  2. (use session_id in POST /pipeline/execute calls)
  3. DELETE /session/{id} — free memory when done with the file

The uploaded file is written to a temp file on disk, loaded by MNE into RAM,
then the temp file is deleted. The in-memory Raw object lives in session_store
until explicitly deleted or the server restarts.

MNE requires a file path (not a file-like object) for most readers,
hence the temp-file pattern. The temp file exists only during the load call.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend import session_store
from backend.rate_limit import limiter
router = APIRouter(prefix="/session", tags=["Session"])

# File extensions supported by mne.io.read_raw()
SUPPORTED_EXTENSIONS = {".edf", ".fif", ".bdf", ".set", ".vhdr", ".cnt"}


@router.post("/load", summary="Load an EEG file into memory")
@limiter.limit("10/minute")
async def load_session(request: Request, file: UploadFile = File(...)) -> dict:
    """
    Accepts an EEG file upload, loads it into memory, and returns a session_id.

    The session_id is required for all subsequent pipeline execution calls.
    The file is read entirely into RAM (preload=True), so ensure sufficient
    memory for large recordings (typical range: 50–500 MB).

    Supported formats: EDF, EDF+, FIF, BDF, BrainVision (.vhdr), CNT.

    Returns:
        {
            "session_id": "<uuid4>",
            "info": {
                "nchan": <int>,
                "sfreq": <float>,
                "ch_names": [<str>, ...],  # first 20 channels
                "ch_names_truncated": <bool>,
                "duration_s": <float>,
                "highpass": <float>,
                "lowpass": <float>,
                "meas_date": "<str> | null"
            }
        }
    """
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: '{ext}'. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Write to a temp file. MNE readers require a path, not a file object.
    # delete=False so we can pass the path to MNE, then delete manually.
    MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds maximum upload size of 500 MB.",
                )
            tmp.write(content)

        session_id, info = session_store.create_session(tmp_path)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=(
                f"Failed to load '{filename}': {str(e)}. "
                "Ensure the file is a valid, uncorrupted EEG recording."
            ),
        )
    finally:
        # Always delete the temp file, even if loading failed
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {"session_id": session_id, "info": info}


@router.get("/{session_id}/info", summary="Get session metadata")
def get_session_info(session_id: str) -> dict:
    """
    Returns metadata about a loaded session (channel count, sampling rate, etc.)
    without making a full copy of the Raw object.
    """
    try:
        info = session_store.get_info(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"session_id": session_id, "info": info}


@router.get("/stats", summary="Session store statistics")
def get_session_stats() -> dict:
    """
    Returns statistics about the session store: number of active sessions,
    sessions directory path, TTL, and max sessions limit.

    Used by the Settings page to display session management info.
    """
    sessions_dir = str(session_store._SESSIONS_DIR)
    active_ids = session_store.list_sessions()

    # Calculate disk usage of sessions directory
    disk_usage_bytes = 0
    try:
        sessions_path = session_store._SESSIONS_DIR
        if sessions_path.exists():
            for f in sessions_path.iterdir():
                if f.is_file():
                    disk_usage_bytes += f.stat().st_size
    except Exception:
        pass

    return {
        "active_sessions": len(active_ids),
        "sessions_dir": sessions_dir,
        "ttl_seconds": session_store.SESSION_TTL_SECONDS,
        "max_sessions": session_store.MAX_SESSIONS,
        "disk_usage_bytes": disk_usage_bytes,
    }


@router.delete("/clear-all", summary="Delete all active sessions")
@limiter.limit("5/hour")
def clear_all_sessions(request: Request) -> dict:
    """
    Deletes all active sessions, freeing memory and removing persisted files.

    Used by the Settings page for bulk session cleanup.
    """
    session_ids = session_store.list_sessions()
    deleted = 0
    for sid in session_ids:
        if session_store.delete_session(sid):
            deleted += 1
    return {"status": "cleared", "deleted_count": deleted}


@router.delete("/{session_id}", summary="Free session memory")
@limiter.limit("30/minute")
def delete_session(request: Request, session_id: str) -> dict:
    """
    Removes the session from the store and frees its memory.

    Call this when the researcher is done with a file to reclaim RAM.
    The frontend calls this automatically when a new file is loaded
    (one active session per canvas).
    """
    found = session_store.delete_session(session_id)
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. It may have already been deleted.",
        )
    return {"status": "deleted", "session_id": session_id}
