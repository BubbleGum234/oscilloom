"""
backend/api/history_routes.py

CRUD routes for pipeline run history.

Run snapshots are persisted as JSON files on the filesystem, replacing
browser-side IndexedDB for cross-session persistence.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import history_store

router = APIRouter(prefix="/history", tags=["History"])


class RunBody(BaseModel):
    """Request body for saving a run."""
    id: str | None = None
    timestamp: str | None = None
    name: str = "Untitled Run"
    nodeResults: dict = {}
    paramSnapshot: dict = {}
    thumbnails: dict = {}
    nodeCount: int = 0
    errorCount: int = 0


class RenameBody(BaseModel):
    name: str


# --- Routes that must come BEFORE /{id} to avoid capture ---


@router.get("/stats", summary="Run history storage statistics")
def get_stats() -> dict:
    """Return count, disk usage, and storage directory path."""
    return history_store.get_stats()


@router.delete("/clear-all", summary="Delete all run history")
def clear_all() -> dict:
    """Delete every saved run from disk."""
    count = history_store.clear_all()
    return {"status": "cleared", "deleted_count": count}


# --- Standard CRUD ---


@router.get("", summary="List all runs")
def list_runs() -> dict:
    """Return all saved runs, sorted by timestamp descending."""
    runs = history_store.list_runs()
    return {"runs": runs, "count": len(runs)}


@router.post("", summary="Save a run")
def create_run(body: RunBody) -> dict:
    """Save a new run snapshot. Auto-trims oldest runs if limit exceeded."""
    run = body.model_dump()
    saved = history_store.save_run(run)
    return {"status": "saved", "run": saved}


@router.get("/{run_id}", summary="Get a run by ID")
def get_run(run_id: str) -> dict:
    """Return a single run by ID."""
    try:
        history_store._validate_id(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    run = history_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run


@router.put("/{run_id}/rename", summary="Rename a run")
def rename_run(run_id: str, body: RenameBody) -> dict:
    """Rename an existing run."""
    try:
        history_store._validate_id(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    updated = history_store.rename_run(run_id, body.name)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {"status": "renamed", "run": updated}


@router.delete("/{run_id}", summary="Delete a run")
def delete_run(run_id: str) -> dict:
    """Delete a single run by ID."""
    try:
        history_store._validate_id(run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    deleted = history_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {"status": "deleted", "run_id": run_id}
