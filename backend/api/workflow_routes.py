"""
backend/api/workflow_routes.py

CRUD routes for saved workflows (pipeline graphs).

Workflows are persisted as JSON files on the filesystem, replacing
browser-side IndexedDB for cross-session persistence.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import workflow_store

router = APIRouter(prefix="/workflows", tags=["Workflows"])


class WorkflowBody(BaseModel):
    """Request body for creating/updating a workflow."""
    id: str | None = None
    name: str = "Untitled Workflow"
    createdAt: str | None = None
    updatedAt: str | None = None
    nodeCount: int = 0
    edgeCount: int = 0
    pipeline: dict = {}


class RenameBody(BaseModel):
    name: str


# --- Routes that must come BEFORE /{id} to avoid capture ---


@router.get("/stats", summary="Workflow storage statistics")
def get_stats() -> dict:
    """Return count, disk usage, and storage directory path."""
    return workflow_store.get_stats()


@router.delete("/clear-all", summary="Delete all workflows")
def clear_all() -> dict:
    """Delete every saved workflow from disk."""
    count = workflow_store.clear_all()
    return {"status": "cleared", "deleted_count": count}


# --- Standard CRUD ---


@router.get("", summary="List all workflows")
def list_workflows() -> dict:
    """Return all saved workflows, sorted by updatedAt descending."""
    workflows = workflow_store.list_workflows()
    return {"workflows": workflows, "count": len(workflows)}


@router.post("", summary="Save a workflow")
def create_workflow(body: WorkflowBody) -> dict:
    """Create or upsert a workflow."""
    workflow = body.model_dump()
    saved = workflow_store.save_workflow(workflow)
    return {"status": "saved", "workflow": saved}


@router.get("/{workflow_id}", summary="Get a workflow by ID")
def get_workflow(workflow_id: str) -> dict:
    """Return a single workflow by ID."""
    try:
        workflow_store._validate_id(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    workflow = workflow_store.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    return workflow


@router.put("/{workflow_id}", summary="Update a workflow")
def update_workflow(workflow_id: str, body: WorkflowBody) -> dict:
    """Update an existing workflow."""
    try:
        workflow_store._validate_id(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    existing = workflow_store.get_workflow(workflow_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")

    workflow = body.model_dump()
    workflow["id"] = workflow_id
    # Preserve original createdAt
    workflow["createdAt"] = existing.get("createdAt", workflow.get("createdAt"))
    saved = workflow_store.save_workflow(workflow)
    return {"status": "updated", "workflow": saved}


@router.delete("/{workflow_id}", summary="Delete a workflow")
def delete_workflow(workflow_id: str) -> dict:
    """Delete a single workflow by ID."""
    try:
        workflow_store._validate_id(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    deleted = workflow_store.delete_workflow(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    return {"status": "deleted", "workflow_id": workflow_id}


@router.post("/{workflow_id}/duplicate", summary="Duplicate a workflow")
def duplicate_workflow(workflow_id: str) -> dict:
    """Create a copy of an existing workflow with a new ID."""
    try:
        workflow_store._validate_id(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        duplicated = workflow_store.duplicate_workflow(workflow_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "duplicated", "workflow": duplicated}
