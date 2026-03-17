"""
backend/api/custom_node_routes.py

CRUD routes for saved custom Python node presets.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend import custom_node_store
from backend.rate_limit import limiter

router = APIRouter(prefix="/custom-nodes", tags=["Custom Nodes"])


class SaveCustomNodeRequest(BaseModel):
    display_name: str
    description: str = ""
    code: str
    timeout_s: int = 60


@router.post("", summary="Save a custom node preset")
def save_custom_node(body: SaveCustomNodeRequest) -> dict:
    """Save a custom Python node preset to disk and register it."""
    try:
        definition = custom_node_store.save_custom_node(
            display_name=body.display_name,
            description=body.description,
            code=body.code,
            timeout_s=body.timeout_s,
        )
        return {"status": "saved", "node": definition}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("", summary="List saved custom nodes")
def list_custom_nodes() -> dict:
    """List all saved custom node presets."""
    nodes = custom_node_store.list_custom_nodes()
    return {"custom_nodes": nodes, "count": len(nodes)}


@router.get("/{slug}", summary="Get a custom node by slug")
def get_custom_node(slug: str) -> dict:
    """Get a single custom node definition."""
    node = custom_node_store.get_custom_node(slug)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Custom node '{slug}' not found.")
    return node


@router.delete("/{slug}", summary="Delete a custom node")
@limiter.limit("10/hour")
def delete_custom_node(request: Request, slug: str) -> dict:
    """Delete a saved custom node preset from disk."""
    deleted = custom_node_store.delete_custom_node(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Custom node '{slug}' not found.")
    return {"status": "deleted", "slug": slug}


@router.get("/{slug}/export", summary="Export custom node as .nfnode")
def export_custom_node(slug: str) -> dict:
    """Export a custom node definition as a downloadable JSON (.nfnode)."""
    node = custom_node_store.get_custom_node(slug)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Custom node '{slug}' not found.")
    # Return the full definition — frontend downloads as .nfnode
    return node


@router.post("/import", summary="Import a .nfnode file")
def import_custom_node(body: SaveCustomNodeRequest) -> dict:
    """Import a custom node from a .nfnode JSON definition."""
    try:
        definition = custom_node_store.save_custom_node(
            display_name=body.display_name,
            description=body.description,
            code=body.code,
            timeout_s=body.timeout_s,
        )
        return {"status": "imported", "node": definition}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
