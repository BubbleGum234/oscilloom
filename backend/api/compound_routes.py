"""
backend/api/compound_routes.py

REST endpoints for compound node management (publish, list, get, delete).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.compound_registry import (
    delete_compound,
    get_compound,
    list_compounds,
    publish_compound,
)

router = APIRouter(prefix="/compound", tags=["Compound Nodes"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class PublishRequest(BaseModel):
    compound_id: str
    display_name: str = ""
    description: str = ""
    tags: list[str] = []
    sub_graph: dict[str, Any]
    entry_node_id: str = ""
    output_node_id: str
    exposed_params: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/publish", summary="Publish a compound node")
def publish(request: PublishRequest) -> dict:
    """
    Validate, register, and persist a new compound node.

    The compound node becomes immediately available in the node registry
    and will be restored on server restart.
    """
    definition = request.model_dump()
    try:
        descriptor = publish_compound(definition)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "status": "published",
        "compound_id": descriptor.node_type,
        "display_name": descriptor.display_name,
    }


@router.get("/list", summary="List all compound nodes")
def list_all() -> dict:
    """Return a summary list of all published compound nodes."""
    return {"compounds": list_compounds()}


@router.get("/{compound_id}", summary="Get compound definition")
def get_by_id(compound_id: str) -> dict:
    """Return the full definition dict for a compound node."""
    defn = get_compound(compound_id)
    if defn is None:
        raise HTTPException(
            status_code=404,
            detail=f"Compound '{compound_id}' not found.",
        )
    return defn


@router.delete("/{compound_id}", summary="Delete a compound node")
def delete_by_id(compound_id: str) -> dict:
    """Remove a compound node from the registry and disk."""
    try:
        delete_compound(compound_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"status": "deleted", "compound_id": compound_id}
