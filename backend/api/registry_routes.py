"""
backend/api/registry_routes.py

Routes for the node registry API.

The frontend fetches GET /registry/nodes once at startup and caches the result.
The registry response drives:
  - The node palette (sidebar): which node types are draggable
  - The GenericNode component: handle positions and labels
  - The NodeParameterPanel: which controls to render for each node type
  - The AI system prompt: what node types the SLM can reference

SERIALIZATION NOTE:
  NodeDescriptor contains an `execute_fn` field (a Python callable) which
  cannot be JSON-serialized. _descriptor_to_dict() explicitly removes it
  before returning the response. This is intentional by design.
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException, Request

from backend.registry import NODE_REGISTRY
from backend.pipeline_templates import PIPELINE_TEMPLATES

router = APIRouter(prefix="/registry", tags=["Node Registry"])


def _descriptor_to_dict(descriptor) -> dict:
    """
    Converts a NodeDescriptor dataclass to a JSON-serializable dict.

    Uses dataclasses.asdict() for deep conversion (handles nested dataclasses
    like ParameterSchema and HandleSchema automatically), then removes
    `execute_fn` which is a Python callable and cannot be serialized.
    """
    d = dataclasses.asdict(descriptor)
    d.pop("execute_fn", None)       # Callable — not serializable, not needed by frontend
    d.pop("code_template", None)    # Callable — used by script exporter, not frontend
    d.pop("methods_template", None) # Callable — used by methods generator, not frontend
    return d


@router.get("/nodes", summary="Get all node types")
def get_all_node_types() -> dict:
    """
    Returns every registered node type and its full schema.

    The response includes: node_type, display_name, category, description,
    inputs (handle schemas), outputs (handle schemas), parameters, and tags.
    The execute_fn field is intentionally excluded.

    This endpoint is called once at frontend startup and the result is cached
    client-side. It does not change at runtime (requires server restart to
    register new node types).
    """
    return {
        "nodes": {
            node_type: _descriptor_to_dict(descriptor)
            for node_type, descriptor in NODE_REGISTRY.items()
        },
        "count": len(NODE_REGISTRY),
    }


@router.get("/nodes/{node_type}", summary="Get a single node type")
def get_node_type(node_type: str) -> dict:
    """
    Returns the full schema for a single node type.

    Returns 404 if the node type is not registered.
    Useful for the frontend to fetch a single descriptor on demand
    (e.g., when loading a saved pipeline that references an unfamiliar type).
    """
    descriptor = NODE_REGISTRY.get(node_type)
    if descriptor is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Node type '{node_type}' is not registered. "
                f"Known types: {sorted(NODE_REGISTRY.keys())}"
            ),
        )
    return _descriptor_to_dict(descriptor)


@router.get("/nodes/{node_type}/code", summary="Get code preview for a node")
def get_node_code(node_type: str, request: Request) -> dict:
    """
    Returns the generated MNE-Python code for a node with given parameters.

    Query parameters are treated as node parameters. Missing params use
    schema defaults. Used by the frontend's per-node code viewer.
    """
    descriptor = NODE_REGISTRY.get(node_type)
    if descriptor is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node type '{node_type}' is not registered.",
        )

    # Merge defaults with query params
    merged_params = {p.name: p.default for p in descriptor.parameters}
    for key, value in request.query_params.items():
        # Try to convert to appropriate type
        try:
            # Try float first, then int
            if '.' in value:
                merged_params[key] = float(value)
            else:
                merged_params[key] = int(value)
        except (ValueError, TypeError):
            if value.lower() in ('true', 'false'):
                merged_params[key] = value.lower() == 'true'
            else:
                merged_params[key] = value

    code = None
    if descriptor.code_template is not None:
        try:
            code = descriptor.code_template(merged_params)
        except Exception:
            code = None

    methods = None
    if descriptor.methods_template is not None:
        try:
            methods = descriptor.methods_template(merged_params)
        except Exception:
            methods = None

    return {
        "code": code,
        "docs_url": getattr(descriptor, "docs_url", None),
        "methods": methods,
    }


@router.get("/templates", summary="Get all pipeline templates")
def get_pipeline_templates() -> dict:
    """
    Returns every available pipeline template.

    Each template is a pre-built set of nodes and edges that can be stamped
    onto the canvas with one click.  The frontend uses this to populate a
    "Templates" section in the node palette.
    """
    return {
        "templates": PIPELINE_TEMPLATES,
        "count": len(PIPELINE_TEMPLATES),
    }
