"""
backend/compound_registry.py

Compound Node Registry — publish, persist, and load user-defined compound nodes.

A compound node wraps an entire sub-graph (pipeline fragment) into a single
reusable NodeDescriptor. Users build a chain of nodes on the canvas, then
"Publish as Node" to collapse it into one drag-and-drop block.

IMPORT ORDER:
    compound_registry -> engine -> registry  (OK)
    registry -> compound_registry            (NEVER — circular import)

This module is imported by main.py at startup. It calls load_compounds_on_startup()
to re-register any previously published compounds from disk.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from backend.path_security import sanitize_id
from backend.models import PipelineGraph
from backend.registry import NODE_REGISTRY
from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
    VALID_HANDLE_TYPES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_COMPOUND_DEPTH = 10

_COMPOUNDS_DIR = Path(__file__).resolve().parent.parent / "compounds"

# In-memory cache: compound_id -> definition dict (mirrors JSON on disk)
_compound_definitions: dict[str, dict[str, Any]] = {}

# Thread safety for NODE_REGISTRY mutations
_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Closure factory
# ---------------------------------------------------------------------------

def _make_compound_execute_fn(
    sub_graph_dict: dict[str, Any],
    output_node_id: str,
    param_routing: list[dict[str, str]],
    depth: int = 0,
) -> Any:
    """
    Creates a closure that executes a sub-graph when called as a node's execute_fn.

    The closure:
    1. Reconstructs a PipelineGraph from the serialized dict.
    2. Routes exposed parameters from the compound's merged_params back
       to the inner nodes (using ``{inner_node_id}__{param_name}`` keys).
    3. Calls ``_execute_graph`` on the sub-graph.
    4. Returns the raw output object of the designated output node.

    Nesting is supported: if the sub-graph itself contains compound nodes,
    each level increments ``depth``. Exceeding ``_MAX_COMPOUND_DEPTH`` raises
    ``RecursionError`` to prevent infinite loops.
    """

    def execute_fn(input_data: Any, params: dict[str, Any]) -> Any:
        if depth >= _MAX_COMPOUND_DEPTH:
            raise RecursionError(
                f"Compound node nesting depth exceeds maximum ({_MAX_COMPOUND_DEPTH}). "
                "Check for circular compound node definitions."
            )

        # Lazy import to avoid circular dependency at module load time
        from backend.engine import _execute_graph

        graph = PipelineGraph(**sub_graph_dict)

        # Route exposed params back to inner nodes
        for routing in param_routing:
            key = f"{routing['inner_node_id']}__{routing['param_name']}"
            if key in params:
                for node in graph.nodes:
                    if node.id == routing["inner_node_id"]:
                        node.parameters[routing["param_name"]] = params[key]
                        break

        _results, node_outputs = _execute_graph(input_data, graph)

        if output_node_id not in node_outputs:
            raise ValueError(
                f"Compound output node '{output_node_id}' did not produce output. "
                f"Available: {list(node_outputs.keys())}"
            )

        return node_outputs[output_node_id]

    return execute_fn


# ---------------------------------------------------------------------------
# Descriptor builder
# ---------------------------------------------------------------------------

def _infer_entry_node(graph_dict: dict[str, Any]) -> str:
    """Find the single node with zero incoming edges in the sub-graph."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])

    target_ids = {e["target_node_id"] for e in edges}
    node_ids = [n["id"] for n in nodes]
    entry_nodes = [nid for nid in node_ids if nid not in target_ids]

    if len(entry_nodes) == 0:
        raise ValueError("Sub-graph has no entry node (all nodes have incoming edges).")
    if len(entry_nodes) > 1:
        raise ValueError(
            f"Sub-graph has multiple entry nodes: {entry_nodes}. "
            "A compound node must have exactly one entry point."
        )
    return entry_nodes[0]


def _build_descriptor(definition: dict[str, Any]) -> NodeDescriptor:
    """
    Build a NodeDescriptor from a compound definition dict.

    Infers input/output handle types from the entry and output nodes
    in the sub-graph.
    """
    compound_id = definition["compound_id"]
    sub_graph = definition["sub_graph"]
    output_node_id = definition["output_node_id"]
    exposed_params = definition.get("exposed_params", [])

    # Find entry node
    entry_node_id = definition.get("entry_node_id")
    if not entry_node_id:
        entry_node_id = _infer_entry_node(sub_graph)

    # Look up inner node types to infer handle types
    inner_nodes = {n["id"]: n for n in sub_graph.get("nodes", [])}

    entry_node = inner_nodes.get(entry_node_id)
    if entry_node is None:
        raise ValueError(f"Entry node '{entry_node_id}' not found in sub-graph.")

    output_node = inner_nodes.get(output_node_id)
    if output_node is None:
        raise ValueError(f"Output node '{output_node_id}' not found in sub-graph.")

    # Infer input handle type from entry node's descriptor
    entry_descriptor = NODE_REGISTRY.get(entry_node["node_type"])
    if entry_descriptor is None:
        raise ValueError(
            f"Entry node type '{entry_node['node_type']}' not in NODE_REGISTRY."
        )

    # Infer output handle type from output node's descriptor
    output_descriptor = NODE_REGISTRY.get(output_node["node_type"])
    if output_descriptor is None:
        raise ValueError(
            f"Output node type '{output_node['node_type']}' not in NODE_REGISTRY."
        )

    # Input handles: use entry node's input handles (empty if it's a loader)
    input_handles = []
    if entry_descriptor.inputs:
        first_input = entry_descriptor.inputs[0]
        input_handles = [
            HandleSchema(
                id="compound_in",
                type=first_input.type,
                label=f"Input ({first_input.type})",
            )
        ]

    # Output handles: use output node's output handles
    output_handles = []
    if output_descriptor.outputs:
        first_output = output_descriptor.outputs[0]
        output_handles = [
            HandleSchema(
                id="compound_out",
                type=first_output.type,
                label=f"Output ({first_output.type})",
            )
        ]

    # Build exposed parameters
    parameters: list[ParameterSchema] = []
    param_routing: list[dict[str, str]] = []
    for ep in exposed_params:
        inner_node_id = ep["inner_node_id"]
        param_name = ep["param_name"]
        display_label = ep.get("display_label", param_name)

        inner_node = inner_nodes.get(inner_node_id)
        if inner_node is None:
            raise ValueError(
                f"Exposed param references node '{inner_node_id}' not in sub-graph."
            )

        inner_descriptor = NODE_REGISTRY.get(inner_node["node_type"])
        if inner_descriptor is None:
            raise ValueError(
                f"Node type '{inner_node['node_type']}' not in NODE_REGISTRY."
            )

        # Find the original ParameterSchema
        orig_param = None
        for p in inner_descriptor.parameters:
            if p.name == param_name:
                orig_param = p
                break
        if orig_param is None:
            raise ValueError(
                f"Parameter '{param_name}' not found on node type "
                f"'{inner_node['node_type']}'."
            )

        # Create compound-level param with namespaced key
        compound_param_name = f"{inner_node_id}__{param_name}"
        parameters.append(
            ParameterSchema(
                name=compound_param_name,
                label=display_label,
                type=orig_param.type,
                default=orig_param.default,
                description=orig_param.description,
                min=orig_param.min,
                max=orig_param.max,
                step=orig_param.step,
                options=orig_param.options,
                unit=orig_param.unit,
                exposed=True,
            )
        )
        param_routing.append({
            "inner_node_id": inner_node_id,
            "param_name": param_name,
        })

    execute_fn = _make_compound_execute_fn(
        sub_graph_dict=sub_graph,
        output_node_id=output_node_id,
        param_routing=param_routing,
    )

    return NodeDescriptor(
        node_type=compound_id,
        display_name=definition.get("display_name", compound_id),
        category="Compound",
        description=definition.get("description", "User-defined compound node."),
        inputs=input_handles,
        outputs=output_handles,
        parameters=parameters,
        execute_fn=execute_fn,
        tags=definition.get("tags", ["compound"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def publish_compound(definition: dict[str, Any]) -> NodeDescriptor:
    """
    Validate, build, register, and persist a compound node definition.

    Args:
        definition: Dict matching the compound JSON schema (see plan).

    Returns:
        The newly created NodeDescriptor.

    Raises:
        ValueError: validation fails (empty ID, builtin collision, bad sub-graph, etc.)
    """
    compound_id = definition.get("compound_id", "").strip()
    if not compound_id:
        raise ValueError("compound_id must not be empty.")

    # Sanitize compound_id for safe filesystem use
    compound_id = sanitize_id(compound_id)

    # Auto-prefix with c_ if missing
    if not compound_id.startswith("c_"):
        compound_id = f"c_{compound_id}"
    definition["compound_id"] = compound_id

    # Check for collision with builtin node types
    with _registry_lock:
        existing = NODE_REGISTRY.get(compound_id)
        if existing is not None and compound_id not in _compound_definitions:
            raise ValueError(
                f"compound_id '{compound_id}' collides with a builtin node type."
            )

    # Validate sub-graph exists and has nodes
    sub_graph = definition.get("sub_graph")
    if not sub_graph or not sub_graph.get("nodes"):
        raise ValueError("sub_graph must contain at least one node.")

    if not definition.get("output_node_id"):
        raise ValueError("output_node_id is required.")

    # Build the descriptor (validates entry/output nodes, exposed params, etc.)
    descriptor = _build_descriptor(definition)

    # Register
    with _registry_lock:
        NODE_REGISTRY[compound_id] = descriptor
        _compound_definitions[compound_id] = definition

    # Persist to disk
    _COMPOUNDS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = _COMPOUNDS_DIR / f"{compound_id}.json"
    filepath.write_text(json.dumps(definition, indent=2), encoding="utf-8")

    return descriptor


def delete_compound(compound_id: str) -> None:
    """
    Remove a compound node from the registry and disk.

    Raises:
        ValueError: if compound_id is not a compound (i.e., it's a builtin).
        KeyError: if compound_id is not found.
    """
    compound_id = sanitize_id(compound_id)
    with _registry_lock:
        if compound_id not in _compound_definitions:
            if compound_id in NODE_REGISTRY:
                raise ValueError(
                    f"'{compound_id}' is a builtin node type and cannot be deleted."
                )
            raise KeyError(f"Compound '{compound_id}' not found.")

        del NODE_REGISTRY[compound_id]
        del _compound_definitions[compound_id]

    filepath = (_COMPOUNDS_DIR / f"{compound_id}.json").resolve()
    if not str(filepath).startswith(str(_COMPOUNDS_DIR.resolve())):
        raise ValueError("Invalid compound_id.")
    if filepath.exists():
        filepath.unlink()


def get_compound(compound_id: str) -> dict[str, Any] | None:
    """Return the full definition dict for a compound, or None if not found."""
    return _compound_definitions.get(compound_id)


def list_compounds() -> list[dict[str, Any]]:
    """Return summary list of all published compounds."""
    result = []
    for cid, defn in _compound_definitions.items():
        descriptor = NODE_REGISTRY.get(cid)
        result.append({
            "compound_id": cid,
            "display_name": defn.get("display_name", cid),
            "description": defn.get("description", ""),
            "category": "Compound",
            "input_type": descriptor.inputs[0].type if descriptor and descriptor.inputs else None,
            "output_type": descriptor.outputs[0].type if descriptor and descriptor.outputs else None,
        })
    return result


def load_compounds_on_startup() -> int:
    """
    Load all compound JSON files from ``_COMPOUNDS_DIR`` and register them.

    Called once at server startup from ``main.py``.

    Returns:
        Number of compounds loaded.
    """
    if not _COMPOUNDS_DIR.exists():
        return 0

    count = 0
    for filepath in sorted(_COMPOUNDS_DIR.glob("*.json")):
        try:
            definition = json.loads(filepath.read_text(encoding="utf-8"))
            descriptor = _build_descriptor(definition)
            compound_id = definition["compound_id"]

            with _registry_lock:
                NODE_REGISTRY[compound_id] = descriptor
                _compound_definitions[compound_id] = definition

            count += 1
        except Exception as exc:
            # Log but don't crash the server for a bad compound file
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load compound '%s': %s", filepath.name, exc
            )

    return count
