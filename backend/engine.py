"""
backend/engine.py

Generic DAG (Directed Acyclic Graph) pipeline execution engine.

INVARIANT: This file must never import from backend.registry.nodes.* directly,
and must never contain node-type-specific logic (no if/elif on node_type).
All node behaviour is encoded in NodeDescriptor.execute_fn. The engine's
only job is to order nodes and call their execute_fn in sequence.

If you find yourself writing:
    if node.node_type == "bandpass_filter": ...
here, that logic belongs in the descriptor's execute_fn instead.

EXECUTION MODEL:
  1. Topologically sort the DAG (Kahn's algorithm).
  2. For each node in order:
     a. Look up its NodeDescriptor in NODE_REGISTRY.
     b. Resolve its input (output of the nearest upstream node, or raw_copy
        for source nodes with no incoming edges).
     c. Merge schema parameter defaults with pipeline-provided values.
     d. Call descriptor.execute_fn(resolved_input, merged_params).
     e. Store the output keyed by node_id for downstream nodes.
  3. Return a result dict mapping node_id → result metadata.

This function is synchronous and runs in a ThreadPoolExecutor thread
(not the asyncio event loop). See pipeline_routes.py for the async wrapper.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from typing import Any

import time

import mne
import traceback

import numpy as np

from backend.models import PipelineGraph
from backend.preview import generate_preview
from backend.registry import NODE_REGISTRY
from backend.execution_cache import ExecutionCache


# ---------------------------------------------------------------------------
# Raw identity hashing
# ---------------------------------------------------------------------------

def _hash_raw_identity(raw: mne.io.BaseRaw) -> str:
    """Create a short hash from raw metadata for cache key chaining."""
    payload = json.dumps({
        "nchan": int(raw.info["nchan"]),
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(raw.n_times),
        "ch_names": list(raw.ch_names[:5]),
    })
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def topological_sort(graph: PipelineGraph) -> list[str]:
    """
    Topologically sorts the pipeline DAG using Kahn's algorithm.

    Returns an ordered list of node IDs such that for every edge (A → B),
    A appears before B in the list.

    Raises:
        ValueError: if the graph contains a cycle (invalid pipeline).
                    Oscilloom pipelines must be acyclic by design.
    """
    # Build in-degree count and adjacency list from edges
    in_degree: dict[str, int] = {n.id: 0 for n in graph.nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in graph.edges:
        adjacency[edge.source_node_id].append(edge.target_node_id)
        in_degree[edge.target_node_id] += 1

    # Start with all nodes that have no incoming edges (source nodes)
    queue: deque[str] = deque(
        nid for nid, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for neighbor in adjacency[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(graph.nodes):
        # Not all nodes were reachable — there is a cycle
        cycled = {n.id for n in graph.nodes} - set(order)
        raise ValueError(
            f"Pipeline graph contains a cycle involving nodes: {cycled}. "
            "Oscilloom pipelines must be acyclic (DAGs). "
            "Check for circular connections on the canvas."
        )

    return order


def _to_native(val: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for JSON serialization."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: _to_native(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_native(v) for v in val]
    return val


def _summarize_output(output: Any) -> dict[str, Any]:
    """
    Build a JSON-serializable summary of a node's output for the inspector.

    NOTE: This function inspects the Python object type, but it does NOT
    check node_type — it is driven purely by the runtime output class.
    This preserves the engine invariant of never branching on node_type.
    """
    summary: dict[str, Any] = {"python_type": type(output).__name__}

    if isinstance(output, mne.io.BaseRaw):
        summary.update({
            "kind": "raw",
            "n_channels": output.info["nchan"],
            "sfreq": output.info["sfreq"],
            "duration_s": round(output.n_times / output.info["sfreq"], 3),
            "shape": [output.info["nchan"], output.n_times],
            "ch_names_preview": list(output.ch_names[:12]),
            "bads": list(output.info["bads"]),
            "highpass": output.info["highpass"],
            "lowpass": output.info["lowpass"],
        })

    elif isinstance(output, mne.Epochs):
        events = output.events[:, 2]
        unique, counts = np.unique(events, return_counts=True)
        event_id_inv = (
            {v: k for k, v in output.event_id.items()} if output.event_id else {}
        )
        event_counts = {
            event_id_inv.get(int(uid), str(int(uid))): int(c)
            for uid, c in zip(unique, counts)
        }
        summary.update({
            "kind": "epochs",
            "n_epochs": len(output),
            "n_channels": output.info["nchan"],
            "tmin": float(output.tmin),
            "tmax": float(output.tmax),
            "sfreq": output.info["sfreq"],
            "event_counts": event_counts,
            "shape": list(output.get_data().shape),
        })
        drop_log = getattr(output, 'drop_log', [])
        if drop_log:
            n_dropped = len([d for d in drop_log if len(d) > 0 and d != ['IGNORED']])
        else:
            n_dropped = 0
        summary["n_dropped"] = n_dropped

    elif isinstance(output, mne.Evoked):
        peak_ch_idx = int(np.argmax(np.abs(output.data).max(axis=1)))
        peak_time_idx = int(np.argmax(np.abs(output.data).max(axis=0)))
        summary.update({
            "kind": "evoked",
            "n_channels": output.info["nchan"],
            "tmin": float(output.times[0]),
            "tmax": float(output.times[-1]),
            "nave": output.nave,
            "comment": output.comment or "",
            "peak_channel": output.ch_names[peak_ch_idx],
            "peak_latency_s": float(output.times[peak_time_idx]),
            "peak_amplitude": float(output.data[peak_ch_idx, peak_time_idx]),
        })

    elif isinstance(output, mne.time_frequency.Spectrum):
        freqs = output.freqs
        summary.update({
            "kind": "psd",
            "n_channels": output.info["nchan"],
            "freq_min": float(freqs[0]),
            "freq_max": float(freqs[-1]),
            "n_freqs": len(freqs),
            "method": getattr(output, "method", "unknown"),
            "shape": list(output.get_data().shape),
        })

    elif isinstance(output, mne.time_frequency.AverageTFR):
        summary.update({
            "kind": "tfr",
            "n_channels": output.info["nchan"],
            "freq_min": float(output.freqs[0]),
            "freq_max": float(output.freqs[-1]),
            "tmin": float(output.times[0]),
            "tmax": float(output.times[-1]),
            "shape": list(output.data.shape),
        })

    elif hasattr(output, "get_data") and type(output).__name__ == "SpectralConnectivity":
        data = output.get_data()
        summary.update({
            "kind": "connectivity",
            "method": getattr(output, "method", "unknown"),
            "n_connections": data.shape[0] if data.ndim >= 1 else 0,
            "n_freqs": data.shape[-1] if data.ndim >= 2 else 0,
            "shape": list(data.shape),
        })

    elif isinstance(output, np.ndarray):
        summary.update({
            "kind": "array",
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "min": float(np.nanmin(output)) if output.size > 0 else None,
            "max": float(np.nanmax(output)) if output.size > 0 else None,
            "mean": float(np.nanmean(output)) if output.size > 0 else None,
        })

    elif isinstance(output, dict):
        safe_metrics: dict[str, Any] = {}
        for k, v in output.items():
            if isinstance(v, (int, float, bool, str, type(None))):
                safe_metrics[str(k)] = v
            elif isinstance(v, np.ndarray):
                safe_metrics[str(k)] = {
                    "type": "array",
                    "shape": list(v.shape),
                    "preview": v.flat[:5].tolist(),
                }
            elif isinstance(v, (list, tuple)):
                safe_metrics[str(k)] = list(v)[:20]
            else:
                safe_metrics[str(k)] = str(v)[:200]
        summary.update({"kind": "metrics", "metrics": safe_metrics})

    elif isinstance(output, (int, float)):
        summary.update({"kind": "scalar", "value": float(output)})

    elif isinstance(output, str) and output.startswith("data:image"):
        summary.update({"kind": "plot"})

    else:
        summary.update({"kind": "unknown", "repr": repr(output)[:300]})

    return _to_native(summary)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _execute_graph(
    input_data: Any,
    graph: PipelineGraph,
    cache: ExecutionCache | None = None,
    generate_previews: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Core execution loop shared by all public entry points.

    Topologically sorts the DAG, resolves inputs, merges parameters,
    and calls each node's execute_fn in order.

    Args:
        input_data:
            The initial input for source nodes (nodes with no incoming
            edges). Typically an mne.io.BaseRaw copy, but may be any
            object when invoked recursively by compound nodes.
        graph:
            A validated PipelineGraph (assumes validate_pipeline() passed).

    Returns:
        A 2-tuple ``(results, node_outputs)`` where:

        - **results** maps node_id → result metadata dict::

              {
                  "node_type": str,
                  "status": "success",
                  "output_type": str,
                  "data": str | None,
              }

        - **node_outputs** maps node_id → the raw Python object produced
          by that node's execute_fn. This is needed by compound nodes
          to retrieve the actual MNE object from the sub-graph.

    Raises:
        ValueError: unknown node_type (defensive — should be caught upstream).
    """
    execution_order = topological_sort(graph)
    node_by_id = {n.id: n for n in graph.nodes}

    # node_outputs maps node_id → the runtime object produced by that node.
    node_outputs: dict[str, Any] = {}

    # Build incoming edge index: target_node_id → list of edges arriving at it
    incoming: dict[str, list] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_node_id].append(edge)

    results: dict[str, Any] = {}

    # Cache support: compute raw identity hash and track per-node content hashes
    raw_identity = _hash_raw_identity(input_data) if isinstance(input_data, mne.io.BaseRaw) else "no_raw"
    node_hashes: dict[str, str] = {}

    for node_id in execution_order:
        node = node_by_id[node_id]
        descriptor = NODE_REGISTRY.get(node.node_type)

        if descriptor is None:
            raise ValueError(
                f"Node '{node_id}' references unknown type '{node.node_type}'. "
                f"Known types: {list(NODE_REGISTRY.keys())}"
            )

        # --- Resolve input ---
        # Source nodes (no incoming edges) receive input_data directly.
        # All other nodes receive the output of their upstream node.
        edges_in = incoming.get(node_id, [])
        if edges_in:
            if len(edges_in) == 1:
                resolved_input = node_outputs[edges_in[0].source_node_id]
            else:
                # Multi-input: select the edge targeting the primary (first declared)
                # input handle so that handle selection on the canvas is respected.
                primary_handle_id = descriptor.inputs[0].id if descriptor.inputs else None
                primary_edge = next(
                    (e for e in edges_in if e.target_handle_id == primary_handle_id),
                    edges_in[0],
                )
                resolved_input = node_outputs[primary_edge.source_node_id]
        else:
            resolved_input = input_data

        # --- Merge parameters ---
        merged_params: dict[str, Any] = {
            p.name: p.default for p in descriptor.parameters
        }
        merged_params.update(node.parameters)

        # --- Cache lookup ---
        cache_hit = False
        content_hash: str | None = None
        if cache is not None:
            edges_in_for_hash = incoming.get(node_id, [])
            if edges_in_for_hash:
                upstream_hash = node_hashes.get(edges_in_for_hash[0].source_node_id, raw_identity)
            else:
                upstream_hash = raw_identity
            content_hash = cache.compute_hash(node.node_type, merged_params, upstream_hash)
            cached_output = cache.get(content_hash)
        else:
            cached_output = None

        if cached_output is not None:
            # Cache hit — skip execution
            output = cached_output
            node_outputs[node_id] = output
            cache_hit = True
            execution_time_ms = 0.0
        else:
            # Cache miss — execute normally
            _t0 = time.perf_counter()
            try:
                output = descriptor.execute_fn(resolved_input, merged_params)
            except Exception as exc:
                _t1 = time.perf_counter()
                exc_type = type(exc).__name__
                error_msg = str(exc)
                results[node_id] = {
                    "node_type": node.node_type,
                    "status": "error",
                    "error": error_msg,
                    "output_type": "error",
                    "data": None,
                    "summary": {
                        "python_type": exc_type,
                        "message": error_msg,
                    },
                    "execution_time_ms": round((_t1 - _t0) * 1000, 2),
                    "cache_hit": False,
                }
                node_outputs[node_id] = None
                continue
            _t1 = time.perf_counter()
            node_outputs[node_id] = output
            execution_time_ms = round((_t1 - _t0) * 1000, 2)
            if cache is not None and content_hash is not None:
                cache.put(content_hash, output)

        # Track content hash for downstream cache key chaining
        if content_hash is not None:
            node_hashes[node_id] = content_hash
        else:
            node_hashes[node_id] = raw_identity

        # --- Build result entry ---
        result_entry: dict[str, Any] = {
            "node_type": node.node_type,
            "status": "success",
            "output_type": type(output).__name__,
            "data": None,
            "summary": _summarize_output(output),
            "execution_time_ms": execution_time_ms,
            "cache_hit": cache_hit,
        }

        if isinstance(output, str) and output.startswith("data:image/png;base64,"):
            result_entry["data"] = output
        elif isinstance(output, dict):
            result_entry["metrics"] = _to_native(output)
        elif isinstance(output, np.ndarray):
            result_entry["metrics"] = _to_native(output)

        if generate_previews and result_entry.get("status") == "success" and not result_entry.get("data"):
            preview = generate_preview(output)
            if preview:
                result_entry["preview"] = preview

        results[node_id] = result_entry

    return results, node_outputs


def execute_pipeline(
    raw_copy: mne.io.BaseRaw,
    graph: PipelineGraph,
    cache: ExecutionCache | None = None,
    generate_previews: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Executes a validated pipeline graph against a pre-copied Raw object.

    This function is synchronous. It is called from pipeline_routes.py
    via asyncio.run_in_executor so it does not block the event loop.

    Args:
        raw_copy:
            A copy of the session's Raw object (already copied by
            session_store.get_raw_copy() before this call). This copy
            is passed as the initial input to source nodes.
        graph:
            A validated PipelineGraph. Assumes validate_pipeline() has
            already been called and returned no errors.
        cache:
            Optional ExecutionCache for content-addressed caching.
            When provided, nodes with matching content hashes skip
            re-execution and return cached outputs.
        generate_previews:
            When True (the default), inline signal preview images are
            generated for each successful non-visualization node via
            ``generate_preview()``.

    Returns:
        A 2-tuple (results, node_outputs) where results maps
        node_id → result dict and node_outputs maps node_id → Python object.

    Raises:
        ValueError: if a node references an unknown node_type.
        Any MNE exception: propagates naturally and is caught by the route
                           handler, which returns it as an error response.
    """
    results, node_outputs = _execute_graph(raw_copy, graph, cache, generate_previews)
    return results, node_outputs


# ---------------------------------------------------------------------------
# FIF export helper
# ---------------------------------------------------------------------------

def execute_pipeline_return_last_raw(
    raw_copy: mne.io.BaseRaw,
    graph: PipelineGraph,
) -> "mne.io.BaseRaw | None":
    """
    Executes the pipeline and returns the last mne.io.BaseRaw output object.

    Used by the /pipeline/download_fif endpoint to produce a processable Raw
    that can be saved to disk as a .fif file.

    Returns None if the pipeline contains no node that outputs a Raw
    (e.g., a pipeline ending at compute_psd or plot_psd with no filter nodes).
    """
    _results, node_outputs = _execute_graph(raw_copy, graph)

    last_raw: "mne.io.BaseRaw | None" = None
    execution_order = topological_sort(graph)
    for node_id in execution_order:
        output = node_outputs.get(node_id)
        if isinstance(output, mne.io.BaseRaw):
            last_raw = output

    return last_raw


# ---------------------------------------------------------------------------
# Partial re-execution — Feature 4
# ---------------------------------------------------------------------------

def _get_descendants(graph: PipelineGraph, start_id: str) -> set[str]:
    """Return all node IDs downstream of start_id (exclusive)."""
    edge_map: dict[str, list[str]] = {}
    for edge in graph.edges:
        edge_map.setdefault(edge.source_node_id, []).append(edge.target_node_id)

    visited: set[str] = set()
    queue = list(edge_map.get(start_id, []))
    while queue:
        nid = queue.pop()
        if nid not in visited:
            visited.add(nid)
            queue.extend(edge_map.get(nid, []))
    return visited


def execute_from_node(
    raw_copy: mne.io.BaseRaw,
    graph: PipelineGraph,
    from_node_id: str,
    cached_outputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Partial re-execution: reuses cached_outputs for nodes upstream of
    from_node_id, re-executes from_node_id and all its descendants.

    Returns (results_dict, updated_node_outputs).
    """
    order = topological_sort(graph)

    if from_node_id not in [n.id for n in graph.nodes]:
        raise ValueError(f"Node '{from_node_id}' not found in graph.")

    descendants = _get_descendants(graph, from_node_id)
    rerun_set = {from_node_id} | descendants

    node_lookup = {n.id: n for n in graph.nodes}
    incoming: dict[str, list] = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.target_node_id].append(edge)

    node_outputs = dict(cached_outputs)
    results: dict[str, Any] = {}

    for node_id in order:
        node = node_lookup[node_id]
        descriptor = NODE_REGISTRY.get(node.node_type)

        if descriptor is None:
            results[node_id] = {
                "node_type": node.node_type,
                "status": "error",
                "error": f"Unknown node type: {node.node_type}",
                "summary": {"kind": "error", "message": f"Unknown node type: {node.node_type}"},
                "rerun": node_id in rerun_set,
                "cache_hit": False,
            }
            continue

        if node_id in rerun_set:
            try:
                edges_in = incoming.get(node_id, [])
                if edges_in:
                    if len(edges_in) == 1:
                        resolved_input = node_outputs[edges_in[0].source_node_id]
                    else:
                        primary_handle_id = descriptor.inputs[0].id if descriptor.inputs else None
                        primary_edge = next(
                            (e for e in edges_in if e.target_handle_id == primary_handle_id),
                            edges_in[0],
                        )
                        resolved_input = node_outputs[primary_edge.source_node_id]
                else:
                    resolved_input = raw_copy

                merged_params: dict[str, Any] = {
                    p.name: p.default for p in descriptor.parameters
                }
                merged_params.update(node.parameters)

                _t0 = time.perf_counter()
                output = descriptor.execute_fn(resolved_input, merged_params)
                _t1 = time.perf_counter()
                node_outputs[node_id] = output

                result_entry: dict[str, Any] = {
                    "node_type": node.node_type,
                    "status": "success",
                    "output_type": type(output).__name__,
                    "data": None,
                    "summary": _summarize_output(output),
                    "rerun": True,
                    "cache_hit": False,
                    "execution_time_ms": round((_t1 - _t0) * 1000, 2),
                }
                if isinstance(output, str) and output.startswith("data:image"):
                    result_entry["data"] = output
                elif isinstance(output, dict):
                    result_entry["metrics"] = _to_native(output)
                elif isinstance(output, np.ndarray):
                    result_entry["metrics"] = _to_native(output)

                if result_entry.get("status") == "success" and not result_entry.get("data"):
                    preview = generate_preview(output)
                    if preview:
                        result_entry["preview"] = preview

                results[node_id] = result_entry

            except Exception as exc:
                results[node_id] = {
                    "node_type": node.node_type,
                    "status": "error",
                    "error": str(exc),
                    "summary": {
                        "kind": "error",
                        "python_type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "rerun": True,
                    "cache_hit": False,
                }
                break
        else:
            output = node_outputs.get(node_id)
            if output is not None:
                result_entry = {
                    "node_type": node.node_type,
                    "status": "success",
                    "output_type": type(output).__name__,
                    "data": None,
                    "summary": _summarize_output(output),
                    "rerun": False,
                    "cache_hit": True,
                }
                if isinstance(output, str) and output.startswith("data:image"):
                    result_entry["data"] = output
                elif isinstance(output, dict):
                    result_entry["metrics"] = _to_native(output)
                elif isinstance(output, np.ndarray):
                    result_entry["metrics"] = _to_native(output)

                if result_entry.get("status") == "success" and not result_entry.get("data"):
                    preview = generate_preview(output)
                    if preview:
                        result_entry["preview"] = preview

                results[node_id] = result_entry

    return results, node_outputs
