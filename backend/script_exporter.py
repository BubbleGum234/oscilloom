"""
backend/script_exporter.py

Generates a standalone, runnable Python script from a PipelineGraph.

The generated script:
  - Reproduces the pipeline using explicit MNE API calls.
  - Makes every parameter explicit — no hidden defaults.
  - Includes the session audit log as a comment block at the top.
  - Is validated with ast.parse() before being returned.
  - Runs in a clean Python environment with only `mne` installed.

Script generation uses a Jinja2 template (templates/pipeline_export.py.j2)
rather than string concatenation for maintainability and testability.

Each NodeDescriptor provides a `code_template` callable that returns the
MNE-Python code snippet for that node type. The exporter pre-computes
the snippet for each node and passes it to the Jinja2 template, which
renders it directly. This eliminates per-node-type elif blocks in the
template and keeps all node-specific logic inside the registry.

Branching and merging pipelines are fully supported — topological sort
ensures correct execution order regardless of graph shape.

To add export support for a new node type, define a `code_template`
callable on the NodeDescriptor. The exporter and template do not need
to change.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from backend.engine import topological_sort
from backend.models import PipelineGraph
from backend.registry import NODE_REGISTRY

TEMPLATES_DIR = Path(__file__).parent / "templates"


def export(pipeline: PipelineGraph, audit_log: list[dict[str, Any]]) -> str:
    """
    Renders a PipelineGraph as a Python script string.

    Args:
        pipeline:  A validated PipelineGraph (validation must run before calling this).
        audit_log: List of AuditLogEntry dicts from the frontend session.
                   Included as comments at the top of the generated script.

    Returns:
        A string containing the complete .py script.

    Raises:
        ValueError: if the rendered script has a syntax error
                    (indicates a template bug, not user error).
    """
    _assert_no_compound_nodes(pipeline)

    execution_order = topological_sort(pipeline)
    node_by_id = {n.id: n for n in pipeline.nodes}

    # Build list of (node, descriptor) in execution order for the template
    ordered_nodes = []
    for node_id in execution_order:
        node = node_by_id[node_id]
        descriptor = NODE_REGISTRY.get(node.node_type)
        if descriptor is None:
            raise ValueError(
                f"Node '{node_id}' has unknown type '{node.node_type}'. "
                "Run validate_pipeline() before export()."
            )

        # Merge schema defaults with pipeline-provided values (same as engine.py)
        merged_params: dict[str, Any] = {
            p.name: p.default for p in descriptor.parameters
        }
        merged_params.update(node.parameters)

        # Pre-compute code snippet from code_template
        code_snippet = None
        if descriptor.code_template is not None:
            try:
                code_snippet = descriptor.code_template(merged_params)
            except Exception:
                code_snippet = None

        ordered_nodes.append({
            "node": node,
            "descriptor": descriptor,
            "params": merged_params,
            "code_snippet": code_snippet,
        })

    # Convert audit log dicts to SimpleNamespace so Jinja2 dot-notation works
    from types import SimpleNamespace
    safe_audit_log = [
        SimpleNamespace(**entry) for entry in audit_log
    ]

    # Render the Jinja2 template
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,  # Raise on undefined variables (catches template bugs)
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("pipeline_export.py.j2")
    rendered = template.render(
        pipeline=pipeline,
        ordered_nodes=ordered_nodes,
        audit_log=safe_audit_log,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Validate that the rendered script is syntactically valid Python.
    # A SyntaxError here is a bug in the template — not a user error.
    try:
        ast.parse(rendered)
    except SyntaxError as e:
        raise ValueError(
            f"Generated script has a syntax error (this is a template bug): "
            f"line {e.lineno}: {e.msg}\n"
            f"--- Script excerpt ---\n"
            f"{_excerpt(rendered, e.lineno)}"
        ) from e

    return rendered


def _assert_no_compound_nodes(pipeline: PipelineGraph) -> None:
    """
    Raises ValueError if the pipeline contains any compound nodes.

    Script export cannot inline-expand compound sub-graphs (MVP limitation).
    """
    compound_nodes = [
        n.id for n in pipeline.nodes if n.node_type.startswith("c_")
    ]
    if compound_nodes:
        raise ValueError(
            "Script export does not support compound nodes yet. "
            f"The following nodes are compound: {compound_nodes}. "
            "Replace them with their inner nodes before exporting."
        )


def _assert_linear_pipeline(pipeline: PipelineGraph) -> None:
    """
    Raises ValueError if the pipeline is not a linear chain.

    A linear pipeline has every node with at most one incoming and one
    outgoing edge. This is the export scope for MVP.
    """
    from collections import Counter

    source_counts = Counter(e.source_node_id for e in pipeline.edges)
    target_counts = Counter(e.target_node_id for e in pipeline.edges)

    branching_nodes = [nid for nid, count in source_counts.items() if count > 1]
    merging_nodes   = [nid for nid, count in target_counts.items() if count > 1]

    if branching_nodes or merging_nodes:
        raise ValueError(
            "Script export currently supports linear pipelines only. "
            "This pipeline has branching or merging connections, which are "
            "not yet supported for export. "
            f"Branching at: {branching_nodes}. Merging at: {merging_nodes}."
        )


def _excerpt(script: str, lineno: int, context: int = 3) -> str:
    """Returns a few lines around a given line number for error context."""
    lines = script.splitlines()
    start = max(0, lineno - context - 1)
    end = min(len(lines), lineno + context)
    return "\n".join(
        f"{'>>>' if i + 1 == lineno else '   '} {i + 1:4d}: {line}"
        for i, line in enumerate(lines[start:end], start=start)
    )
