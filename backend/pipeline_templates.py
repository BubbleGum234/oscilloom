"""
backend/pipeline_templates.py

Pre-built pipeline templates that researchers can drop onto the canvas with
one click.  Each template is a dict with a name, description, category, and
a list of nodes + edges using the same schema as PipelineGraph.

Templates reference only node types that already exist in NODE_REGISTRY.
The frontend requests GET /registry/templates on startup and renders them
in a dedicated "Templates" section of the node palette.

To add a new template:
  1. Define a new dict following the pattern below.
  2. Append it to PIPELINE_TEMPLATES.
  3. Restart the server (or wait for --reload).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Artifact Rejection Pipeline
# ---------------------------------------------------------------------------
# Common pattern: amplitude threshold -> flatline detection -> gradient
# detection -> summarize annotations.  Saves 4 drag-drops and 3 edge
# connections.

ARTIFACT_REJECTION_TEMPLATE: dict[str, Any] = {
    "id": "artifact_rejection",
    "name": "Artifact Rejection Pipeline",
    "description": (
        "Standard multi-method artifact detection: amplitude threshold "
        "\u2192 flatline \u2192 gradient \u2192 annotation summary. "
        "Connect filtered EEG into the first node."
    ),
    "category": "Preprocessing",
    "nodes": [
        {
            "id": "tpl_1",
            "node_type": "detect_bad_segments",
            "label": "Detect Bad Segments",
            "params": {
                "threshold_uv": 150.0,
                "window_s": 1.0,
                "step_s": 0.5,
            },
        },
        {
            "id": "tpl_2",
            "node_type": "detect_flatline",
            "label": "Detect Flatline",
            "params": {
                "min_std_uv": 0.5,
                "window_s": 5.0,
            },
        },
        {
            "id": "tpl_3",
            "node_type": "detect_bad_gradient",
            "label": "Detect Bad Gradient",
            "params": {
                "max_gradient_uv_per_ms": 10.0,
                "window_s": 0.5,
            },
        },
        {
            "id": "tpl_4",
            "node_type": "summarize_annotations",
            "label": "Summarize Annotations",
            "params": {},
        },
    ],
    "edges": [
        {
            "source": "tpl_1",
            "source_handle": "raw_out",
            "target": "tpl_2",
            "target_handle": "raw_in",
            "handle_type": "filtered_eeg",
        },
        {
            "source": "tpl_2",
            "source_handle": "raw_out",
            "target": "tpl_3",
            "target_handle": "raw_in",
            "handle_type": "filtered_eeg",
        },
        {
            "source": "tpl_3",
            "source_handle": "raw_out",
            "target": "tpl_4",
            "target_handle": "raw_in",
            "handle_type": "filtered_eeg",
        },
    ],
}

# ---------------------------------------------------------------------------
# Master list
# ---------------------------------------------------------------------------

PIPELINE_TEMPLATES: list[dict[str, Any]] = [
    ARTIFACT_REJECTION_TEMPLATE,
]
