"""
backend/custom_node_store.py

Persists user-saved custom Python node presets to disk.

Storage: ~/.oscilloom/custom_nodes/<slug>.json
Each file contains: { slug, display_name, description, code, timeout_s, created_at }

On startup, load_custom_nodes_on_startup() reads all JSON files and registers
each as a NodeDescriptor with node_type = "custom__<slug>".
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.registry import NODE_REGISTRY
from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)
from backend.registry.nodes.custom import _execute_custom_python

# Storage directory
_CUSTOM_NODES_DIR = Path.home() / ".oscilloom" / "custom_nodes"

# Thread safety for registry mutations
_lock = threading.Lock()


def _slugify(name: str) -> str:
    """Convert display name to a safe slug for node_type and filename."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:50] or "untitled"


def _node_type_from_slug(slug: str) -> str:
    return f"custom__{slug}"


def _build_descriptor(definition: dict[str, Any]) -> NodeDescriptor:
    """Build a NodeDescriptor from a saved custom node definition."""
    slug = definition["slug"]
    code = definition["code"]
    timeout_s = definition.get("timeout_s", 60)

    return NodeDescriptor(
        node_type=_node_type_from_slug(slug),
        display_name=definition["display_name"],
        category="Custom",
        description=definition.get("description", "User-saved custom node."),
        tags=["custom", "saved", slug],
        inputs=[
            HandleSchema(id="data_in", type="raw_eeg", label="Raw EEG In", required=False),
            HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG In", required=False),
            HandleSchema(id="epochs_in", type="epochs", label="Epochs In", required=False),
        ],
        outputs=[
            HandleSchema(id="data_out", type="filtered_eeg", label="Data Out"),
        ],
        parameters=[
            ParameterSchema(
                name="code",
                label="Python Code",
                type="string",
                default=code,
                description="MNE-Python code. The variable `data` holds the input.",
            ),
            ParameterSchema(
                name="timeout_s",
                label="Timeout",
                type="int",
                default=timeout_s,
                min=5,
                max=120,
                step=5,
                unit="s",
                description="Maximum execution time in seconds.",
            ),
        ],
        execute_fn=_execute_custom_python,
        code_template=lambda p: p.get("code", "# No code"),
        methods_template=lambda p: (
            "A custom processing step was applied using user-defined "
            "MNE-Python code."
        ),
    )


def save_custom_node(
    display_name: str,
    description: str,
    code: str,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """
    Save a custom node preset to disk and register it.

    Returns the saved definition dict.
    Raises ValueError if the slug conflicts with an existing non-custom node.
    """
    slug = _slugify(display_name)
    node_type = _node_type_from_slug(slug)

    # Don't overwrite built-in nodes
    if node_type in NODE_REGISTRY and not node_type.startswith("custom__"):
        raise ValueError(f"Node type '{node_type}' conflicts with a built-in node.")

    definition = {
        "slug": slug,
        "display_name": display_name,
        "description": description,
        "code": code,
        "timeout_s": min(int(timeout_s), 120),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Ensure directory exists
    _CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)

    # Write to disk
    file_path = _CUSTOM_NODES_DIR / f"{slug}.json"
    with open(file_path, "w") as f:
        json.dump(definition, f, indent=2)

    # Register in NODE_REGISTRY
    descriptor = _build_descriptor(definition)
    with _lock:
        NODE_REGISTRY[node_type] = descriptor

    return definition


def list_custom_nodes() -> list[dict[str, Any]]:
    """List all saved custom node definitions."""
    if not _CUSTOM_NODES_DIR.exists():
        return []

    nodes = []
    for file_path in sorted(_CUSTOM_NODES_DIR.glob("*.json")):
        try:
            with open(file_path) as f:
                nodes.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return nodes


def get_custom_node(slug: str) -> dict[str, Any] | None:
    """Get a single custom node definition by slug."""
    file_path = _CUSTOM_NODES_DIR / f"{slug}.json"
    if not file_path.exists():
        return None
    with open(file_path) as f:
        return json.load(f)


def delete_custom_node(slug: str) -> bool:
    """
    Delete a custom node from disk and unregister it.
    Returns True if deleted, False if not found.
    """
    file_path = _CUSTOM_NODES_DIR / f"{slug}.json"
    node_type = _node_type_from_slug(slug)

    if not file_path.exists():
        return False

    file_path.unlink()

    with _lock:
        NODE_REGISTRY.pop(node_type, None)

    return True


def load_custom_nodes_on_startup() -> int:
    """
    Load all saved custom nodes from disk and register them.
    Called once at server startup from main.py.
    Returns the number of nodes loaded.
    """
    if not _CUSTOM_NODES_DIR.exists():
        return 0

    count = 0
    for file_path in _CUSTOM_NODES_DIR.glob("*.json"):
        try:
            with open(file_path) as f:
                definition = json.load(f)
            descriptor = _build_descriptor(definition)
            with _lock:
                NODE_REGISTRY[descriptor.node_type] = descriptor
            count += 1
        except Exception:
            # Don't crash server for a bad custom node file
            logging.getLogger(__name__).warning(
                "Failed to load custom node from %s", file_path, exc_info=True
            )
    return count
