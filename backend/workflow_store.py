"""
backend/workflow_store.py

Filesystem-backed storage for saved workflows (pipeline graphs).

Each workflow is persisted as a JSON file under ~/.oscilloom/workflows/.
This replaces browser-side IndexedDB storage so workflows survive across
browsers and devices sharing the same machine.

THREAD SAFETY:
  - All write operations are protected by a threading.Lock.
  - Reads do not require the lock (atomic file reads).

STORAGE LOCATION:
  - Default: ~/.oscilloom/workflows/
  - Override via OSCILLOOM_WORKFLOWS_DIR environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_WORKFLOWS_DIR = pathlib.Path(
    os.environ.get(
        "OSCILLOOM_WORKFLOWS_DIR",
        os.path.expanduser("~/.oscilloom/workflows"),
    )
)

_lock = threading.Lock()

# UUID pattern for path traversal prevention
_UUID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _ensure_dir() -> None:
    """Create the workflows directory if it doesn't exist."""
    _WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)


def _validate_id(workflow_id: str) -> None:
    """Validate ID to prevent path traversal."""
    if not workflow_id or not _UUID_RE.match(workflow_id):
        raise ValueError(f"Invalid workflow ID: '{workflow_id}'")


def _read_workflow_file(path: pathlib.Path) -> dict | None:
    """Read and parse a single workflow JSON file. Returns None on error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Skipping corrupted workflow file '%s': %s", path.name, e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_workflows() -> list[dict]:
    """Return all workflows sorted by updatedAt descending."""
    _ensure_dir()
    workflows = []
    for path in _WORKFLOWS_DIR.glob("*.json"):
        data = _read_workflow_file(path)
        if data is not None:
            workflows.append(data)
    workflows.sort(key=lambda w: w.get("updatedAt", ""), reverse=True)
    return workflows


def get_workflow(workflow_id: str) -> dict | None:
    """Return a single workflow by ID, or None if not found."""
    _validate_id(workflow_id)
    path = _WORKFLOWS_DIR / f"{workflow_id}.json"
    if not path.exists():
        return None
    return _read_workflow_file(path)


def save_workflow(workflow: dict) -> dict:
    """
    Upsert a workflow — create or update.

    If no 'id' is provided, a new UUID is generated.
    Timestamps are set automatically if missing.
    """
    _ensure_dir()

    if not workflow.get("id"):
        workflow["id"] = str(uuid.uuid4())

    _validate_id(workflow["id"])

    now = datetime.now(timezone.utc).isoformat()
    if not workflow.get("createdAt"):
        workflow["createdAt"] = now
    workflow["updatedAt"] = now

    # Ensure required fields have defaults
    workflow.setdefault("name", "Untitled Workflow")
    workflow.setdefault("nodeCount", 0)
    workflow.setdefault("edgeCount", 0)
    workflow.setdefault("pipeline", {"nodes": [], "edges": []})

    path = _WORKFLOWS_DIR / f"{workflow['id']}.json"
    with _lock:
        with open(path, "w") as f:
            json.dump(workflow, f, indent=2)

    logger.info("Saved workflow '%s' (%s)", workflow["id"][:8], workflow.get("name", ""))
    return workflow


def delete_workflow(workflow_id: str) -> bool:
    """Delete a workflow by ID. Returns True if it existed."""
    _validate_id(workflow_id)
    path = _WORKFLOWS_DIR / f"{workflow_id}.json"
    with _lock:
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.warning("Failed to delete workflow '%s': %s", workflow_id, e)
            return False


def duplicate_workflow(workflow_id: str) -> dict:
    """
    Duplicate a workflow with a new ID and "(copy)" name suffix.

    Raises:
        KeyError: if the source workflow does not exist.
    """
    _validate_id(workflow_id)
    original = get_workflow(workflow_id)
    if original is None:
        raise KeyError(f"Workflow '{workflow_id}' not found.")

    now = datetime.now(timezone.utc).isoformat()
    duplicate = dict(original)
    duplicate["id"] = str(uuid.uuid4())
    duplicate["name"] = f"{original.get('name', 'Untitled')} (copy)"
    duplicate["createdAt"] = now
    duplicate["updatedAt"] = now

    return save_workflow(duplicate)


def clear_all() -> int:
    """Delete all workflows. Returns the count deleted."""
    _ensure_dir()
    count = 0
    with _lock:
        for path in _WORKFLOWS_DIR.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError as e:
                logger.warning("Failed to delete '%s': %s", path.name, e)
    logger.info("Cleared %d workflow(s)", count)
    return count


def get_stats() -> dict:
    """Return storage statistics."""
    _ensure_dir()
    count = 0
    disk_usage_bytes = 0
    for path in _WORKFLOWS_DIR.glob("*.json"):
        count += 1
        try:
            disk_usage_bytes += path.stat().st_size
        except OSError:
            pass
    return {
        "count": count,
        "disk_usage_bytes": disk_usage_bytes,
        "workflows_dir": str(_WORKFLOWS_DIR),
    }
