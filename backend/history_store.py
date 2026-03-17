"""
backend/history_store.py

Filesystem-backed storage for pipeline run history.

Each run is persisted as a JSON file under ~/.oscilloom/history/.
When the number of stored runs exceeds MAX_RUNS, the oldest runs
are automatically deleted.

THREAD SAFETY:
  - All write operations are protected by a threading.Lock.

STORAGE LOCATION:
  - Default: ~/.oscilloom/history/
  - Override via OSCILLOOM_HISTORY_DIR environment variable.
  - Max runs configurable via OSCILLOOM_MAX_HISTORY_RUNS (default: 50).
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

_HISTORY_DIR = pathlib.Path(
    os.environ.get(
        "OSCILLOOM_HISTORY_DIR",
        os.path.expanduser("~/.oscilloom/history"),
    )
)

MAX_RUNS = int(os.environ.get("OSCILLOOM_MAX_HISTORY_RUNS", "50"))

_lock = threading.Lock()

# Safe ID pattern for path traversal prevention
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _ensure_dir() -> None:
    """Create the history directory if it doesn't exist."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _validate_id(run_id: str) -> None:
    """Validate ID to prevent path traversal."""
    if not run_id or not _SAFE_ID_RE.match(run_id):
        raise ValueError(f"Invalid run ID: '{run_id}'")


def _read_run_file(path: pathlib.Path) -> dict | None:
    """Read and parse a single run JSON file. Returns None on error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Skipping corrupted history file '%s': %s", path.name, e)
        return None


def _trim_old_runs() -> None:
    """
    Delete oldest runs if we exceed MAX_RUNS.
    Must be called with _lock held.
    """
    runs = []
    for path in _HISTORY_DIR.glob("*.json"):
        data = _read_run_file(path)
        if data is not None:
            runs.append((data.get("timestamp", ""), path))

    if len(runs) <= MAX_RUNS:
        return

    # Sort by timestamp ascending (oldest first)
    runs.sort(key=lambda x: x[0])
    to_delete = runs[: len(runs) - MAX_RUNS]
    for _, path in to_delete:
        try:
            path.unlink()
            logger.info("Auto-trimmed old run: %s", path.stem[:8])
        except OSError as e:
            logger.warning("Failed to trim run '%s': %s", path.name, e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_runs() -> list[dict]:
    """Return all runs sorted by timestamp descending."""
    _ensure_dir()
    runs = []
    for path in _HISTORY_DIR.glob("*.json"):
        data = _read_run_file(path)
        if data is not None:
            runs.append(data)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


def get_run(run_id: str) -> dict | None:
    """Return a single run by ID, or None if not found."""
    _validate_id(run_id)
    path = _HISTORY_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return _read_run_file(path)


def save_run(run: dict) -> dict:
    """
    Save a new run to disk. Auto-trims oldest runs if MAX_RUNS exceeded.

    If no 'id' is provided, a new UUID is generated.
    """
    _ensure_dir()

    if not run.get("id"):
        run["id"] = str(uuid.uuid4())

    _validate_id(run["id"])

    now = datetime.now(timezone.utc).isoformat()
    if not run.get("timestamp"):
        run["timestamp"] = now

    # Ensure required fields have defaults
    run.setdefault("name", "Untitled Run")
    run.setdefault("nodeResults", {})
    run.setdefault("paramSnapshot", {})
    run.setdefault("thumbnails", {})
    run.setdefault("nodeCount", 0)
    run.setdefault("errorCount", 0)

    path = _HISTORY_DIR / f"{run['id']}.json"
    with _lock:
        with open(path, "w") as f:
            json.dump(run, f, indent=2)
        _trim_old_runs()

    logger.info("Saved run '%s' (%s)", run["id"][:8], run.get("name", ""))
    return run


def delete_run(run_id: str) -> bool:
    """Delete a run by ID. Returns True if it existed."""
    _validate_id(run_id)
    path = _HISTORY_DIR / f"{run_id}.json"
    with _lock:
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.warning("Failed to delete run '%s': %s", run_id, e)
            return False


def rename_run(run_id: str, name: str) -> dict | None:
    """
    Rename a run. Returns the updated run dict, or None if not found.
    """
    _validate_id(run_id)
    path = _HISTORY_DIR / f"{run_id}.json"

    with _lock:
        if not path.exists():
            return None
        data = _read_run_file(path)
        if data is None:
            return None
        data["name"] = name
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return data


def clear_all() -> int:
    """Delete all runs. Returns the count deleted."""
    _ensure_dir()
    count = 0
    with _lock:
        for path in _HISTORY_DIR.glob("*.json"):
            try:
                path.unlink()
                count += 1
            except OSError as e:
                logger.warning("Failed to delete '%s': %s", path.name, e)
    logger.info("Cleared %d run(s)", count)
    return count


def get_stats() -> dict:
    """Return storage statistics."""
    _ensure_dir()
    count = 0
    disk_usage_bytes = 0
    for path in _HISTORY_DIR.glob("*.json"):
        count += 1
        try:
            disk_usage_bytes += path.stat().st_size
        except OSError:
            pass
    return {
        "count": count,
        "disk_usage_bytes": disk_usage_bytes,
        "history_dir": str(_HISTORY_DIR),
    }
