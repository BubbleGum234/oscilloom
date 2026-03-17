"""
backend/session_store.py

In-memory session store for loaded EEG Raw objects, with optional disk
persistence so sessions survive server restarts.

DESIGN:
  - A session is created when a researcher uploads an EEG file.
  - The Raw object is loaded once (preload=True) and held in RAM.
  - All pipeline executions receive a .copy() of the stored Raw — the
    stored original is NEVER mutated. This enables re-running with
    different parameters without reloading the file.
  - Sessions are evicted after SESSION_TTL_SECONDS of inactivity.

PERSISTENCE:
  - When a session is created, the Raw is saved as a .fif file and
    metadata as a .json file under ~/.oscilloom/sessions/.
  - On server startup, load_persisted_sessions() reloads any sessions
    that are still within the TTL window.
  - When a session is deleted, the on-disk files are also removed.

THREAD SAFETY:
  - FastAPI handles requests concurrently via asyncio.
  - The session dict is protected by a threading.Lock for mutations.
  - Reads via get_raw_copy() hold the lock briefly to fetch the reference,
    then release it before calling .copy() (which is safe without a lock
    since the stored object is immutable after creation).

MEMORY:
  - Each session holds one mne.io.BaseRaw in RAM (~50–500 MB typical).
  - Automatic TTL eviction removes sessions idle for > SESSION_TTL_SECONDS.
  - Max MAX_SESSIONS concurrent sessions; oldest evicted when exceeded.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
import uuid
from typing import Any, Optional

import mne

from backend.execution_cache import ExecutionCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = int(os.environ.get("OSCILLOOM_SESSION_TTL", "3600"))  # 1 hour default
MAX_SESSIONS = int(os.environ.get("OSCILLOOM_MAX_SESSIONS", "10"))

# Directory for persisted session files (.fif + .json)
_SESSIONS_DIR = pathlib.Path(
    os.environ.get("OSCILLOOM_SESSIONS_DIR", os.path.expanduser("~/.oscilloom/sessions"))
)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_sessions: dict[str, mne.io.BaseRaw] = {}
_session_last_access: dict[str, float] = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Node output cache — populated after each pipeline execution
# ---------------------------------------------------------------------------
# Stores the Python objects produced by each node in the last pipeline run.
# Keyed by session_id → {node_id → Python output object}.
# Used by the inspector, MNE browser, export, and re-run features to
# retrieve outputs without re-executing the pipeline.

_node_caches: dict[str, dict[str, Any]] = {}
_node_caches_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Execution cache — stores per-node output hashes for incremental re-runs
# ---------------------------------------------------------------------------

_execution_caches: dict[str, ExecutionCache] = {}


# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------

def _ensure_sessions_dir() -> None:
    """Create the sessions directory if it doesn't exist."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _persist_session(session_id: str, raw: mne.io.BaseRaw, info_dict: dict) -> None:
    """
    Save a session's Raw data and metadata to disk.

    Runs outside the lock — the Raw object is not mutated.
    Errors are logged but do not prevent the session from working in-memory.
    """
    try:
        _ensure_sessions_dir()
        fif_path = _SESSIONS_DIR / f"{session_id}.fif"
        json_path = _SESSIONS_DIR / f"{session_id}.json"

        # Save the Raw as .fif (verbose=False per project rules)
        raw.save(str(fif_path), overwrite=True, verbose=False)

        # Build metadata JSON
        metadata = {
            "session_id": session_id,
            "original_filename": info_dict.get("original_filename", ""),
            "upload_timestamp": time.time(),
            "sfreq": info_dict.get("sfreq"),
            "nchan": info_dict.get("nchan"),
            "duration_s": info_dict.get("duration_s"),
            "ch_names": info_dict.get("ch_names", []),
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Session '%s' persisted to disk", session_id[:8])
    except Exception:
        logger.warning("Failed to persist session '%s' to disk", session_id[:8], exc_info=True)


def _remove_persisted_session(session_id: str) -> None:
    """Delete the .fif and .json files for a session from disk."""
    for suffix in (".fif", ".json"):
        path = _SESSIONS_DIR / f"{session_id}{suffix}"
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to delete %s", path, exc_info=True)


def load_persisted_sessions() -> int:
    """
    Scan ~/.oscilloom/sessions/ and reload valid sessions into memory.

    Called once during server startup. Sessions older than TTL are cleaned
    up rather than reloaded. Corrupted .fif files or missing .json metadata
    files are skipped with a warning.

    Returns:
        The number of sessions successfully reloaded.
    """
    if not _SESSIONS_DIR.exists():
        return 0

    loaded = 0
    now = time.time()

    for json_path in _SESSIONS_DIR.glob("*.json"):
        session_id = json_path.stem
        fif_path = _SESSIONS_DIR / f"{session_id}.fif"

        # Skip if .fif is missing
        if not fif_path.exists():
            logger.warning(
                "Skipping persisted session '%s': .fif file missing", session_id[:8]
            )
            # Clean up orphaned .json
            try:
                json_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        # Read metadata
        try:
            with open(json_path) as f:
                metadata = json.load(f)
        except Exception:
            logger.warning(
                "Skipping persisted session '%s': cannot read .json metadata",
                session_id[:8],
                exc_info=True,
            )
            continue

        # Check TTL — use upload_timestamp from metadata
        upload_ts = metadata.get("upload_timestamp", 0)
        if now - upload_ts > SESSION_TTL_SECONDS:
            logger.info(
                "Cleaning up stale persisted session '%s' (age %.0fs > TTL %ds)",
                session_id[:8],
                now - upload_ts,
                SESSION_TTL_SECONDS,
            )
            _remove_persisted_session(session_id)
            continue

        # Check MAX_SESSIONS limit
        with _lock:
            if len(_sessions) >= MAX_SESSIONS:
                logger.info(
                    "Skipping persisted session '%s': MAX_SESSIONS (%d) reached",
                    session_id[:8],
                    MAX_SESSIONS,
                )
                continue

        # Load the .fif file
        try:
            raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
            raw._filenames = [None]
        except Exception:
            logger.warning(
                "Skipping persisted session '%s': corrupted .fif file",
                session_id[:8],
                exc_info=True,
            )
            # Clean up corrupted files
            _remove_persisted_session(session_id)
            continue

        # Insert into in-memory store.
        # Set last_access to *now* (not the original upload_ts) so the
        # reloaded session gets a full TTL window.  Using upload_ts would
        # cause near-expiry sessions to be evicted almost immediately
        # after being loaded.
        with _lock:
            _sessions[session_id] = raw
            _session_last_access[session_id] = now
            _execution_caches[session_id] = ExecutionCache()

        loaded += 1
        logger.info("Reloaded persisted session '%s'", session_id[:8])

    return loaded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _evict_expired_sessions() -> list[str]:
    """
    Remove sessions that have exceeded the TTL. Must be called with _lock held.

    Returns a list of evicted session IDs so the caller can perform disk
    cleanup and node-cache cleanup AFTER releasing _lock.  This avoids
    doing file I/O while holding the lock and prevents a nested-lock
    deadlock (_lock -> _node_caches_lock vs the reverse order).
    """
    now = time.time()
    expired = [
        sid for sid, last in _session_last_access.items()
        if now - last > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_last_access.pop(sid, None)
        _execution_caches.pop(sid, None)
        logger.info("Session '%s' evicted (idle for >%ds)", sid[:8], SESSION_TTL_SECONDS)
    return expired


def _evict_oldest_if_full() -> str | None:
    """
    Evict the oldest session if we've hit MAX_SESSIONS. Must be called with _lock held.

    Returns the evicted session ID (or None) so the caller can perform disk
    cleanup and node-cache cleanup AFTER releasing _lock.
    """
    if len(_sessions) >= MAX_SESSIONS:
        oldest_sid = min(_session_last_access, key=_session_last_access.get)  # type: ignore[arg-type]
        _sessions.pop(oldest_sid, None)
        _session_last_access.pop(oldest_sid, None)
        _execution_caches.pop(oldest_sid, None)
        logger.info(
            "Session '%s' evicted (max %d sessions reached)", oldest_sid[:8], MAX_SESSIONS
        )
        return oldest_sid
    return None


def _cleanup_evicted(evicted_ids: list[str]) -> None:
    """
    Perform disk and node-cache cleanup for evicted sessions.

    Called OUTSIDE _lock to avoid holding the lock during file I/O
    and to prevent deadlock from nested lock acquisition.
    """
    if not evicted_ids:
        return
    with _node_caches_lock:
        for sid in evicted_ids:
            _node_caches.pop(sid, None)
    for sid in evicted_ids:
        _remove_persisted_session(sid)


def create_session(file_path: str) -> tuple[str, dict]:
    """
    Loads an EEG file into memory and registers a new session.

    Args:
        file_path: Absolute path to an EDF or FIF file on disk.
                   The file is loaded with preload=True.

    Returns:
        A tuple of (session_id, info_dict) where:
          - session_id is a UUID4 string identifying this session.
          - info_dict contains a JSON-serializable summary of Raw.info
            for display in the InputNode (channel count, sfreq, etc.).

    Raises:
        Any exception from mne.io.read_raw (FileNotFoundError, RuntimeError,
        etc.) propagates to the caller. The session is not created on error.
    """
    # mne.io.read_raw dispatches to the correct reader based on file extension.
    # This handles .edf, .fif, .bdf, .set, etc.
    raw = mne.io.read_raw(file_path, preload=True, verbose=False)

    # Replace the internal file reference with None. Data is fully preloaded
    # into RAM, but MNE still stores the original temp file path in
    # raw._filenames. Since the temp file is deleted after upload,
    # operations like ICA's raw.copy().crop() raise FileNotFoundError when
    # MNE's filenames setter checks file existence. Using [None] preserves
    # the expected list length (one entry per data segment) so crop()'s
    # index-based slicing works, while None entries skip the existence check
    # in the setter (see mne/io/base.py filenames.setter).
    raw._filenames = [None]

    session_id = str(uuid.uuid4())

    with _lock:
        expired = _evict_expired_sessions()
        oldest = _evict_oldest_if_full()
        _sessions[session_id] = raw
        _session_last_access[session_id] = time.time()
        _execution_caches[session_id] = ExecutionCache()

    # Disk + node-cache cleanup outside the lock
    evicted = expired + ([oldest] if oldest else [])
    _cleanup_evicted(evicted)

    info_dict = _build_info_dict(raw)

    # Persist to disk (runs outside the lock; failures are non-fatal)
    _persist_session(session_id, raw, info_dict)

    return session_id, info_dict


def get_raw_copy(session_id: str) -> mne.io.BaseRaw:
    """
    Returns a COPY of the stored Raw object for a session.

    CRITICAL: Always use this function. Never access _sessions directly
    from engine.py or routes. The copy is what the pipeline mutates;
    the original stays clean for future re-runs.

    Args:
        session_id: A session ID previously returned by create_session().

    Returns:
        A copy of the stored mne.io.BaseRaw.

    Raises:
        KeyError: if the session_id is not found (file needs to be reloaded).
    """
    with _lock:
        expired = _evict_expired_sessions()
        raw = _sessions.get(session_id)
        if raw is not None:
            _session_last_access[session_id] = time.time()

    # Disk + node-cache cleanup outside the lock
    _cleanup_evicted(expired)

    if raw is None:
        raise KeyError(
            f"Session '{session_id}' not found. "
            "The file may need to be reloaded — the server may have restarted "
            "or the session expired after inactivity."
        )

    # .copy() is called outside the lock: it is a read-only operation on
    # the stored object, and we don't want to hold the lock during a
    # potentially expensive memory copy.
    return raw.copy()


def get_info(session_id: str) -> dict:
    """
    Returns the info dict for a session without making a full copy.

    Faster than get_raw_copy() when only metadata is needed.

    Raises:
        KeyError: if the session_id is not found.
    """
    with _lock:
        raw = _sessions.get(session_id)

    if raw is None:
        raise KeyError(f"Session '{session_id}' not found.")

    return _build_info_dict(raw)


def cache_node_outputs(session_id: str, outputs: dict[str, Any]) -> None:
    """Store node outputs from the last pipeline execution."""
    with _node_caches_lock:
        _node_caches[session_id] = outputs


def get_cached_output(session_id: str, node_id: str) -> Any:
    """
    Retrieve a cached node output. Raises KeyError if not found.

    Used by the MNE browser and per-node export to avoid re-execution.
    """
    with _node_caches_lock:
        cache = _node_caches.get(session_id)
    if cache is None:
        raise KeyError(f"No cached outputs for session '{session_id}'. Run the pipeline first.")
    output = cache.get(node_id)
    if output is None:
        raise KeyError(f"No cached output for node '{node_id}' in session '{session_id}'.")
    return output


def get_all_cached_outputs(session_id: str) -> dict[str, Any] | None:
    """Return full cache for a session, or None if no cache exists."""
    with _node_caches_lock:
        return _node_caches.get(session_id)


def get_execution_cache(session_id: str) -> ExecutionCache:
    """Get or create the execution cache for a session."""
    with _lock:
        cache = _execution_caches.get(session_id)
        if cache is None:
            cache = ExecutionCache()
            _execution_caches[session_id] = cache
        return cache


def clear_node_cache(session_id: str) -> None:
    """Clear cache when session is deleted or pipeline changes."""
    with _node_caches_lock:
        _node_caches.pop(session_id, None)


def update_session_annotations(session_id: str, annotations: "mne.Annotations") -> None:
    """
    Persist user-added annotations onto the session store's Raw object.

    This is an intentional user edit (like marking bad channels), NOT a
    pipeline mutation. Annotations set here survive full pipeline re-runs
    because get_raw_copy() will include them in the copy.

    Raises:
        KeyError: if the session_id is not found.
    """
    with _lock:
        raw = _sessions.get(session_id)
        if raw is None:
            raise KeyError(f"Session '{session_id}' not found.")
        raw.set_annotations(annotations)
        _session_last_access[session_id] = time.time()
    logger.info("Updated annotations on session '%s' (%d annotations)", session_id[:8], len(annotations))


def delete_session(session_id: str) -> bool:
    """
    Removes a session from the store, freeing its memory and deleting
    any persisted .fif/.json files from disk.

    Returns:
        True if the session existed and was removed.
        False if the session was not found (already deleted or never existed).
    """
    clear_node_cache(session_id)
    with _lock:
        _session_last_access.pop(session_id, None)
        _execution_caches.pop(session_id, None)
        found = _sessions.pop(session_id, None) is not None

    # Remove persisted files regardless of whether the in-memory session
    # existed — handles the case where the session was evicted from memory
    # but files remain on disk.
    _remove_persisted_session(session_id)

    return found


def list_sessions() -> list[str]:
    """Returns all active session IDs. For debugging/admin use."""
    with _lock:
        return list(_sessions.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_info_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Builds a JSON-serializable summary of an MNE Raw.info object.

    Includes all metadata a researcher needs before building a pipeline:
    channel summary, hardware filter settings, recording date, event
    annotations, channel type breakdown, and pre-marked bad channels.

    Limits ch_names to the first 20 to keep the response size reasonable
    for recordings with many channels (e.g., 256-channel HD-EEG).
    """
    from collections import Counter

    duration_s = raw.n_times / raw.info["sfreq"]

    # Channel type breakdown — e.g. {"eeg": 62, "eog": 2, "ecg": 1}
    ch_types: dict[str, int] = dict(
        Counter(mne.channel_type(raw.info, i) for i in range(raw.info["nchan"]))
    )

    # Annotation labels and count — critical for choosing event_id in Epoch node.
    # EDF+ files encode stimulus events as string annotations (e.g. "T0","T1","769").
    annotations = raw.annotations
    annotation_labels: list[str] = sorted(set(annotations.description)) if len(annotations) > 0 else []
    n_annotations: int = len(annotations)

    # Auto-detect channel naming convention (prefixes / suffixes)
    from backend.registry.nodes._channel_utils import detect_naming_convention
    naming_hints = detect_naming_convention(raw.ch_names)

    # Per-channel type list — for Set Channel Types table UI
    ch_name_type_list = [
        {"name": raw.ch_names[i], "type": mne.channel_type(raw.info, i)}
        for i in range(raw.info["nchan"])
    ]

    return {
        "nchan": raw.info["nchan"],
        "sfreq": raw.info["sfreq"],
        "ch_names": list(raw.ch_names),           # full list for dropdown population
        "ch_names_truncated": raw.info["nchan"] > 20,  # hint for display truncation
        "duration_s": round(duration_s, 3),
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "meas_date": str(raw.info["meas_date"]) if raw.info.get("meas_date") else None,
        # New fields
        "ch_types": ch_types,
        "ch_name_type_list": ch_name_type_list,  # per-channel [{name, type}, ...]
        "bads": list(raw.info["bads"]),          # pre-marked bad channels from file header
        "n_annotations": n_annotations,
        "annotation_labels": annotation_labels,   # unique event labels, e.g. ["T0","T1","T2"]
        "naming_hints": naming_hints,             # auto-detected naming convention
    }
