"""
backend/batch_processor.py

Core batch processing engine for Tier 4.

Processes multiple EEG files through a single pipeline, collecting results
and generating an aggregated metrics CSV. Files are processed sequentially
by default (max_workers=1) to keep peak memory at one Raw object.

DESIGN:
  - The batch processor is a loop over engine.execute_pipeline().
  - The edf_loader passthrough pattern means we load each file into a
    BaseRaw, then pass it as raw_copy to execute_pipeline().
  - No node changes required. The engine is completely unaware of batching.
  - Jobs are stored in an in-memory dict, same pattern as session_store.
  - Progress is tracked via a BatchJob dataclass with a threading.Lock.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from concurrent.futures import ThreadPoolExecutor as _FileExecutor, TimeoutError as _FuturesTimeout

import mne

from backend.engine import execute_pipeline
from backend.models import PipelineGraph
from backend.path_security import sanitize_id


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatchJob:
    """Mutable state for a single batch job."""
    batch_id: str
    status: str = "running"
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_file: Optional[str] = None
    file_results: list[dict[str, Any]] = field(default_factory=list)
    failed_files: list[dict[str, str]] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

FILE_TIMEOUT_S = 300  # 5 minutes max per file in batch processing

MAX_JOBS = 20  # Evict oldest completed jobs when exceeded

_jobs: dict[str, BatchJob] = {}
_job_order: list[str] = []  # Insertion order for eviction
_jobs_lock = threading.Lock()


def get_job(batch_id: str) -> Optional[BatchJob]:
    """Get a batch job by ID. Returns None if not found."""
    with _jobs_lock:
        return _jobs.get(batch_id)


def _register_job(job: BatchJob) -> None:
    with _jobs_lock:
        _jobs[job.batch_id] = job
        _job_order.append(job.batch_id)
        # Evict oldest finished jobs when over limit
        while len(_jobs) > MAX_JOBS:
            _evict_oldest_finished()


def _evict_oldest_finished() -> None:
    """Remove the oldest completed/failed/cancelled job. Must hold _jobs_lock."""
    for bid in list(_job_order):
        j = _jobs.get(bid)
        if j and j.status in ("complete", "failed", "cancelled"):
            del _jobs[bid]
            _job_order.remove(bid)
            return
    # All jobs are still running — nothing to evict
    return


def delete_job(batch_id: str) -> bool:
    """Explicitly delete a batch job. Returns True if found and removed."""
    with _jobs_lock:
        if batch_id in _jobs:
            del _jobs[batch_id]
            if batch_id in _job_order:
                _job_order.remove(batch_id)
            return True
        return False


def count_running_jobs() -> int:
    """Count batch jobs currently in 'running' status."""
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.status == "running")


# ---------------------------------------------------------------------------
# File staging store
# ---------------------------------------------------------------------------

_staged_files: dict[str, dict[str, Any]] = {}
_staged_lock = threading.Lock()


STAGED_TTL_S = 7200  # 2 hours — auto-cleanup threshold for stale staged files


def stage_file(tmp_path: str, original_filename: str) -> str:
    """Register a staged file and return its ID."""
    file_id = str(uuid.uuid4())
    with _staged_lock:
        _staged_files[file_id] = {
            "path": tmp_path,
            "filename": original_filename,
            "metadata": {},
            "staged_at": time.time(),
        }
    return file_id


def cleanup_stale_staged(max_age_seconds: int = STAGED_TTL_S) -> int:
    """Remove staged files older than max_age_seconds. Returns count removed."""
    cutoff = time.time() - max_age_seconds
    to_remove: list[str] = []
    with _staged_lock:
        for fid, info in _staged_files.items():
            if info.get("staged_at", 0) < cutoff:
                to_remove.append(fid)
        removed_infos = []
        for fid in to_remove:
            removed_infos.append(_staged_files.pop(fid))
    # Delete temp files outside the lock
    for info in removed_infos:
        if os.path.exists(info["path"]):
            os.unlink(info["path"])
    return len(to_remove)


def get_staged_file(file_id: str) -> Optional[dict[str, Any]]:
    """Get staged file info by ID. Returns None if not found."""
    with _staged_lock:
        return _staged_files.get(file_id)


def update_file_metadata(file_id: str, metadata: dict[str, str]) -> bool:
    """Update metadata for a staged file. Returns False if file_id not found."""
    with _staged_lock:
        if file_id not in _staged_files:
            return False
        _staged_files[file_id]["metadata"] = metadata
        return True


def get_file_metadata(file_id: str) -> dict[str, str]:
    """Get metadata for a staged file. Returns empty dict if not found."""
    with _staged_lock:
        entry = _staged_files.get(file_id)
        return dict(entry["metadata"]) if entry else {}


def remove_staged_file(file_id: str) -> None:
    """Remove staged file entry and delete the temp file."""
    with _staged_lock:
        info = _staged_files.pop(file_id, None)
    if info and os.path.exists(info["path"]):
        os.unlink(info["path"])


def list_staged_files() -> list[dict[str, Any]]:
    """Return all staged files as [{file_id, filename, metadata}]."""
    with _staged_lock:
        return [
            {"file_id": fid, "filename": info["filename"],
             "metadata": info.get("metadata", {})}
            for fid, info in _staged_files.items()
        ]


def clear_staged_files() -> int:
    """Remove all staged files. Returns count removed."""
    with _staged_lock:
        items = list(_staged_files.items())
        _staged_files.clear()
    count = 0
    for _, info in items:
        if os.path.exists(info["path"]):
            os.unlink(info["path"])
            count += 1
    return count


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _flatten_metrics(node_results: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and flatten all metrics from node_results into a single dict.

    Each metric key is prefixed with the node_type to avoid collisions.
    E.g., {"compute_alpha_peak.iaf_hz": 10.2, "compute_asymmetry.asymmetry": -0.3}
    """
    flat: dict[str, Any] = {}
    for node_id, result in node_results.items():
        if (
            result.get("status") == "success"
            and isinstance(result.get("metrics"), dict)
        ):
            prefix = result.get("node_type", node_id)
            for key, value in result["metrics"].items():
                flat[f"{prefix}.{key}"] = value
    return flat


def _generate_metrics_csv(file_results: list[dict[str, Any]]) -> str:
    """
    Generate a CSV string from batch file results.

    Fixed columns: filename, status, n_channels, sfreq, duration_s,
    processing_time_s, error.
    Then: metadata columns (subject_id, group, condition, etc.).
    Then: metric columns from clinical/analysis nodes.
    Missing values get empty cells.
    """
    if not file_results:
        return ""

    # Collect all metadata keys across all files (preserve insertion order)
    meta_keys: list[str] = []
    meta_seen: set[str] = set()
    for fr in file_results:
        for key in fr.get("metadata", {}).keys():
            if key not in meta_seen:
                meta_keys.append(key)
                meta_seen.add(key)

    # Collect all metric keys across all files (preserve insertion order)
    all_keys: list[str] = []
    seen: set[str] = set()
    for fr in file_results:
        for key in fr.get("metrics", {}).keys():
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    output = io.StringIO()
    writer = csv.writer(output)

    # Fixed columns + metadata + metrics
    fixed = ["filename", "status", "n_channels", "sfreq", "duration_s",
             "processing_time_s", "error"]
    writer.writerow(fixed + meta_keys + all_keys)

    # Data rows
    for fr in file_results:
        file_info = fr.get("file_info", {})
        proc_time = fr.get("processing_time_s")
        error = fr.get("error") or ""

        row: list[str] = [
            fr.get("filename", ""),
            fr.get("status", ""),
            str(file_info.get("n_channels", "")),
            str(file_info.get("sfreq", "")),
            str(file_info.get("duration_s", "")),
            f"{proc_time:.3f}" if proc_time is not None else "",
            error,
        ]
        # Metadata columns
        metadata = fr.get("metadata", {})
        for key in meta_keys:
            row.append(str(metadata.get(key, "")))
        # Metric columns
        metrics = fr.get("metrics", {})
        for key in all_keys:
            val = metrics.get(key, "")
            if isinstance(val, float):
                row.append(f"{val:.6f}")
            elif isinstance(val, list):
                row.append(str(val))
            else:
                row.append(str(val) if val != "" else "")
        writer.writerow(row)

    return output.getvalue()


def compute_aggregate_statistics(
    file_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute mean/std/min/max/median for all numeric metrics across successful
    files. Also computes group-level breakdowns if metadata includes a 'group'
    field.

    Returns:
        {
          "overall": { "metric.name": { count, mean, std, min, max, median } },
          "by_group": {
            "control": { "metric.name": { count, mean, std, ... } },
            ...
          }
        }
    """
    import numpy as np
    from collections import defaultdict

    overall: dict[str, list[float]] = defaultdict(list)
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fr in file_results:
        if fr.get("status") != "success":
            continue
        group = fr.get("metadata", {}).get("group", "")
        for key, val in fr.get("metrics", {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                overall[key].append(float(val))
                if group:
                    by_group[group][key].append(float(val))

    def _summarize(values: list[float]) -> dict[str, Any]:
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=1)) if len(values) > 1 else 0.0, 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "median": round(float(np.median(arr)), 6),
        }

    result: dict[str, Any] = {
        "overall": {k: _summarize(v) for k, v in overall.items()},
        "by_group": {},
    }
    for grp, metrics in by_group.items():
        result["by_group"][grp] = {k: _summarize(v) for k, v in metrics.items()}

    return result


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_batch(file_ids: list[str], graph: PipelineGraph) -> str:
    """
    Create a batch job and return its batch_id.

    The actual processing is NOT started here -- the caller (the route handler)
    starts it in a background thread via the ThreadPoolExecutor.
    """
    batch_id = str(uuid.uuid4())
    job = BatchJob(
        batch_id=batch_id,
        total=len(file_ids),
        started_at=time.time(),
    )
    _register_job(job)
    return batch_id


def execute_batch(
    batch_id: str,
    file_ids: list[str],
    graph: PipelineGraph,
) -> None:
    """
    Process all files sequentially. Called in a background thread.

    For each file:
      1. Load the file into an MNE Raw object.
      2. Call engine.execute_pipeline(raw, graph).
      3. Extract flattened metrics and store results.
      4. Delete the Raw to free memory.
      5. Update job progress.

    NOTE: Do NOT remove staged files after execution.
    Tier 4B selective re-run depends on staged files persisting.
    """
    job = get_job(batch_id)
    if job is None:
        return

    for file_id in file_ids:
        # Check for cancellation
        with job.lock:
            if job.status == "cancelled":
                break

        staged = get_staged_file(file_id)
        if staged is None:
            with job.lock:
                job.failed += 1
                job.failed_files.append({
                    "file_id": file_id,
                    "filename": "unknown",
                    "error": f"Staged file '{file_id}' not found.",
                })
            continue

        filename = staged["filename"]
        file_path = staged["path"]

        with job.lock:
            job.current_file = filename

        try:
            # Load file -- creates a fresh Raw each time
            raw = mne.io.read_raw(file_path, preload=True, verbose=False)

            # Capture file info before pipeline may alter the Raw
            file_info = {
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info["sfreq"],
                "duration_s": round(raw.times[-1], 2) if len(raw.times) > 0 else 0.0,
            }

            # Execute pipeline (raw is used once then discarded)
            t0 = time.time()
            with _FileExecutor(max_workers=1) as file_exec:
                future = file_exec.submit(execute_pipeline, raw, graph)
                try:
                    node_results, _node_outputs = future.result(timeout=FILE_TIMEOUT_S)
                except _FuturesTimeout:
                    raise TimeoutError(
                        f"Processing timed out after {FILE_TIMEOUT_S}s"
                    )
            processing_time_s = round(time.time() - t0, 3)

            # Extract flattened metrics
            flat_metrics = _flatten_metrics(node_results)

            file_result = {
                "file_id": file_id,
                "filename": filename,
                "status": "success",
                "node_results": node_results,
                "metrics": flat_metrics,
                "metadata": staged.get("metadata", {}),
                "file_info": file_info,
                "processing_time_s": processing_time_s,
                "error": None,
            }

            with job.lock:
                job.completed += 1
                job.file_results.append(file_result)

            # Free memory
            del raw
            del node_results

        except Exception as e:
            traceback.print_exc()
            with job.lock:
                job.failed += 1
                job.file_results.append({
                    "file_id": file_id,
                    "filename": filename,
                    "status": "error",
                    "node_results": {},
                    "metrics": {},
                    "metadata": staged.get("metadata", {}) if staged else {},
                    "file_info": {},
                    "processing_time_s": None,
                    "error": str(e),
                })
                job.failed_files.append({
                    "file_id": file_id,
                    "filename": filename,
                    "error": str(e),
                })

    # Finalize job
    with job.lock:
        job.current_file = None
        job.finished_at = time.time()
        if job.status != "cancelled":
            job.status = "complete" if job.failed < job.total else "failed"


# ---------------------------------------------------------------------------
# Persistent batch results (Tier 4B — Enhancement 5)
# ---------------------------------------------------------------------------

_BATCH_RESULTS_DIR = os.environ.get(
    "OSCILLOOM_BATCH_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "batch_results"),
)


def _ensure_batch_dir() -> str:
    """Create the batch_results directory if it does not exist."""
    os.makedirs(_BATCH_RESULTS_DIR, exist_ok=True)
    return _BATCH_RESULTS_DIR


def save_batch_results(batch_id: str) -> str:
    """
    Serialize a completed batch job to disk as gzipped JSON.
    Returns the file path. Raises ValueError if job not found or not complete.
    """
    job = get_job(batch_id)
    if job is None:
        raise ValueError(f"Batch job '{batch_id}' not found.")

    with job.lock:
        if job.status not in ("complete", "failed", "cancelled"):
            raise ValueError(f"Batch job is still {job.status}.")

        data: dict[str, Any] = {
            "batch_id": job.batch_id,
            "status": job.status,
            "total": job.total,
            "completed": job.completed,
            "failed": job.failed,
            "file_results": job.file_results,
            "failed_files": list(job.failed_files),
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "saved_at": time.time(),
        }

    data["metrics_csv"] = _generate_metrics_csv(data["file_results"])

    batch_dir = _ensure_batch_dir()
    file_path = os.path.join(batch_dir, f"{batch_id}.json.gz")

    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    return file_path


def list_saved_batches() -> list[dict[str, Any]]:
    """
    List all saved batch results as summary dicts.
    Does NOT load full file_results (would be too expensive).
    """
    batch_dir = _ensure_batch_dir()
    results: list[dict[str, Any]] = []

    for fname in os.listdir(batch_dir):
        if not fname.endswith(".json.gz"):
            continue
        fpath = os.path.join(batch_dir, fname)
        try:
            with gzip.open(fpath, "rt", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "batch_id": data["batch_id"],
                "status": data["status"],
                "total": data["total"],
                "completed": data["completed"],
                "failed": data["failed"],
                "saved_at": data.get("saved_at", 0),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    results.sort(key=lambda x: x.get("saved_at", 0), reverse=True)
    return results


def load_saved_batch(batch_id: str) -> dict[str, Any]:
    """
    Load a saved batch result from disk.
    Raises FileNotFoundError if not found.
    """
    batch_id = sanitize_id(batch_id)
    batch_dir = _ensure_batch_dir()
    file_path = os.path.join(batch_dir, f"{batch_id}.json.gz")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Saved batch '{batch_id}' not found.")

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        return json.load(f)
