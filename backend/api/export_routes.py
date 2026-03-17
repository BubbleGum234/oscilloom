"""
backend/api/export_routes.py

Routes for per-node data export in multiple formats (CSV, NPZ, MAT, JSON, FIF).

All exports read from the node output cache — no re-execution needed.
The pipeline must be executed at least once before exporting.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from typing import Any

import mne
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from backend import session_store


router = APIRouter(prefix="/pipeline", tags=["Export"])

# Accepted export formats
VALID_FORMATS = {"csv", "npz", "mat", "json", "fif", "png"}


class NodeExportRequest(BaseModel):
    session_id: str
    target_node_id: str
    format: str
    node_label: str = "export"


def _sanitize_label(label: str) -> str:
    """Make a node label safe for filenames."""
    from backend.path_security import sanitize_filename
    return sanitize_filename(label)


def _to_matlab_dict(output: Any) -> dict[str, Any]:
    """Convert an MNE/numpy output to a dict suitable for scipy.io.savemat."""
    if isinstance(output, mne.io.BaseRaw):
        return {
            "data": output.get_data(),
            "times": output.times,
            "ch_names": np.array(output.ch_names, dtype=object),
            "sfreq": output.info["sfreq"],
        }
    elif isinstance(output, mne.Epochs):
        return {
            "data": output.get_data(),
            "times": output.times,
            "ch_names": np.array(output.ch_names, dtype=object),
            "events": output.events,
            "sfreq": output.info["sfreq"],
        }
    elif isinstance(output, mne.Evoked):
        return {
            "data": output.data,
            "times": output.times,
            "ch_names": np.array(output.ch_names, dtype=object),
            "nave": output.nave,
        }
    elif isinstance(output, mne.time_frequency.Spectrum):
        return {
            "data": output.get_data(),
            "freqs": output.freqs,
            "ch_names": np.array(output.ch_names, dtype=object),
        }
    elif isinstance(output, np.ndarray):
        return {"data": output}
    elif isinstance(output, dict):
        mat: dict[str, Any] = {}
        for k, v in output.items():
            key = str(k).replace(" ", "_")[:31]  # MATLAB field name limits
            if isinstance(v, (int, float)):
                mat[key] = v
            elif isinstance(v, np.ndarray):
                mat[key] = v
            elif isinstance(v, (list, tuple)):
                mat[key] = np.array(v)
            elif isinstance(v, str):
                mat[key] = v
            else:
                mat[key] = str(v)
        return mat
    else:
        raise ValueError(f"Cannot convert {type(output).__name__} to MATLAB format.")


@router.post("/node/export", summary="Export a node's output in various formats")
async def export_node_output(request: NodeExportRequest):
    """
    Export a single node's cached output in the requested format.

    Supported formats: csv, npz, mat, json, fif.
    Requires pipeline to have been executed at least once.
    """
    fmt = request.format.lower()
    if fmt not in VALID_FORMATS:
        raise HTTPException(
            422,
            f"Unsupported format: '{fmt}'. Supported: {sorted(VALID_FORMATS)}"
        )

    try:
        output = session_store.get_cached_output(
            request.session_id, request.target_node_id
        )
    except KeyError as e:
        raise HTTPException(404, str(e))

    label = _sanitize_label(request.node_label)
    filename = f"{label}_output.{fmt}"

    try:
        if fmt == "png":
            return _export_png(output, filename)
        elif fmt == "csv":
            return _export_csv(output, filename)
        elif fmt == "npz":
            return _export_npz(output, filename)
        elif fmt == "mat":
            return _export_mat(output, filename)
        elif fmt == "json":
            return _export_json(output, filename)
        elif fmt == "fif":
            return _export_fif(output, filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(500, "Export failed due to an internal error.")


def _export_png(output: Any, filename: str) -> StreamingResponse:
    """Export a base64 PNG data URI as a downloadable PNG file."""
    import base64

    if not isinstance(output, str) or not output.startswith("data:image/png;base64,"):
        raise ValueError(
            f"PNG export only available for plot nodes (got {type(output).__name__})."
        )

    parts = output.split(",", 1)
    if len(parts) < 2:
        raise ValueError("Malformed PNG data URI — expected 'data:image/png;base64,...'")
    try:
        png_bytes = base64.b64decode(parts[1])
    except Exception:
        raise ValueError("Invalid base64 encoding in PNG data URI.")
    png_filename = filename.replace(".png", "") + ".png"

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={png_filename}"},
    )


def _export_csv(output: Any, filename: str) -> StreamingResponse:
    """Export as CSV."""
    import pandas as pd

    if isinstance(output, mne.io.BaseRaw):
        df = pd.DataFrame(output.get_data().T, columns=output.ch_names)
        df.insert(0, "time_s", output.times)
    elif isinstance(output, mne.Epochs):
        data = output.get_data()
        rows = []
        for ep_idx in range(data.shape[0]):
            for t_idx in range(data.shape[2]):
                rows.append(
                    [ep_idx, output.times[t_idx]] + data[ep_idx, :, t_idx].tolist()
                )
        df = pd.DataFrame(rows, columns=["epoch", "time_s"] + list(output.ch_names))
    elif isinstance(output, mne.Evoked):
        df = pd.DataFrame(output.data.T, columns=output.ch_names)
        df.insert(0, "time_s", output.times)
    elif isinstance(output, mne.time_frequency.Spectrum):
        psd_data = output.get_data()
        df = pd.DataFrame(psd_data.T, columns=output.ch_names)
        df.insert(0, "freq_hz", output.freqs)
    elif isinstance(output, np.ndarray):
        df = pd.DataFrame(output)
    elif isinstance(output, dict):
        df = pd.DataFrame([output])
    else:
        raise ValueError(f"Cannot export {type(output).__name__} as CSV.")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    content = buf.getvalue().encode("utf-8")

    return StreamingResponse(
        io.BytesIO(content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _export_npz(output: Any, filename: str) -> StreamingResponse:
    """Export as NumPy .npz archive."""
    buf = io.BytesIO()

    if isinstance(output, mne.io.BaseRaw):
        np.savez(buf, data=output.get_data(), times=output.times,
                 ch_names=np.array(output.ch_names))
    elif isinstance(output, mne.Epochs):
        np.savez(buf, data=output.get_data(), times=output.times,
                 ch_names=np.array(output.ch_names), events=output.events)
    elif isinstance(output, mne.Evoked):
        np.savez(buf, data=output.data, times=output.times,
                 ch_names=np.array(output.ch_names))
    elif isinstance(output, mne.time_frequency.Spectrum):
        np.savez(buf, data=output.get_data(), freqs=output.freqs,
                 ch_names=np.array(output.ch_names))
    elif isinstance(output, np.ndarray):
        np.savez(buf, data=output)
    else:
        raise ValueError(f"Cannot export {type(output).__name__} as NPZ.")

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _export_mat(output: Any, filename: str) -> StreamingResponse:
    """Export as MATLAB .mat file."""
    from scipy.io import savemat

    mat_dict = _to_matlab_dict(output)
    buf = io.BytesIO()
    savemat(buf, mat_dict)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _export_json(output: Any, filename: str) -> StreamingResponse:
    """Export as JSON (metrics dicts only)."""
    if not isinstance(output, dict):
        raise ValueError(
            f"JSON export only available for metrics nodes (got {type(output).__name__})."
        )

    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    content = json.dumps(output, indent=2, default=_default).encode("utf-8")

    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _export_fif(output: Any, filename: str) -> FileResponse:
    """Export as MNE .fif file."""
    if isinstance(output, mne.io.BaseRaw):
        suffix = ".fif"
    elif isinstance(output, mne.Epochs):
        suffix = "-epo.fif"
        filename = filename.replace(".fif", "-epo.fif")
    elif isinstance(output, mne.Evoked):
        suffix = "-ave.fif"
        filename = filename.replace(".fif", "-ave.fif")
    else:
        raise ValueError(
            f"FIF export only available for Raw/Epochs/Evoked (got {type(output).__name__})."
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    if isinstance(output, mne.Evoked):
        mne.write_evokeds(tmp_path, output, overwrite=True, verbose=False)
    else:
        output.save(tmp_path, overwrite=True, verbose=False)

    return FileResponse(
        tmp_path,
        media_type="application/octet-stream",
        filename=filename,
        background=BackgroundTask(os.unlink, tmp_path),
    )
