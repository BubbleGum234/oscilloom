"""
backend/registry/nodes/custom.py

Custom Python node — lets power users run arbitrary MNE-Python code
inside a pipeline. Executes in a subprocess with restricted builtins.

Security note: This is safety, not security. The sandbox prevents
accidental damage (import os, open()), not malicious intent. Oscilloom
is local-first — the user runs code on their own machine.
"""

# SECURITY NOTE — Custom Python Sandbox
# The exec() call in this module uses restricted builtins to prevent
# accidental damage (e.g., no file I/O, no imports beyond numpy/scipy).
# This is a SAFETY measure, not a SECURITY boundary.
# Oscilloom is local-first: users execute code on their own machine.
# Do NOT expose this endpoint to untrusted users in a multi-tenant deployment
# without adding a proper sandbox (e.g., subprocess with seccomp, Docker container).

import multiprocessing
import os
import pickle
import tempfile
from typing import Any

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


def _execute_custom_python(input_data: Any, params: dict) -> Any:
    user_code = params.get("code", "").strip()
    if not user_code:
        raise ValueError("No code provided in custom node")
    timeout = min(int(params.get("timeout_s", 60)), 120)  # Cap at 2 minutes
    return _run_in_subprocess(input_data, user_code, timeout)


def _run_in_subprocess(data: Any, code: str, timeout_s: int) -> Any:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.pkl")
        out_path = os.path.join(tmpdir, "out.pkl")
        err_path = os.path.join(tmpdir, "err.txt")

        with open(in_path, "wb") as f:
            pickle.dump(data, f)

        proc = multiprocessing.Process(
            target=_worker, args=(in_path, out_path, err_path, code)
        )
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            raise TimeoutError(f"Custom node timed out after {timeout_s}s")

        # Check for error
        if os.path.exists(err_path):
            with open(err_path) as f:
                raise RuntimeError(f"Custom node error: {f.read()}")

        if proc.exitcode != 0:
            raise RuntimeError(f"Custom node crashed (exit code {proc.exitcode})")

        with open(out_path, "rb") as f:
            return pickle.load(f)


def _worker(in_path: str, out_path: str, err_path: str, code: str) -> None:
    try:
        import mne
        import numpy as np

        with open(in_path, "rb") as f:
            data = pickle.load(f)

        allowed = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "list": list,
                "dict": dict, "tuple": tuple, "set": set, "int": int,
                "float": float, "str": str, "bool": bool, "type": type,
                "isinstance": isinstance, "enumerate": enumerate, "zip": zip,
                "min": min, "max": max, "abs": abs, "round": round,
                "sorted": sorted, "reversed": reversed, "sum": sum,
                "ValueError": ValueError, "TypeError": TypeError,
                "RuntimeError": RuntimeError,
            },
            "mne": mne,
            "np": np,
            "data": data,
        }

        exec(code, allowed)
        result = allowed.get("data", data)

        with open(out_path, "wb") as f:
            pickle.dump(result, f)

    except Exception as e:
        with open(err_path, "w") as f:
            f.write(str(e))


CUSTOM_PYTHON = NodeDescriptor(
    node_type="custom_python",
    display_name="Custom Python",
    category="Custom",
    description=(
        "Run custom MNE-Python code in a sandboxed subprocess. "
        "WARNING: This node executes code on your machine — only use code "
        "you understand and trust. Your code receives `data` (the upstream "
        "MNE object) and should assign your result to `data`. "
        "Available: mne, numpy (as np). Max timeout: 2 minutes."
    ),
    tags=["custom", "python", "code", "advanced", "script"],
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
            default=(
                "# `data` is your MNE object (Raw, Epochs, etc.)\n"
                "# Modify it and reassign to `data`\n"
                "data = data.copy().filter(l_freq=1, h_freq=40, verbose=False)\n"
            ),
            description=(
                "MNE-Python code. The variable `data` holds the input. "
                "Reassign `data` with your result. "
                "Available: mne, np (numpy)."
            ),
        ),
        ParameterSchema(
            name="timeout_s",
            label="Timeout",
            type="int",
            default=60,
            min=5,
            max=120,
            step=5,
            unit="s",
            description="Maximum execution time in seconds. Capped at 120s.",
        ),
    ],
    execute_fn=_execute_custom_python,
    code_template=lambda p: p.get("code", "# No code"),
    methods_template=lambda p: (
        "A custom processing step was applied using user-defined "
        "MNE-Python code."
    ),
)
