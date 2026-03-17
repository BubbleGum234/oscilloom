"""
backend/path_security.py

Reusable path and ID sanitization utilities for security hardening.

Used by io.py, io_extended.py, compound_registry.py, batch_processor.py,
and export_routes.py to prevent path traversal, arbitrary file access,
and filename injection attacks.
"""

from __future__ import annotations

import re
from pathlib import Path


def validate_read_path(path: str, allowed_dirs: list[Path] | None = None) -> Path:
    """
    Validate a file path for reading. Rejects path traversal and null bytes.

    Args:
        path: Raw file path string from user input.
        allowed_dirs: If provided, the resolved path must be under one of these.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the path is invalid or not within allowed directories.
    """
    if not path or not path.strip():
        raise ValueError("File path must not be empty.")

    if "\x00" in path:
        raise ValueError("Invalid file path.")

    if ".." in path:
        raise ValueError("Invalid file path.")

    resolved = Path(path).resolve()

    if allowed_dirs:
        if not any(
            str(resolved).startswith(str(allowed.resolve()))
            for allowed in allowed_dirs
        ):
            raise ValueError("Invalid file path.")

    return resolved


def validate_write_path(
    path: str,
    allowed_extensions: list[str] | None = None,
) -> Path:
    """
    Validate a file path for writing. Rejects traversal, null bytes, and
    optionally enforces allowed file extensions.

    Args:
        path: Raw file path string from user input.
        allowed_extensions: If provided, file must end with one of these (e.g., [".fif"]).

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the path is invalid or has a disallowed extension.
    """
    if not path or not path.strip():
        raise ValueError("Output path must not be empty.")

    if "\x00" in path:
        raise ValueError("Invalid output path.")

    if ".." in path:
        raise ValueError("Invalid output path.")

    resolved = Path(path).resolve()

    if not resolved.parent.exists():
        raise ValueError("Invalid output path: parent directory does not exist.")

    if allowed_extensions:
        if not any(resolved.name.endswith(ext) for ext in allowed_extensions):
            raise ValueError(
                f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
            )

    return resolved


def sanitize_id(raw_id: str) -> str:
    """
    Sanitize an identifier (compound_id, batch_id) for safe use in file paths.

    Allows only alphanumeric characters, underscores, and hyphens.

    Args:
        raw_id: The raw identifier string.

    Returns:
        The sanitized identifier.

    Raises:
        ValueError: If the ID is empty or contains only invalid characters.
    """
    if not raw_id or not raw_id.strip():
        raise ValueError("ID must not be empty.")

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", raw_id)

    if not sanitized:
        raise ValueError("ID contains only invalid characters.")

    if sanitized.startswith("."):
        raise ValueError("ID must not start with a dot.")

    return sanitized


def sanitize_filename(label: str) -> str:
    """
    Sanitize a label for safe use as a filename in Content-Disposition headers.

    Strips path separators, null bytes, traversal sequences, and non-safe characters.
    Truncates to 200 characters. Falls back to "export" if empty after sanitization.

    Args:
        label: Raw label string (e.g., node label, pipeline name).

    Returns:
        A safe filename string.
    """
    if not label:
        return "export"

    # Remove null bytes and traversal sequences
    safe = label.replace("\x00", "").replace("..", "")

    # Keep only safe characters: alphanumeric, underscore, hyphen, dot, space
    safe = re.sub(r"[^a-zA-Z0-9_\-. ]", "", safe)

    # Replace spaces with underscores
    safe = safe.replace(" ", "_")

    # Strip leading/trailing underscores and dots
    safe = safe.strip("_.")

    # Truncate
    safe = safe[:200]

    return safe or "export"
