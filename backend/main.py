"""
backend/main.py

FastAPI application factory for Oscilloom.

This file is a wiring file — it assembles routers and middleware.
Keep business logic out of here; it belongs in the route files or engine.py.

To start the server:
  source .venv/bin/activate
  uvicorn backend.main:app --reload --port 8000

OpenAPI docs (auto-generated):
  http://localhost:8000/docs       — Swagger UI
  http://localhost:8000/redoc      — ReDoc UI
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mne
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.api import (
    batch_routes, compound_routes, custom_node_routes,
    export_routes, history_routes, inspect_routes, pipeline_routes,
    registry_routes, report_routes, session_routes, workflow_routes,
)
from backend.compound_registry import load_compounds_on_startup
from backend.custom_node_store import load_custom_nodes_on_startup
from backend.rate_limit import limiter
from backend.session_store import load_persisted_sessions

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data directory migration (~/.neuroflow → ~/.oscilloom)
# ---------------------------------------------------------------------------

def _migrate_data_dir() -> None:
    """Rename ~/.neuroflow to ~/.oscilloom if the old directory exists and the
    new one does not."""
    old_dir = Path.home() / ".neuroflow"
    new_dir = Path.home() / ".oscilloom"

    if old_dir.exists() and not new_dir.exists():
        old_dir.rename(new_dir)
        _logger.info("Migrated data directory: %s -> %s", old_dir, new_dir)
    elif old_dir.exists() and new_dir.exists():
        _logger.warning(
            "Both %s and %s exist. Skipping migration to avoid data loss. "
            "Please manually merge or remove the old directory.",
            old_dir, new_dir,
        )


_migrate_data_dir()

_enable_docs = os.environ.get("OSCILLOOM_ENABLE_DOCS", "true").lower() == "true"

app = FastAPI(
    title="Oscilloom API",
    description=(
        "Local backend for the Oscilloom EEG pipeline builder. "
        "All data is processed on-device — nothing leaves the machine."
    ),
    version="0.1.0-beta",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Allow requests from the Vite dev server and its preview build.
# In a future Electron build, the origin will be a file:// URL or a
# custom protocol — update this list accordingly.

_default_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:4173",   # Vite preview build
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]
_extra = os.environ.get("OSCILLOOM_CORS_ORIGINS", "")
_cors_origins = _default_origins + [o.strip() for o in _extra.split(",") if o.strip()]

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

# ---------------------------------------------------------------------------
# Security response headers
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ---------------------------------------------------------------------------
# Shared thread pool executor
# ---------------------------------------------------------------------------
# Attached to app.state so route handlers can access it via request.app.state.executor.
# max_workers=2: allows one active MNE execution + one queued request.
# Increase this for multi-user or batch processing scenarios.
# See ARCHITECTURE.md Section 9 for the rationale for ThreadPoolExecutor
# over multiprocessing.

app.state.executor = ThreadPoolExecutor(max_workers=3)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(session_routes.router)
app.include_router(registry_routes.router)
app.include_router(pipeline_routes.router)
app.include_router(report_routes.router)
app.include_router(batch_routes.router)
app.include_router(compound_routes.router)
app.include_router(inspect_routes.router)
app.include_router(export_routes.router)
app.include_router(custom_node_routes.router)
app.include_router(workflow_routes.router)
app.include_router(history_routes.router)

# ---------------------------------------------------------------------------
# Load persisted compound nodes and custom nodes on startup
# ---------------------------------------------------------------------------
load_compounds_on_startup()
load_custom_nodes_on_startup()

# ---------------------------------------------------------------------------
# Reload persisted EEG sessions from disk
# ---------------------------------------------------------------------------
_n_reloaded = load_persisted_sessions()
if _n_reloaded:
    _logger.info("Reloaded %d persisted EEG session(s)", _n_reloaded)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/status", tags=["Health"], summary="Server health check")
def status() -> dict:
    """
    Returns server status and the installed MNE-Python version.

    Use this endpoint to verify the backend is running before the frontend
    attempts to load the registry or execute a pipeline.

    Expected response: {"status": "ok", "mne_version": "1.x.x"}
    """
    return {
        "status": "ok",
        "mne_version": mne.__version__,
    }
