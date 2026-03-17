"""
backend/tests/conftest.py

Shared pytest fixtures for all backend tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    A TestClient wrapping the FastAPI app.
    Scoped to the session so the app is only initialized once per test run.
    """
    return TestClient(app)
