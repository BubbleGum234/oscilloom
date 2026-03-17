"""
backend/tests/test_hardening.py

Tests for backend hardening: session clear-all, session stats, and
custom node deletion of nonexistent slugs.
"""

from __future__ import annotations

import time

import mne
import numpy as np
import pytest

from backend import session_store
from backend.execution_cache import ExecutionCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_raw(n_channels: int = 5, sfreq: float = 256.0, duration: float = 2.0):
    """Create a minimal synthetic Raw for testing (no files on disk)."""
    data = np.random.RandomState(42).randn(n_channels, int(sfreq * duration)) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg", verbose=False)
    return mne.io.RawArray(data, info, verbose=False)


def _inject_session(monkeypatch, ss, session_id: str, raw) -> None:
    """Directly inject a session into session_store internals."""
    ss._sessions[session_id] = raw
    ss._session_last_access[session_id] = time.time()
    ss._execution_caches[session_id] = ExecutionCache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionClearAll:
    def test_session_clear_all_removes_sessions(self, client, tmp_path, monkeypatch):
        """POST two sessions, call DELETE /session/clear-all, verify both are gone."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)

        raw1 = _make_synthetic_raw()
        raw2 = _make_synthetic_raw()

        _inject_session(monkeypatch, ss, "sess-clear-1", raw1)
        _inject_session(monkeypatch, ss, "sess-clear-2", raw2)

        # Verify both sessions exist
        resp1 = client.get("/session/sess-clear-1/info")
        assert resp1.status_code == 200
        resp2 = client.get("/session/sess-clear-2/info")
        assert resp2.status_code == 200

        # Clear all
        resp = client.delete("/session/clear-all")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "cleared"
        assert body["deleted_count"] >= 2

        # Verify both are gone
        resp1 = client.get("/session/sess-clear-1/info")
        assert resp1.status_code == 404
        resp2 = client.get("/session/sess-clear-2/info")
        assert resp2.status_code == 404


class TestSessionStats:
    def test_session_stats_returns_counts(self, client, tmp_path, monkeypatch):
        """POST a session, call GET /session/stats, verify expected fields."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)

        raw = _make_synthetic_raw()
        _inject_session(monkeypatch, ss, "sess-stats-1", raw)

        resp = client.get("/session/stats")
        assert resp.status_code == 200
        body = resp.json()

        # Check expected fields exist
        assert "active_sessions" in body
        assert "sessions_dir" in body
        assert "ttl_seconds" in body
        assert "max_sessions" in body
        assert "disk_usage_bytes" in body

        # At least our injected session should be counted
        assert body["active_sessions"] >= 1


class TestDeleteCustomNodeNonexistent:
    def test_delete_custom_node_nonexistent_returns_404(self, client):
        """DELETE /custom-nodes/nonexistent-slug should return 404."""
        resp = client.delete("/custom-nodes/nonexistent-slug")
        assert resp.status_code == 404
