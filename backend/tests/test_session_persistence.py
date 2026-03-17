"""
backend/tests/test_session_persistence.py

Integration tests for session disk persistence: create, persist, reload, delete.
Uses a temporary directory so tests never touch ~/.oscilloom/sessions/.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time

import mne
import numpy as np
import pytest

from backend.execution_cache import ExecutionCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_raw(n_channels: int = 5, sfreq: float = 256.0, duration: float = 2.0):
    """Create a minimal synthetic Raw for testing (no files on disk)."""
    data = np.random.RandomState(42).randn(n_channels, int(sfreq * duration)) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg", verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """End-to-end tests for the persistence lifecycle."""

    def test_create_persists_fif_and_json(self, tmp_path, monkeypatch):
        """create_session() should write .fif and .json to the sessions dir."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        # Also reset internal state to avoid cross-test pollution
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        # Write a synthetic .fif file that MNE can read back
        raw = _make_synthetic_raw()
        fif_path = str(tmp_path / "input.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        session_id, info = ss.create_session(fif_path)

        # Verify in-memory session exists
        assert session_id in ss._sessions
        assert info["nchan"] == 5
        assert info["sfreq"] == 256.0

        # Verify .fif and .json exist on disk
        persisted_fif = tmp_path / f"{session_id}.fif"
        persisted_json = tmp_path / f"{session_id}.json"
        assert persisted_fif.exists(), ".fif file not persisted"
        assert persisted_json.exists(), ".json file not persisted"

        # Verify JSON metadata content
        with open(persisted_json) as f:
            meta = json.load(f)
        assert meta["session_id"] == session_id
        assert meta["nchan"] == 5
        assert meta["sfreq"] == 256.0

    def test_load_persisted_sessions_restores(self, tmp_path, monkeypatch):
        """load_persisted_sessions() should reload sessions from disk."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        # Manually persist a session (simulating a previous server run)
        raw = _make_synthetic_raw()
        sid = "test-session-001"
        fif_path = tmp_path / f"{sid}.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)
        meta = {
            "session_id": sid,
            "upload_timestamp": time.time(),
            "sfreq": 256.0,
            "nchan": 5,
            "duration_s": 2.0,
            "ch_names": [f"EEG{i:03d}" for i in range(5)],
        }
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        # Reload
        count = ss.load_persisted_sessions()
        assert count == 1
        assert sid in ss._sessions

        # Verify copy-on-write compliance
        copy1 = ss.get_raw_copy(sid)
        copy2 = ss.get_raw_copy(sid)
        assert copy1 is not copy2
        assert copy1 is not ss._sessions[sid]

    def test_load_skips_stale_sessions(self, tmp_path, monkeypatch):
        """Sessions older than TTL should be cleaned up, not reloaded."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})
        monkeypatch.setattr(ss, "SESSION_TTL_SECONDS", 60)

        raw = _make_synthetic_raw()
        sid = "stale-session"
        fif_path = tmp_path / f"{sid}.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)
        meta = {
            "session_id": sid,
            "upload_timestamp": time.time() - 120,  # 2 minutes old, TTL is 60s
            "sfreq": 256.0,
            "nchan": 5,
        }
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        count = ss.load_persisted_sessions()
        assert count == 0
        assert sid not in ss._sessions
        # Stale files should be cleaned up
        assert not fif_path.exists()

    def test_load_skips_missing_fif(self, tmp_path, monkeypatch):
        """If .fif is missing but .json exists, skip gracefully."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        sid = "orphan-json"
        meta = {"session_id": sid, "upload_timestamp": time.time()}
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        count = ss.load_persisted_sessions()
        assert count == 0
        assert sid not in ss._sessions

    def test_load_skips_corrupted_fif(self, tmp_path, monkeypatch):
        """If .fif is corrupted, skip with warning and clean up."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        sid = "corrupt-session"
        # Write garbage to .fif
        (tmp_path / f"{sid}.fif").write_bytes(b"NOT A REAL FIF FILE")
        meta = {"session_id": sid, "upload_timestamp": time.time()}
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        count = ss.load_persisted_sessions()
        assert count == 0
        assert sid not in ss._sessions

    def test_delete_removes_disk_files(self, tmp_path, monkeypatch):
        """delete_session() should remove .fif and .json from disk."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        # Create a session
        raw = _make_synthetic_raw()
        fif_path = str(tmp_path / "input.fif")
        raw.save(fif_path, overwrite=True, verbose=False)
        session_id, _ = ss.create_session(fif_path)

        persisted_fif = tmp_path / f"{session_id}.fif"
        persisted_json = tmp_path / f"{session_id}.json"
        assert persisted_fif.exists()
        assert persisted_json.exists()

        # Delete and verify cleanup
        found = ss.delete_session(session_id)
        assert found is True
        assert session_id not in ss._sessions
        assert not persisted_fif.exists(), ".fif not deleted from disk"
        assert not persisted_json.exists(), ".json not deleted from disk"

    def test_reloaded_session_has_filenames_none(self, tmp_path, monkeypatch):
        """Reloaded sessions must have raw._filenames = [None] to avoid MNE errors."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        raw = _make_synthetic_raw()
        sid = "filenames-test"
        fif_path = tmp_path / f"{sid}.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)
        meta = {"session_id": sid, "upload_timestamp": time.time()}
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        ss.load_persisted_sessions()
        reloaded_raw = ss._sessions[sid]
        assert reloaded_raw._filenames == [None]

        # Verify copy works without FileNotFoundError
        copy = reloaded_raw.copy()
        assert copy is not reloaded_raw

    def test_reload_sets_last_access_to_now(self, tmp_path, monkeypatch):
        """Reloaded sessions should get last_access = now, not the original upload time."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})
        monkeypatch.setattr(ss, "SESSION_TTL_SECONDS", 3600)

        raw = _make_synthetic_raw()
        sid = "access-time-test"
        upload_ts = time.time() - 1800  # 30 minutes ago
        fif_path = tmp_path / f"{sid}.fif"
        raw.save(str(fif_path), overwrite=True, verbose=False)
        meta = {"session_id": sid, "upload_timestamp": upload_ts}
        with open(tmp_path / f"{sid}.json", "w") as f:
            json.dump(meta, f)

        before = time.time()
        ss.load_persisted_sessions()
        after = time.time()

        last_access = ss._session_last_access[sid]
        # last_access should be approximately now, not upload_ts
        assert last_access >= before
        assert last_access <= after

    def test_persist_failure_does_not_affect_in_memory(self, tmp_path, monkeypatch):
        """If disk persistence fails, the in-memory session should still work."""
        import backend.session_store as ss

        # Point to a non-writable directory
        bad_dir = tmp_path / "readonly"
        bad_dir.mkdir()
        bad_dir.chmod(0o444)

        monkeypatch.setattr(ss, "_SESSIONS_DIR", bad_dir)
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        raw = _make_synthetic_raw()
        fif_path = str(tmp_path / "input.fif")
        raw.save(fif_path, overwrite=True, verbose=False)

        # Should not raise despite disk failure
        session_id, info = ss.create_session(fif_path)
        assert session_id in ss._sessions
        assert info["nchan"] == 5

        # get_raw_copy should still work
        copy = ss.get_raw_copy(session_id)
        assert copy is not ss._sessions[session_id]

        # Cleanup chmod for tmp_path cleanup
        bad_dir.chmod(0o755)

    def test_nonexistent_sessions_dir_returns_zero(self, tmp_path, monkeypatch):
        """load_persisted_sessions() should return 0 if dir doesn't exist."""
        import backend.session_store as ss

        monkeypatch.setattr(ss, "_SESSIONS_DIR", tmp_path / "nonexistent")
        monkeypatch.setattr(ss, "_sessions", {})
        monkeypatch.setattr(ss, "_session_last_access", {})
        monkeypatch.setattr(ss, "_execution_caches", {})

        count = ss.load_persisted_sessions()
        assert count == 0
