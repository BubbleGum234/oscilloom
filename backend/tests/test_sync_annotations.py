"""
backend/tests/test_sync_annotations.py

Tests for the interactive annotation sync feature (P3-11).

Tests cover:
- The sync-annotations endpoint returns 404 when no temp file exists
- The sync-annotations endpoint returns 400 when browser is still open
- End-to-end annotation sync via temp .fif file exchange
- Annotations are correctly applied to the cached Raw object
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from backend.api.inspect_routes import (
    _active_browsers,
    _browser_lock,
    _browser_temp_files,
    _temp_files_lock,
)
from backend import session_store


@pytest.fixture()
def synthetic_raw() -> mne.io.BaseRaw:
    """Create a small synthetic Raw for testing."""
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3"],
        sfreq=256.0,
        ch_types=["eeg", "eeg", "eeg"],
        verbose=False,
    )
    data = np.random.RandomState(42).randn(3, 256 * 5)  # 5 seconds
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture()
def session_with_cache(synthetic_raw: mne.io.BaseRaw):
    """Create a session and populate its node cache with a Raw object."""
    sid = "test-session-annot"
    node_id = "node-raw-1"
    # Register the Raw in the session store so update_session_annotations works
    with session_store._lock:
        session_store._sessions[sid] = synthetic_raw.copy()
        session_store._session_last_access[sid] = __import__("time").time()
    # Also put raw in the node cache
    session_store.cache_node_outputs(sid, {node_id: synthetic_raw})
    yield sid, node_id, synthetic_raw
    # Cleanup
    session_store.clear_node_cache(sid)
    session_store.delete_session(sid)


class TestSyncAnnotationsRoute:
    """Tests for POST /pipeline/inspect/browser/sync-annotations."""

    def test_sync_returns_404_when_no_temp_file(self, client, session_with_cache):
        """Sync should return 404 when no temp file exists for the key."""
        sid, node_id, _ = session_with_cache
        # Ensure no temp file is registered
        key = f"{sid}:{node_id}"
        with _temp_files_lock:
            _browser_temp_files.pop(key, None)
        with _browser_lock:
            _active_browsers.pop(key, None)

        resp = client.post(
            "/pipeline/inspect/browser/sync-annotations",
            json={"session_id": sid, "target_node_id": node_id},
        )
        assert resp.status_code == 404
        assert "already synced" in resp.json()["detail"]

    def test_sync_returns_400_when_browser_still_open(self, client, session_with_cache):
        """Sync should return 400 if the browser process is still alive."""
        sid, node_id, _ = session_with_cache
        key = f"{sid}:{node_id}"

        # Mock a live process
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        with _browser_lock:
            _active_browsers[key] = mock_proc

        try:
            resp = client.post(
                "/pipeline/inspect/browser/sync-annotations",
                json={"session_id": sid, "target_node_id": node_id},
            )
            assert resp.status_code == 400
            assert "still open" in resp.json()["detail"]
        finally:
            with _browser_lock:
                _active_browsers.pop(key, None)

    def test_sync_reads_annotations_from_temp_file(self, client, session_with_cache):
        """Sync should read annotations from the temp .fif file and return them."""
        sid, node_id, raw = session_with_cache
        key = f"{sid}:{node_id}"

        # Create a temp .fif file with annotations (simulates what the browser subprocess writes)
        annotated_raw = raw.copy()
        annotations = mne.Annotations(
            onset=[1.0, 2.5],
            duration=[0.5, 1.0],
            description=["BAD_artifact", "BAD_eye_blink"],
        )
        annotated_raw.set_annotations(annotations)

        temp_fd, temp_path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(temp_fd)
        annotated_raw.save(temp_path, overwrite=True, verbose=False)

        # Register the temp file as if the browser had created it
        with _temp_files_lock:
            _browser_temp_files[key] = temp_path
        with _browser_lock:
            _active_browsers.pop(key, None)  # Ensure no active browser

        try:
            resp = client.post(
                "/pipeline/inspect/browser/sync-annotations",
                json={"session_id": sid, "target_node_id": node_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "synced"
            assert data["node_id"] == node_id
            assert data["n_annotations"] == 2
            assert len(data["annotations"]) == 2

            # Verify annotation content
            descs = [a["description"] for a in data["annotations"]]
            assert "BAD_artifact" in descs
            assert "BAD_eye_blink" in descs

            # Verify the cached Raw object was updated with annotations
            cached = session_store.get_cached_output(sid, node_id)
            assert len(cached.annotations) == 2
        finally:
            # Cleanup temp file if it still exists
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            with _temp_files_lock:
                _browser_temp_files.pop(key, None)

    def test_sync_cleans_up_temp_file(self, client, session_with_cache):
        """After successful sync, the temp file should be deleted."""
        sid, node_id, raw = session_with_cache
        key = f"{sid}:{node_id}"

        # Create a temp .fif file with no annotations
        temp_fd, temp_path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(temp_fd)
        raw.save(temp_path, overwrite=True, verbose=False)

        with _temp_files_lock:
            _browser_temp_files[key] = temp_path
        with _browser_lock:
            _active_browsers.pop(key, None)

        resp = client.post(
            "/pipeline/inspect/browser/sync-annotations",
            json={"session_id": sid, "target_node_id": node_id},
        )
        assert resp.status_code == 200
        # Temp file should be cleaned up
        assert not os.path.exists(temp_path)
        # Entry should be removed from tracking dict
        with _temp_files_lock:
            assert key not in _browser_temp_files

    def test_sync_with_zero_annotations(self, client, session_with_cache):
        """Sync should succeed even when the Raw has no annotations."""
        sid, node_id, raw = session_with_cache
        key = f"{sid}:{node_id}"

        temp_fd, temp_path = tempfile.mkstemp(suffix="_raw.fif")
        os.close(temp_fd)
        raw.save(temp_path, overwrite=True, verbose=False)

        with _temp_files_lock:
            _browser_temp_files[key] = temp_path
        with _browser_lock:
            _active_browsers.pop(key, None)

        try:
            resp = client.post(
                "/pipeline/inspect/browser/sync-annotations",
                json={"session_id": sid, "target_node_id": node_id},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["n_annotations"] == 0
            assert data["annotations"] == []
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            with _temp_files_lock:
                _browser_temp_files.pop(key, None)
