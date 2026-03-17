"""
backend/tests/test_security.py

Security-focused tests for path sanitization, ID validation, and API hardening.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from backend.path_security import (
    sanitize_filename,
    sanitize_id,
    validate_read_path,
    validate_write_path,
)


# ---------------------------------------------------------------------------
# validate_read_path
# ---------------------------------------------------------------------------


class TestValidateReadPath:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_read_path("")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="Invalid file path"):
            validate_read_path("/tmp/file\x00.edf")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="Invalid file path"):
            validate_read_path("../../../etc/passwd")

    def test_accepts_normal_path(self):
        # Use a path that exists
        result = validate_read_path(os.path.abspath(__file__))
        assert result.exists()

    def test_rejects_outside_allowed_dirs(self, tmp_path):
        allowed = [tmp_path / "safe"]
        allowed[0].mkdir()
        test_file = tmp_path / "unsafe" / "file.txt"
        test_file.parent.mkdir()
        test_file.touch()

        with pytest.raises(ValueError, match="Invalid file path"):
            validate_read_path(str(test_file), allowed_dirs=allowed)

    def test_accepts_within_allowed_dirs(self, tmp_path):
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        test_file = safe_dir / "data.edf"
        test_file.touch()

        result = validate_read_path(str(test_file), allowed_dirs=[safe_dir])
        assert result == test_file.resolve()


# ---------------------------------------------------------------------------
# validate_write_path
# ---------------------------------------------------------------------------


class TestValidateWritePath:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_write_path("")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="Invalid output path"):
            validate_write_path("../../evil.fif")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="Invalid output path"):
            validate_write_path("/tmp/out\x00.fif")

    def test_rejects_wrong_extension(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file extension"):
            validate_write_path(
                str(tmp_path / "output.exe"), allowed_extensions=[".fif"]
            )

    def test_accepts_correct_extension(self, tmp_path):
        result = validate_write_path(
            str(tmp_path / "output.fif"), allowed_extensions=[".fif"]
        )
        assert str(result).endswith(".fif")

    def test_rejects_nonexistent_parent(self):
        with pytest.raises(ValueError, match="parent directory does not exist"):
            validate_write_path("/nonexistent/dir/output.fif")


# ---------------------------------------------------------------------------
# sanitize_id
# ---------------------------------------------------------------------------


class TestSanitizeId:
    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            sanitize_id("")

    def test_rejects_traversal(self):
        result = sanitize_id("c_../../evil")
        assert "/" not in result
        assert ".." not in result
        assert result == "c_evil"

    def test_rejects_path_separators(self):
        result = sanitize_id("c_foo/bar\\baz")
        assert result == "c_foobarbaz"

    def test_accepts_valid_id(self):
        assert sanitize_id("c_my_node") == "c_my_node"
        assert sanitize_id("c_test-123") == "c_test-123"

    def test_rejects_only_invalid_chars(self):
        with pytest.raises(ValueError, match="only invalid characters"):
            sanitize_id("../../")


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_removes_traversal(self):
        result = sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_removes_null_bytes(self):
        result = sanitize_filename("file\x00name")
        assert "\x00" not in result

    def test_truncates_long_names(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_fallback_to_export(self):
        assert sanitize_filename("") == "export"
        assert sanitize_filename("///") == "export"

    def test_replaces_spaces(self):
        result = sanitize_filename("my file name")
        assert " " not in result
        assert result == "my_file_name"

    def test_preserves_safe_chars(self):
        result = sanitize_filename("node-output_v2.3")
        assert result == "node-output_v2.3"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestAPISecurityIntegration:
    """Tests that verify security fixes are applied at the API level."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_batch_save_no_path_in_response(self, client):
        """Verify the save endpoint doesn't leak filesystem paths."""
        resp = client.post("/pipeline/batch/nonexistent/save")
        # Should be 400 (job not found) — not 500
        assert resp.status_code in (400, 404)
        # Verify no filesystem path leaked in response body
        assert "/Users/" not in resp.text
        assert "/home/" not in resp.text

    def test_compound_publish_sanitizes_traversal_id(self, client):
        """Verify compound publish sanitizes path traversal in compound_id."""
        resp = client.post("/compound/publish", json={
            "compound_id": "../../evil",
            "display_name": "Evil",
            "description": "test",
            "sub_graph": {"nodes": [{"id": "n1", "node_type": "edf_loader", "parameters": {}}], "edges": []},
            "output_node_id": "n1",
        })
        # sanitize_id strips dots/slashes → "evil" → "c_evil" (valid)
        if resp.status_code == 200:
            data = resp.json()
            node_type = data.get("node_type", "")
            assert ".." not in node_type
            assert "/" not in node_type
            # Clean up
            client.delete(f"/compound/{node_type}")
