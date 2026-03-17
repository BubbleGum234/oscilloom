"""Tests for custom node persistence (B5) and import/export (B6)."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.custom_node_store import (
    save_custom_node,
    list_custom_nodes,
    get_custom_node,
    delete_custom_node,
    load_custom_nodes_on_startup,
    _slugify,
    _CUSTOM_NODES_DIR,
)
from backend.registry import NODE_REGISTRY


class TestSlugify:
    def test_simple_name(self):
        assert _slugify("My Filter") == "my_filter"

    def test_special_chars(self):
        assert _slugify("EEG Pre-Process v2.0!") == "eeg_pre_process_v2_0"

    def test_empty_name(self):
        assert _slugify("") == "untitled"

    def test_long_name_truncated(self):
        result = _slugify("a" * 100)
        assert len(result) <= 50


class TestCustomNodeStore:
    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path):
        """Redirect storage to a temp directory for test isolation."""
        with patch("backend.custom_node_store._CUSTOM_NODES_DIR", tmp_path / "custom_nodes"):
            yield tmp_path / "custom_nodes"
        # Clean up any custom__ entries left in NODE_REGISTRY
        to_remove = [k for k in NODE_REGISTRY if k.startswith("custom__")]
        for k in to_remove:
            NODE_REGISTRY.pop(k, None)

    def test_save_creates_file(self, use_temp_dir):
        result = save_custom_node(
            display_name="Test Filter",
            description="A test filter",
            code="data = data.copy()",
            timeout_s=30,
        )
        assert result["slug"] == "test_filter"
        assert result["display_name"] == "Test Filter"
        assert (use_temp_dir / "test_filter.json").exists()

    def test_save_registers_in_registry(self, use_temp_dir):
        save_custom_node(
            display_name="Reg Test",
            description="",
            code="data = data.copy()",
        )
        assert "custom__reg_test" in NODE_REGISTRY
        # Clean up
        delete_custom_node("reg_test")

    def test_list_returns_saved_nodes(self, use_temp_dir):
        save_custom_node("Node A", "", "data = data.copy()")
        save_custom_node("Node B", "", "data = data.copy()")
        nodes = list_custom_nodes()
        names = [n["display_name"] for n in nodes]
        assert "Node A" in names
        assert "Node B" in names

    def test_get_returns_definition(self, use_temp_dir):
        save_custom_node("Get Test", "desc", "data = data.copy()")
        result = get_custom_node("get_test")
        assert result is not None
        assert result["display_name"] == "Get Test"
        assert result["description"] == "desc"

    def test_get_missing_returns_none(self, use_temp_dir):
        assert get_custom_node("nonexistent") is None

    def test_delete_removes_file_and_registry(self, use_temp_dir):
        save_custom_node("Delete Me", "", "data = data.copy()")
        assert "custom__delete_me" in NODE_REGISTRY
        assert delete_custom_node("delete_me") is True
        assert "custom__delete_me" not in NODE_REGISTRY
        assert not (use_temp_dir / "delete_me.json").exists()

    def test_delete_missing_returns_false(self, use_temp_dir):
        assert delete_custom_node("nonexistent") is False

    def test_save_overwrites_existing(self, use_temp_dir):
        save_custom_node("Same Name", "", "code_v1")
        save_custom_node("Same Name", "", "code_v2")
        result = get_custom_node("same_name")
        assert result["code"] == "code_v2"

    def test_timeout_capped_at_120(self, use_temp_dir):
        result = save_custom_node("Cap Test", "", "data = data.copy()", timeout_s=999)
        assert result["timeout_s"] == 120

    def test_load_on_startup(self, use_temp_dir):
        # Manually create a JSON file
        use_temp_dir.mkdir(parents=True, exist_ok=True)
        definition = {
            "slug": "startup_test",
            "display_name": "Startup Test",
            "description": "Test",
            "code": "data = data.copy()",
            "timeout_s": 60,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        with open(use_temp_dir / "startup_test.json", "w") as f:
            json.dump(definition, f)

        count = load_custom_nodes_on_startup()
        assert count >= 1
        assert "custom__startup_test" in NODE_REGISTRY
        # Clean up
        NODE_REGISTRY.pop("custom__startup_test", None)


class TestCustomNodeRoutes:
    """Test the REST API routes via TestClient."""

    @pytest.fixture(autouse=True)
    def use_temp_dir(self, tmp_path):
        with patch("backend.custom_node_store._CUSTOM_NODES_DIR", tmp_path / "custom_nodes"):
            yield
        # Clean up any custom__ entries left in NODE_REGISTRY
        to_remove = [k for k in NODE_REGISTRY if k.startswith("custom__")]
        for k in to_remove:
            NODE_REGISTRY.pop(k, None)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_save_endpoint(self, client):
        res = client.post("/custom-nodes", json={
            "display_name": "API Test",
            "description": "From API",
            "code": "data = data.copy()",
            "timeout_s": 30,
        })
        assert res.status_code == 200
        assert res.json()["status"] == "saved"
        # Clean up
        delete_custom_node("api_test")

    def test_list_endpoint(self, client):
        client.post("/custom-nodes", json={
            "display_name": "List Test",
            "code": "data = data.copy()",
        })
        res = client.get("/custom-nodes")
        assert res.status_code == 200
        assert res.json()["count"] >= 1
        # Clean up
        delete_custom_node("list_test")

    def test_get_endpoint(self, client):
        client.post("/custom-nodes", json={
            "display_name": "Get API",
            "code": "data = data.copy()",
        })
        res = client.get("/custom-nodes/get_api")
        assert res.status_code == 200
        assert res.json()["display_name"] == "Get API"
        # Clean up
        delete_custom_node("get_api")

    def test_get_404(self, client):
        res = client.get("/custom-nodes/nonexistent")
        assert res.status_code == 404

    def test_delete_endpoint(self, client):
        client.post("/custom-nodes", json={
            "display_name": "Del API",
            "code": "data = data.copy()",
        })
        res = client.delete("/custom-nodes/del_api")
        assert res.status_code == 200
        assert res.json()["status"] == "deleted"

    def test_delete_404(self, client):
        res = client.delete("/custom-nodes/nonexistent")
        assert res.status_code == 404

    def test_export_endpoint(self, client):
        client.post("/custom-nodes", json={
            "display_name": "Export Test",
            "code": "data = data.copy()",
        })
        res = client.get("/custom-nodes/export_test/export")
        assert res.status_code == 200
        assert res.json()["slug"] == "export_test"
        # Clean up
        delete_custom_node("export_test")

    def test_import_endpoint(self, client):
        res = client.post("/custom-nodes/import", json={
            "display_name": "Import Test",
            "description": "Imported",
            "code": "data = data.copy()",
            "timeout_s": 30,
        })
        assert res.status_code == 200
        assert res.json()["status"] == "imported"
        # Clean up
        delete_custom_node("import_test")
