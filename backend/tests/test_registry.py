"""
backend/tests/test_registry.py

Tests for the node registry API and descriptor integrity.

These tests verify:
  - The registry API endpoints return correct responses
  - All 6 MVP node types are registered
  - Every descriptor has the required fields
  - execute_fn is never exposed in the API response
  - All handle types are valid (no typos in descriptor definitions)
  - Parameter schemas are well-formed
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.registry import NODE_REGISTRY
from backend.registry.node_descriptor import VALID_HANDLE_TYPES

# ---------------------------------------------------------------------------
# Expected MVP node types
# ---------------------------------------------------------------------------

MVP_NODE_TYPES = {
    "edf_loader",
    "bandpass_filter",
    "notch_filter",
    "resample",
    "compute_psd",
    "plot_psd",
}

REQUIRED_DESCRIPTOR_FIELDS = {
    "node_type",
    "display_name",
    "category",
    "description",
    "inputs",
    "outputs",
    "parameters",
    "tags",
}


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestRegistryEndpoints:
    def test_get_all_nodes_returns_200(self, client: TestClient):
        response = client.get("/registry/nodes")
        assert response.status_code == 200

    def test_get_all_nodes_has_nodes_key(self, client: TestClient):
        response = client.get("/registry/nodes")
        data = response.json()
        assert "nodes" in data, "Response must have a 'nodes' key"

    def test_get_all_nodes_has_count_key(self, client: TestClient):
        response = client.get("/registry/nodes")
        data = response.json()
        assert "count" in data
        assert data["count"] == len(data["nodes"])

    def test_all_mvp_node_types_are_registered(self, client: TestClient):
        response = client.get("/registry/nodes")
        registered = set(response.json()["nodes"].keys())
        missing = MVP_NODE_TYPES - registered
        assert not missing, (
            f"Missing MVP node types in registry: {missing}. "
            f"Registered: {registered}"
        )

    def test_execute_fn_never_in_api_response(self, client: TestClient):
        """
        execute_fn is a Python callable and must never be serialized.
        If it appears in the response, the serialization logic is broken.
        """
        response = client.get("/registry/nodes")
        nodes = response.json()["nodes"]
        for node_type, descriptor in nodes.items():
            assert "execute_fn" not in descriptor, (
                f"Node '{node_type}' exposed execute_fn in API response. "
                f"Check _descriptor_to_dict() in registry_routes.py."
            )

    def test_all_descriptors_have_required_fields(self, client: TestClient):
        response = client.get("/registry/nodes")
        nodes = response.json()["nodes"]
        for node_type, descriptor in nodes.items():
            missing = REQUIRED_DESCRIPTOR_FIELDS - set(descriptor.keys())
            assert not missing, (
                f"Node '{node_type}' is missing required fields: {missing}"
            )

    def test_get_single_node_type_returns_200(self, client: TestClient):
        response = client.get("/registry/nodes/bandpass_filter")
        assert response.status_code == 200

    def test_get_single_node_type_returns_correct_data(self, client: TestClient):
        response = client.get("/registry/nodes/bandpass_filter")
        data = response.json()
        assert data["node_type"] == "bandpass_filter"
        assert data["display_name"] == "Bandpass Filter"
        assert data["category"] == "Preprocessing"

    def test_get_unknown_node_type_returns_404(self, client: TestClient):
        response = client.get("/registry/nodes/nonexistent_node")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Descriptor integrity tests (test the Python objects, not the API)
# ---------------------------------------------------------------------------

class TestDescriptorIntegrity:
    def test_all_handle_types_are_valid(self):
        """
        Every handle type declared on every descriptor must be a valid HandleType.
        Catches typos like "raw-eeg" or "filteredEEG".
        """
        for node_type, descriptor in NODE_REGISTRY.items():
            for handle in descriptor.inputs + descriptor.outputs:
                assert handle.type in VALID_HANDLE_TYPES, (
                    f"Node '{node_type}': handle '{handle.id}' has invalid type "
                    f"'{handle.type}'. Valid types: {VALID_HANDLE_TYPES}"
                )

    def test_all_node_types_match_registry_key(self):
        """
        The descriptor.node_type must match the key used to register it.
        A mismatch would cause the engine to look up the wrong descriptor.
        """
        for key, descriptor in NODE_REGISTRY.items():
            assert key == descriptor.node_type, (
                f"Registry key '{key}' does not match descriptor.node_type "
                f"'{descriptor.node_type}'. They must be identical."
            )

    def test_no_empty_descriptions(self):
        """All descriptions must be non-empty strings."""
        for node_type, descriptor in NODE_REGISTRY.items():
            assert descriptor.description.strip(), (
                f"Node '{node_type}' has an empty description."
            )
            for param in descriptor.parameters:
                assert param.description.strip(), (
                    f"Node '{node_type}', parameter '{param.name}' has an empty description."
                )

    def test_select_parameters_have_options(self):
        """Parameters with type='select' must have at least 2 options."""
        for node_type, descriptor in NODE_REGISTRY.items():
            for param in descriptor.parameters:
                if param.type == "select":
                    assert param.options and len(param.options) >= 2, (
                        f"Node '{node_type}', parameter '{param.name}' is type "
                        f"'select' but has fewer than 2 options: {param.options}"
                    )

    def test_parameter_defaults_match_type(self):
        """Parameter defaults must be consistent with their declared type."""
        type_checks = {
            "float": (int, float),
            "int": (int,),
            "bool": (bool,),
            "string": (str,),
            "select": (str,),
        }
        for node_type, descriptor in NODE_REGISTRY.items():
            for param in descriptor.parameters:
                expected_types = type_checks.get(param.type)
                if expected_types:
                    # bool is a subclass of int in Python, so check bool first
                    if param.type == "int" and isinstance(param.default, bool):
                        raise AssertionError(
                            f"Node '{node_type}', param '{param.name}': "
                            f"default is bool but type is 'int'."
                        )
                    assert isinstance(param.default, expected_types), (
                        f"Node '{node_type}', param '{param.name}': "
                        f"default value {param.default!r} is not consistent "
                        f"with declared type '{param.type}'."
                    )

    def test_execute_fn_is_callable(self):
        """execute_fn must be a callable for every registered node."""
        for node_type, descriptor in NODE_REGISTRY.items():
            assert callable(descriptor.execute_fn), (
                f"Node '{node_type}'.execute_fn is not callable: "
                f"{descriptor.execute_fn!r}"
            )

    def test_source_nodes_have_no_inputs(self):
        """edf_loader should have no inputs (it is a DAG source)."""
        loader = NODE_REGISTRY.get("edf_loader")
        assert loader is not None
        assert loader.inputs == [], (
            "edf_loader should have no inputs. It is a source node."
        )

    def test_terminal_nodes_have_plot_output(self):
        """plot_psd should output a 'plot' handle type (terminal)."""
        plotter = NODE_REGISTRY.get("plot_psd")
        assert plotter is not None
        output_types = [h.type for h in plotter.outputs]
        assert "plot" in output_types, (
            "plot_psd should have at least one output handle of type 'plot'."
        )


# ---------------------------------------------------------------------------
# Status endpoint test
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    def test_status_returns_200(self, client: TestClient):
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_has_ok(self, client: TestClient):
        response = client.get("/status")
        data = response.json()
        assert data["status"] == "ok"

    def test_status_has_mne_version(self, client: TestClient):
        response = client.get("/status")
        data = response.json()
        assert "mne_version" in data
        assert data["mne_version"].startswith("1.")


# ---------------------------------------------------------------------------
# Pipeline templates endpoint tests
# ---------------------------------------------------------------------------

class TestTemplatesEndpoint:
    def test_get_templates_returns_200(self, client: TestClient):
        response = client.get("/registry/templates")
        assert response.status_code == 200

    def test_get_templates_has_templates_key(self, client: TestClient):
        response = client.get("/registry/templates")
        data = response.json()
        assert "templates" in data
        assert "count" in data
        assert data["count"] == len(data["templates"])

    def test_artifact_rejection_template_present(self, client: TestClient):
        response = client.get("/registry/templates")
        templates = response.json()["templates"]
        ids = [t["id"] for t in templates]
        assert "artifact_rejection" in ids, (
            f"artifact_rejection template not found. Available: {ids}"
        )

    def test_template_nodes_reference_valid_node_types(self, client: TestClient):
        """Every node_type in a template must exist in NODE_REGISTRY."""
        response = client.get("/registry/templates")
        templates = response.json()["templates"]
        for tpl in templates:
            for node in tpl["nodes"]:
                assert node["node_type"] in NODE_REGISTRY, (
                    f"Template '{tpl['id']}' references unknown node_type "
                    f"'{node['node_type']}'. Available: {sorted(NODE_REGISTRY.keys())}"
                )

    def test_template_has_required_fields(self, client: TestClient):
        response = client.get("/registry/templates")
        templates = response.json()["templates"]
        required = {"id", "name", "description", "category", "nodes", "edges"}
        for tpl in templates:
            missing = required - set(tpl.keys())
            assert not missing, (
                f"Template '{tpl.get('id', '?')}' missing fields: {missing}"
            )
