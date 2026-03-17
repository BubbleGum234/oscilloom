"""
backend/validation.py

Pipeline graph validation logic.

validate_pipeline() runs before every execution and export. It returns a
list of human-readable error strings — an empty list means the pipeline is
valid. The engine never receives an invalid pipeline.

Validation checks:
  1. Pipeline has at least one node.
  2. All node_types exist in NODE_REGISTRY.
  3. Edge source_handle_type equals target_handle_type (type mismatch).
  4. Edge source_handle_id exists on the source node's descriptor outputs.
  5. Edge target_handle_id exists on the target node's descriptor inputs.
  6. Parameter values within declared min/max ranges.
  7. Required input handles have at least one incoming edge.

Does NOT check:
  - Cycles (handled by engine.py's topological sort, which raises ValueError).
"""

from __future__ import annotations

from backend.models import PipelineGraph
from backend.registry import NODE_REGISTRY


def validate_pipeline(graph: PipelineGraph) -> list[str]:
    """
    Validates a PipelineGraph before execution or export.

    Args:
        graph: The PipelineGraph to validate.

    Returns:
        A list of human-readable error strings.
        Empty list means the pipeline is valid.
    """
    errors: list[str] = []
    node_by_id = {n.id: n for n in graph.nodes}

    # Check 1: Pipeline must have nodes
    if not graph.nodes:
        errors.append(
            "The pipeline has no nodes. Add at least one node to the canvas."
        )
        return errors  # No point checking edges on an empty graph

    # Check 2: All node_types must be registered
    for node in graph.nodes:
        if node.node_type not in NODE_REGISTRY:
            errors.append(
                f"Node '{node.id}' (label: '{node.label}') has unknown type "
                f"'{node.node_type}'. "
                f"Known types: {sorted(NODE_REGISTRY.keys())}"
            )

    # Check 3, 4, 5: Edge validity
    for edge in graph.edges:
        # Check 3: Handle type compatibility
        if edge.source_handle_type != edge.target_handle_type:
            errors.append(
                f"Edge '{edge.id}': type mismatch. "
                f"Source '{edge.source_node_id}.{edge.source_handle_id}' "
                f"outputs '{edge.source_handle_type}', but "
                f"target '{edge.target_node_id}.{edge.target_handle_id}' "
                f"expects '{edge.target_handle_type}'. "
                f"Connect nodes with matching handle types."
            )

        # Check 4: Source handle ID must exist on source descriptor
        source_node = node_by_id.get(edge.source_node_id)
        if source_node:
            descriptor = NODE_REGISTRY.get(source_node.node_type)
            if descriptor:
                valid_output_ids = {h.id for h in descriptor.outputs}
                if edge.source_handle_id not in valid_output_ids:
                    errors.append(
                        f"Edge '{edge.id}': source handle '{edge.source_handle_id}' "
                        f"does not exist on node type '{source_node.node_type}'. "
                        f"Valid output handles: {sorted(valid_output_ids)}"
                    )

        # Check 5: Target handle ID must exist on target descriptor
        target_node = node_by_id.get(edge.target_node_id)
        if target_node:
            descriptor = NODE_REGISTRY.get(target_node.node_type)
            if descriptor:
                valid_input_ids = {h.id for h in descriptor.inputs}
                if edge.target_handle_id not in valid_input_ids:
                    errors.append(
                        f"Edge '{edge.id}': target handle '{edge.target_handle_id}' "
                        f"does not exist on node type '{target_node.node_type}'. "
                        f"Valid input handles: {sorted(valid_input_ids)}"
                    )

    # Check 6: Parameter value ranges
    for node in graph.nodes:
        descriptor = NODE_REGISTRY.get(node.node_type)
        if not descriptor:
            continue
        for param_schema in descriptor.parameters:
            value = node.parameters.get(param_schema.name)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                continue
            if param_schema.min is not None and value < param_schema.min:
                errors.append(
                    f"Node '{node.id}' ('{node.label}'): parameter '{param_schema.label}' "
                    f"value {value} is below minimum {param_schema.min}."
                )
            if param_schema.max is not None and value > param_schema.max:
                errors.append(
                    f"Node '{node.id}' ('{node.label}'): parameter '{param_schema.label}' "
                    f"value {value} exceeds maximum {param_schema.max}."
                )

    # Check 7: Nodes with required inputs must have at least one incoming edge
    #
    # Many nodes declare multiple required input handles as *alternatives*
    # (e.g., "Raw EEG" OR "Filtered EEG"). A node is valid if at least one
    # of its required inputs has an incoming edge. Only flag an error when a
    # node has required inputs but NONE of them are connected.
    connected_nodes: set[str] = {edge.target_node_id for edge in graph.edges}

    for node in graph.nodes:
        descriptor = NODE_REGISTRY.get(node.node_type)
        if not descriptor:
            continue
        required_inputs = [h for h in descriptor.inputs if h.required]
        if required_inputs and node.id not in connected_nodes:
            handle_names = ", ".join(f"'{h.label}'" for h in required_inputs)
            errors.append(
                f"Node '{node.id}' ('{node.label}'): has required inputs "
                f"({handle_names}) but no incoming connections. "
                f"Connect an upstream node or remove this node."
            )

    return errors
