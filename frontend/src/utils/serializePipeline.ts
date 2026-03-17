import type { Node, Edge } from "@xyflow/react";
import type { PipelineGraph, NodeTypeDescriptor, NodeData } from "../types/pipeline";

/**
 * Converts React Flow node/edge state into the backend PipelineGraph schema.
 * This is a pure transformation — no side effects.
 */
export function serializePipeline(
  nodes: Node[],
  edges: Edge[],
  name = "My Pipeline",
  description = ""
): PipelineGraph {
  return {
    metadata: {
      name,
      description,
      created_by: "human",
      schema_version: "1.0",
    },
    nodes: nodes.map((n) => {
      const nodeData = n.data as NodeData;
      const descriptor = nodeData.descriptor;
      if (!descriptor?.node_type) {
        throw new Error(
          `Node "${n.id}" has no descriptor. It may have been deleted from the registry.`
        );
      }
      return {
        id: n.id,
        node_type: descriptor.node_type,
        label: nodeData.label || descriptor?.display_name || n.id,
        parameters: nodeData.parameters ?? {},
        position: n.position,
      };
    }),
    edges: edges.map((e) => {
      const sourceNode = nodes.find((n) => n.id === e.source);
      const targetNode = nodes.find((n) => n.id === e.target);
      const sourceDesc = (sourceNode?.data as NodeData | undefined)?.descriptor;
      const targetDesc = (targetNode?.data as NodeData | undefined)?.descriptor;
      const sourceHandle = sourceDesc?.outputs.find((h) => h.id === e.sourceHandle);
      const targetHandle = targetDesc?.inputs.find((h) => h.id === e.targetHandle);
      return {
        id: e.id,
        source_node_id: e.source,
        source_handle_id: e.sourceHandle ?? "",
        source_handle_type: sourceHandle?.type ?? "",
        target_node_id: e.target,
        target_handle_id: e.targetHandle ?? "",
        target_handle_type: targetHandle?.type ?? "",
      };
    }),
  };
}

/**
 * Converts a backend PipelineGraph into React Flow nodes and edges.
 * Used when loading a saved pipeline or applying AI-generated pipelines.
 */
export function deserializePipeline(
  graph: PipelineGraph,
  registry: Record<string, NodeTypeDescriptor>
): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = graph.nodes.map((n) => ({
    id: n.id,
    type: "genericNode",
    position: n.position,
    data: {
      descriptor: registry[n.node_type],
      parameters: { ...n.parameters },
      label: n.label,
      nodeResult: null,
    },
  }));

  const edges: Edge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.source_node_id,
    sourceHandle: e.source_handle_id,
    target: e.target_node_id,
    targetHandle: e.target_handle_id,
    type: "smoothstep",
    animated: false,
  }));

  return { nodes, edges };
}
