import { useCallback, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type Connection,
  addEdge,
  useReactFlow,
} from "@xyflow/react";
import { GenericNode } from "../nodes/GenericNode";
import type { NodeRegistry, NodeTypeDescriptor } from "../../types/pipeline";
import { getNextNodeId } from "../../utils/nodeId";
import { SURFACE, BORDER } from "../../constants/theme";

const nodeTypes = { genericNode: GenericNode };

function validateConnection(
  connection: Connection,
  nodes: Node[]
): { valid: boolean; reason?: string } {
  const sourceNode = nodes.find((n) => n.id === connection.source);
  const targetNode = nodes.find((n) => n.id === connection.target);
  if (!sourceNode || !targetNode) return { valid: false, reason: "Unknown node" };

  const sourceDesc = sourceNode.data.descriptor as NodeTypeDescriptor | undefined;
  const targetDesc = targetNode.data.descriptor as NodeTypeDescriptor | undefined;
  if (!sourceDesc || !targetDesc) return { valid: false, reason: "Missing descriptor" };

  const srcHandle = sourceDesc.outputs.find((h) => h.id === connection.sourceHandle);
  const tgtHandle = targetDesc.inputs.find((h) => h.id === connection.targetHandle);
  if (!srcHandle || !tgtHandle) return { valid: false, reason: "Unknown handle" };

  if (srcHandle.type !== tgtHandle.type) {
    return {
      valid: false,
      reason: `Cannot connect ${srcHandle.type} \u2192 ${tgtHandle.type}`,
    };
  }

  return { valid: true };
}

interface CanvasPaneProps {
  nodes: Node[];
  edges: Edge[];
  registry: NodeRegistry;
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onNodesSet: (nodes: Node[]) => void;
  onEdgesSet: (edges: Edge[]) => void;
  onNodeSelect: (node: Node | null) => void;
  onSelectionCountChange?: (count: number) => void;
  onNodeDoubleClick?: (node: Node) => void;
  onConnectionRejected?: (reason: string) => void;
}

export function CanvasPane({
  nodes,
  edges,
  registry,
  onNodesChange,
  onEdgesChange,
  onNodesSet,
  onEdgesSet,
  onNodeSelect,
  onSelectionCountChange,
  onNodeDoubleClick,
  onConnectionRejected,
}: CanvasPaneProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const onConnect = useCallback(
    (connection: Connection) => {
      const result = validateConnection(connection, nodes);
      if (!result.valid) {
        onConnectionRejected?.(result.reason ?? "Invalid connection");
        return;
      }
      onEdgesSet(
        addEdge({ ...connection, type: "smoothstep", animated: false }, edges)
      );
    },
    [nodes, edges, onEdgesSet, onConnectionRejected]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const nodeType = event.dataTransfer.getData("application/oscilloom-node");
      if (!nodeType || !registry[nodeType]) return;

      const descriptor = registry[nodeType];
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const id = getNextNodeId();
      const defaultParams: Record<string, unknown> = {};
      for (const p of descriptor.parameters) {
        defaultParams[p.name] = p.default;
      }

      const newNode: Node = {
        id,
        type: "genericNode",
        position,
        data: {
          descriptor,
          parameters: defaultParams,
          label: descriptor.display_name,
          nodeResult: null,
          sessionInfo: null,
        },
      };

      onNodesSet([...nodes, newNode]);
    },
    [registry, nodes, screenToFlowPosition, onNodesSet]
  );

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodeDoubleClick={(_event, node) => onNodeDoubleClick?.(node)}
        onSelectionChange={({ nodes: selected }) => {
          onNodeSelect(selected.length >= 1 ? selected[selected.length - 1] : null);
          onSelectionCountChange?.(selected.length);
        }}
        fitView
        deleteKeyCode={["Delete", "Backspace"]}
        colorMode="dark"
      >
        <Background variant={BackgroundVariant.Dots} color={SURFACE.canvas} gap={20} />
        <Controls />
        <MiniMap
          style={{ background: SURFACE.minimap, border: `1px solid ${BORDER.subtle}` }}
          nodeColor={BORDER.default}
        />
      </ReactFlow>
    </div>
  );
}
