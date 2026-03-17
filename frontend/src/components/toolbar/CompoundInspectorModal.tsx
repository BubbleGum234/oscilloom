import { useState, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  BackgroundVariant,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeProps,
  ReactFlowProvider,
} from "@xyflow/react";
import type { NodeTypeDescriptor } from "../../types/pipeline";
import { CATEGORY_COLORS, CATEGORY_HEADER_COLORS, getHandleHex } from "../../utils/handleColors";
import { getCompound } from "../../api/client";
import { SURFACE, BORDER } from "../../constants/theme";

interface CompoundInspectorModalProps {
  compoundId: string;
  displayName: string;
  registry: Record<string, NodeTypeDescriptor>;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// InspectorNode — lightweight read-only node for the mini-canvas
// ---------------------------------------------------------------------------

interface InspectorNodeData {
  label: string;
  category: string;
  isEntry: boolean;
  isOutput: boolean;
  descriptor: NodeTypeDescriptor | null;
}

const InspectorNode = ({ data }: NodeProps) => {
  const { label, category, isEntry, isOutput, descriptor } =
    data as InspectorNodeData;

  const borderClass = CATEGORY_COLORS[category] ?? "border-gray-600 bg-gray-900";
  const headerClass = CATEGORY_HEADER_COLORS[category] ?? "bg-gray-700 text-gray-100";

  let ringClass = "";
  if (isEntry) ringClass = "ring-2 ring-cyan-400 ring-offset-1 ring-offset-transparent";
  else if (isOutput) ringClass = "ring-2 ring-orange-400 ring-offset-1 ring-offset-transparent";

  return (
    <div
      className={`min-w-[160px] max-w-[220px] rounded-lg border-2 text-xs shadow-lg ${borderClass} ${ringClass}`}
    >
      {/* Input handles */}
      {descriptor?.inputs.map((handle, i) => (
        <Handle
          key={handle.id}
          type="target"
          position={Position.Left}
          id={handle.id}
          style={{
            top: `${((i + 1) / ((descriptor?.inputs.length ?? 0) + 1)) * 100}%`,
            background: getHandleHex(handle.type),
            width: 8,
            height: 8,
            border: `2px solid ${SURFACE.node}`,
          }}
        />
      ))}

      {/* Header */}
      <div className={`px-2 py-1.5 rounded-t-md flex items-center gap-2 ${headerClass}`}>
        <span className="font-semibold truncate text-[11px]">{label}</span>
        {isEntry && (
          <span className="text-[9px] bg-cyan-600 text-cyan-100 px-1 rounded flex-shrink-0">
            IN
          </span>
        )}
        {isOutput && (
          <span className="text-[9px] bg-orange-600 text-orange-100 px-1 rounded flex-shrink-0">
            OUT
          </span>
        )}
      </div>

      {/* Body — just show category */}
      <div className="px-2 py-1 text-[10px] text-slate-400">
        {category}
      </div>

      {/* Output handles */}
      {descriptor?.outputs.map((handle, i) => (
        <Handle
          key={handle.id}
          type="source"
          position={Position.Right}
          id={handle.id}
          style={{
            top: `${((i + 1) / ((descriptor?.outputs.length ?? 0) + 1)) * 100}%`,
            background: getHandleHex(handle.type),
            width: 8,
            height: 8,
            border: `2px solid ${SURFACE.node}`,
          }}
        />
      ))}
    </div>
  );
};

const inspectorNodeTypes = { inspectorNode: InspectorNode };

// ---------------------------------------------------------------------------
// Mini-canvas wrapper (needs its own ReactFlowProvider)
// ---------------------------------------------------------------------------

function MiniCanvas({
  nodes,
  edges,
}: {
  nodes: Node[];
  edges: Edge[];
}) {
  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={inspectorNodeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag
        zoomOnScroll
        fitView
        fitViewOptions={{ padding: 0.3 }}
        colorMode="dark"
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} color={BORDER.subtle} gap={16} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </ReactFlowProvider>
  );
}

// ---------------------------------------------------------------------------
// CompoundInspectorModal
// ---------------------------------------------------------------------------

export function CompoundInspectorModal({
  compoundId,
  displayName,
  registry,
  onClose,
}: CompoundInspectorModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [definition, setDefinition] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    getCompound(compoundId)
      .then((defn) => {
        if (!cancelled) setDefinition(defn);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [compoundId]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Parse sub_graph into React Flow nodes and edges
  const { flowNodes, flowEdges, entryNodeId: _entryNodeId, outputNodeId: _outputNodeId, exposedParams, description } =
    useMemo(() => {
      if (!definition) {
        return {
          flowNodes: [] as Node[],
          flowEdges: [] as Edge[],
          entryNodeId: "",
          outputNodeId: "",
          exposedParams: [] as Array<{
            inner_node_id: string;
            param_name: string;
            display_label: string;
          }>,
          description: "",
        };
      }

      const subGraph = definition.sub_graph as {
        nodes: Array<{
          id: string;
          node_type: string;
          label: string;
          position: { x: number; y: number };
        }>;
        edges: Array<{
          id: string;
          source_node_id: string;
          source_handle_id: string;
          target_node_id: string;
          target_handle_id: string;
        }>;
      };

      const entryId = (definition.entry_node_id as string) || "";
      const outputId = (definition.output_node_id as string) || "";
      const exposed = (definition.exposed_params as Array<{
        inner_node_id: string;
        param_name: string;
        display_label: string;
      }>) || [];

      const nodes: Node[] = (subGraph?.nodes || []).map((n) => {
        const desc = registry[n.node_type] ?? null;
        return {
          id: n.id,
          type: "inspectorNode",
          position: n.position || { x: 0, y: 0 },
          data: {
            label: desc?.display_name ?? n.label ?? n.node_type,
            category: desc?.category ?? "Unknown",
            isEntry: n.id === entryId,
            isOutput: n.id === outputId,
            descriptor: desc,
          } satisfies InspectorNodeData,
        };
      });

      const edges: Edge[] = (subGraph?.edges || []).map((e) => ({
        id: e.id,
        source: e.source_node_id,
        sourceHandle: e.source_handle_id,
        target: e.target_node_id,
        targetHandle: e.target_handle_id,
        type: "smoothstep",
        animated: false,
      }));

      return {
        flowNodes: nodes,
        flowEdges: edges,
        entryNodeId: entryId,
        outputNodeId: outputId,
        exposedParams: exposed,
        description: (definition.description as string) || "",
      };
    }, [definition, registry]);

  // Look up compound descriptor for input/output handle types
  const compoundDescriptor = registry[compoundId];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-[700px] max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <h2 className="text-slate-100 font-semibold text-sm">{displayName}</h2>
            <span className="text-[10px] bg-teal-800 text-teal-200 px-1.5 py-0.5 rounded font-mono">
              {compoundId}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg leading-none"
          >
            x
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
          {loading && (
            <div className="flex-1 flex items-center justify-center text-slate-400 text-sm py-12">
              Loading compound definition...
            </div>
          )}

          {error && (
            <div className="px-5 py-4">
              <div className="text-red-400 text-xs bg-red-950 border border-red-800 rounded px-3 py-2">
                {error}
              </div>
            </div>
          )}

          {!loading && !error && definition && (
            <>
              {/* Metadata */}
              <div className="px-5 py-3 space-y-2 border-b border-slate-800">
                {description && (
                  <p className="text-slate-400 text-xs">{description}</p>
                )}
                <div className="flex gap-4 text-[10px]">
                  {compoundDescriptor?.inputs?.[0] && (
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500">Input:</span>
                      <span
                        className="font-mono px-1 rounded"
                        style={{
                          color: getHandleHex(compoundDescriptor.inputs[0].type),
                        }}
                      >
                        {compoundDescriptor.inputs[0].type}
                      </span>
                    </div>
                  )}
                  {compoundDescriptor?.outputs?.[0] && (
                    <div className="flex items-center gap-1">
                      <span className="text-slate-500">Output:</span>
                      <span
                        className="font-mono px-1 rounded"
                        style={{
                          color: getHandleHex(compoundDescriptor.outputs[0].type),
                        }}
                      >
                        {compoundDescriptor.outputs[0].type}
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-1">
                    <span className="text-slate-500">Nodes:</span>
                    <span className="text-slate-300 font-mono">{flowNodes.length}</span>
                  </div>
                </div>
              </div>

              {/* Mini-canvas */}
              <div className="h-[320px] border-b border-slate-800">
                <MiniCanvas nodes={flowNodes} edges={flowEdges} />
              </div>

              {/* Legend */}
              <div className="px-5 py-2 flex gap-4 text-[10px] text-slate-500 border-b border-slate-800">
                <div className="flex items-center gap-1">
                  <span className="inline-block w-2.5 h-2.5 rounded border-2 border-cyan-400" />
                  Entry node
                </div>
                <div className="flex items-center gap-1">
                  <span className="inline-block w-2.5 h-2.5 rounded border-2 border-orange-400" />
                  Output node
                </div>
              </div>

              {/* Exposed parameters */}
              {exposedParams.length > 0 && (
                <div className="px-5 py-3">
                  <h3 className="text-slate-400 text-[10px] uppercase tracking-wider mb-2">
                    Exposed Parameters ({exposedParams.length})
                  </h3>
                  <div className="bg-slate-800 rounded border border-slate-700 divide-y divide-slate-700">
                    {exposedParams.map((ep) => {
                      const innerDesc = flowNodes.find((n) => n.id === ep.inner_node_id);
                      const innerLabel =
                        (innerDesc?.data as InspectorNodeData | undefined)?.label ??
                        ep.inner_node_id;
                      return (
                        <div
                          key={`${ep.inner_node_id}__${ep.param_name}`}
                          className="flex items-center justify-between px-3 py-1.5 text-xs"
                        >
                          <span className="text-slate-200">{ep.display_label}</span>
                          <span className="text-slate-500 text-[10px] font-mono">
                            {innerLabel} / {ep.param_name}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {exposedParams.length === 0 && (
                <div className="px-5 py-3">
                  <p className="text-slate-500 text-xs italic">
                    No parameters exposed. This compound runs with fixed internal settings.
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-700 flex justify-end">
          <button
            onClick={onClose}
            className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-4 py-1.5 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
