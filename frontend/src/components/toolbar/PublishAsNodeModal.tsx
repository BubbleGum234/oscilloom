import { useState, useMemo } from "react";
import type { Node, Edge } from "@xyflow/react";
import type { NodeTypeDescriptor, NodeData } from "../../types/pipeline";
import { publishCompound } from "../../api/client";
import { serializePipeline } from "../../utils/serializePipeline";

interface PublishAsNodeModalProps {
  nodes: Node[];
  edges: Edge[];
  registry: Record<string, NodeTypeDescriptor>;
  onClose: () => void;
  onPublished: () => void;
}

interface ExposedParam {
  innerNodeId: string;
  paramName: string;
  displayLabel: string;
  checked: boolean;
}

export function PublishAsNodeModal({
  nodes,
  edges,
  registry,
  onClose,
  onPublished,
}: PublishAsNodeModalProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Find entry node (zero incoming edges)
  const entryNodes = useMemo(() => {
    const targetIds = new Set(edges.map((e) => e.target));
    return nodes.filter((n) => !targetIds.has(n.id));
  }, [nodes, edges]);

  // Topological order for output node default
  const topoOrder = useMemo(() => {
    const sourceIds = new Set(edges.map((e) => e.source));
    // Last node = no outgoing edges
    const terminalNodes = nodes.filter((n) => !sourceIds.has(n.id));
    return terminalNodes;
  }, [nodes, edges]);

  const [outputNodeId, setOutputNodeId] = useState(
    topoOrder[0]?.id ?? nodes[nodes.length - 1]?.id ?? ""
  );

  // Build exposable params list
  const allParams = useMemo(() => {
    const result: ExposedParam[] = [];
    for (const node of nodes) {
      const nodeData = node.data as NodeData;
      const nodeType = nodeData.nodeType;
      const descriptor = nodeType ? registry[nodeType] : null;
      if (!descriptor) continue;
      for (const param of descriptor.parameters) {
        if (param.hidden) continue;
        result.push({
          innerNodeId: node.id,
          paramName: param.name,
          displayLabel: `${descriptor.display_name}: ${param.label}`,
          checked: false,
        });
      }
    }
    return result;
  }, [nodes, registry]);

  const [exposedParams, setExposedParams] = useState(allParams);

  const toggleParam = (idx: number) => {
    setExposedParams((prev) =>
      prev.map((p, i) => (i === idx ? { ...p, checked: !p.checked } : p))
    );
  };

  const compoundId = useMemo(() => {
    return "c_" + name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  }, [name]);

  const handlePublish = async () => {
    if (!name.trim()) {
      setError("Name is required.");
      return;
    }
    if (nodes.length === 0) {
      setError("Canvas is empty — nothing to publish.");
      return;
    }
    if (entryNodes.length !== 1) {
      setError(
        `Expected exactly 1 entry node (node with no incoming edges), found ${entryNodes.length}.`
      );
      return;
    }

    setPublishing(true);
    setError(null);

    try {
      const pipeline = serializePipeline(nodes, edges, name);
      const selected = exposedParams.filter((p) => p.checked);

      await publishCompound({
        compound_id: compoundId,
        display_name: name.trim(),
        description: description.trim() || `Compound node: ${name.trim()}`,
        tags: ["compound"],
        sub_graph: pipeline as unknown as Record<string, unknown>,
        entry_node_id: entryNodes[0].id,
        output_node_id: outputNodeId,
        exposed_params: selected.map((p) => ({
          inner_node_id: p.innerNodeId,
          param_name: p.paramName,
          display_label: p.displayLabel,
        })),
      });

      onPublished();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPublishing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl w-[480px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="px-5 py-3 border-b border-slate-700 flex items-center justify-between">
          <h2 className="text-slate-100 font-semibold text-sm">Publish as Node</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-300 text-lg leading-none"
          >
            x
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-slate-400 text-xs mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Clean & Filter"
              className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
            />
            {name.trim() && (
              <div className="text-slate-600 text-[10px] mt-0.5">
                ID: {compoundId}
              </div>
            )}
          </div>

          {/* Description */}
          <div>
            <label className="block text-slate-400 text-xs mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this compound node do?"
              rows={2}
              className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400 resize-none"
            />
          </div>

          {/* Entry node info */}
          <div>
            <label className="block text-slate-400 text-xs mb-1">Entry Node</label>
            <div className="text-slate-300 text-xs bg-slate-800 rounded px-3 py-1.5 border border-slate-700">
              {entryNodes.length === 1
                ? `${(entryNodes[0].data as NodeData).label || entryNodes[0].id} (auto-detected)`
                : entryNodes.length === 0
                ? "No entry node found"
                : `${entryNodes.length} entry nodes found (must be exactly 1)`}
            </div>
          </div>

          {/* Output node selector */}
          <div>
            <label className="block text-slate-400 text-xs mb-1">Output Node</label>
            <select
              value={outputNodeId}
              onChange={(e) => setOutputNodeId(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
            >
              {nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {(n.data as NodeData).label || n.id}
                </option>
              ))}
            </select>
          </div>

          {/* Exposed parameters */}
          {allParams.length > 0 && (
            <div>
              <label className="block text-slate-400 text-xs mb-1">
                Exposed Parameters ({exposedParams.filter((p) => p.checked).length} selected)
              </label>
              <div className="max-h-40 overflow-y-auto bg-slate-800 border border-slate-700 rounded p-2 space-y-1">
                {exposedParams.map((p, i) => (
                  <label
                    key={`${p.innerNodeId}__${p.paramName}`}
                    className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer hover:text-slate-100"
                  >
                    <input
                      type="checkbox"
                      checked={p.checked}
                      onChange={() => toggleParam(i)}
                      className="rounded border-slate-600"
                    />
                    {p.displayLabel}
                  </label>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="text-red-400 text-xs bg-red-950/50 border border-red-900 rounded px-3 py-2">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-700 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-4 py-1.5 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handlePublish}
            disabled={publishing || !name.trim()}
            className="bg-teal-700 hover:bg-teal-600 disabled:bg-slate-800 disabled:text-slate-600 text-white text-xs rounded px-4 py-1.5 transition-colors font-medium"
          >
            {publishing ? "Publishing..." : "Publish"}
          </button>
        </div>
      </div>
    </div>
  );
}
