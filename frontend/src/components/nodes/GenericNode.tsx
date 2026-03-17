import { memo, useState, useCallback } from "react";
import { Handle, Position, type NodeProps, useReactFlow } from "@xyflow/react";
import type { NodeResult, NodeData } from "../../types/pipeline";
import { getHandleHex, CATEGORY_COLORS, CATEGORY_STRIPE_HEX } from "../../utils/handleColors";
import { SURFACE } from "../../constants/theme";
import { SessionInfoPanel } from "./SessionInfoPanel";
import { NodeCodePreview, CodeToggleButton, CodePreviewPanel } from "./NodeCodePreview";
import { SaveCustomNodeModal } from "./SaveCustomNodeModal";
import { ImageLightbox } from "./ImageLightbox";
import { ConfirmDialog } from "../ui/ConfirmDialog";

// ---------------------------------------------------------------------------
// StatusBadge — execution result indicator
// ---------------------------------------------------------------------------

function StatusBadge({ result }: { result?: NodeResult | null }) {
  if (!result) return null;
  if (result.status === "success") {
    return (
      <div className="flex items-center gap-1">
        {result.cache_hit && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-sky-900/60 text-sky-300 font-mono flex items-center gap-0.5" title="Served from cache">
            ⚡ cached
          </span>
        )}
        {result.rerun && !result.cache_hit && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-amber-900/60 text-amber-300 font-mono flex items-center gap-0.5" title="Re-executed from this node">
            ↻ re-ran
          </span>
        )}
        {!result.cache_hit && result.execution_time_ms != null && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-slate-700/60 text-slate-400 font-mono" title="Execution time">
            {result.execution_time_ms < 1000
              ? `${Math.round(result.execution_time_ms)}ms`
              : `${(result.execution_time_ms / 1000).toFixed(1)}s`}
          </span>
        )}
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-700 text-cyan-100 font-mono">
          {result.output_type}
        </span>
      </div>
    );
  }
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-800 text-red-100 font-mono">
      error
    </span>
  );
}

// ---------------------------------------------------------------------------
// GenericNode — single component that renders ALL node types
// ---------------------------------------------------------------------------

export const GenericNode = memo(({ id, data, selected }: NodeProps) => {
  const { descriptor, parameters, customLabel, nodeResult, sessionInfo, isRunning, batchMode, onRename } = data as NodeData;
  const { deleteElements } = useReactFlow();
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const closeLightbox = useCallback(() => setLightboxSrc(null), []);
  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [showSave, setShowSave] = useState(false);
  const [saveResult, setSaveResult] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Code preview hook
  const { showCode, codeLoading, codeData, handleToggleCode } = NodeCodePreview({
    nodeType: descriptor?.node_type ?? "",
    parameters: parameters || {},
  });

  if (!descriptor) return null;

  const borderClass = CATEGORY_COLORS[descriptor.category] ?? `border-slate-700/50 bg-[${SURFACE.category}]`;
  const stripeColor = CATEGORY_STRIPE_HEX[descriptor.category] ?? "#6b7280";

  const stateClass = isRunning && !nodeResult
    ? "nf-node-running"
    : nodeResult?.status === "success"
      ? "nf-node-success"
      : nodeResult?.status === "error"
        ? "nf-node-error"
        : "";

  const isSourceNode = descriptor.inputs.length === 0;
  const batchSourceClass = batchMode && isSourceNode ? "border-l-2 border-amber-400/50" : "";

  return (
    <div
      data-presentation-node
      className={`
        min-w-[200px] max-w-[260px] rounded-lg border text-xs shadow-xl relative overflow-hidden
        ${borderClass}
        ${selected ? "ring-2 ring-white ring-offset-1 ring-offset-transparent" : ""}
        ${stateClass}
        ${batchSourceClass}
      `}
    >
      {/* Left accent stripe */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-lg"
        style={{ background: stripeColor }}
      />

      {/* Input handles */}
      {descriptor.inputs.map((handle, i) => (
        <Handle
          key={handle.id}
          type="target"
          position={Position.Left}
          id={handle.id}
          style={{
            top: `${((i + 1) / (descriptor.inputs.length + 1)) * 100}%`,
            background: getHandleHex(handle.type),
            width: 10,
            height: 10,
            border: `2px solid ${SURFACE.node}`,
          }}
          title={`${handle.label} (${handle.type})`}
        />
      ))}

      {/* Header */}
      <div data-presentation-header className="px-3 py-2 pl-4 rounded-t-md flex items-center justify-between gap-2 text-slate-100" style={{ background: SURFACE.nodeHeader }}>
        {editing ? (
          <input
            autoFocus
            className="font-semibold bg-transparent border-b border-white/40 outline-none text-inherit text-xs w-full nodrag"
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={() => {
              setEditing(false);
              onRename?.(id, editValue.trim());
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") { setEditing(false); onRename?.(id, editValue.trim()); }
              if (e.key === "Escape") { setEditing(false); }
            }}
          />
        ) : (
          <span
            className="font-medium text-[13px] truncate cursor-text"
            title="Double-click to rename"
            onDoubleClick={(e) => {
              e.stopPropagation();
              setEditValue(customLabel || descriptor.display_name);
              setEditing(true);
            }}
          >
            {customLabel || descriptor.display_name}
          </span>
        )}
        <div className="flex items-center gap-1 flex-shrink-0">
          <StatusBadge result={nodeResult} />
          <CodeToggleButton showCode={showCode} onClick={handleToggleCode} />
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="ml-1 text-[11px] leading-none w-4 h-4 flex items-center justify-center rounded hover:bg-black/30 opacity-50 hover:opacity-100 transition-opacity nodrag"
            title="Delete node"
          >
            ×
          </button>
        </div>
      </div>

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        isOpen={showDeleteConfirm}
        title="Delete Node"
        message={`Delete "${descriptor.display_name}"?`}
        confirmLabel="Delete"
        destructive
        onConfirm={() => {
          setShowDeleteConfirm(false);
          deleteElements({ nodes: [{ id }] });
        }}
        onCancel={() => setShowDeleteConfirm(false)}
      />

      {/* Body */}
      <div className="px-3 py-2 pl-4 space-y-1">
        {/* Source nodes (no inputs): batch mode indicator */}
        {isSourceNode && batchMode && (
          <div className="text-amber-400/70 italic text-center py-1 text-[10px]">
            Batch mode — files managed in panel
          </div>
        )}

        {/* Source nodes (no inputs): show full session info after load */}
        {isSourceNode && !batchMode && sessionInfo && (
          <SessionInfoPanel info={sessionInfo} />
        )}

        {/* Source nodes (no inputs): show file path if set, prompt if not */}
        {isSourceNode && !batchMode && !sessionInfo && (
          <div className="text-slate-400 italic text-center py-1">
            {parameters.file_path ? (
              <span className="text-slate-300 break-all">
                {String(parameters.file_path).split("/").pop()}
              </span>
            ) : (
              "No file loaded — select node to browse"
            )}
          </div>
        )}

        {/* Plot node: thumbnail (click to expand) + download button */}
        {descriptor.category === "Visualization" && nodeResult?.data && (
          <>
            <img
              src={nodeResult.data}
              alt="Pipeline output"
              className="w-full rounded mt-1 border border-slate-700 cursor-zoom-in"
              title="Click to expand"
              onClick={() => setLightboxSrc(nodeResult.data as string)}
            />
            <button
              onClick={() => {
                const a = document.createElement("a");
                a.href = nodeResult.data as string;
                a.download = `${descriptor.display_name.replace(/\s+/g, "_")}.png`;
                a.click();
              }}
              className="mt-1 w-full text-[10px] bg-slate-700 hover:bg-slate-600 text-slate-300 rounded px-2 py-0.5 transition-colors text-center"
            >
              ↓ Save PNG
            </button>
          </>
        )}

        {/* Inline signal preview for non-visualization nodes */}
        {descriptor.category !== "Visualization" && nodeResult?.preview && (
          <div className="mt-1">
            <img
              src={nodeResult.preview}
              alt="Signal preview"
              className="w-full rounded border border-slate-700 cursor-zoom-in hover:border-blue-500/60 transition-colors"
              title="Click to expand full size"
              onClick={() => setLightboxSrc(nodeResult.preview as string)}
            />
            <div className="text-[9px] text-slate-500 mt-0.5 text-center">
              Click to expand
            </div>
          </div>
        )}

        {/* Lightbox overlay for expanded plot/preview images */}
        {lightboxSrc && (
          <ImageLightbox
            src={lightboxSrc}
            title={customLabel || descriptor.display_name}
            onClose={closeLightbox}
          />
        )}

        {/* Custom Python node: show first 3 lines of code as preview */}
        {descriptor.node_type === "custom_python" && parameters.code && (
          <pre
            className="text-[9px] font-mono text-orange-300/80 bg-slate-900/50 rounded px-1.5 py-1 whitespace-pre-wrap overflow-hidden leading-tight"
            style={{ maxHeight: "3.6em" }}
          >
            {String(parameters.code).split("\n").filter(l => l.trim() && !l.trim().startsWith("#")).slice(0, 3).join("\n") || "# No code"}
          </pre>
        )}

        {/* Show key params as compact summary (non-plot, non-loader nodes) */}
        {descriptor.category !== "Visualization" &&
          descriptor.node_type !== "custom_python" &&
          descriptor.inputs.length > 0 &&
          descriptor.parameters.slice(0, 3).map((param) => {
            const val = parameters[param.name] ?? param.default;
            return (
              <div key={param.name} className="flex justify-between text-slate-400">
                <span className="truncate">{param.label}</span>
                <span className="font-mono text-slate-200 ml-2">
                  {String(val)}
                  {param.unit ? ` ${param.unit}` : ""}
                </span>
              </div>
            );
          })}

        {/* Compound node: double-click hint */}
        {descriptor.category === "Compound" && (
          <div className="text-[10px] text-teal-400/60 text-center italic mt-1">
            Double-click to inspect
          </div>
        )}

        {/* Save to My Nodes — only for custom_python */}
        {descriptor.node_type === "custom_python" && (
          <div className="mt-1">
            {!showSave && !saveResult && (
              <button
                onClick={() => setShowSave(true)}
                className="w-full text-[10px] bg-orange-800/50 hover:bg-orange-700/50 text-orange-200 rounded px-2 py-1 transition-colors nodrag"
              >
                Save to My Nodes
              </button>
            )}
            {showSave && (
              <SaveCustomNodeModal
                code={String(parameters.code ?? "")}
                timeout={Number(parameters.timeout_s ?? 60)}
                onClose={() => {
                  setShowSave(false);
                  setSaveResult("Saved! Reload palette to see it.");
                }}
              />
            )}
            {!showSave && saveResult && (
              <div className={`text-[9px] mt-0.5 px-1 ${saveResult.includes("Saved") ? "text-cyan-400" : "text-red-400"}`}>
                {saveResult}
              </div>
            )}
          </div>
        )}

        {/* Error display */}
        {nodeResult?.status === "error" && nodeResult.error && (
          <div className="mt-1 text-red-300 text-[10px] break-words border border-red-800 rounded px-1.5 py-1 bg-red-950">
            {nodeResult.error}
          </div>
        )}
      </div>

      {/* Code preview panel */}
      <CodePreviewPanel showCode={showCode} codeLoading={codeLoading} codeData={codeData} />

      {/* Output handles */}
      {descriptor.outputs.map((handle, i) => (
        <Handle
          key={handle.id}
          type="source"
          position={Position.Right}
          id={handle.id}
          style={{
            top: `${((i + 1) / (descriptor.outputs.length + 1)) * 100}%`,
            background: getHandleHex(handle.type),
            width: 10,
            height: 10,
            border: `2px solid ${SURFACE.node}`,
          }}
          title={`${handle.label} (${handle.type})`}
        />
      ))}
    </div>
  );
});

GenericNode.displayName = "GenericNode";
