import { useState } from "react";
import type { RunSnapshot } from "../../hooks/useRunHistory";
import { ConfirmDialog } from "../ui/ConfirmDialog";

interface RunHistoryPanelProps {
  history: RunSnapshot[];
  onRestore: (runId: string) => void;
  onRename: (runId: string, name: string) => void;
  onRemove: (runId: string) => void;
  onCompare: (runIdA: string, runIdB: string) => void;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function RunHistoryPanel({
  history,
  onRestore,
  onRename,
  onRemove,
  onCompare,
}: RunHistoryPanelProps) {
  const [collapsed, setCollapsed] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [compareSet, setCompareSet] = useState<Set<string>>(new Set());
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);

  if (history.length === 0) return null;

  const toggleCompare = (runId: string) => {
    setCompareSet((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) {
        next.delete(runId);
      } else if (next.size < 2) {
        next.add(runId);
      }
      return next;
    });
  };

  const handleCompare = () => {
    const ids = Array.from(compareSet);
    if (ids.length === 2) {
      onCompare(ids[0], ids[1]);
      setCompareSet(new Set());
    }
  };

  return (
    <div>
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full px-3 py-2 text-left flex items-center justify-between hover:bg-slate-800/50 transition-colors"
      >
        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
          {collapsed ? "▸" : "▾"} Run History ({history.length})
        </span>
      </button>

      {!collapsed && (
        <div className="px-3 pb-2 space-y-1.5 max-h-[240px] overflow-y-auto">
          {compareSet.size === 2 && (
            <button
              onClick={handleCompare}
              className="w-full px-2 py-1 text-[10px] rounded bg-purple-600 hover:bg-purple-700 text-white transition-colors mb-1"
            >
              Compare Selected Runs
            </button>
          )}

          {history.map((run) => (
            <div
              key={run.id}
              className={`rounded border p-2 text-[10px] ${
                compareSet.has(run.id)
                  ? "border-purple-500/50 bg-purple-950/20"
                  : "border-slate-700/50 bg-slate-800/30"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                {editingId === run.id ? (
                  <input
                    autoFocus
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onBlur={() => {
                      onRename(run.id, editName);
                      setEditingId(null);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        onRename(run.id, editName);
                        setEditingId(null);
                      }
                    }}
                    className="bg-slate-900 border border-slate-600 rounded px-1 py-0.5 text-[10px] text-slate-200 w-24 outline-none"
                  />
                ) : (
                  <span
                    className="text-slate-300 cursor-pointer hover:text-slate-100"
                    onClick={() => {
                      setEditingId(run.id);
                      setEditName(run.name);
                    }}
                  >
                    {run.name}
                  </span>
                )}
                <span className="text-slate-500">{relativeTime(run.timestamp)}</span>
              </div>

              <div className="flex items-center gap-1 mb-1.5">
                <span className="text-slate-500">
                  {run.nodeCount} nodes
                </span>
                {run.errorCount > 0 ? (
                  <span className="text-red-400">· {run.errorCount} errors</span>
                ) : (
                  <span className="text-emerald-400">· all passed</span>
                )}
              </div>

              <div className="flex gap-1">
                <button
                  onClick={() => onRestore(run.id)}
                  className="flex-1 px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
                >
                  View
                </button>
                <button
                  onClick={() => toggleCompare(run.id)}
                  className={`flex-1 px-1.5 py-0.5 rounded transition-colors ${
                    compareSet.has(run.id)
                      ? "bg-purple-600 text-white"
                      : "bg-slate-700 hover:bg-slate-600 text-slate-300"
                  }`}
                >
                  {compareSet.has(run.id) ? "Selected" : "Compare"}
                </button>
                <button
                  onClick={() => setRemoveTarget(run.id)}
                  className="px-1.5 py-0.5 rounded bg-slate-700 hover:bg-red-800 text-slate-400 hover:text-red-300 transition-colors"
                >
                  ×
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        isOpen={removeTarget !== null}
        title="Remove Run"
        message="Remove this run from history?"
        confirmLabel="Remove"
        destructive
        onConfirm={() => {
          if (removeTarget) onRemove(removeTarget);
          setRemoveTarget(null);
        }}
        onCancel={() => setRemoveTarget(null)}
      />
    </div>
  );
}
