import { useState, useRef, useCallback, useEffect } from "react";
import type { Node } from "@xyflow/react";
import type { NodeResult, AuditLogEntry, SessionInfo } from "../../types/pipeline";
import type { RunSnapshot } from "../../hooks/useRunHistory";
import { NodeParameterPanel } from "./NodeParameterPanel";
import { NodeOutputInspector } from "./NodeOutputInspector";
import { RunHistoryPanel } from "./RunHistoryPanel";
import { AuditLogPanel } from "./AuditLogPanel";
import { SURFACE } from "../../constants/theme";
import { STORAGE_KEYS } from "../../constants/storageKeys";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BottomTab = "output" | "history" | "log";

const STORAGE_KEY = STORAGE_KEYS.RIGHT_PANEL_SPLIT;
const DEFAULT_SPLIT = 0.45; // 45% top, 55% bottom
const MIN_TOP_PX = 120;
const MIN_BOTTOM_PX = 120;

interface RightPanelProps {
  // Parameters panel
  selectedNode: Node | null;
  onParamChange: (nodeId: string, paramName: string, value: unknown) => void;
  onFileLoad: (file: File, nodeId: string) => void;
  onDownloadFif?: () => void;
  selectionCount?: number;
  sessionInfo?: SessionInfo | null;
  // Output inspector
  nodeResult: NodeResult | null;
  nodeLabel?: string;
  nodeId?: string | null;
  sessionId?: string | null;
  onOpenBrowser?: (nodeId: string) => void | Promise<void>;
  onExport?: (nodeId: string, format: string, label: string) => void;
  onRerunFrom?: (nodeId: string) => void;
  onSessionInfoUpdate?: (info: Record<string, unknown>) => void;
  isRunning?: boolean;
  // Run history
  runHistory: RunSnapshot[];
  onHistoryRestore: (runId: string) => void;
  onHistoryRename: (runId: string, name: string) => void;
  onHistoryRemove: (runId: string) => void;
  onHistoryCompare: (runIdA: string, runIdB: string) => void;
  // Audit log
  auditLog: AuditLogEntry[];
  onAuditClear: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function RightPanel({
  selectedNode,
  onParamChange,
  onFileLoad,
  onDownloadFif,
  selectionCount,
  sessionInfo,
  nodeResult,
  nodeLabel,
  nodeId,
  sessionId,
  onOpenBrowser,
  onExport,
  onRerunFrom,
  onSessionInfoUpdate,
  isRunning,
  runHistory,
  onHistoryRestore,
  onHistoryRename,
  onHistoryRemove,
  onHistoryCompare,
  auditLog,
  onAuditClear,
}: RightPanelProps) {
  // ── Split ratio (persisted) ─────────────────────────────────────────────
  const [splitRatio, setSplitRatio] = useState<number>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const v = parseFloat(saved);
        if (v >= 0.15 && v <= 0.85) return v;
      }
    } catch { /* ignore */ }
    return DEFAULT_SPLIT;
  });

  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  // Persist split ratio
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, String(splitRatio));
    } catch { /* ignore */ }
  }, [splitRatio]);

  // ── Drag handling ───────────────────────────────────────────────────────
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const totalH = rect.height;
      const y = e.clientY - rect.top;

      // Enforce min heights
      const topPx = Math.max(MIN_TOP_PX, Math.min(y, totalH - MIN_BOTTOM_PX));
      setSplitRatio(topPx / totalH);
    };

    const onMouseUp = () => {
      if (dragging.current) {
        dragging.current = false;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      }
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  // ── Tab state ───────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<BottomTab>("output");

  // Auto-switch to output tab when a new result arrives
  const prevResultRef = useRef(nodeResult);
  useEffect(() => {
    if (nodeResult && nodeResult !== prevResultRef.current) {
      setActiveTab("output");
    }
    prevResultRef.current = nodeResult;
  }, [nodeResult]);

  // Badge counts
  const historyCount = runHistory.length;
  const logCount = auditLog.length;

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex flex-col border-l border-slate-700"
      style={{ background: SURFACE.panel }}
    >
      {/* ── TOP ZONE: Parameters ──────────────────────────────────────────── */}
      <div
        className="overflow-y-auto min-h-0"
        style={{ height: `${splitRatio * 100}%` }}
      >
        <NodeParameterPanel
          node={selectedNode}
          onParamChange={onParamChange}
          onFileLoad={onFileLoad}
          onDownloadFif={onDownloadFif}
          selectionCount={selectionCount}
          sessionInfo={sessionInfo}
        />
      </div>

      {/* ── DIVIDER ───────────────────────────────────────────────────────── */}
      <div
        onMouseDown={onMouseDown}
        className="h-1 flex-shrink-0 cursor-row-resize group relative"
      >
        <div className="absolute inset-x-0 -top-1 -bottom-1" />
        <div className="h-full bg-slate-700 group-hover:bg-slate-500 transition-colors" />
      </div>

      {/* ── BOTTOM ZONE: Tabs ─────────────────────────────────────────────── */}
      <div
        className="flex flex-col min-h-0"
        style={{ height: `${(1 - splitRatio) * 100}%` }}
      >
        {/* Tab bar */}
        <div className="flex items-center border-b border-slate-700 flex-shrink-0" style={{ background: SURFACE.panelHeader }}>
          <TabButton
            active={activeTab === "output"}
            onClick={() => setActiveTab("output")}
            label="Output"
          />
          <TabButton
            active={activeTab === "history"}
            onClick={() => setActiveTab("history")}
            label="History"
            badge={historyCount > 0 ? historyCount : undefined}
          />
          <TabButton
            active={activeTab === "log"}
            onClick={() => setActiveTab("log")}
            label="Log"
            badge={logCount > 0 ? logCount : undefined}
          />
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {activeTab === "output" && (
            <NodeOutputInspector
              nodeResult={nodeId ? nodeResult : null}
              nodeLabel={nodeLabel}
              nodeId={nodeId ?? undefined}
              sessionId={sessionId}
              onOpenBrowser={onOpenBrowser}
              onExport={onExport}
              onRerunFrom={onRerunFrom}
              onSessionInfoUpdate={onSessionInfoUpdate}
              isRunning={isRunning}
            />
          )}
          {activeTab === "history" && (
            <RunHistoryPanel
              history={runHistory}
              onRestore={onHistoryRestore}
              onRename={onHistoryRename}
              onRemove={onHistoryRemove}
              onCompare={onHistoryCompare}
            />
          )}
          {activeTab === "log" && (
            <AuditLogPanel log={auditLog} onClear={onAuditClear} />
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab Button
// ---------------------------------------------------------------------------

function TabButton({
  active,
  onClick,
  label,
  badge,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  badge?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`
        relative px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide transition-colors
        ${active
          ? "text-slate-200"
          : "text-slate-500 hover:text-slate-300"
        }
      `}
    >
      {label}
      {badge !== undefined && badge > 0 && (
        <span className="ml-1 px-1 py-px text-[9px] rounded-full bg-slate-700 text-slate-400 font-mono">
          {badge > 99 ? "99+" : badge}
        </span>
      )}
      {/* Active indicator */}
      {active && (
        <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-emerald-500 rounded-full" />
      )}
    </button>
  );
}
