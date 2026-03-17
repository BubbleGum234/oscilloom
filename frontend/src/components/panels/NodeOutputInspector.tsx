import { useState, useEffect, useRef, useCallback } from "react";
import type { NodeResult } from "../../types/pipeline";
import { checkBrowserStatus, syncBrowserAnnotations } from "../../api/client";

// Format map: which export formats are available for each output kind
const FORMAT_MAP: Record<string, string[]> = {
  raw: ["fif", "csv"],
  epochs: ["fif", "csv", "mat", "npz"],
  evoked: ["csv", "mat", "npz"],
  psd: ["csv", "npz", "mat"],
  tfr: ["npz", "mat"],
  connectivity: ["csv", "npz", "mat"],
  metrics: ["csv", "json"],
  plot: ["png"],
  array: ["csv", "npz", "mat"],
  scalar: ["json"],
};

interface NodeOutputInspectorProps {
  nodeResult: NodeResult | null;
  nodeLabel?: string;
  nodeId?: string;
  sessionId?: string | null;
  onOpenBrowser?: (nodeId: string) => void | Promise<void>;
  onExport?: (nodeId: string, format: string, label: string) => void;
  onRerunFrom?: (nodeId: string) => void;
  onSessionInfoUpdate?: (info: Record<string, unknown>) => void;
  isRunning?: boolean;
}

export function NodeOutputInspector({
  nodeResult,
  nodeLabel,
  nodeId,
  sessionId,
  onOpenBrowser,
  onExport,
  onRerunFrom,
  onSessionInfoUpdate,
  isRunning,
}: NodeOutputInspectorProps) {
  const [exportOpen, setExportOpen] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const [showTraceback, setShowTraceback] = useState(false);
  const [syncStatus, setSyncStatus] = useState<"idle" | "syncing" | "done" | "error">("idle");
  const [syncResult, setSyncResult] = useState<string | null>(null);
  // Track whether the browser has been opened and closed (annotations may exist)
  const [browserWasClosed, setBrowserWasClosed] = useState(false);

  // All hooks must be called before any early return (Rules of Hooks).
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Clean up polling on unmount or when selected node changes
  useEffect(() => {
    stopPolling();
    setBrowserOpen(false);
    setBrowserWasClosed(false);
    setSyncStatus("idle");
    setSyncResult(null);
    return stopPolling;
  }, [nodeId, stopPolling]);

  if (!nodeId) {
    return (
      <div className="px-3 py-4 text-xs text-slate-500 text-center">
        Select a node to inspect its output
      </div>
    );
  }

  if (!nodeResult) {
    return (
      <div className="px-3 py-4">
        <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Output Inspector
        </div>
        <div className="text-xs text-slate-500 text-center py-3">
          Run pipeline to see output
        </div>
      </div>
    );
  }

  const summary = nodeResult.summary;
  const kind = summary?.kind ?? "unknown";
  const formats = FORMAT_MAP[kind] ?? [];

  const handleBrowserClick = async () => {
    if (!nodeId || !onOpenBrowser) return;
    setBrowserOpen(true);
    try {
      await onOpenBrowser(nodeId);
      // Start polling to detect when the browser window closes
      if (sessionId) {
        stopPolling();
        pollRef.current = setInterval(async () => {
          const isOpen = await checkBrowserStatus(sessionId, nodeId);
          if (!isOpen) {
            setBrowserOpen(false);
            setBrowserWasClosed(true);
            stopPolling();
          }
        }, 2000);
      }
    } catch {
      setBrowserOpen(false);
    }
  };

  const handleSyncAnnotations = async () => {
    if (!nodeId || !sessionId) return;
    setSyncStatus("syncing");
    setSyncResult(null);
    try {
      const result = await syncBrowserAnnotations(sessionId, nodeId);
      setSyncStatus("done");
      setSyncResult(`Synced ${result.n_annotations} annotation(s) — re-running pipeline…`);
      setBrowserWasClosed(false);

      // Refresh session info so annotation chips update in the UI
      if (result.session_info && onSessionInfoUpdate) {
        onSessionInfoUpdate(result.session_info);
      }

      // Auto re-run from this node so downstream nodes pick up the annotations
      if (onRerunFrom) {
        onRerunFrom(nodeId);
      }
    } catch (err) {
      setSyncStatus("error");
      setSyncResult(err instanceof Error ? err.message : "Sync failed");
    }
  };

  return (
    <div className="px-3 py-3">
      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-2 flex items-center justify-between">
        <span>Output Inspector</span>
        <div className="flex items-center gap-1">
          {nodeResult.cache_hit !== undefined && (
            <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono ${
              nodeResult.cache_hit
                ? "bg-sky-900/50 text-sky-300"
                : "bg-slate-700 text-slate-400"
            }`}>
              {nodeResult.cache_hit ? "⚡ cached" : (
                nodeResult.execution_time_ms != null
                  ? (nodeResult.execution_time_ms < 1000
                      ? `${Math.round(nodeResult.execution_time_ms)}ms`
                      : `${(nodeResult.execution_time_ms / 1000).toFixed(1)}s`)
                  : "executed"
              )}
            </span>
          )}
          {nodeResult.rerun !== undefined && (
            <span className={`px-1.5 py-0.5 rounded text-[9px] ${
              nodeResult.rerun ? "bg-amber-800/50 text-amber-300" : "bg-slate-700 text-slate-400"
            }`}>
              {nodeResult.rerun ? "re-ran" : "cached"}
            </span>
          )}
        </div>
      </div>

      {/* Status badge */}
      {nodeResult.status === "error" ? (
        <div className="mb-2">
          <div className="bg-red-950/50 border border-red-900/50 rounded p-2">
            <div className="text-[10px] font-mono text-red-400 mb-1">
              {summary?.python_type ?? "Error"}: {summary?.message ?? nodeResult.error}
            </div>
            {summary?.traceback_preview && (
              <>
                <button
                  onClick={() => setShowTraceback(!showTraceback)}
                  className="text-[9px] text-red-500 hover:text-red-400 underline"
                >
                  {showTraceback ? "Hide traceback" : "Show traceback"}
                </button>
                {showTraceback && (
                  <pre className="text-[9px] text-red-500/70 mt-1 whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                    {summary.traceback_preview}
                  </pre>
                )}
              </>
            )}
          </div>
        </div>
      ) : (
        <>
          {/* Data summary based on kind */}
          <div className="space-y-1 mb-2">
            {kind === "raw" && (
              <KVGrid entries={[
                ["Channels", `${summary!.n_channels}`],
                ["Sampling Rate", `${summary!.sfreq} Hz`],
                ["Duration", `${summary!.duration_s} s`],
                ["Shape", `${summary!.shape?.[0]} × ${summary!.shape?.[1]}`],
                ["Filters", `${summary!.highpass}–${summary!.lowpass} Hz`],
                ...(summary!.bads && summary!.bads.length > 0
                  ? [["Bad Channels", summary!.bads.join(", ")] as [string, string]]
                  : []),
              ]} />
            )}

            {kind === "epochs" && (
              <>
                <KVGrid entries={[
                  ["Epochs", `${summary!.n_epochs}`],
                  ...(summary!.n_dropped != null && summary!.n_dropped > 0
                    ? [["Dropped", `${summary!.n_dropped} (${((summary!.n_dropped / ((summary!.n_epochs ?? 0) + summary!.n_dropped)) * 100).toFixed(1)}%)`] as [string, string]]
                    : []),
                  ["Channels", `${summary!.n_channels}`],
                  ["Time Range", `${summary!.tmin} – ${summary!.tmax} s`],
                  ["Shape", summary!.shape?.join(" × ") ?? ""],
                ]} />
                {summary!.event_counts && Object.keys(summary!.event_counts).length > 0 && (
                  <div className="mt-1">
                    <div className="text-[9px] text-slate-500 mb-0.5">Events:</div>
                    <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
                      {Object.entries(summary!.event_counts).map(([k, v]) => (
                        <div key={k} className="flex justify-between text-[10px]">
                          <span className="text-slate-400 truncate">{k}</span>
                          <span className="text-slate-300 font-mono ml-1">{v as number}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {kind === "evoked" && (
              <KVGrid entries={[
                ["Channels", `${summary!.n_channels}`],
                ["Time Range", `${summary!.tmin} – ${summary!.tmax} s`],
                ["Averages", `${summary!.nave}`],
                ["Peak Channel", summary!.peak_channel ?? ""],
                ["Peak Latency", `${summary!.peak_latency_s} s`],
                ["Peak Amplitude", `${summary!.peak_amplitude?.toExponential(2)}`],
              ]} />
            )}

            {kind === "psd" && (
              <KVGrid entries={[
                ["Channels", `${summary!.n_channels}`],
                ["Freq Range", `${summary!.freq_min} – ${summary!.freq_max} Hz`],
                ["Frequencies", `${summary!.n_freqs}`],
                ["Method", summary!.method ?? ""],
                ["Shape", summary!.shape?.join(" × ") ?? ""],
              ]} />
            )}

            {kind === "tfr" && (
              <KVGrid entries={[
                ["Channels", `${summary!.n_channels}`],
                ["Freq Range", `${summary!.freq_min} – ${summary!.freq_max} Hz`],
                ["Time Range", `${summary!.tmin} – ${summary!.tmax} s`],
                ["Shape", summary!.shape?.join(" × ") ?? ""],
              ]} />
            )}

            {kind === "connectivity" && (
              <KVGrid entries={[
                ["Method", summary!.method ?? ""],
                ["Connections", `${summary!.n_connections}`],
                ["Frequencies", `${summary!.n_freqs}`],
                ["Shape", summary!.shape?.join(" × ") ?? ""],
              ]} />
            )}

            {kind === "array" && (
              <KVGrid entries={[
                ["Shape", summary!.shape?.join(" × ") ?? ""],
                ["Dtype", summary!.dtype ?? ""],
                ["Min", summary!.min?.toFixed(4) ?? ""],
                ["Max", summary!.max?.toFixed(4) ?? ""],
                ["Mean", summary!.mean?.toFixed(4) ?? ""],
              ]} />
            )}

            {kind === "metrics" && summary!.metrics && (
              <div className="max-h-40 overflow-y-auto">
                <table className="w-full text-[10px]">
                  <tbody>
                    {Object.entries(summary!.metrics).map(([k, v]) => (
                      <tr key={k} className="border-b border-slate-800/50">
                        <td className="text-slate-400 py-0.5 pr-2 whitespace-nowrap">{k}</td>
                        <td className="text-slate-200 py-0.5 font-mono text-right">
                          {formatMetricValue(v)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {kind === "scalar" && (
              <div className="text-center py-2">
                <span className="text-lg font-mono text-slate-200">
                  {summary!.value?.toFixed(4)}
                </span>
              </div>
            )}

            {kind === "plot" && (
              <div className="text-xs text-slate-400 text-center py-1">
                View plot in the node card lightbox
              </div>
            )}

            {kind === "unknown" && (
              <pre className="text-[9px] font-mono text-slate-500 whitespace-pre-wrap max-h-20 overflow-y-auto bg-slate-900/50 rounded p-1.5">
                {summary?.repr ?? nodeResult.output_type}
              </pre>
            )}
          </div>

          {/* Action buttons */}
          <div className="space-y-1.5 mt-3">
            {/* MNE Browser button for Raw nodes */}
            {kind === "raw" && onOpenBrowser && sessionId && (
              <div>
                <button
                  onClick={handleBrowserClick}
                  disabled={browserOpen}
                  className="w-full px-2 py-1.5 text-[11px] rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
                >
                  {browserOpen ? "Browser Open…" : "Open MNE Browser"}
                </button>
                {browserOpen && (
                  <p className="text-[9px] text-slate-400 mt-0.5 px-0.5">
                    Press <kbd className="px-0.5 py-px bg-slate-700 rounded text-slate-300">a</kbd> to
                    enter annotation mode, then click+drag to mark regions.
                  </p>
                )}
              </div>
            )}

            {/* Sync Annotations button — visible after the browser closes */}
            {kind === "raw" && sessionId && browserWasClosed && (
              <button
                onClick={handleSyncAnnotations}
                disabled={syncStatus === "syncing"}
                className="w-full px-2 py-1.5 text-[11px] rounded bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
              >
                {syncStatus === "syncing" ? "Syncing…" : "Sync Annotations"}
              </button>
            )}
            {syncResult && (
              <div className={`text-[10px] px-1 py-0.5 rounded ${
                syncStatus === "done" ? "text-green-400 bg-green-950/40" : "text-red-400 bg-red-950/40"
              }`}>
                {syncResult}
              </div>
            )}

            {/* .fif download for Raw/Epochs */}
            {(kind === "raw" || kind === "epochs") && onExport && nodeId && (
              <button
                onClick={() => onExport(nodeId, "fif", nodeLabel ?? "export")}
                className="w-full px-2 py-1.5 text-[11px] rounded bg-emerald-600 hover:bg-emerald-700 text-white transition-colors"
              >
                ↓ Download .fif
              </button>
            )}

            {/* Export dropdown */}
            {formats.length > 0 && onExport && nodeId && (
              <div className="relative">
                <button
                  onClick={() => setExportOpen(!exportOpen)}
                  className="w-full px-2 py-1.5 text-[11px] rounded bg-slate-700 hover:bg-slate-600 text-slate-200 transition-colors flex items-center justify-between"
                >
                  <span>Export As…</span>
                  <span className="text-[9px] text-slate-400">{formats.join(" / ").toUpperCase()}</span>
                </button>
                {exportOpen && (
                  <div className="absolute top-full left-0 right-0 mt-0.5 bg-slate-800 border border-slate-700 rounded shadow-lg z-10">
                    {formats.map((fmt) => (
                      <button
                        key={fmt}
                        onClick={() => {
                          onExport(nodeId, fmt, nodeLabel ?? "export");
                          setExportOpen(false);
                        }}
                        className="w-full text-left px-3 py-1.5 text-[11px] text-slate-300 hover:bg-slate-700 first:rounded-t last:rounded-b transition-colors"
                      >
                        .{fmt.toUpperCase()}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Re-run from here button */}
            {onRerunFrom && nodeId && nodeResult.status === "success" && (
              <button
                onClick={() => onRerunFrom(nodeId)}
                disabled={isRunning}
                className="w-full px-2 py-1.5 text-[11px] rounded bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
              >
                {isRunning ? "Re-running…" : "▶ Re-run from here"}
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper components
// ---------------------------------------------------------------------------

function KVGrid({ entries }: { entries: [string, string][] }) {
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[12px]">
      {entries.map(([label, value]) => (
        <div key={label} className="contents">
          <span className="text-slate-500">{label}</span>
          <span className="text-slate-300 font-mono text-right truncate">{value}</span>
        </div>
      ))}
    </div>
  );
}

function formatMetricValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return v.length > 40 ? v.slice(0, 40) + "…" : v;
  if (typeof v === "object" && v !== null) {
    const obj = v as Record<string, unknown>;
    if (obj.type === "array" && Array.isArray(obj.shape)) {
      return `array(${(obj.shape as number[]).join("×")})`;
    }
    if (Array.isArray(v)) {
      return `[${v.slice(0, 3).join(", ")}${v.length > 3 ? "…" : ""}]`;
    }
    return JSON.stringify(v).slice(0, 40);
  }
  return String(v);
}
