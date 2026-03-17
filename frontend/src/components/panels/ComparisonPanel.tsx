import { useMemo, useState } from "react";
import type { RunSnapshot } from "../../hooks/useRunHistory";
import type { NodeResult } from "../../types/pipeline";

interface ComparisonPanelProps {
  runA: RunSnapshot;
  runB: RunSnapshot;
  onClose: () => void;
}

interface ParamDiff {
  nodeId: string;
  nodeType: string;
  param: string;
  valueA: unknown;
  valueB: unknown;
}

interface MetricDiff {
  nodeId: string;
  nodeType: string;
  metric: string;
  valueA: number;
  valueB: number;
  delta: number;
  pct: number | null;
}

interface SummaryDiff {
  nodeId: string;
  nodeType: string;
  field: string;
  valueA: string;
  valueB: string;
}

interface PlotPair {
  nodeId: string;
  nodeType: string;
  imageA?: string;
  imageB?: string;
}

interface EpochRejection {
  nodeId: string;
  nodeType: string;
  totalA: number;
  droppedA: number;
  pctA: number;
  totalB: number;
  droppedB: number;
  pctB: number;
  deltaPct: number;
}

// Parameters that are not useful to compare (file paths, internal IDs)
const IGNORED_PARAMS = new Set(["file_path", "preload"]);

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "\u2014";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "boolean") return v ? "true" : "false";
  const s = String(v);
  return s.length > 50 ? s.slice(0, 47) + "\u2026" : s;
}

function nodeDisplayName(nr: NodeResult | undefined): string {
  if (!nr) return "unknown";
  // Convert node_type like "bandpass_filter" to "Bandpass Filter"
  return nr.node_type
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function summaryFieldLabel(field: string): string {
  const map: Record<string, string> = {
    n_channels: "Channels",
    sfreq: "Sampling Rate (Hz)",
    duration_s: "Duration (s)",
    n_epochs: "Epochs",
    tmin: "Time Min (s)",
    tmax: "Time Max (s)",
    nave: "Averages",
    freq_min: "Freq Min (Hz)",
    freq_max: "Freq Max (Hz)",
    n_freqs: "Frequencies",
    method: "Method",
    highpass: "Highpass (Hz)",
    lowpass: "Lowpass (Hz)",
    n_connections: "Connections",
  };
  return map[field] ?? field;
}

// Fields from NodeOutputSummary worth comparing
const SUMMARY_FIELDS = [
  "n_channels", "sfreq", "duration_s", "n_epochs", "tmin", "tmax",
  "nave", "freq_min", "freq_max", "n_freqs", "method", "highpass",
  "lowpass", "n_connections",
];

export function ComparisonPanel({ runA, runB, onClose }: ComparisonPanelProps) {
  const { paramDiffs, metricsDiffs, summaryDiffs, plotPairs, epochRejections } = useMemo(() => {
    const allNodeIds = new Set([
      ...Object.keys(runA.paramSnapshot),
      ...Object.keys(runB.paramSnapshot),
      ...Object.keys(runA.nodeResults),
      ...Object.keys(runB.nodeResults),
    ]);

    // 1. Parameter diffs (filtering out noise like file_path)
    const paramDiffs: ParamDiff[] = [];
    for (const nodeId of allNodeIds) {
      const paramsA = runA.paramSnapshot[nodeId] || {};
      const paramsB = runB.paramSnapshot[nodeId] || {};
      const nodeType = nodeDisplayName(runA.nodeResults[nodeId] ?? runB.nodeResults[nodeId]);
      const allKeys = new Set([...Object.keys(paramsA), ...Object.keys(paramsB)]);
      for (const key of allKeys) {
        if (IGNORED_PARAMS.has(key)) continue;
        if (JSON.stringify(paramsA[key]) !== JSON.stringify(paramsB[key])) {
          paramDiffs.push({ nodeId, nodeType, param: key, valueA: paramsA[key], valueB: paramsB[key] });
        }
      }
    }

    // 2. Metrics diffs
    const metricsDiffs: MetricDiff[] = [];
    for (const nodeId of allNodeIds) {
      const nrA = runA.nodeResults[nodeId];
      const nrB = runB.nodeResults[nodeId];
      const metricsA = nrA?.metrics;
      const metricsB = nrB?.metrics;
      if (!metricsA && !metricsB) continue;
      const nodeType = nodeDisplayName(nrA ?? nrB);
      const allKeys = new Set([
        ...Object.keys(metricsA ?? {}),
        ...Object.keys(metricsB ?? {}),
      ]);
      for (const key of allKeys) {
        const a = metricsA?.[key];
        const b = metricsB?.[key];
        if (typeof a === "number" && typeof b === "number") {
          const delta = b - a;
          const pct = a !== 0 ? (delta / Math.abs(a)) * 100 : null;
          metricsDiffs.push({ nodeId, nodeType, metric: key, valueA: a, valueB: b, delta, pct });
        }
      }
    }

    // 3. Output summary diffs (channels, sfreq, duration, etc.)
    const summaryDiffs: SummaryDiff[] = [];
    for (const nodeId of allNodeIds) {
      const nrA = runA.nodeResults[nodeId];
      const nrB = runB.nodeResults[nodeId];
      if (!nrA?.summary && !nrB?.summary) continue;
      const nodeType = nodeDisplayName(nrA ?? nrB);
      const sumA = (nrA?.summary ?? {}) as Record<string, unknown>;
      const sumB = (nrB?.summary ?? {}) as Record<string, unknown>;
      for (const field of SUMMARY_FIELDS) {
        const a = sumA[field];
        const b = sumB[field];
        if (a === undefined && b === undefined) continue;
        if (JSON.stringify(a) !== JSON.stringify(b)) {
          summaryDiffs.push({
            nodeId,
            nodeType,
            field,
            valueA: a !== undefined ? String(a) : "\u2014",
            valueB: b !== undefined ? String(b) : "\u2014",
          });
        }
      }
    }

    // 4. Plot pairs
    const plotPairs: PlotPair[] = [];
    for (const nodeId of allNodeIds) {
      const imgA = runA.thumbnails[nodeId];
      const imgB = runB.thumbnails[nodeId];
      if (imgA || imgB) {
        const nodeType = nodeDisplayName(runA.nodeResults[nodeId] ?? runB.nodeResults[nodeId]);
        plotPairs.push({ nodeId, nodeType, imageA: imgA, imageB: imgB });
      }
    }

    // 5. Epoch rejection comparison
    const epochRejections: EpochRejection[] = [];
    for (const nodeId of allNodeIds) {
      const nrA = runA.nodeResults[nodeId];
      const nrB = runB.nodeResults[nodeId];
      const sumA = nrA?.summary;
      const sumB = nrB?.summary;
      if (sumA?.kind !== "epochs" || sumB?.kind !== "epochs") continue;
      if (sumA.n_epochs == null || sumB.n_epochs == null) continue;
      const droppedA = sumA.n_dropped ?? 0;
      const droppedB = sumB.n_dropped ?? 0;
      const totalA = sumA.n_epochs + droppedA;
      const totalB = sumB.n_epochs + droppedB;
      if (totalA === 0 || totalB === 0) continue;
      const pctA = (droppedA / totalA) * 100;
      const pctB = (droppedB / totalB) * 100;
      epochRejections.push({
        nodeId,
        nodeType: nodeDisplayName(nrA ?? nrB),
        totalA, droppedA, pctA,
        totalB, droppedB, pctB,
        deltaPct: pctB - pctA,
      });
    }

    return { paramDiffs, metricsDiffs, summaryDiffs, plotPairs, epochRejections };
  }, [runA, runB]);

  const hasDiffs = paramDiffs.length > 0 || metricsDiffs.length > 0 || summaryDiffs.length > 0 || epochRejections.length > 0;

  // ── Copy to clipboard ────────────────────────────────────────────────────
  const [copyLabel, setCopyLabel] = useState("Copy Summary");

  const handleCopy = async () => {
    try {
      const lines: string[] = [
        `Oscilloom Run Comparison`,
        `${"=".repeat(50)}`,
        `Run A: ${runA.name}  (${new Date(runA.timestamp).toLocaleString()})`,
        `Run B: ${runB.name}  (${new Date(runB.timestamp).toLocaleString()})`,
        `Nodes: ${runA.nodeCount} | ${runB.nodeCount}    Errors: ${runA.errorCount} | ${runB.errorCount}`,
        "",
      ];

      if (paramDiffs.length > 0) {
        lines.push("PARAMETER CHANGES");
        lines.push("-".repeat(50));
        for (const d of paramDiffs) {
          lines.push(`  [${d.nodeType}] ${d.param}`);
          lines.push(`    ${runA.name}: ${formatValue(d.valueA)}`);
          lines.push(`    ${runB.name}: ${formatValue(d.valueB)}`);
        }
        lines.push("");
      }

      if (summaryDiffs.length > 0) {
        lines.push("OUTPUT CHANGES");
        lines.push("-".repeat(50));
        for (const d of summaryDiffs) {
          lines.push(`  [${d.nodeType}] ${summaryFieldLabel(d.field)}: ${d.valueA} -> ${d.valueB}`);
        }
        lines.push("");
      }

      if (epochRejections.length > 0) {
        lines.push("EPOCH REJECTION");
        lines.push("-".repeat(50));
        for (const e of epochRejections) {
          lines.push(`  [${e.nodeType}]`);
          lines.push(`    ${runA.name}: ${e.droppedA}/${e.totalA} dropped (${e.pctA.toFixed(1)}% lost)`);
          lines.push(`    ${runB.name}: ${e.droppedB}/${e.totalB} dropped (${e.pctB.toFixed(1)}% lost)`);
          lines.push(`    Change: ${e.deltaPct > 0 ? "+" : ""}${e.deltaPct.toFixed(1)}pp`);
        }
        lines.push("");
      }

      if (metricsDiffs.length > 0) {
        lines.push("METRICS");
        lines.push("-".repeat(50));
        for (const d of metricsDiffs) {
          const delta = `${d.delta > 0 ? "+" : ""}${d.delta.toFixed(4)}`;
          const pct = d.pct !== null ? ` (${d.pct > 0 ? "+" : ""}${d.pct.toFixed(1)}%)` : "";
          lines.push(`  [${d.nodeType}] ${d.metric}: ${d.valueA.toFixed(4)} -> ${d.valueB.toFixed(4)}  ${delta}${pct}`);
        }
        lines.push("");
      }

      if (!hasDiffs) {
        lines.push("No differences found between these runs.");
      }

      await navigator.clipboard.writeText(lines.join("\n"));
      setCopyLabel("Copied!");
      setTimeout(() => setCopyLabel("Copy Summary"), 2000);
    } catch {
      setCopyLabel("Copy failed");
      setTimeout(() => setCopyLabel("Copy Summary"), 2000);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-2xl w-[90vw] max-w-4xl max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
          <div>
            <div className="text-sm text-slate-200 font-semibold">
              Compare Runs
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              {runA.name} ({runA.nodeCount} nodes) vs {runB.name} ({runB.nodeCount} nodes)
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg px-2"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div
          id="comparison-content"
          className="flex-1 overflow-y-auto p-4 space-y-5"
        >
          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-slate-800/60 rounded-lg p-3 text-center">
              <div className="text-lg font-mono text-amber-400">{paramDiffs.length}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Param Changes</div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 text-center">
              <div className="text-lg font-mono text-blue-400">{summaryDiffs.length}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Output Changes</div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 text-center">
              <div className="text-lg font-mono text-emerald-400">{metricsDiffs.length}</div>
              <div className="text-[10px] text-slate-500 mt-0.5">Metric Diffs</div>
            </div>
          </div>

          {/* Parameter Diffs */}
          {paramDiffs.length > 0 && (
            <section>
              <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Parameter Changes
              </h3>
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700">
                    <th className="text-left py-1.5 pr-2">Node / Parameter</th>
                    <th className="text-right py-1.5 px-2">{runA.name}</th>
                    <th className="text-right py-1.5 pl-2">{runB.name}</th>
                  </tr>
                </thead>
                <tbody>
                  {paramDiffs.map((d, i) => (
                    <tr
                      key={i}
                      className="border-b border-slate-800/50 hover:bg-slate-800/30"
                    >
                      <td className="text-slate-400 py-1.5 pr-2">
                        <span className="text-slate-500 text-[10px]">
                          {d.nodeType}
                        </span>
                        <span className="text-slate-600 mx-1">/</span>
                        <span className="text-slate-300">{d.param}</span>
                      </td>
                      <td className="text-slate-400 py-1.5 px-2 font-mono text-right">
                        {formatValue(d.valueA)}
                      </td>
                      <td className="text-slate-200 py-1.5 pl-2 font-mono text-right font-semibold">
                        {formatValue(d.valueB)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* Output Summary Diffs */}
          {summaryDiffs.length > 0 && (
            <section>
              <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Output Changes
              </h3>
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700">
                    <th className="text-left py-1.5 pr-2">Node / Field</th>
                    <th className="text-right py-1.5 px-2">{runA.name}</th>
                    <th className="text-right py-1.5 pl-2">{runB.name}</th>
                  </tr>
                </thead>
                <tbody>
                  {summaryDiffs.map((d, i) => (
                    <tr
                      key={i}
                      className="border-b border-slate-800/50 hover:bg-slate-800/30"
                    >
                      <td className="text-slate-400 py-1.5 pr-2">
                        <span className="text-slate-500 text-[10px]">
                          {d.nodeType}
                        </span>
                        <span className="text-slate-600 mx-1">/</span>
                        <span className="text-slate-300">{summaryFieldLabel(d.field)}</span>
                      </td>
                      <td className="text-slate-400 py-1.5 px-2 font-mono text-right">
                        {d.valueA}
                      </td>
                      <td className="text-slate-200 py-1.5 pl-2 font-mono text-right font-semibold">
                        {d.valueB}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* Epoch Rejection Comparison */}
          {epochRejections.length > 0 && (
            <section>
              <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Epoch Rejection
              </h3>
              <div className="space-y-2">
                {epochRejections.map((e) => (
                  <div key={e.nodeId} className="bg-slate-800/40 rounded-lg p-3">
                    <div className="text-[10px] text-slate-500 mb-2">{e.nodeType}</div>
                    <div className="grid grid-cols-2 gap-3 mb-2">
                      <div className="text-center">
                        <div className="text-[9px] text-slate-500 mb-0.5">{runA.name}</div>
                        <div className="text-sm font-mono text-slate-200">
                          {e.droppedA}/{e.totalA} dropped
                        </div>
                        <div className="text-[10px] font-mono text-amber-400">{e.pctA.toFixed(1)}% lost</div>
                      </div>
                      <div className="text-center">
                        <div className="text-[9px] text-slate-500 mb-0.5">{runB.name}</div>
                        <div className="text-sm font-mono text-slate-200">
                          {e.droppedB}/{e.totalB} dropped
                        </div>
                        <div className="text-[10px] font-mono text-amber-400">{e.pctB.toFixed(1)}% lost</div>
                      </div>
                    </div>
                    <div className={`text-center text-xs font-semibold rounded py-1 ${
                      e.deltaPct < 0 ? "bg-emerald-900/40 text-emerald-400" : e.deltaPct > 0 ? "bg-red-900/40 text-red-400" : "bg-slate-800 text-slate-500"
                    }`}>
                      {e.deltaPct > 0 ? "+" : ""}{e.deltaPct.toFixed(1)}pp rejection change
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Metrics Diffs */}
          {metricsDiffs.length > 0 && (
            <section>
              <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Metrics Comparison
              </h3>
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700">
                    <th className="text-left py-1.5 pr-2">Node / Metric</th>
                    <th className="text-right py-1.5 px-2">{runA.name}</th>
                    <th className="text-right py-1.5 px-2">{runB.name}</th>
                    <th className="text-right py-1.5 pl-2">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsDiffs.map((d, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="text-slate-400 py-1.5 pr-2">
                        <span className="text-slate-500 text-[10px]">
                          {d.nodeType}
                        </span>
                        <span className="text-slate-600 mx-1">/</span>
                        <span className="text-slate-300">{d.metric}</span>
                      </td>
                      <td className="text-slate-300 py-1.5 px-2 font-mono text-right">
                        {d.valueA.toFixed(4)}
                      </td>
                      <td className="text-slate-200 py-1.5 px-2 font-mono text-right">
                        {d.valueB.toFixed(4)}
                      </td>
                      <td className="py-1.5 pl-2 font-mono text-right">
                        <span
                          className={
                            d.delta > 0
                              ? "text-emerald-400"
                              : d.delta < 0
                              ? "text-red-400"
                              : "text-slate-500"
                          }
                        >
                          {d.delta > 0 ? "+" : ""}
                          {d.delta.toFixed(4)}
                          {d.pct !== null && (
                            <span className="text-[9px] ml-1 text-slate-500">
                              ({d.pct > 0 ? "+" : ""}{d.pct.toFixed(1)}%)
                            </span>
                          )}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          {/* Empty state */}
          {!hasDiffs && plotPairs.length === 0 && (
            <div className="text-center py-8">
              <div className="text-slate-500 text-sm">No differences found between these runs.</div>
              <div className="text-slate-600 text-[11px] mt-1">
                Both runs used the same parameters and produced identical outputs.
              </div>
            </div>
          )}

          {/* Plot Comparisons */}
          {plotPairs.length > 0 && (
            <section>
              <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Plot Comparison
              </h3>
              <div className="space-y-3">
                {plotPairs.map((p) => (
                  <div key={p.nodeId}>
                    <div className="text-[10px] text-slate-500 mb-1">{p.nodeType}</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-800/50 rounded p-1.5">
                        <div className="text-[9px] text-slate-500 mb-1">
                          {runA.name}
                        </div>
                        {p.imageA ? (
                          <img
                            src={p.imageA}
                            alt={`${runA.name} - ${p.nodeType}`}
                            className="w-full rounded"
                          />
                        ) : (
                          <div className="h-24 flex items-center justify-center text-[10px] text-slate-600">
                            No plot
                          </div>
                        )}
                      </div>
                      <div className="bg-slate-800/50 rounded p-1.5">
                        <div className="text-[9px] text-slate-500 mb-1">
                          {runB.name}
                        </div>
                        {p.imageB ? (
                          <img
                            src={p.imageB}
                            alt={`${runB.name} - ${p.nodeType}`}
                            className="w-full rounded"
                          />
                        ) : (
                          <div className="h-24 flex items-center justify-center text-[10px] text-slate-600">
                            No plot
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-slate-700">
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 text-[11px] rounded bg-slate-700 hover:bg-slate-600 text-slate-300 transition-colors"
          >
            {copyLabel}
          </button>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-[11px] rounded bg-slate-600 hover:bg-slate-500 text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
