import React, { useRef, useCallback } from "react";
import type { PipelineGraph, MetricStats } from "../../types/pipeline";
import type { UseBatchMode } from "../../hooks/useBatchMode";
import type { ReportSections } from "../../api/client";

type BatchTab = "files" | "progress" | "results";

const SECTION_LABELS: Record<
  keyof ReportSections,
  { label: string; description: string }
> = {
  data_quality: {
    label: "Data Quality Summary",
    description: "Recording metadata (not available in batch mode)",
  },
  pipeline_config: {
    label: "Pipeline Configuration",
    description: "Processing steps with parameter values",
  },
  analysis_results: {
    label: "Analysis Results",
    description: "Metrics tables from clinical and analysis nodes",
  },
  clinical_interpretation: {
    label: "Clinical Interpretation",
    description: "Reference ranges and status flags for known metrics",
  },
  visualizations: {
    label: "Visualizations",
    description: "PSD plots, topomaps, and other generated figures",
  },
  audit_trail: {
    label: "Audit Trail",
    description: "Parameter change history (not available in batch mode)",
  },
  notes: {
    label: "Clinician Notes",
    description: "Free-text observations included in every report",
  },
};

const SECTION_ORDER: (keyof ReportSections)[] = [
  "pipeline_config",
  "analysis_results",
  "clinical_interpretation",
  "visualizations",
  "notes",
  "data_quality",
  "audit_trail",
];

const SECTION_AVAILABILITY: Record<keyof ReportSections, boolean> = {
  data_quality: false,
  pipeline_config: true,
  analysis_results: true,
  clinical_interpretation: true,
  visualizations: true,
  audit_trail: false,
  notes: true,
};

interface BatchPanelProps {
  batch: UseBatchMode;
  pipeline: PipelineGraph;
}

function activeTab(phase: string): BatchTab {
  if (phase === "running" || phase === "uploading") return "progress";
  if (phase === "results" || phase === "report_config") return "results";
  return "files";
}

function MetricStatsRow({ name, stats }: { name: string; stats: MetricStats }) {
  return (
    <tr className="border-b border-slate-700/50 text-xs">
      <td className="py-1 px-1 text-slate-300 truncate max-w-[10rem]" title={name}>
        {name.split(".").pop()}
      </td>
      <td className="py-1 px-1 text-slate-400 text-right">{stats.count}</td>
      <td className="py-1 px-1 text-cyan-400 text-right font-mono">
        {stats.mean.toFixed(3)}
      </td>
      <td className="py-1 px-1 text-slate-400 text-right font-mono">
        {stats.std.toFixed(3)}
      </td>
      <td className="py-1 px-1 text-slate-500 text-right font-mono">
        {stats.min.toFixed(3)}
      </td>
      <td className="py-1 px-1 text-slate-500 text-right font-mono">
        {stats.max.toFixed(3)}
      </td>
    </tr>
  );
}

export function BatchPanel({ batch, pipeline }: BatchPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const tab = activeTab(batch.phase);

  const handleFilesSelected = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = e.target.files;
      if (!fileList) return;
      batch.addFiles(Array.from(fileList));
    },
    [batch.addFiles]
  );

  const enabledSectionCount = SECTION_ORDER.filter(
    (k) => batch.reportSections[k] && SECTION_AVAILABILITY[k]
  ).length;

  return (
    <div className="flex flex-col h-full bg-slate-900 border-l border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700 bg-slate-800/50">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-amber-400" />
          <span className="text-slate-200 text-xs font-semibold">
            Batch Processing
          </span>
          {batch.phase === "running" && batch.progress && (
            <span className="text-amber-400 text-[10px] font-mono">
              {batch.progress.completed + batch.progress.failed}/
              {batch.progress.total}
            </span>
          )}
        </div>
        <button
          onClick={batch.toggleBatchMode}
          className="text-slate-400 hover:text-slate-200 text-sm"
          title="Exit Batch Mode"
        >
          &times;
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-700">
        {(["files", "progress", "results"] as BatchTab[]).map((t) => (
          <button
            key={t}
            className={`flex-1 text-[10px] font-medium py-1.5 transition-colors capitalize ${
              tab === t
                ? "text-amber-400 border-b-2 border-amber-400 bg-slate-800/30"
                : "text-slate-500 hover:text-slate-300"
            }`}
            disabled
          >
            {t}
            {t === "files" && batch.selectedFiles.length > 0 && (
              <span className="ml-1 text-slate-500">
                ({batch.selectedFiles.length})
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* ========== FILES TAB ========== */}
        {tab === "files" && (
          <>
            <p className="text-slate-500 text-[10px]">
              Select EEG files to process through the current pipeline.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".edf,.fif,.bdf,.set,.vhdr,.cnt"
              className="hidden"
              onChange={handleFilesSelected}
            />
            <div className="flex gap-1.5">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-[10px] rounded px-2 py-1.5 transition-colors"
              >
                Select Files...
              </button>
              <button
                onClick={batch.loadSavedResults}
                className="bg-slate-700 hover:bg-slate-600 text-slate-400 text-[10px] rounded px-2 py-1.5 transition-colors"
              >
                Load Previous
              </button>
            </div>

            {/* Saved batches */}
            {batch.showSaved && batch.savedBatches.length > 0 && (
              <div className="max-h-24 overflow-y-auto text-[10px]">
                <div className="text-slate-400 mb-1 font-semibold">
                  Saved Results:
                </div>
                {batch.savedBatches.map((sb) => (
                  <button
                    key={sb.batch_id}
                    onClick={() => batch.selectSavedBatch(sb.batch_id)}
                    className="w-full text-left bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded px-2 py-1 mb-0.5 text-[10px]"
                  >
                    {sb.batch_id.slice(0, 8)}... {sb.completed}/{sb.total}{" "}
                    completed (
                    {new Date(sb.saved_at * 1000).toLocaleDateString()})
                  </button>
                ))}
              </div>
            )}
            {batch.showSaved && batch.savedBatches.length === 0 && (
              <div className="text-slate-500 text-[10px]">
                No saved results found.
              </div>
            )}

            {/* File table with metadata */}
            {batch.selectedFiles.length > 0 && (
              <div className="max-h-[50vh] overflow-y-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-1 px-1">File</th>
                      <th className="text-left py-1 px-1 w-16">Subject</th>
                      <th className="text-left py-1 px-1 w-16">Group</th>
                      <th className="text-left py-1 px-1 w-16">Condition</th>
                      <th className="w-5"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.selectedFiles.map((f, i) => (
                      <tr key={i} className="border-b border-slate-700/50">
                        <td
                          className="py-1 px-1 text-slate-300 truncate max-w-[8rem]"
                          title={f.name}
                        >
                          {f.name}
                        </td>
                        {(
                          ["subject_id", "group", "condition"] as const
                        ).map((field) => (
                          <td key={field} className="py-0.5 px-1">
                            <input
                              type="text"
                              className="bg-slate-700 text-slate-200 text-[10px] rounded px-1 py-0.5 w-full"
                              placeholder={
                                field === "subject_id"
                                  ? "ID"
                                  : field === "group"
                                  ? "Grp"
                                  : "Cond"
                              }
                              value={batch.fileMetadata[i]?.[field] || ""}
                              onChange={(e) =>
                                batch.updateMetadata(i, field, e.target.value)
                              }
                            />
                          </td>
                        ))}
                        <td className="py-0.5 px-1">
                          <button
                            onClick={() => batch.removeFile(i)}
                            className="text-slate-500 hover:text-red-400 text-xs"
                            title="Remove file"
                          >
                            &times;
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {/* ========== PROGRESS TAB ========== */}
        {tab === "progress" && (
          <>
            {batch.phase === "uploading" && (
              <div className="text-slate-400 text-xs py-8 text-center">
                Uploading {batch.selectedFiles.length} file(s)...
              </div>
            )}
            {batch.phase === "running" && batch.progress && (
              <>
                <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                  <span className="truncate">
                    {batch.progress.current_file || "..."}
                  </span>
                  <span className="font-mono">
                    {batch.progress.completed + batch.progress.failed}/
                    {batch.progress.total}
                  </span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-1.5">
                  <div
                    className="bg-amber-400 h-1.5 rounded-full transition-all"
                    style={{
                      width: `${
                        ((batch.progress.completed + batch.progress.failed) /
                          Math.max(batch.progress.total, 1)) *
                        100
                      }%`,
                    }}
                  />
                </div>
                <div className="flex gap-3 text-[10px] mt-1">
                  <span className="text-cyan-400">
                    {batch.progress.completed} done
                  </span>
                  {batch.progress.failed > 0 && (
                    <span className="text-red-400">
                      {batch.progress.failed} failed
                    </span>
                  )}
                </div>
              </>
            )}
            {batch.phase === "running" && !batch.progress && (
              <div className="text-slate-400 text-xs py-8 text-center">
                Starting batch...
              </div>
            )}
          </>
        )}

        {/* ========== RESULTS TAB ========== */}
        {tab === "results" && batch.results && batch.phase !== "report_config" && (
          <>
            {/* Summary cards */}
            <div className="grid grid-cols-3 gap-1.5">
              <div className="bg-slate-800 rounded p-2 text-center">
                <div className="text-cyan-400 text-sm font-bold">
                  {batch.results.summary.completed}
                </div>
                <div className="text-slate-500 text-[10px]">Done</div>
              </div>
              <div className="bg-slate-800 rounded p-2 text-center">
                <div className="text-red-400 text-sm font-bold">
                  {batch.results.summary.failed}
                </div>
                <div className="text-slate-500 text-[10px]">Failed</div>
              </div>
              <div className="bg-slate-800 rounded p-2 text-center">
                <div className="text-slate-200 text-sm font-bold">
                  {batch.results.summary.runtime_s}s
                </div>
                <div className="text-slate-500 text-[10px]">Time</div>
              </div>
            </div>

            {/* Aggregate statistics */}
            {batch.statistics &&
              Object.keys(batch.statistics.overall).length > 0 && (
                <div>
                  <div className="text-slate-400 text-[10px] font-semibold mb-1">
                    Aggregate Statistics
                  </div>
                  <div className="bg-slate-800 rounded overflow-hidden">
                    <table className="w-full">
                      <thead>
                        <tr className="text-slate-500 text-[10px] border-b border-slate-700">
                          <th className="text-left py-1 px-1">Metric</th>
                          <th className="text-right py-1 px-1">N</th>
                          <th className="text-right py-1 px-1">Mean</th>
                          <th className="text-right py-1 px-1">SD</th>
                          <th className="text-right py-1 px-1">Min</th>
                          <th className="text-right py-1 px-1">Max</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(batch.statistics.overall).map(
                          ([name, stats]) => (
                            <MetricStatsRow
                              key={name}
                              name={name}
                              stats={stats}
                            />
                          )
                        )}
                      </tbody>
                    </table>
                  </div>

                  {/* Group-level breakdowns */}
                  {Object.keys(batch.statistics.by_group).length > 0 && (
                    <div className="mt-2 space-y-2">
                      {Object.entries(batch.statistics.by_group).map(
                        ([group, metrics]) => (
                          <div key={group}>
                            <div className="text-amber-400/80 text-[10px] font-semibold mb-0.5">
                              Group: {group}
                            </div>
                            <div className="bg-slate-800 rounded overflow-hidden">
                              <table className="w-full">
                                <thead>
                                  <tr className="text-slate-500 text-[10px] border-b border-slate-700">
                                    <th className="text-left py-1 px-1">
                                      Metric
                                    </th>
                                    <th className="text-right py-1 px-1">N</th>
                                    <th className="text-right py-1 px-1">
                                      Mean
                                    </th>
                                    <th className="text-right py-1 px-1">SD</th>
                                    <th className="text-right py-1 px-1">
                                      Min
                                    </th>
                                    <th className="text-right py-1 px-1">
                                      Max
                                    </th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(metrics).map(
                                    ([name, stats]) => (
                                      <MetricStatsRow
                                        key={name}
                                        name={name}
                                        stats={stats}
                                      />
                                    )
                                  )}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  )}
                </div>
              )}

            {/* Per-file results table */}
            <div>
              <div className="text-slate-400 text-[10px] font-semibold mb-1">
                Per-File Results
              </div>
              <div className="max-h-48 overflow-y-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700">
                      <th className="py-1 px-1 w-5"></th>
                      <th className="text-left py-1 px-1">File</th>
                      <th className="text-left py-1 px-1">Status</th>
                      <th className="py-1 px-1 w-8"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.results.file_results.map((fr) => (
                      <React.Fragment key={fr.file_id}>
                        <tr className="border-b border-slate-700/50">
                          <td className="py-1 px-1">
                            <input
                              type="checkbox"
                              checked={batch.selectedForRetry.has(fr.file_id)}
                              onChange={() =>
                                batch.toggleRetrySelection(fr.file_id)
                              }
                              className="accent-amber-500"
                            />
                          </td>
                          <td
                            className="py-1 px-1 text-slate-300 truncate max-w-[10rem]"
                            title={fr.filename}
                          >
                            {fr.filename}
                            {fr.processing_time_s != null && (
                              <span className="text-slate-600 ml-1">
                                ({fr.processing_time_s}s)
                              </span>
                            )}
                          </td>
                          <td className="py-1 px-1">
                            {fr.status === "success" ? (
                              <span className="text-cyan-400">OK</span>
                            ) : (
                              <span
                                className="text-red-400"
                                title={fr.error || ""}
                              >
                                Fail
                              </span>
                            )}
                          </td>
                          <td className="py-1 px-1">
                            {fr.status === "success" && (
                              <button
                                onClick={() =>
                                  batch.viewFileDetail(fr.file_id)
                                }
                                className="text-blue-400 hover:text-blue-300 underline"
                              >
                                {batch.expandedFileId === fr.file_id
                                  ? "Hide"
                                  : "View"}
                              </button>
                            )}
                          </td>
                        </tr>
                        {batch.expandedFileId === fr.file_id && (
                          <tr>
                            <td colSpan={4} className="p-2 bg-slate-800/50">
                              {batch.detailLoading ? (
                                <div className="text-slate-400 text-[10px]">
                                  Loading...
                                </div>
                              ) : batch.fileDetail ? (
                                <div className="space-y-2">
                                  {Object.keys(batch.fileDetail.metrics)
                                    .length > 0 && (
                                    <div>
                                      <div className="text-slate-400 text-[10px] font-semibold mb-0.5">
                                        Metrics
                                      </div>
                                      {Object.entries(
                                        batch.fileDetail.metrics
                                      ).map(([key, val]) => (
                                        <div
                                          key={key}
                                          className="text-[10px] text-slate-300 ml-1"
                                        >
                                          <span className="text-slate-500">
                                            {key}:
                                          </span>{" "}
                                          {typeof val === "number"
                                            ? Number(val).toFixed(4)
                                            : String(val)}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                  {Object.entries(
                                    batch.fileDetail.node_results
                                  )
                                    .filter(
                                      ([, nr]) =>
                                        typeof nr.data === "string" &&
                                        nr.data.startsWith("data:image")
                                    )
                                    .map(([nodeId, nr]) => (
                                      <div key={nodeId}>
                                        <div className="text-[10px] text-slate-400 mb-0.5">
                                          {nr.node_type}
                                        </div>
                                        <img
                                          src={nr.data!}
                                          alt={nr.node_type}
                                          className="max-w-full rounded"
                                        />
                                      </div>
                                    ))}
                                </div>
                              ) : null}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Failed files detail */}
            {batch.results.failed_files.length > 0 && (
              <div>
                <div className="text-red-400 text-[10px] font-semibold mb-1">
                  Failed Files
                </div>
                {batch.results.failed_files.map((ff, i) => (
                  <div
                    key={i}
                    className="text-[10px] text-red-300 bg-red-950/50 rounded px-2 py-1 mb-0.5"
                  >
                    {ff.filename}: {ff.error}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ========== REPORT CONFIG ========== */}
        {batch.phase === "report_config" && (
          <div className="space-y-3">
            <p className="text-slate-500 text-[10px]">
              Configure PDF reports for {batch.results?.summary.completed ?? 0}{" "}
              successful files.
            </p>

            <div>
              <label className="block text-slate-400 text-[10px] mb-1">
                Clinic / Lab
              </label>
              <input
                type="text"
                value={batch.reportClinicName}
                onChange={(e) => batch.setReportClinicName(e.target.value)}
                placeholder="e.g. Neurology Dept."
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-slate-200 text-[10px] focus:outline-none focus:border-slate-400"
              />
            </div>

            <div>
              <label className="block text-slate-400 text-[10px] mb-1">
                Sections ({enabledSectionCount})
              </label>
              <div className="bg-slate-800 border border-slate-700 rounded p-1.5 space-y-0.5">
                {SECTION_ORDER.map((key) => {
                  const available = SECTION_AVAILABILITY[key];
                  const enabled = batch.reportSections[key];
                  const info = SECTION_LABELS[key];
                  return (
                    <label
                      key={key}
                      className={`flex items-start gap-1.5 px-1.5 py-1 rounded cursor-pointer text-[10px] ${
                        available
                          ? "hover:bg-slate-700"
                          : "opacity-40 cursor-not-allowed"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={enabled && available}
                        disabled={!available}
                        onChange={() => batch.toggleReportSection(key)}
                        className="mt-0.5"
                      />
                      <div>
                        <div className="text-slate-200">
                          {info.label}
                          {!available && (
                            <span className="text-slate-600 ml-1">(N/A)</span>
                          )}
                        </div>
                        <div className="text-slate-600">{info.description}</div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            <div>
              <label className="block text-slate-400 text-[10px] mb-1">
                Notes
              </label>
              <textarea
                value={batch.reportNotes}
                onChange={(e) => batch.setReportNotes(e.target.value)}
                placeholder="Included in every report..."
                rows={2}
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1 text-slate-200 text-[10px] focus:outline-none focus:border-slate-400 resize-none"
              />
            </div>
          </div>
        )}
      </div>

      {/* Error display */}
      {batch.error && (
        <div className="mx-3 mb-2 text-red-400 text-[10px] bg-red-950/50 border border-red-800/50 rounded px-2 py-1">
          {batch.error}
        </div>
      )}

      {/* Action bar */}
      <div className="flex flex-wrap gap-1.5 p-3 border-t border-slate-700 bg-slate-800/30">
        {tab === "files" && (
          <>
            <button
              onClick={() => batch.startBatchRun(pipeline)}
              disabled={batch.selectedFiles.length === 0}
              className="flex-1 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
            >
              Start Batch ({batch.selectedFiles.length})
            </button>
            {batch.selectedFiles.length > 0 && (
              <button
                onClick={batch.reset}
                className="bg-slate-700 hover:bg-slate-600 text-slate-400 text-[10px] rounded px-2 py-1.5 transition-colors"
              >
                Clear
              </button>
            )}
          </>
        )}

        {tab === "progress" && batch.phase === "running" && (
          <button
            onClick={batch.cancelBatchRun}
            className="flex-1 bg-red-800 hover:bg-red-700 text-white text-[10px] rounded px-2 py-1.5 transition-colors"
          >
            Cancel
          </button>
        )}

        {tab === "results" && batch.phase !== "report_config" && (
          <>
            <button
              onClick={batch.downloadCsv}
              disabled={!batch.results?.metrics_csv}
              className="bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
            >
              CSV
            </button>
            <button
              onClick={batch.showReportConfig}
              disabled={
                !batch.results || batch.results.summary.completed === 0
              }
              className="bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
            >
              Reports
            </button>
            <button
              onClick={batch.saveResults}
              disabled={batch.saved}
              className="bg-sky-700 hover:bg-sky-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
            >
              {batch.saved ? "Saved" : "Save"}
            </button>
            {batch.results &&
              batch.results.failed_files.length > 0 && (
                <button
                  onClick={() => batch.retryFailed(pipeline)}
                  className="bg-amber-700 hover:bg-amber-600 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
                >
                  Retry ({batch.results.failed_files.length})
                </button>
              )}
            {batch.selectedForRetry.size > 0 && (
              <button
                onClick={() => batch.retrySelected(pipeline)}
                className="bg-blue-700 hover:bg-blue-600 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
              >
                Retry Sel. ({batch.selectedForRetry.size})
              </button>
            )}
            <button
              onClick={batch.reset}
              className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-[10px] rounded px-2 py-1.5 transition-colors"
            >
              New Batch
            </button>
          </>
        )}

        {batch.phase === "report_config" && (
          <>
            <button
              onClick={() => batch.generateReports(pipeline)}
              disabled={batch.generatingReports || enabledSectionCount === 0}
              className="flex-1 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-[10px] rounded px-2 py-1.5 transition-colors font-medium"
            >
              {batch.generatingReports ? "Generating..." : "Download ZIP"}
            </button>
            <button
              onClick={() => batch.setPhase("results")}
              className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-[10px] rounded px-2 py-1.5 transition-colors"
            >
              Back
            </button>
          </>
        )}
      </div>
    </div>
  );
}
