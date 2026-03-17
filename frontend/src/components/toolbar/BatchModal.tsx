import React, { useState, useRef, useEffect, useCallback } from "react";
import type { PipelineGraph, BatchProgress, BatchFileResult, BatchResults, SavedBatchSummary } from "../../types/pipeline";
import type { ReportSections, BatchReportConfig } from "../../api/client";
import {
  stageFiles,
  startBatch,
  getBatchProgress,
  getBatchResults,
  cancelBatch,
  updateFileMetadata,
  getBatchFileDetail,
  generateBatchReports,
  saveBatchResults,
  listSavedBatches,
  loadSavedBatch,
} from "../../api/client";

interface BatchModalProps {
  onClose: () => void;
  pipeline: PipelineGraph;
}

type BatchPhase = "select" | "uploading" | "running" | "results" | "report_config";

const DEFAULT_SECTIONS: ReportSections = {
  data_quality: false,       // No session_info in batch context
  pipeline_config: true,
  analysis_results: true,
  clinical_interpretation: true,
  visualizations: true,
  audit_trail: false,        // No audit log in batch context
  notes: true,
};

const SECTION_LABELS: Record<keyof ReportSections, { label: string; description: string }> = {
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

export function BatchModal({ onClose, pipeline }: BatchModalProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [phase, setPhase] = useState<BatchPhase>("select");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [results, setResults] = useState<BatchResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileMetadata, setFileMetadata] = useState<
    Record<number, Record<string, string>>
  >({});
  const [expandedFileId, setExpandedFileId] = useState<string | null>(null);
  const [fileDetail, setFileDetail] = useState<BatchFileResult | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedForRetry, setSelectedForRetry] = useState<Set<string>>(
    new Set()
  );
  const [saved, setSaved] = useState(false);
  const [savedBatches, setSavedBatches] = useState<SavedBatchSummary[]>([]);
  const [showSaved, setShowSaved] = useState(false);

  // Report configuration state
  const [reportClinicName, setReportClinicName] = useState("");
  const [reportNotes, setReportNotes] = useState("");
  const [reportSections, setReportSections] = useState<ReportSections>(DEFAULT_SECTIONS);
  const [generatingReports, setGeneratingReports] = useState(false);

  // Poll progress every 2 seconds while running
  useEffect(() => {
    if (phase !== "running" || !batchId) return;

    const interval = setInterval(async () => {
      try {
        const prog = await getBatchProgress(batchId);
        setProgress(prog);
        if (
          prog.status === "complete" ||
          prog.status === "failed" ||
          prog.status === "cancelled"
        ) {
          clearInterval(interval);
          const res = await getBatchResults(batchId);
          setResults(res);
          setPhase("results");
        }
      } catch (pollErr) {
        console.warn("Batch progress poll failed (will retry):", pollErr);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [phase, batchId]);

  const handleFilesSelected = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const fileList = e.target.files;
      if (!fileList) return;
      setSelectedFiles(Array.from(fileList));
      setError(null);
    },
    []
  );

  const handleStart = useCallback(async () => {
    if (selectedFiles.length === 0) {
      setError("Select at least one EEG file.");
      return;
    }
    if (pipeline.nodes.length === 0) {
      setError("Build a pipeline on the canvas first.");
      return;
    }

    try {
      setPhase("uploading");
      setError(null);

      const { staged_files } = await stageFiles(selectedFiles);

      // Send metadata for each staged file (Tier 4B)
      for (let i = 0; i < staged_files.length; i++) {
        const meta = fileMetadata[i];
        if (meta && Object.values(meta).some((v) => v)) {
          await updateFileMetadata(staged_files[i].file_id, meta);
        }
      }

      const fileIds = staged_files.map((f) => f.file_id);

      const { batch_id } = await startBatch(fileIds, pipeline);
      setBatchId(batch_id);
      setPhase("running");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Batch failed to start."
      );
      setPhase("select");
    }
  }, [selectedFiles, pipeline]);

  const handleCancel = useCallback(async () => {
    if (batchId) {
      try {
        await cancelBatch(batchId);
      } catch (err) {
        console.warn("Batch cancel failed:", err instanceof Error ? err.message : String(err));
      }
    }
    onClose();
  }, [batchId, onClose]);

  const handleDownloadCsv = useCallback(() => {
    if (!results?.metrics_csv) return;
    const blob = new Blob([results.metrics_csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_metrics_${batchId?.slice(0, 8) ?? "unknown"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results, batchId]);

  const handleViewDetail = useCallback(
    async (fileId: string) => {
      if (expandedFileId === fileId) {
        setExpandedFileId(null);
        setFileDetail(null);
        return;
      }
      setExpandedFileId(fileId);
      setDetailLoading(true);
      try {
        const detail = await getBatchFileDetail(batchId!, fileId);
        setFileDetail(detail);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load detail."
        );
      } finally {
        setDetailLoading(false);
      }
    },
    [expandedFileId, batchId]
  );

  const handleRetryFailed = useCallback(async () => {
    if (!results) return;
    const failedIds = results.failed_files.map((ff) => ff.file_id);
    if (failedIds.length === 0) return;
    try {
      setPhase("running");
      setError(null);
      setProgress(null);
      setResults(null);
      setExpandedFileId(null);
      setFileDetail(null);
      setSelectedForRetry(new Set());
      const { batch_id } = await startBatch(failedIds, pipeline);
      setBatchId(batch_id);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Retry failed to start."
      );
      setPhase("results");
    }
  }, [results, pipeline]);

  const handleRetrySelected = useCallback(async () => {
    if (selectedForRetry.size === 0) return;
    try {
      setPhase("running");
      setError(null);
      setProgress(null);
      setResults(null);
      setExpandedFileId(null);
      setFileDetail(null);
      const retryIds = Array.from(selectedForRetry);
      setSelectedForRetry(new Set());
      const { batch_id } = await startBatch(retryIds, pipeline);
      setBatchId(batch_id);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Retry failed to start."
      );
      setPhase("results");
    }
  }, [selectedForRetry, pipeline]);

  // Show report config instead of directly downloading
  const handleShowReportConfig = useCallback(() => {
    setPhase("report_config");
    setError(null);
  }, []);

  const handleGenerateReports = useCallback(async () => {
    if (!batchId) return;
    setGeneratingReports(true);
    setError(null);
    try {
      // Build pipeline config from the pipeline prop
      const pipelineConfig = pipeline.nodes.map((n) => ({
        node_id: n.id,
        node_type: n.node_type,
        label: n.label,
        parameters: n.parameters,
      }));

      const config: BatchReportConfig = {
        clinic_name: reportClinicName,
        notes: reportNotes,
        pipeline_config: reportSections.pipeline_config ? pipelineConfig : null,
        sections: reportSections,
      };

      const blob = await generateBatchReports(batchId, config);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch_reports_${batchId.slice(0, 8)}.zip`;
      a.click();
      URL.revokeObjectURL(url);

      // Go back to results
      setPhase("results");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to generate reports."
      );
    } finally {
      setGeneratingReports(false);
    }
  }, [batchId, pipeline, reportClinicName, reportNotes, reportSections]);

  const handleSaveResults = useCallback(async () => {
    if (!batchId) return;
    try {
      await saveBatchResults(batchId);
      setSaved(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to save."
      );
    }
  }, [batchId]);

  const handleLoadSaved = useCallback(async () => {
    try {
      const { saved_batches } = await listSavedBatches();
      setSavedBatches(saved_batches);
      setShowSaved(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to list saved batches."
      );
    }
  }, []);

  const handleSelectSaved = useCallback(async (savedBatchId: string) => {
    try {
      const res = await loadSavedBatch(savedBatchId);
      setBatchId(savedBatchId);
      setResults(res);
      setPhase("results");
      setShowSaved(false);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load saved batch."
      );
    }
  }, []);

  const toggleReportSection = (key: keyof ReportSections) => {
    setReportSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Sections not available in batch context
  const sectionAvailability: Record<keyof ReportSections, boolean> = {
    data_quality: false,       // No session_info per-file in batch
    pipeline_config: true,
    analysis_results: true,
    clinical_interpretation: true,
    visualizations: true,
    audit_trail: false,        // No audit log in batch
    notes: true,
  };

  const enabledSectionCount = SECTION_ORDER.filter(
    (k) => reportSections[k] && sectionAvailability[k]
  ).length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-2xl w-[32rem] max-h-[80vh] flex flex-col p-5">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-slate-100 font-semibold text-sm">
            {phase === "report_config" ? "Configure Batch Reports" : "Batch Processing"}
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg leading-none"
          >
            &times;
          </button>
        </div>

        {/* Phase: File Selection */}
        {phase === "select" && (
          <>
            <p className="text-slate-400 text-xs mb-3">
              Select multiple EEG files to process through the current pipeline.
              Files are processed sequentially to minimize memory usage.
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".edf,.fif,.bdf,.set,.vhdr,.cnt"
              className="hidden"
              onChange={handleFilesSelected}
            />
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs rounded px-3 py-2 transition-colors"
              >
                Select EEG Files...
              </button>
              <button
                onClick={handleLoadSaved}
                className="bg-slate-700 hover:bg-slate-600 text-slate-400 text-xs rounded px-3 py-2 transition-colors"
              >
                Load Previous
              </button>
            </div>

            {showSaved && savedBatches.length > 0 && (
              <div className="mb-3 max-h-32 overflow-y-auto text-xs">
                <div className="text-slate-400 mb-1 font-semibold">Saved Results:</div>
                {savedBatches.map((sb) => (
                  <button
                    key={sb.batch_id}
                    onClick={() => handleSelectSaved(sb.batch_id)}
                    className="w-full text-left bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded px-2 py-1 mb-0.5 text-xs"
                  >
                    {sb.batch_id.slice(0, 8)}... — {sb.completed}/{sb.total} completed
                    {" "}({new Date(sb.saved_at * 1000).toLocaleDateString()})
                  </button>
                ))}
              </div>
            )}
            {showSaved && savedBatches.length === 0 && (
              <div className="mb-3 text-slate-500 text-xs">No saved results found.</div>
            )}

            {selectedFiles.length > 0 && (
              <div className="mb-3 max-h-60 overflow-y-auto text-xs">
                <div className="text-slate-400 mb-1">
                  {selectedFiles.length} file(s) selected:
                </div>
                <table className="w-full">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-1 px-1">File</th>
                      <th className="text-left py-1 px-1 w-20">Subject ID</th>
                      <th className="text-left py-1 px-1 w-20">Group</th>
                      <th className="text-left py-1 px-1 w-20">Condition</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedFiles.map((f, i) => (
                      <tr key={i} className="border-b border-slate-700/50">
                        <td className="py-1 px-1 text-slate-300 truncate max-w-[10rem]" title={f.name}>
                          {f.name}
                        </td>
                        {(["subject_id", "group", "condition"] as const).map(
                          (field) => (
                            <td key={field} className="py-1 px-1">
                              <input
                                type="text"
                                className="bg-slate-700 text-slate-200 text-xs rounded px-1 py-0.5 w-full"
                                placeholder={field === "subject_id" ? "ID" : field === "group" ? "Group" : "Cond"}
                                value={fileMetadata[i]?.[field] || ""}
                                onChange={(e) =>
                                  setFileMetadata((prev) => ({
                                    ...prev,
                                    [i]: { ...prev[i], [field]: e.target.value },
                                  }))
                                }
                              />
                            </td>
                          )
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {/* Phase: Uploading */}
        {phase === "uploading" && (
          <div className="text-slate-400 text-xs py-8 text-center">
            Uploading {selectedFiles.length} file(s) to server...
          </div>
        )}

        {/* Phase: Running */}
        {phase === "running" && progress && (
          <div className="mb-3">
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Processing: {progress.current_file || "..."}</span>
              <span>
                {progress.completed + progress.failed} / {progress.total}
              </span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-cyan-500 h-2 rounded-full transition-all"
                style={{
                  width: `${((progress.completed + progress.failed) / Math.max(progress.total, 1)) * 100}%`,
                }}
              />
            </div>
            {progress.failed > 0 && (
              <div className="text-red-400 text-xs mt-1">
                {progress.failed} file(s) failed
              </div>
            )}
          </div>
        )}

        {/* Phase: Running but no progress yet */}
        {phase === "running" && !progress && (
          <div className="text-slate-400 text-xs py-8 text-center">
            Starting batch processing...
          </div>
        )}

        {/* Phase: Results */}
        {phase === "results" && results && (
          <div className="flex-1 overflow-y-auto">
            <div className="grid grid-cols-3 gap-2 mb-3">
              <div className="bg-slate-700 rounded p-2 text-center">
                <div className="text-cyan-400 text-lg font-bold">
                  {results.summary.completed}
                </div>
                <div className="text-slate-400 text-xs">Completed</div>
              </div>
              <div className="bg-slate-700 rounded p-2 text-center">
                <div className="text-red-400 text-lg font-bold">
                  {results.summary.failed}
                </div>
                <div className="text-slate-400 text-xs">Failed</div>
              </div>
              <div className="bg-slate-700 rounded p-2 text-center">
                <div className="text-slate-200 text-lg font-bold">
                  {results.summary.runtime_s}s
                </div>
                <div className="text-slate-400 text-xs">Runtime</div>
              </div>
            </div>

            {/* Per-file results table */}
            <div className="max-h-64 overflow-y-auto mb-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700">
                    <th className="py-1 px-1 w-6"></th>
                    <th className="text-left py-1 px-2">File</th>
                    <th className="text-left py-1 px-2">Status</th>
                    <th className="text-left py-1 px-2 w-12"></th>
                  </tr>
                </thead>
                <tbody>
                  {results.file_results.map((fr, i) => (
                    <React.Fragment key={i}>
                      <tr className="border-b border-slate-700/50">
                        <td className="py-1 px-1">
                          <input
                            type="checkbox"
                            checked={selectedForRetry.has(fr.file_id)}
                            onChange={(e) =>
                              setSelectedForRetry((prev) => {
                                const next = new Set(prev);
                                if (e.target.checked) next.add(fr.file_id);
                                else next.delete(fr.file_id);
                                return next;
                              })
                            }
                            className="accent-cyan-500"
                          />
                        </td>
                        <td className="py-1 px-2 text-slate-300">
                          {fr.filename}
                        </td>
                        <td className="py-1 px-2">
                          {fr.status === "success" ? (
                            <span className="text-cyan-400">Success</span>
                          ) : (
                            <span
                              className="text-red-400"
                              title={fr.error || ""}
                            >
                              Failed
                            </span>
                          )}
                        </td>
                        <td className="py-1 px-2">
                          {fr.status === "success" && (
                            <button
                              onClick={() => handleViewDetail(fr.file_id)}
                              className="text-blue-400 hover:text-blue-300 text-xs underline"
                            >
                              {expandedFileId === fr.file_id ? "Hide" : "View"}
                            </button>
                          )}
                        </td>
                      </tr>
                      {expandedFileId === fr.file_id && (
                        <tr>
                          <td colSpan={4} className="p-2 bg-slate-700/30">
                            {detailLoading ? (
                              <div className="text-slate-400 text-xs">Loading...</div>
                            ) : fileDetail ? (
                              <div className="space-y-2">
                                {Object.keys(fileDetail.metrics).length > 0 && (
                                  <div>
                                    <div className="text-slate-400 text-xs font-semibold mb-1">Metrics</div>
                                    {Object.entries(fileDetail.metrics).map(([key, val]) => (
                                      <div key={key} className="text-xs text-slate-300 ml-2">
                                        <span className="text-slate-400">{key}:</span>{" "}
                                        {typeof val === "number" ? Number(val).toFixed(4) : String(val)}
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {Object.entries(fileDetail.node_results)
                                  .filter(
                                    ([, nr]) =>
                                      typeof nr.data === "string" &&
                                      nr.data.startsWith("data:image")
                                  )
                                  .map(([nodeId, nr]) => (
                                    <div key={nodeId}>
                                      <div className="text-xs text-slate-400 mb-1">
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

            {/* Failed file details */}
            {results.failed_files.length > 0 && (
              <div className="mb-3">
                <div className="text-red-400 text-xs font-semibold mb-1">
                  Failed Files:
                </div>
                {results.failed_files.map((ff, i) => (
                  <div
                    key={i}
                    className="text-xs text-red-300 bg-red-950 rounded px-2 py-1 mb-1"
                  >
                    {ff.filename}: {ff.error}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Phase: Report Configuration */}
        {phase === "report_config" && (
          <div className="flex-1 overflow-y-auto space-y-4">
            <p className="text-slate-400 text-xs">
              Configure the PDF reports that will be generated for each
              successful file ({results?.summary.completed ?? 0} files).
              Each file gets its own PDF, bundled into a ZIP download.
            </p>

            {/* Clinic name */}
            <div>
              <label className="block text-slate-400 text-xs mb-1">
                Clinic / Lab (applied to all reports)
              </label>
              <input
                type="text"
                value={reportClinicName}
                onChange={(e) => setReportClinicName(e.target.value)}
                placeholder="e.g. Neurology Dept., City Hospital"
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
              />
            </div>

            {/* Section toggles */}
            <div>
              <label className="block text-slate-400 text-xs mb-2">
                Report Sections ({enabledSectionCount} enabled)
              </label>
              <div className="bg-slate-900 border border-slate-700 rounded p-2 space-y-1">
                {SECTION_ORDER.map((key) => {
                  const available = sectionAvailability[key];
                  const enabled = reportSections[key];
                  const info = SECTION_LABELS[key];

                  return (
                    <label
                      key={key}
                      className={`flex items-start gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                        available
                          ? "hover:bg-slate-800"
                          : "opacity-40 cursor-not-allowed"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={enabled && available}
                        disabled={!available}
                        onChange={() => toggleReportSection(key)}
                        className="mt-0.5 rounded border-slate-600"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs text-slate-200 flex items-center gap-2">
                          {info.label}
                          {!available && (
                            <span className="text-[10px] text-slate-500 bg-slate-800 px-1.5 rounded">
                              not available
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-slate-500 mt-0.5">
                          {info.description}
                        </div>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Notes */}
            <div>
              <label className="block text-slate-400 text-xs mb-1">
                Clinician Notes (included in every report)
              </label>
              <textarea
                value={reportNotes}
                onChange={(e) => setReportNotes(e.target.value)}
                placeholder="e.g. Resting-state protocol, eyes closed. Study cohort: healthy controls."
                rows={3}
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400 resize-none"
              />
            </div>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="mt-2 text-red-400 text-xs bg-red-950 border border-red-800 rounded px-2 py-1.5">
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2 mt-4">
          {phase === "select" && (
            <>
              <button
                onClick={handleStart}
                disabled={selectedFiles.length === 0}
                className="flex-1 bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
              >
                Start Batch ({selectedFiles.length} files)
              </button>
              <button
                onClick={onClose}
                className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-3 py-2 transition-colors"
              >
                Cancel
              </button>
            </>
          )}

          {phase === "uploading" && (
            <button
              onClick={onClose}
              className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-3 py-2 transition-colors"
            >
              Cancel
            </button>
          )}

          {phase === "running" && (
            <button
              onClick={handleCancel}
              className="flex-1 bg-red-800 hover:bg-red-700 text-white text-xs rounded px-3 py-2 transition-colors"
            >
              Cancel Batch
            </button>
          )}

          {phase === "results" && (
            <>
              <button
                onClick={handleDownloadCsv}
                disabled={!results?.metrics_csv}
                className="flex-1 bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
              >
                Download CSV
              </button>
              <button
                onClick={handleShowReportConfig}
                disabled={!results || results.summary.completed === 0}
                className="flex-1 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
              >
                Reports (ZIP)
              </button>
              <button
                onClick={handleSaveResults}
                disabled={saved}
                className="bg-sky-700 hover:bg-sky-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
              >
                {saved ? "Saved" : "Save"}
              </button>
              {results && results.failed_files.length > 0 && (
                <button
                  onClick={handleRetryFailed}
                  className="bg-amber-700 hover:bg-amber-600 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
                >
                  Retry Failed ({results.failed_files.length})
                </button>
              )}
              {selectedForRetry.size > 0 && (
                <button
                  onClick={handleRetrySelected}
                  className="bg-blue-700 hover:bg-blue-600 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
                >
                  Retry Selected ({selectedForRetry.size})
                </button>
              )}
              <button
                onClick={onClose}
                className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-3 py-2 transition-colors"
              >
                Close
              </button>
            </>
          )}

          {phase === "report_config" && (
            <>
              <button
                onClick={handleGenerateReports}
                disabled={generatingReports || enabledSectionCount === 0}
                className="flex-1 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
              >
                {generatingReports ? "Generating..." : "Download Reports (ZIP)"}
              </button>
              <button
                onClick={() => setPhase("results")}
                className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-3 py-2 transition-colors"
              >
                Back
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
