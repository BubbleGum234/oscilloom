import { useState, useCallback, useEffect, useRef } from "react";
import type {
  PipelineGraph,
  BatchProgress,
  BatchResults,
  BatchFileResult,
  AggregateStatistics,
  SavedBatchSummary,
} from "../types/pipeline";
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
} from "../api/client";
import type { ReportSections, BatchReportConfig } from "../api/client";

export type BatchPhase =
  | "idle"
  | "files"
  | "uploading"
  | "running"
  | "results"
  | "report_config";

export interface UseBatchMode {
  // State
  active: boolean;
  phase: BatchPhase;
  selectedFiles: File[];
  fileMetadata: Record<number, Record<string, string>>;
  batchId: string | null;
  progress: BatchProgress | null;
  results: BatchResults | null;
  statistics: AggregateStatistics | null;
  error: string | null;
  saved: boolean;
  expandedFileId: string | null;
  fileDetail: BatchFileResult | null;
  detailLoading: boolean;
  selectedForRetry: Set<string>;
  savedBatches: SavedBatchSummary[];
  showSaved: boolean;
  // Report config
  reportClinicName: string;
  reportNotes: string;
  reportSections: ReportSections;
  generatingReports: boolean;
  // Actions
  toggleBatchMode: () => void;
  setPhase: (phase: BatchPhase) => void;
  addFiles: (files: File[]) => void;
  removeFile: (index: number) => void;
  updateMetadata: (index: number, field: string, value: string) => void;
  startBatchRun: (pipeline: PipelineGraph) => Promise<void>;
  cancelBatchRun: () => Promise<void>;
  downloadCsv: () => void;
  saveResults: () => Promise<void>;
  retryFailed: (pipeline: PipelineGraph) => Promise<void>;
  retrySelected: (pipeline: PipelineGraph) => Promise<void>;
  toggleRetrySelection: (fileId: string) => void;
  viewFileDetail: (fileId: string) => Promise<void>;
  loadSavedResults: () => Promise<void>;
  selectSavedBatch: (savedBatchId: string) => Promise<void>;
  showReportConfig: () => void;
  setReportClinicName: (name: string) => void;
  setReportNotes: (notes: string) => void;
  toggleReportSection: (key: keyof ReportSections) => void;
  generateReports: (pipeline: PipelineGraph) => Promise<void>;
  reset: () => void;
}

const DEFAULT_SECTIONS: ReportSections = {
  data_quality: false,
  pipeline_config: true,
  analysis_results: true,
  clinical_interpretation: true,
  visualizations: true,
  audit_trail: false,
  notes: true,
};

export function useBatchMode(): UseBatchMode {
  const [active, setActive] = useState(false);
  const [phase, setPhase] = useState<BatchPhase>("idle");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [fileMetadata, setFileMetadata] = useState<
    Record<number, Record<string, string>>
  >({});
  const [batchId, setBatchId] = useState<string | null>(null);
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [results, setResults] = useState<BatchResults | null>(null);
  const [statistics, setStatistics] = useState<AggregateStatistics | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [expandedFileId, setExpandedFileId] = useState<string | null>(null);
  const [fileDetail, setFileDetail] = useState<BatchFileResult | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedForRetry, setSelectedForRetry] = useState<Set<string>>(
    new Set()
  );
  const [savedBatches, setSavedBatches] = useState<SavedBatchSummary[]>([]);
  const [showSaved, setShowSaved] = useState(false);

  // Report config
  const [reportClinicName, setReportClinicName] = useState("");
  const [reportNotes, setReportNotes] = useState("");
  const [reportSections, setReportSections] =
    useState<ReportSections>(DEFAULT_SECTIONS);
  const [generatingReports, setGeneratingReports] = useState(false);

  // Track original results for retry merging
  const originalResultsRef = useRef<BatchResults | null>(null);

  // Poll progress every 2s while running
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
          // Merge with original results if this was a retry
          if (originalResultsRef.current) {
            const orig = originalResultsRef.current;
            const retryFileIds = new Set(
              res.file_results.map((fr) => fr.file_id)
            );
            const keptResults = orig.file_results.filter(
              (fr) => !retryFileIds.has(fr.file_id)
            );
            res.file_results = [...keptResults, ...res.file_results];
            res.summary.total = res.file_results.length;
            res.summary.completed = res.file_results.filter(
              (fr) => fr.status === "success"
            ).length;
            res.summary.failed = res.file_results.filter(
              (fr) => fr.status === "error"
            ).length;
            originalResultsRef.current = null;
          }
          setResults(res);
          setStatistics(res.statistics ?? null);
          setPhase("results");
        }
      } catch (err) {
        console.warn("Batch poll failed, will retry:", err instanceof Error ? err.message : String(err));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [phase, batchId]);

  const toggleBatchMode = useCallback(() => {
    setActive((prev) => {
      if (prev) {
        // Exiting batch mode — reset to idle
        setPhase("idle");
      } else {
        setPhase("files");
      }
      return !prev;
    });
  }, []);

  const addFiles = useCallback((files: File[]) => {
    setSelectedFiles((prev) => [...prev, ...files]);
    setError(null);
  }, []);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFileMetadata((prev) => {
      const next = { ...prev };
      delete next[index];
      // Re-index metadata after removal
      const reindexed: Record<number, Record<string, string>> = {};
      let newIdx = 0;
      for (let i = 0; i < Object.keys(prev).length + 1; i++) {
        if (i === index) continue;
        if (prev[i]) reindexed[newIdx] = prev[i];
        newIdx++;
      }
      return reindexed;
    });
  }, []);

  const updateMetadataFn = useCallback(
    (index: number, field: string, value: string) => {
      setFileMetadata((prev) => ({
        ...prev,
        [index]: { ...prev[index], [field]: value },
      }));
    },
    []
  );

  const startBatchRun = useCallback(
    async (pipeline: PipelineGraph) => {
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
        originalResultsRef.current = null;

        const { staged_files } = await stageFiles(selectedFiles);

        // Send metadata for each staged file
        for (let i = 0; i < staged_files.length; i++) {
          const meta = fileMetadata[i];
          if (meta && Object.values(meta).some((v) => v)) {
            await updateFileMetadata(staged_files[i].file_id, meta);
          }
        }

        const fileIds = staged_files.map((f) => f.file_id);
        const { batch_id } = await startBatch(fileIds, pipeline);
        setBatchId(batch_id);
        setSaved(false);
        setPhase("running");
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Batch failed to start."
        );
        setPhase("files");
      }
    },
    [selectedFiles, fileMetadata]
  );

  const cancelBatchRun = useCallback(async () => {
    if (batchId) {
      try {
        await cancelBatch(batchId);
      } catch (err) {
        console.warn("Batch cancel failed:", err instanceof Error ? err.message : String(err));
      }
    }
  }, [batchId]);

  const downloadCsv = useCallback(() => {
    if (!results?.metrics_csv) return;
    const blob = new Blob([results.metrics_csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_metrics_${batchId?.slice(0, 8) ?? "unknown"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results, batchId]);

  const saveResultsFn = useCallback(async () => {
    if (!batchId) return;
    try {
      await saveBatchResults(batchId);
      setSaved(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save.");
    }
  }, [batchId]);

  const retryFailed = useCallback(
    async (pipeline: PipelineGraph) => {
      if (!results) return;
      const failedIds = results.failed_files.map((ff) => ff.file_id);
      if (failedIds.length === 0) return;
      try {
        originalResultsRef.current = results;
        setPhase("running");
        setError(null);
        setProgress(null);
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
        originalResultsRef.current = null;
      }
    },
    [results]
  );

  const retrySelected = useCallback(
    async (pipeline: PipelineGraph) => {
      if (selectedForRetry.size === 0) return;
      try {
        originalResultsRef.current = results;
        setPhase("running");
        setError(null);
        setProgress(null);
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
        originalResultsRef.current = null;
      }
    },
    [selectedForRetry, results]
  );

  const toggleRetrySelection = useCallback((fileId: string) => {
    setSelectedForRetry((prev) => {
      const next = new Set(prev);
      if (next.has(fileId)) next.delete(fileId);
      else next.add(fileId);
      return next;
    });
  }, []);

  const viewFileDetail = useCallback(
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

  const loadSavedResults = useCallback(async () => {
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

  const selectSavedBatch = useCallback(async (savedBatchId: string) => {
    try {
      const res = await loadSavedBatch(savedBatchId);
      setBatchId(savedBatchId);
      setResults(res);
      setStatistics(res.statistics ?? null);
      setPhase("results");
      setShowSaved(false);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load saved batch."
      );
    }
  }, []);

  const showReportConfig = useCallback(() => {
    setPhase("report_config");
    setError(null);
  }, []);

  const toggleReportSection = useCallback((key: keyof ReportSections) => {
    setReportSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const generateReportsFn = useCallback(
    async (pipeline: PipelineGraph) => {
      if (!batchId) return;
      setGeneratingReports(true);
      setError(null);
      try {
        const pipelineConfig = pipeline.nodes.map((n) => ({
          node_id: n.id,
          node_type: n.node_type,
          label: n.label,
          parameters: n.parameters,
        }));

        const config: BatchReportConfig = {
          clinic_name: reportClinicName,
          notes: reportNotes,
          pipeline_config: reportSections.pipeline_config
            ? pipelineConfig
            : null,
          sections: reportSections,
        };

        const blob = await generateBatchReports(batchId, config);
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `batch_reports_${batchId.slice(0, 8)}.zip`;
        a.click();
        URL.revokeObjectURL(url);
        setPhase("results");
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to generate reports."
        );
      } finally {
        setGeneratingReports(false);
      }
    },
    [batchId, reportClinicName, reportNotes, reportSections]
  );

  const reset = useCallback(() => {
    setPhase("files");
    setSelectedFiles([]);
    setFileMetadata({});
    setBatchId(null);
    setProgress(null);
    setResults(null);
    setStatistics(null);
    setError(null);
    setSaved(false);
    setExpandedFileId(null);
    setFileDetail(null);
    setSelectedForRetry(new Set());
    setShowSaved(false);
    originalResultsRef.current = null;
  }, []);

  return {
    active,
    phase,
    selectedFiles,
    fileMetadata,
    batchId,
    progress,
    results,
    statistics,
    error,
    saved,
    expandedFileId,
    fileDetail,
    detailLoading,
    selectedForRetry,
    savedBatches,
    showSaved,
    reportClinicName,
    reportNotes,
    reportSections,
    generatingReports,
    toggleBatchMode,
    setPhase,
    addFiles,
    removeFile,
    updateMetadata: updateMetadataFn,
    startBatchRun,
    cancelBatchRun,
    downloadCsv,
    saveResults: saveResultsFn,
    retryFailed,
    retrySelected,
    toggleRetrySelection,
    viewFileDetail,
    loadSavedResults,
    selectSavedBatch,
    showReportConfig,
    setReportClinicName,
    setReportNotes,
    toggleReportSection,
    generateReports: generateReportsFn,
    reset,
  };
}
