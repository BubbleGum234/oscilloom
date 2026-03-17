import { useRef } from "react";
import { useNavigate } from "react-router-dom";
import { SURFACE } from "../../constants/theme";

interface ToolbarProps {
  running: boolean;
  sessionId: string | null;
  hasRawOutput: boolean;
  hasMetricsOutput: boolean;
  pipelineName: string;
  onPipelineNameChange: (name: string) => void;
  onRun: () => void;
  onExport: () => void;
  onExportModal: () => void;
  onDownloadFif: () => void;
  onClear: () => void;
  // TASK-18: pipeline save/load
  onSavePipeline: () => void;
  onLoadPipeline: (file: File) => void;
  onGenerateReport: () => void;
  onBatchProcess: () => void;
  batchActive?: boolean;
  onBatchToggle?: () => void;
  batchProgress?: { completed: number; total: number } | null;
  onPublishAsNode: () => void;
  onUndo: () => void;
  canUndo?: boolean;
  loadingFile?: boolean;
  loadingExport?: boolean;
  loadingFif?: boolean;
  loadingReport?: boolean;
  presentationMode?: boolean;
  onTogglePresentationMode?: () => void;
  onBidsExport?: () => void;
}

export function Toolbar({
  running,
  sessionId,
  hasRawOutput,
  hasMetricsOutput,
  pipelineName,
  onPipelineNameChange,
  onRun,
  onExport: _onExport,
  onExportModal,
  onDownloadFif,
  onClear,
  onSavePipeline,
  onLoadPipeline,
  onGenerateReport,
  onBatchProcess,
  batchActive,
  onBatchToggle,
  batchProgress,
  onPublishAsNode,
  onUndo,
  canUndo,
  loadingFile,
  loadingExport: _loadingExport,
  loadingFif,
  loadingReport,
  presentationMode,
  onTogglePresentationMode,
  onBidsExport,
}: ToolbarProps) {
  const loadInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b flex-shrink-0 h-12" style={{ borderColor: SURFACE.elevated, background: SURFACE.toolbar }}>
      <button
        onClick={() => navigate("/")}
        title="Back to Home"
        className="text-slate-400 hover:text-slate-200 transition-colors mr-1"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2z" />
        </svg>
      </button>
      <img src="/favicon.png" alt="Oscilloom" className="w-6 h-6 rounded" />
      <span className="text-slate-100 font-bold text-sm tracking-tight mr-1">
        Oscilloom
      </span>
      <div
        className={`w-2 h-2 rounded-full flex-shrink-0 transition-colors ${
          sessionId ? "bg-cyan-500" : "bg-slate-600"
        }`}
        title={sessionId ? "EEG session active" : "No file loaded"}
      />

      <input
        type="text"
        value={pipelineName}
        onChange={(e) => onPipelineNameChange(e.target.value)}
        placeholder="Pipeline name"
        className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs w-36 focus:outline-none focus:border-slate-400"
        title="Pipeline name — used as the exported filename"
      />

      <div className="w-px h-5 bg-slate-700 mx-1" />

      <button
        onClick={onUndo}
        disabled={!canUndo}
        title="Undo (Ctrl+Z)"
        className="bg-slate-800 hover:bg-slate-700 disabled:bg-slate-800 disabled:text-slate-600 text-slate-400 hover:text-slate-200 text-xs rounded px-2 py-1.5 transition-colors"
      >
        ↩ Undo
      </button>

      <button
        onClick={onRun}
        disabled={running || !sessionId || loadingFile}
        className="bg-cyan-700 hover:bg-cyan-600 disabled:bg-slate-800 disabled:text-slate-600 text-white text-[13px] rounded px-3 py-1.5 transition-colors font-medium"
      >
        {running ? "Running\u2026" : loadingFile ? "Loading\u2026" : "\u25B6  Run"}
      </button>

      <button
        onClick={onExportModal}
        disabled={!sessionId}
        className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-slate-200 text-xs rounded px-3 py-1.5 transition-colors"
      >
        Export
      </button>

      <button
        onClick={onDownloadFif}
        disabled={running || !sessionId || !hasRawOutput || loadingFif}
        title={hasRawOutput ? "Download processed EEG as .fif" : "Run pipeline first to enable download"}
        className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-slate-200 text-xs rounded px-3 py-1.5 transition-colors"
      >
        {loadingFif ? "Downloading\u2026" : "\u2193 Download .fif"}
      </button>

      {onBidsExport && (
        <button
          onClick={onBidsExport}
          disabled={!sessionId}
          title="Export processed data in BIDS format"
          className="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-slate-200 text-xs rounded px-3 py-1.5 transition-colors"
        >
          BIDS Export
        </button>
      )}

      <div className="w-px h-5 bg-slate-700 mx-1" />

      {/* TASK-18: Pipeline save/load */}
      <button
        onClick={onSavePipeline}
        title="Save pipeline structure to a .json file"
        className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs rounded px-3 py-1.5 transition-colors"
      >
        Save Pipeline
      </button>

      <input
        ref={loadInputRef}
        type="file"
        accept=".json"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onLoadPipeline(file);
          e.target.value = "";
        }}
      />
      <button
        onClick={() => loadInputRef.current?.click()}
        title="Load a previously saved pipeline .json file"
        className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs rounded px-3 py-1.5 transition-colors"
      >
        Load Pipeline
      </button>

      <button
        onClick={onGenerateReport}
        disabled={!hasMetricsOutput || loadingReport}
        title={
          hasMetricsOutput
            ? "Generate a PDF report from metrics and plots"
            : "Run a pipeline with Clinical metric nodes first"
        }
        className="bg-slate-700/60 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-600 text-slate-200 border border-slate-600 text-xs rounded px-3 py-1.5 transition-colors"
      >
        {loadingReport ? "Generating\u2026" : "Generate Report"}
      </button>

      <button
        onClick={onBatchToggle ?? onBatchProcess}
        title={batchActive ? "Exit Batch Mode" : "Run a pipeline against multiple EEG files"}
        className={
          batchActive
            ? "bg-amber-600 hover:bg-amber-500 text-white border border-amber-500 text-xs rounded px-3 py-1.5 transition-colors"
            : "bg-slate-700/60 hover:bg-slate-600 text-slate-200 border border-slate-600 text-xs rounded px-3 py-1.5 transition-colors"
        }
      >
        {batchActive ? "Exit Batch" : "Batch"}
        {batchProgress && (
          <span className="ml-1 font-mono text-[10px]">
            {batchProgress.completed}/{batchProgress.total}
          </span>
        )}
      </button>

      <button
        onClick={onPublishAsNode}
        title="Wrap the current pipeline into a reusable compound node"
        className="bg-slate-700/60 hover:bg-slate-600 text-slate-200 border border-slate-600 text-xs rounded px-3 py-1.5 transition-colors"
      >
        Publish as Node
      </button>

      <button
        onClick={onClear}
        className="bg-red-500/20 hover:bg-red-500/30 text-red-400 hover:text-red-300 text-xs rounded px-3 py-1.5 transition-colors"
      >
        Clear
      </button>

      <div className="flex-1" />

      {onTogglePresentationMode && (
        <button
          onClick={onTogglePresentationMode}
          title="Toggle Presentation Mode (light theme for screenshots)"
          className={`text-xs rounded px-3 py-1.5 transition-colors border ${
            presentationMode
              ? "bg-white text-slate-900 border-slate-300"
              : "bg-slate-700/60 text-slate-200 border-slate-600 hover:bg-slate-600"
          }`}
        >
          {presentationMode ? "☀ Presentation" : "☀ Present"}
        </button>
      )}
    </div>
  );
}
