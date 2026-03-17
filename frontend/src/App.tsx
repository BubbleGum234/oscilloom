import { useState, useCallback, useEffect, useRef } from "react";
import { useNodesState, useEdgesState } from "@xyflow/react";
import type { Node, Edge, NodeChange, EdgeChange } from "@xyflow/react";
import { useParams, useNavigate } from "react-router-dom";

import { useNodeRegistry } from "./hooks/useNodeRegistry";
import { PublishAsNodeModal } from "./components/toolbar/PublishAsNodeModal";
import { usePipelineRunner } from "./hooks/usePipelineRunner";
import { useAuditLog } from "./hooks/useAuditLog";
import { useHistory } from "./hooks/useHistory";
import { useBatchMode } from "./hooks/useBatchMode";
import { useToast } from "./components/ui/Toast";
import { usePanelResize } from "./hooks/usePanelResize";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useSessionManager } from "./hooks/useSessionManager";
import { useExportActions } from "./hooks/useExportActions";
import { useWorkflowPersistence } from "./hooks/useWorkflowPersistence";

import { NodePalette } from "./components/canvas/NodePalette";
import { CanvasPane } from "./components/canvas/CanvasPane";
import { RightPanel } from "./components/panels/RightPanel";
import { Toolbar } from "./components/toolbar/Toolbar";
import { ReportModal } from "./components/toolbar/ReportModal";
import type { NodeOutput } from "./components/toolbar/ReportModal";
import { BatchPanel } from "./components/panels/BatchPanel";
import { CompoundInspectorModal } from "./components/toolbar/CompoundInspectorModal";
import ExportModal from "./components/toolbar/ExportModal";
import BidsExportModal from "./components/toolbar/BidsExportModal";

import { openMneBrowser, exportNodeOutput, executeFromNode } from "./api/client";
import { serializePipeline } from "./utils/serializePipeline";
import { getNextNodeId } from "./utils/nodeId";
import type { ParameterSchema, NodeData } from "./types/pipeline";

import { ComparisonPanel } from "./components/panels/ComparisonPanel";
import { useRunHistory } from "./hooks/useRunHistory";
import { ConfirmDialog } from "./components/ui/ConfirmDialog";

export default function App() {
  const { workflowId } = useParams<{ workflowId?: string }>();
  const navigate = useNavigate();
  const { registry, loading, error, refresh: refreshRegistry } = useNodeRegistry();
  const { toast } = useToast();

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Store only the ID so the panel always reads fresh node data from `nodes`.
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const selectedNode = selectedNodeId
    ? (nodes as Node[]).find((n) => n.id === selectedNodeId) ?? null
    : null;

  const [presentationMode, setPresentationMode] = useState(false);

  // ── Extracted hooks ─────────────────────────────────────────────────────
  const { leftPanelWidth, rightPanelWidth, startLeftDrag, startRightDrag } = usePanelResize();

  const { run, running, result, error: runError } = usePipelineRunner();
  const { log: auditLog, append: appendAudit, clear: clearAudit } = useAuditLog();

  // TASK-17: undo history (node/edge deletions)
  const { push: pushHistory, pop: popHistory, canUndo } = useHistory();

  const batch = useBatchMode();

  // ── Research Workbench: run history + comparison ─────────────────────────
  const { history: runHistory, capture: captureRun, restore: restoreRun, rename: renameRun, remove: removeRun } = useRunHistory();
  const [comparisonRuns, setComparisonRuns] = useState<[string, string] | null>(null);
  const [viewingHistoryResult, setViewingHistoryResult] = useState<Record<string, import("./types/pipeline").NodeResult> | null>(null);

  // ── Workflow persistence (must come before hooks that need markDirty) ──
  const {
    pipelineName, setPipelineName,
    markDirty,
    handleSavePipeline, handleLoadPipeline, handleClear,
    showClearConfirm, confirmClear, cancelClear,
  } = useWorkflowPersistence(
    nodes as Node[], edges as Edge[],
    setNodes, setEdges, registry,
    workflowId, navigate, clearAudit, setSelectedNodeId, toast,
  );

  // ── Session management ──────────────────────────────────────────────────
  const {
    sessionId, sessionInfo, loadingFile, handleFileLoad, setSessionInfo,
  } = useSessionManager(setNodes, markDirty, toast);

  // ── Export actions ──────────────────────────────────────────────────────
  const {
    loadingExport, loadingFif, loadingReport,
    handleExport, handleDownloadFif, handleGenerateReport,
  } = useExportActions(
    nodes as Node[], edges as Edge[],
    sessionId, pipelineName, auditLog, result, sessionInfo, toast,
  );

  // ── Multi-select count ──────────────────────────────────────────────────
  const [selectionCount, setSelectionCount] = useState(0);

  // True when the last run produced at least one Raw output (enables Download .fif).
  const hasRawOutput = result
    ? Object.values(result.node_results).some(
      (r) => r.status === "success" && r.output_type === "Raw"
    )
    : false;

  // True when the last run produced at least one metrics output (enables Generate Report).
  const hasMetricsOutput = result
    ? Object.values(result.node_results).some(
      (r) => r.status === "success" && (r.output_type === "dict" || r.metrics != null)
    )
    : false;

  // Build the list of reportable outputs for the ReportModal output selector.
  const availableOutputs: NodeOutput[] = result
    ? Object.entries(result.node_results)
      .filter(([, r]) => r.status === "success")
      .map(([nodeId, r]) => {
        const isPlot =
          typeof r.data === "string" &&
          r.data.startsWith("data:image/png;base64,");
        const isMetrics = r.output_type === "dict" || r.metrics != null;
        if (!isPlot && !isMetrics) return null;

        // Find the canvas node to get the user-visible label
        const canvasNode = (nodes as Node[]).find((n) => n.id === nodeId);
        const label = canvasNode
          ? (canvasNode.data as NodeData).label || nodeId
          : nodeId;

        return {
          nodeId,
          label,
          nodeType: r.node_type || "unknown",
          kind: isPlot ? "plot" : "metrics",
        } as NodeOutput;
      })
      .filter((o): o is NodeOutput => o !== null)
    : [];

  const [showReportModal, setShowReportModal] = useState(false);
  const [showPublishModal, setShowPublishModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [showBidsModal, setShowBidsModal] = useState(false);
  const [inspectCompoundId, setInspectCompoundId] = useState<string | null>(null);
  const [inspectCompoundName, setInspectCompoundName] = useState("");

  // Ref for rename callback — populated after handleNodeRename is defined.
  const nodeRenameRef = useRef<(nodeId: string, label: string) => void>(() => { });

  // Sync pipeline results + callbacks into nodes state so React Flow v12
  // properly receives and renders nodeResult in each GenericNode.
  useEffect(() => {
    setNodes((nds) =>
      nds.map((n) => ({
        ...n,
        data: {
          ...n.data,
          nodeResult: result?.node_results?.[n.id] ?? null,
          isRunning: running,
          batchMode: batch.active,
          onRename: (nodeId: string, label: string) => nodeRenameRef.current(nodeId, label),
        },
      }))
    );
  }, [result, running, batch.active, setNodes]);

  // ── TASK-17: Undo (Ctrl+Z / Cmd+Z) ──────────────────────────────────────
  // Wrap change handlers: push a snapshot before any removal so it can be restored.
  const handleNodesChange = useCallback(
    (changes: NodeChange<Node>[]) => {
      if (changes.some((c) => c.type === "remove")) {
        pushHistory(nodes as Node[], edges as Edge[]);
      }
      markDirty();
      onNodesChange(changes);
    },
    [nodes, edges, onNodesChange, pushHistory, markDirty]
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      if (changes.some((c) => c.type === "remove")) {
        pushHistory(nodes as Node[], edges as Edge[]);
      }
      markDirty();
      onEdgesChange(changes);
    },
    [nodes, edges, onEdgesChange, pushHistory, markDirty]
  );

  const handleUndo = useCallback(() => {
    const snapshot = popHistory();
    if (snapshot) {
      setNodes(snapshot.nodes);
      setEdges(snapshot.edges);
    }
  }, [popHistory, setNodes, setEdges]);

  // ── Duplicate selected node ─────────────────────────────────────────────
  const handleDuplicateNode = useCallback(() => {
    if (!selectedNodeId) {
      toast("Select a node first to duplicate", "warning");
      return;
    }
    const sourceNode = (nodes as Node[]).find((n) => n.id === selectedNodeId);
    if (!sourceNode) return;

    const id = getNextNodeId();
    const clone: Node = {
      ...sourceNode,
      id,
      position: {
        x: sourceNode.position.x + 40,
        y: sourceNode.position.y + 40,
      },
      selected: false,
      data: {
        ...sourceNode.data,
        nodeResult: null,
        customLabel: undefined,
      },
    };
    setNodes((nds) => [...nds, clone]);
    setSelectedNodeId(id);
    markDirty();
    toast(`Duplicated ${(sourceNode.data as NodeData).label ?? "node"}`, "success");
  }, [selectedNodeId, nodes, setNodes, markDirty, toast]);

  // ── Parameter changes (also appends to audit log) ────────────────────────
  const handleParamChange = useCallback(
    (nodeId: string, paramName: string, value: unknown) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== nodeId) return n;
          const nodeData = n.data as NodeData;
          const descriptor = nodeData.descriptor;
          const param = descriptor?.parameters?.find(
            (p: ParameterSchema) => p.name === paramName
          );
          const oldValue = nodeData.parameters[paramName];
          appendAudit({
            nodeId,
            nodeDisplayName: descriptor?.display_name ?? nodeId,
            paramLabel: param?.label ?? paramName,
            oldValue,
            newValue: value,
            unit: param?.unit,
          });
          return {
            ...n,
            data: {
              ...n.data,
              parameters: {
                ...((n.data as NodeData).parameters),
                [paramName]: value,
              },
            },
          };
        })
      );
      markDirty();
    },
    [setNodes, appendAudit, markDirty]
  );

  // ── Run pipeline ──────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!sessionId) {
      toast('Load an EEG file first \u2014 drag a loader node onto the canvas and browse for a file.', "warning");
      return;
    }
    setViewingHistoryResult(null);
    await run(nodes as Node[], edges as Edge[], sessionId);
  }, [nodes, edges, sessionId, run, toast]);

  // Capture run snapshot for history when result changes (Feature 6)
  const captureRunRef = useRef(captureRun);
  captureRunRef.current = captureRun;
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  useEffect(() => {
    if (result && result.status === "success") {
      captureRunRef.current(result, nodesRef.current as Node[]).catch(console.error);
    }
  }, [result]);

  // ── Research Workbench: Inspector actions ────────────────────────────────
  const handleOpenBrowser = useCallback(async (nodeId: string) => {
    if (!sessionId) return;
    const label = ((nodes as Node[]).find((n) => n.id === nodeId)?.data as NodeData | undefined)?.label;
    try {
      await openMneBrowser(sessionId, nodeId, label);
      toast("MNE Browser opened", "success");
    } catch (err) {
      toast(`Browser failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }, [sessionId, nodes, toast]);

  const handleExportNode = useCallback(async (nodeId: string, format: string, label: string) => {
    if (!sessionId) return;
    try {
      // PNG plots: download directly from the base64 data already in the result
      if (format === "png") {
        const nodeResult = (viewingHistoryResult ?? result?.node_results)?.[nodeId];
        const dataUri = nodeResult?.data;
        if (dataUri && dataUri.startsWith("data:image/png;base64,")) {
          const a = document.createElement("a");
          a.href = dataUri;
          a.download = `${label.replace(/\s+/g, "_")}.png`;
          a.click();
          toast("Exported as .png", "success");
          return;
        }
      }
      await exportNodeOutput(sessionId, nodeId, format, label);
      toast(`Exported as .${format}`, "success");
    } catch (err) {
      toast(`Export failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }, [sessionId, viewingHistoryResult, result, toast]);

  const handleRerunFrom = useCallback(async (nodeId: string) => {
    if (!sessionId) return;
    try {
      const pipeline = serializePipeline(nodes as Node[], edges as Edge[], pipelineName);
      const response = await executeFromNode(sessionId, pipeline, nodeId);
      if (response.status === "success") {
        setNodes((nds) =>
          nds.map((n) => ({
            ...n,
            data: {
              ...n.data,
              nodeResult: response.node_results?.[n.id] ?? (n.data as NodeData).nodeResult ?? null,
            },
          }))
        );
        toast("Re-run complete", "success");
      } else {
        toast(`Re-run error: ${response.error}`, "error");
      }
    } catch (err) {
      toast(`Re-run failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }, [sessionId, nodes, edges, pipelineName, setNodes, toast]);

  // ── Run History: restore + compare ──────────────────────────────────────
  const handleHistoryRestore = useCallback((runId: string) => {
    const snapshot = restoreRun(runId);
    if (snapshot) {
      setViewingHistoryResult(snapshot.nodeResults);
      // Update node badges with historical results
      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          data: {
            ...n.data,
            nodeResult: snapshot.nodeResults[n.id] ?? null,
          },
        }))
      );
      toast(`Viewing: ${snapshot.name}`, "info");
    }
  }, [restoreRun, setNodes, toast]);

  const handleHistoryCompare = useCallback((runIdA: string, runIdB: string) => {
    setComparisonRuns([runIdA, runIdB]);
  }, []);

  // ── Keyboard shortcuts (extracted hook) ─────────────────────────────────
  useKeyboardShortcuts({
    run: handleRun,
    exportPy: handleExport,
    save: handleSavePipeline,
    duplicate: handleDuplicateNode,
    undo: handleUndo,
  });

  // ── Node renaming ────────────────────────────────────────────────────────
  const handleNodeRename = useCallback(
    (nodeId: string, newLabel: string) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, customLabel: newLabel || undefined } }
            : n
        )
      );
      markDirty();
    },
    [setNodes, markDirty]
  );
  nodeRenameRef.current = handleNodeRename;

  // ── Compound node inspection (double-click) ─────────────────────────────
  const handleNodeDoubleClick = useCallback((node: Node) => {
    const descriptor = (node.data as NodeData).descriptor;
    if (descriptor?.category === "Compound") {
      setInspectCompoundId(descriptor.node_type);
      setInspectCompoundName(descriptor.display_name);
    }
  }, []);

  // ── Loading / error states ────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950 text-slate-400 text-sm">
        Connecting to backend…
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950 text-red-400 text-sm text-center px-8">
        <div>
          <div className="font-semibold mb-1">Backend unreachable</div>
          <div className="text-slate-500 text-xs">{error}</div>
          <div className="text-slate-600 text-xs mt-2">
            Run:{" "}
            <code className="font-mono">
              uvicorn backend.main:app --reload --port 8000
            </code>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-screen bg-slate-950 text-slate-200 ${presentationMode ? "presentation-mode" : ""}`}>
      {runError && (
        <div className="bg-red-950 text-red-300 text-xs px-4 py-2 border-b border-red-900 flex-shrink-0">
          Pipeline error: {runError}
        </div>
      )}
      <Toolbar
        running={running}
        sessionId={sessionId}
        hasRawOutput={hasRawOutput}
        hasMetricsOutput={hasMetricsOutput}
        pipelineName={pipelineName}
        onPipelineNameChange={(name) => { setPipelineName(name); markDirty(); }}
        onRun={handleRun}
        onExport={handleExport}
        onExportModal={() => setShowExportModal(true)}
        onDownloadFif={handleDownloadFif}
        onClear={handleClear}
        onSavePipeline={handleSavePipeline}
        onLoadPipeline={handleLoadPipeline}
        onGenerateReport={() => setShowReportModal(true)}
        onBatchProcess={() => batch.toggleBatchMode()}
        batchActive={batch.active}
        onBatchToggle={batch.toggleBatchMode}
        batchProgress={batch.progress && batch.phase === "running" ? { completed: batch.progress.completed + batch.progress.failed, total: batch.progress.total } : null}
        onPublishAsNode={() => setShowPublishModal(true)}
        onUndo={handleUndo}
        canUndo={canUndo}
        loadingFile={loadingFile}
        loadingExport={loadingExport}
        loadingFif={loadingFif}
        loadingReport={loadingReport}
        presentationMode={presentationMode}
        onTogglePresentationMode={() => setPresentationMode((p) => !p)}
        onBidsExport={() => setShowBidsModal(true)}
      />
      {showReportModal && (
        <ReportModal
          onClose={() => setShowReportModal(false)}
          onGenerate={handleGenerateReport}
          hasSessionInfo={sessionInfo !== null}
          hasPipelineConfig={(nodes as Node[]).length > 0}
          hasAuditLog={auditLog.length > 0}
          availableOutputs={availableOutputs}
        />
      )}
      {showPublishModal && registry && (
        <PublishAsNodeModal
          nodes={nodes as Node[]}
          edges={edges as Edge[]}
          registry={registry}
          onClose={() => setShowPublishModal(false)}
          onPublished={refreshRegistry}
        />
      )}
      {sessionId && (
        <ExportModal
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
          sessionId={sessionId}
          pipeline={serializePipeline(nodes as Node[], edges as Edge[], pipelineName)}
          auditLog={auditLog}
        />
      )}
      {sessionId && (
        <BidsExportModal
          isOpen={showBidsModal}
          onClose={() => setShowBidsModal(false)}
          sessionId={sessionId}
          pipeline={serializePipeline(nodes as Node[], edges as Edge[], pipelineName)}
        />
      )}
      {inspectCompoundId && registry && (
        <CompoundInspectorModal
          compoundId={inspectCompoundId}
          displayName={inspectCompoundName}
          registry={registry}
          onClose={() => setInspectCompoundId(null)}
        />
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Left: node palette */}
        <div style={{ width: leftPanelWidth }} className="flex-shrink-0 relative">
          <NodePalette registry={registry!} onRegistryChanged={refreshRegistry} />
          {/* Drag handle — right edge of left panel */}
          <div
            className="absolute top-0 right-0 bottom-0 w-1 cursor-col-resize hover:bg-slate-500 transition-colors z-10"
            onMouseDown={startLeftDrag}
          />
        </div>

        {/* Centre: React Flow canvas */}
        <CanvasPane
          nodes={nodes as Node[]}
          edges={edges as Edge[]}
          registry={registry!}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onNodesSet={setNodes}
          onEdgesSet={setEdges}
          onNodeSelect={(node) => setSelectedNodeId(node?.id ?? null)}
          onSelectionCountChange={setSelectionCount}
          onNodeDoubleClick={handleNodeDoubleClick}
          onConnectionRejected={(reason) => toast(reason, "warning")}
        />

        {/* Right: split panel — parameters on top, tabbed output/history/log on bottom */}
        <div style={{ width: rightPanelWidth }} className="flex-shrink-0 relative">
          {/* Drag handle — left edge of right panel */}
          <div
            className="absolute top-0 left-0 bottom-0 w-1 cursor-col-resize hover:bg-slate-500 transition-colors z-10"
            onMouseDown={startRightDrag}
          />
          {batch.active ? (
            <BatchPanel batch={batch} pipeline={serializePipeline(nodes as Node[], edges as Edge[], pipelineName)} />
          ) : (
            <RightPanel
              selectedNode={selectedNode}
              onParamChange={handleParamChange}
              onFileLoad={handleFileLoad}
              onDownloadFif={handleDownloadFif}
              selectionCount={selectionCount}
              sessionInfo={sessionInfo as unknown as import("./types/pipeline").SessionInfo | undefined}
              nodeResult={selectedNodeId
                ? (viewingHistoryResult ?? result?.node_results)?.[selectedNodeId] ?? null
                : null}
              nodeLabel={selectedNode ? (selectedNode.data as NodeData).label : undefined}
              nodeId={selectedNodeId}
              sessionId={sessionId}
              onOpenBrowser={handleOpenBrowser}
              onExport={handleExportNode}
              onRerunFrom={handleRerunFrom}
              onSessionInfoUpdate={(info: Record<string, unknown>) => setSessionInfo(info)}
              isRunning={running}
              runHistory={runHistory}
              onHistoryRestore={handleHistoryRestore}
              onHistoryRename={renameRun}
              onHistoryRemove={removeRun}
              onHistoryCompare={handleHistoryCompare}
              auditLog={auditLog}
              onAuditClear={clearAudit}
            />
          )}
        </div>
      </div>

      {/* Clear canvas confirmation */}
      <ConfirmDialog
        isOpen={showClearConfirm}
        title="Clear Canvas"
        message="Clear the entire canvas? This cannot be undone."
        confirmLabel="Clear"
        destructive
        onConfirm={confirmClear}
        onCancel={cancelClear}
      />

      {/* Comparison modal (Feature 7) */}
      {comparisonRuns && (() => {
        const snapA = restoreRun(comparisonRuns[0]);
        const snapB = restoreRun(comparisonRuns[1]);
        if (!snapA || !snapB) return null;
        return (
          <ComparisonPanel
            runA={snapA}
            runB={snapB}
            onClose={() => setComparisonRuns(null)}
          />
        );
      })()}
    </div>
  );
}
