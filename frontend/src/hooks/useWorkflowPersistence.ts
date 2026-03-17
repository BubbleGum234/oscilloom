// Extracted from App.tsx — workflow save/load, dirty-state tracking
import { useState, useCallback, useEffect, useRef } from "react";
import type { Node, Edge } from "@xyflow/react";
import type { NodeRegistry } from "../types/pipeline";
import { serializePipeline, deserializePipeline } from "../utils/serializePipeline";
import { syncNodeCounter } from "../utils/nodeId";
import { getWorkflow, saveWorkflow } from "../store/workflowStore";
import type { SavedWorkflow } from "../store/workflowStore";

export interface UseWorkflowPersistenceReturn {
  pipelineName: string;
  setPipelineName: (name: string) => void;
  currentWorkflowId: string | null;
  markDirty: () => void;
  markClean: () => void;
  handleSavePipeline: () => Promise<void>;
  handleLoadPipeline: (file: File) => void;
  handleClear: () => void;
  showClearConfirm: boolean;
  confirmClear: () => void;
  cancelClear: () => void;
}

export function useWorkflowPersistence(
  nodes: Node[],
  edges: Edge[],
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>,
  setEdges: React.Dispatch<React.SetStateAction<Edge[]>>,
  registry: NodeRegistry | null,
  workflowId: string | undefined,  // from useParams
  navigate: (path: string, opts?: { replace?: boolean }) => void,
  clearAudit: () => void,
  setSelectedNodeId: (id: string | null) => void,
  toast: (msg: string, type: string) => void,
): UseWorkflowPersistenceReturn {
  // ── State ──────────────────────────────────────────────────────────────────
  const [pipelineName, setPipelineName] = useState("My Pipeline");
  const [currentWorkflowId, setCurrentWorkflowId] = useState<string | null>(workflowId ?? null);

  // ── Dirty-state tracking (unsaved changes warning) ──────────────────────
  const isDirtyRef = useRef(false);

  const markDirty = useCallback(() => {
    isDirtyRef.current = true;
  }, []);

  const markClean = useCallback(() => {
    isDirtyRef.current = false;
  }, []);

  // ── beforeunload handler ────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (isDirtyRef.current) {
        e.preventDefault();
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  // ── Load workflow from IndexedDB if workflowId is in the URL ───────────
  const workflowLoadedRef = useRef(false);
  useEffect(() => {
    if (!workflowId || !registry || workflowLoadedRef.current) return;
    workflowLoadedRef.current = true;
    getWorkflow(workflowId).then((wf) => {
      if (!wf) {
        toast("Workflow not found", "error");
        navigate("/");
        return;
      }
      const { nodes: newNodes, edges: newEdges } = deserializePipeline(wf.pipeline, registry);
      setNodes(newNodes);
      setEdges(newEdges);
      syncNodeCounter(newNodes);
      setPipelineName(wf.name);
      setCurrentWorkflowId(wf.id);
      markClean();
    }).catch((err) => {
      console.error("Failed to load workflow:", err);
      toast("Failed to load workflow", "error");
    });
  }, [workflowId, registry, setNodes, setEdges, navigate, toast, markClean]);

  // ── TASK-18: Save pipeline to JSON ────────────────────────────────────────
  const handleSavePipeline = useCallback(async () => {
    const pipeline = serializePipeline(nodes as Node[], edges as Edge[], pipelineName);

    // Persist to IndexedDB workflow store
    const now = new Date().toISOString();
    const wfId = currentWorkflowId || crypto.randomUUID();
    const wf: SavedWorkflow = {
      id: wfId,
      name: pipelineName,
      createdAt: currentWorkflowId ? now : now, // preserved on update below
      updatedAt: now,
      nodeCount: (nodes as Node[]).length,
      edgeCount: (edges as Edge[]).length,
      pipeline,
    };

    // If updating existing, preserve original createdAt
    if (currentWorkflowId) {
      try {
        const existing = await getWorkflow(currentWorkflowId);
        if (existing) wf.createdAt = existing.createdAt;
      } catch { /* use current time */ }
    }

    await saveWorkflow(wf);
    if (!currentWorkflowId) {
      setCurrentWorkflowId(wfId);
      navigate(`/editor/${wfId}`, { replace: true });
    }

    // Also download as .json file
    const blob = new Blob([JSON.stringify(pipeline, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${pipelineName.replace(/\s+/g, "_") || "pipeline"}.json`;
    a.click();
    URL.revokeObjectURL(url);
    markClean();
    toast("Pipeline saved", "success");
  }, [nodes, edges, pipelineName, currentWorkflowId, markClean, toast, navigate]);

  // ── TASK-18: Load pipeline from JSON ─────────────────────────────────────
  const handleLoadPipeline = useCallback(
    (file: File) => {
      if (!registry) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const pipeline = JSON.parse(e.target?.result as string);
          const { nodes: newNodes, edges: newEdges } = deserializePipeline(
            pipeline,
            registry
          );
          setNodes(newNodes);
          setEdges(newEdges);
          syncNodeCounter(newNodes);
          if (pipeline.name) setPipelineName(pipeline.name);
          markClean();
          toast(`Loaded pipeline: ${pipeline.name || file.name}`, "success");
        } catch {
          toast("Failed to load pipeline: the file is not valid Oscilloom JSON.", "error");
        }
      };
      reader.readAsText(file);
    },
    [registry, setNodes, setEdges, markClean, toast]
  );

  // ── Clear canvas ──────────────────────────────────────────────────────────
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const handleClear = useCallback(() => {
    if (nodes.length === 0 && edges.length === 0) return;
    setShowClearConfirm(true);
  }, [nodes, edges]);

  const confirmClear = useCallback(() => {
    setShowClearConfirm(false);
    setNodes([]);
    setEdges([]);
    setSelectedNodeId(null);
    clearAudit();
    markClean();
    toast("Canvas cleared", "info");
  }, [setNodes, setEdges, setSelectedNodeId, clearAudit, markClean, toast]);

  const cancelClear = useCallback(() => {
    setShowClearConfirm(false);
  }, []);

  return {
    pipelineName,
    setPipelineName,
    currentWorkflowId,
    markDirty,
    markClean,
    handleSavePipeline,
    handleLoadPipeline,
    handleClear,
    showClearConfirm,
    confirmClear,
    cancelClear,
  };
}
