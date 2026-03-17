import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { ConfirmDialog } from "../components/ui/ConfirmDialog";
import {
  Plus,
  Clock,
  Folder,
  Activity,
  Settings,
  Search,
  Upload,
  Trash2,
  Copy,
  Pencil,
  GitBranch,
  ChevronRight,
  Wifi,
  WifiOff,
  LayoutGrid,
  FileText,
} from "lucide-react";

import type { SavedWorkflow } from "../store/workflowStore";
import {
  listWorkflows,
  saveWorkflow,
  deleteWorkflow,
  duplicateWorkflow,
} from "../store/workflowStore";
import { getStatus } from "../api/client";
import type { PipelineGraph } from "../types/pipeline";

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SkeletonCard() {
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 animate-pulse">
      <div className="h-4 bg-slate-800 rounded w-3/4 mb-3" />
      <div className="h-3 bg-slate-800 rounded w-1/2 mb-4" />
      <div className="flex gap-3">
        <div className="h-3 bg-slate-800 rounded w-16" />
        <div className="h-3 bg-slate-800 rounded w-16" />
      </div>
    </div>
  );
}

function SkeletonRow({ count }: { count: number }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function Home() {
  const navigate = useNavigate();

  // Workflow state
  const [workflows, setWorkflows] = useState<SavedWorkflow[]>([]);
  const [loadingWorkflows, setLoadingWorkflows] = useState(true);
  const [search, setSearch] = useState("");
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameRef = useRef<HTMLInputElement>(null);
  const importRef = useRef<HTMLInputElement>(null);

  // Backend status
  const [backendStatus, setBackendStatus] = useState<{
    connected: boolean;
    mneVersion: string | null;
  }>({ connected: false, mneVersion: null });

  // Node count from registry
  const [nodeCount, setNodeCount] = useState<number | null>(null);

  // ── Data loading ──────────────────────────────────────────────────────────

  const loadWorkflows = useCallback(async () => {
    try {
      const wfs = await listWorkflows();
      setWorkflows(wfs);
    } catch (err) {
      console.error("Failed to load workflows:", err);
    } finally {
      setLoadingWorkflows(false);
    }
  }, []);

  useEffect(() => {
    loadWorkflows();
  }, [loadWorkflows]);

  useEffect(() => {
    getStatus()
      .then((data) => {
        setBackendStatus({ connected: true, mneVersion: data.mne_version });
      })
      .catch(() => {
        setBackendStatus({ connected: false, mneVersion: null });
      });
  }, []);

  // Fetch node count from registry
  useEffect(() => {
    fetch((import.meta.env.VITE_API_URL || "http://localhost:8000") + "/registry/nodes")
      .then((res) => res.json())
      .then((data: { nodes: Record<string, unknown> }) => {
        setNodeCount(Object.keys(data.nodes).length);
      })
      .catch(() => {
        // Silently ignore — status indicator already shows disconnected
      });
  }, []);

  // Focus rename input when editing
  useEffect(() => {
    if (renamingId && renameRef.current) {
      renameRef.current.focus();
      renameRef.current.select();
    }
  }, [renamingId]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleNewWorkflow = useCallback(() => {
    navigate("/editor");
  }, [navigate]);

  const handleOpen = useCallback(
    (id: string) => {
      navigate(`/editor/${id}`);
    },
    [navigate]
  );

  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);

  const handleDeleteConfirmed = useCallback(async () => {
    if (!deleteTarget) return;
    await deleteWorkflow(deleteTarget.id);
    setWorkflows((prev) => prev.filter((w) => w.id !== deleteTarget.id));
    setDeleteTarget(null);
  }, [deleteTarget]);

  const handleDelete = useCallback(
    (id: string, name: string) => {
      setDeleteTarget({ id, name });
    },
    []
  );

  const handleDuplicate = useCallback(async (id: string) => {
    const copy = await duplicateWorkflow(id);
    setWorkflows((prev) => [copy, ...prev]);
  }, []);

  const handleRenameStart = useCallback((id: string, currentName: string) => {
    setRenamingId(id);
    setRenameValue(currentName);
  }, []);

  const handleRenameSubmit = useCallback(
    async (wf: SavedWorkflow) => {
      if (renameValue.trim() && renameValue !== wf.name) {
        const updated = {
          ...wf,
          name: renameValue.trim(),
          updatedAt: new Date().toISOString(),
        };
        await saveWorkflow(updated);
        setWorkflows((prev) =>
          prev.map((w) => (w.id === wf.id ? updated : w))
        );
      }
      setRenamingId(null);
    },
    [renameValue]
  );

  const handleImport = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async (ev) => {
        try {
          const pipeline = JSON.parse(ev.target?.result as string) as PipelineGraph;
          const now = new Date().toISOString();
          const wf: SavedWorkflow = {
            id: crypto.randomUUID(),
            name: pipeline.metadata?.name || file.name.replace(".json", ""),
            createdAt: now,
            updatedAt: now,
            nodeCount: pipeline.nodes?.length || 0,
            edgeCount: pipeline.edges?.length || 0,
            pipeline,
          };
          await saveWorkflow(wf);
          setWorkflows((prev) => [wf, ...prev]);
        } catch {
          alert("Invalid pipeline JSON file.");
        }
      };
      reader.readAsText(file);
      e.target.value = "";
    },
    []
  );

  // ── Derived state ─────────────────────────────────────────────────────────

  const filtered = search
    ? workflows.filter((w) =>
        w.name.toLowerCase().includes(search.toLowerCase())
      )
    : workflows;

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    if (diff < 60000) return "Just now";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    return d.toLocaleDateString();
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="h-screen overflow-y-auto bg-[#0f1117] text-slate-200">
      {/* ── Header / Nav Bar ────────────────────────────────────────────── */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0f1117]/90 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img src="/favicon.png" alt="Oscilloom" className="w-9 h-9 rounded-lg shadow-lg shadow-cyan-500/20" />
              <div className="flex items-center gap-2.5">
                <h1 className="text-xl font-bold text-slate-100 tracking-tight">
                  Oscilloom
                </h1>
                <span className="text-[10px] font-medium tracking-wide uppercase px-1.5 py-0.5 rounded bg-slate-800 text-slate-500 border border-slate-700/50">
                  v0.1.0-beta
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => navigate("/settings")}
                className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/60 transition-colors"
                title="Settings"
              >
                <Settings className="w-4.5 h-4.5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6">
        {/* ── Hero Section ──────────────────────────────────────────────── */}
        <section className="pt-12 pb-10">
          <div className="max-w-2xl">
            <h2 className="text-3xl sm:text-4xl font-bold text-slate-100 tracking-tight leading-tight">
              Local-first EEG Pipeline Builder
            </h2>
            <p className="mt-3 text-base text-slate-400 leading-relaxed max-w-xl">
              Design, execute, and export reproducible EEG processing pipelines
              with a visual node editor. All processing runs on-device — your
              data never leaves this machine.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                onClick={handleNewWorkflow}
                className="inline-flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-white text-sm font-medium rounded-lg px-5 py-2.5 transition-all shadow-lg shadow-cyan-600/20 hover:shadow-cyan-500/30"
              >
                <Plus className="w-4 h-4" />
                New Pipeline
              </button>
              <input
                ref={importRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImport}
              />
              <button
                onClick={() => importRef.current?.click()}
                className="inline-flex items-center gap-2 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 text-sm rounded-lg px-4 py-2.5 transition-colors border border-slate-700/60"
              >
                <Upload className="w-4 h-4" />
                Import JSON
              </button>
            </div>
          </div>
        </section>

        {/* ── Recent Workflows ──────────────────────────────────────────── */}
        <section className="pb-10">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2.5">
              <Clock className="w-4.5 h-4.5 text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
                Recent Workflows
              </h3>
              {!loadingWorkflows && workflows.length > 0 && (
                <span className="text-xs text-slate-600 font-medium">
                  ({workflows.length})
                </span>
              )}
            </div>
            {workflows.length > 3 && (
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search..."
                  className="w-48 bg-slate-900/60 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-slate-600 transition-colors"
                />
              </div>
            )}
          </div>

          {/* Loading skeleton */}
          {loadingWorkflows && <SkeletonRow count={3} />}

          {/* Empty state */}
          {!loadingWorkflows && workflows.length === 0 && (
            <div className="border border-dashed border-slate-800 rounded-xl py-14 px-6 text-center">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 rounded-xl bg-slate-800/60 flex items-center justify-center">
                  <Folder className="w-6 h-6 text-slate-600" />
                </div>
              </div>
              <h4 className="text-sm font-semibold text-slate-400 mb-1.5">
                No pipelines yet
              </h4>
              <p className="text-xs text-slate-600 max-w-sm mx-auto leading-relaxed">
                Create your first pipeline to start building EEG processing
                workflows, or import an existing pipeline JSON file.
              </p>
              <div className="mt-5 flex items-center justify-center gap-3">
                <button
                  onClick={handleNewWorkflow}
                  className="inline-flex items-center gap-1.5 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium rounded-lg px-4 py-2 transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                  Create Pipeline
                </button>
                <button
                  onClick={() => importRef.current?.click()}
                  className="inline-flex items-center gap-1.5 bg-slate-800/80 hover:bg-slate-700/80 text-slate-400 text-xs rounded-lg px-4 py-2 transition-colors border border-slate-700/60"
                >
                  <Upload className="w-3.5 h-3.5" />
                  Import
                </button>
              </div>
            </div>
          )}

          {/* Workflow grid */}
          {!loadingWorkflows && filtered.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {filtered.map((wf) => (
                <div
                  key={wf.id}
                  className="group relative bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 hover:border-slate-600/80 hover:bg-slate-900/80 transition-all duration-200 cursor-pointer"
                  onClick={() => handleOpen(wf.id)}
                >
                  {/* Card header */}
                  <div className="flex items-start justify-between mb-3">
                    {renamingId === wf.id ? (
                      <input
                        ref={renameRef}
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onBlur={() => handleRenameSubmit(wf)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleRenameSubmit(wf);
                          if (e.key === "Escape") setRenamingId(null);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-cyan-500 flex-1 mr-2"
                      />
                    ) : (
                      <div className="flex items-center gap-2 flex-1 min-w-0 mr-2">
                        <FileText className="w-4 h-4 text-slate-600 flex-shrink-0" />
                        <h4 className="text-sm font-semibold text-slate-200 truncate">
                          {wf.name}
                        </h4>
                      </div>
                    )}
                    {/* Actions */}
                    <div
                      className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        onClick={() => handleRenameStart(wf.id, wf.name)}
                        title="Rename"
                        className="text-slate-600 hover:text-slate-300 p-1.5 rounded-md hover:bg-slate-800 transition-colors"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleDuplicate(wf.id)}
                        title="Duplicate"
                        className="text-slate-600 hover:text-slate-300 p-1.5 rounded-md hover:bg-slate-800 transition-colors"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleDelete(wf.id, wf.name)}
                        title="Delete"
                        className="text-slate-600 hover:text-red-400 p-1.5 rounded-md hover:bg-slate-800 transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center gap-3 text-xs text-slate-500 mb-3">
                    <span className="inline-flex items-center gap-1">
                      <LayoutGrid className="w-3 h-3" />
                      {wf.nodeCount} node{wf.nodeCount !== 1 ? "s" : ""}
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <GitBranch className="w-3 h-3" />
                      {wf.edgeCount} edge{wf.edgeCount !== 1 ? "s" : ""}
                    </span>
                  </div>

                  {/* Timestamps */}
                  <div className="flex items-center justify-between text-[11px] text-slate-600">
                    <span>Updated {formatDate(wf.updatedAt)}</span>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-700 group-hover:text-slate-500 transition-colors" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* No search results */}
          {!loadingWorkflows && workflows.length > 0 && filtered.length === 0 && (
            <div className="text-center py-10 text-slate-600 text-sm">
              No workflows matching &ldquo;{search}&rdquo;
            </div>
          )}
        </section>

        {/* ── Stats / Info Footer ───────────────────────────────────────── */}
        <footer className="border-t border-slate-800/60 py-6 mb-4">
          <div className="flex flex-wrap items-center justify-between gap-4 text-xs text-slate-600">
            <div className="flex items-center gap-5">
              {/* Connection status */}
              <div className="inline-flex items-center gap-1.5">
                {backendStatus.connected ? (
                  <>
                    <Wifi className="w-3.5 h-3.5 text-cyan-500" />
                    <span className="text-cyan-500/80">Backend connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-red-400/80">Backend disconnected</span>
                  </>
                )}
              </div>

              {/* MNE version */}
              {backendStatus.mneVersion && (
                <div className="inline-flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-slate-600" />
                  <span>MNE {backendStatus.mneVersion}</span>
                </div>
              )}

              {/* Node count */}
              {nodeCount !== null && (
                <div className="inline-flex items-center gap-1.5">
                  <LayoutGrid className="w-3.5 h-3.5 text-slate-600" />
                  <span>{nodeCount} node type{nodeCount !== 1 ? "s" : ""} available</span>
                </div>
              )}
            </div>

            <div className="text-slate-700">
              Oscilloom -- Local-first EEG processing
            </div>
          </div>
        </footer>
      </main>

      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Delete Workflow"
        message={`Delete "${deleteTarget?.name ?? ""}"? This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        onConfirm={handleDeleteConfirmed}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
