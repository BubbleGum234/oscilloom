import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Brain,
  Server,
  Database,
  Palette,
  Info,
  Trash2,
  HardDrive,
  Wifi,
  WifiOff,
  Activity,
  LayoutGrid,
  Moon,
  Sun,
  Loader2,
  AlertTriangle,
  Check,
  FolderOpen,
  Blocks,
} from "lucide-react";

import {
  getStatus,
  getSessionStats,
  clearAllSessions,
  listCustomNodes,
  apiClearAllRuns,
  apiClearAllWorkflows,
  apiGetWorkflowStats,
  apiGetHistoryStats,
} from "../api/client";
import type { SessionStats } from "../api/client";
import { STORAGE_KEYS } from "../constants/storageKeys";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

// ---------------------------------------------------------------------------
// Section card wrapper
// ---------------------------------------------------------------------------

function SectionCard({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ElementType;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-6">
      <div className="flex items-center gap-2.5 mb-5">
        <Icon className="w-4.5 h-4.5 text-slate-400" />
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Info row component
// ---------------------------------------------------------------------------

function InfoRow({
  label,
  value,
  muted,
}: {
  label: string;
  value: React.ReactNode;
  muted?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-slate-800/50 last:border-b-0">
      <span className="text-sm text-slate-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-200 font-medium">{value}</span>
        {muted && (
          <span className="text-[10px] text-slate-600 italic">{muted}</span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Confirm dialog (inline)
// ---------------------------------------------------------------------------

function ConfirmButton({
  label,
  confirmLabel,
  onConfirm,
  loading,
  variant = "danger",
}: {
  label: string;
  confirmLabel: string;
  onConfirm: () => void;
  loading?: boolean;
  variant?: "danger" | "warning";
}) {
  const [confirming, setConfirming] = useState(false);

  const baseClasses =
    variant === "danger"
      ? "border-red-800/60 text-red-400 hover:bg-red-950/40"
      : "border-amber-800/60 text-amber-400 hover:bg-amber-950/40";

  const confirmClasses =
    variant === "danger"
      ? "bg-red-600 hover:bg-red-500 text-white border-red-600"
      : "bg-amber-600 hover:bg-amber-500 text-white border-amber-600";

  if (loading) {
    return (
      <button
        disabled
        className="inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border border-slate-700 text-slate-500 cursor-not-allowed"
      >
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        Working...
      </button>
    );
  }

  if (confirming) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-slate-500 flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" />
          Are you sure?
        </span>
        <button
          onClick={() => {
            setConfirming(false);
            onConfirm();
          }}
          className={`inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors ${confirmClasses}`}
        >
          <Check className="w-3 h-3" />
          {confirmLabel}
        </button>
        <button
          onClick={() => setConfirming(false)}
          className="text-xs px-2.5 py-1.5 rounded-lg border border-slate-700 text-slate-400 hover:bg-slate-800 transition-colors"
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => setConfirming(true)}
      className={`inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition-colors ${baseClasses}`}
    >
      <Trash2 className="w-3 h-3" />
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function Settings() {
  const navigate = useNavigate();

  // Backend status
  const [backendStatus, setBackendStatus] = useState<{
    connected: boolean;
    mneVersion: string | null;
    loading: boolean;
  }>({ connected: false, mneVersion: null, loading: true });

  // Session stats
  const [sessionStats, setSessionStats] = useState<SessionStats | null>(null);
  const [sessionStatsLoading, setSessionStatsLoading] = useState(true);

  // Node count
  const [nodeCount, setNodeCount] = useState<number | null>(null);

  // Custom nodes count
  const [customNodeCount, setCustomNodeCount] = useState<number | null>(null);

  // Workflow & history stats
  const [workflowStats, setWorkflowStats] = useState<{ count: number; disk_usage_bytes: number; workflows_dir: string } | null>(null);
  const [historyStats, setHistoryStats] = useState<{ count: number; disk_usage_bytes: number; history_dir: string } | null>(null);

  // Action states
  const [clearingSessions, setClearingSessions] = useState(false);
  const [clearingRuns, setClearingRuns] = useState(false);
  const [clearingWorkflows, setClearingWorkflows] = useState(false);
  const [clearSessionsResult, setClearSessionsResult] = useState<string | null>(null);

  // Theme
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return (localStorage.getItem(STORAGE_KEYS.THEME) as "dark" | "light") || "dark";
    } catch {
      return "dark";
    }
  });

  // -- Data fetching ---------------------------------------------------------

  useEffect(() => {
    getStatus()
      .then((data) => {
        setBackendStatus({ connected: true, mneVersion: data.mne_version, loading: false });
      })
      .catch(() => {
        setBackendStatus({ connected: false, mneVersion: null, loading: false });
      });
  }, []);

  const fetchSessionStats = useCallback(() => {
    setSessionStatsLoading(true);
    getSessionStats()
      .then((stats) => {
        setSessionStats(stats);
      })
      .catch(() => {
        setSessionStats(null);
      })
      .finally(() => setSessionStatsLoading(false));
  }, []);

  useEffect(() => {
    fetchSessionStats();
  }, [fetchSessionStats]);

  useEffect(() => {
    fetch(
      (import.meta.env.VITE_API_URL || "http://localhost:8000") +
        "/registry/nodes"
    )
      .then((res) => res.json())
      .then((data: { nodes: Record<string, unknown> }) => {
        setNodeCount(Object.keys(data.nodes).length);
      })
      .catch(() => setNodeCount(null));
  }, []);

  useEffect(() => {
    listCustomNodes()
      .then((data) => setCustomNodeCount(data.count))
      .catch(() => setCustomNodeCount(null));
  }, []);

  const fetchStorageStats = useCallback(() => {
    apiGetWorkflowStats()
      .then(setWorkflowStats)
      .catch(() => setWorkflowStats(null));
    apiGetHistoryStats()
      .then(setHistoryStats)
      .catch(() => setHistoryStats(null));
  }, []);

  useEffect(() => {
    fetchStorageStats();
  }, [fetchStorageStats]);

  // -- Handlers --------------------------------------------------------------

  const handleClearAllSessions = useCallback(async () => {
    setClearingSessions(true);
    setClearSessionsResult(null);
    try {
      const result = await clearAllSessions();
      setClearSessionsResult(
        `Cleared ${result.deleted_count} session${result.deleted_count !== 1 ? "s" : ""}`
      );
      fetchSessionStats();
    } catch {
      setClearSessionsResult("Failed to clear sessions");
    } finally {
      setClearingSessions(false);
    }
  }, [fetchSessionStats]);

  const handleClearRunHistory = useCallback(async () => {
    setClearingRuns(true);
    setClearSessionsResult(null);
    try {
      const result = await apiClearAllRuns();
      setClearSessionsResult(
        `Cleared ${result.deleted_count} run${result.deleted_count !== 1 ? "s" : ""}`
      );
      fetchStorageStats();
    } catch {
      setClearSessionsResult("Failed to clear run history");
    } finally {
      setClearingRuns(false);
    }
  }, [fetchStorageStats]);

  const handleClearWorkflows = useCallback(async () => {
    setClearingWorkflows(true);
    setClearSessionsResult(null);
    try {
      const result = await apiClearAllWorkflows();
      setClearSessionsResult(
        `Cleared ${result.deleted_count} workflow${result.deleted_count !== 1 ? "s" : ""}`
      );
      fetchStorageStats();
    } catch {
      setClearSessionsResult("Failed to clear workflows");
    } finally {
      setClearingWorkflows(false);
    }
  }, [fetchStorageStats]);

  const handleThemeToggle = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(STORAGE_KEYS.THEME, next);
      } catch {}
      if (next === "light") {
        document.documentElement.classList.add("light-theme");
      } else {
        document.documentElement.classList.remove("light-theme");
      }
      return next;
    });
  }, []);

  // Apply theme on mount
  useEffect(() => {
    if (theme === "light") {
      document.documentElement.classList.add("light-theme");
    } else {
      document.documentElement.classList.remove("light-theme");
    }
  }, [theme]);

  // -- Render ----------------------------------------------------------------

  return (
    <div className="h-screen overflow-y-auto bg-[#0f1117] text-slate-200">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0f1117]/90 backdrop-blur-md">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => navigate("/")}
                className="p-2 -ml-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-colors"
                title="Back to Home"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="flex items-center gap-2.5">
                <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/20">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-slate-100 tracking-tight">
                    Settings
                  </h1>
                  <p className="text-[11px] text-slate-500 -mt-0.5">
                    Oscilloom configuration
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8 space-y-6">
        {/* Status toast */}
        {clearSessionsResult && (
          <div className="bg-slate-800/80 border border-slate-700/60 rounded-lg px-4 py-2.5 text-xs text-slate-300 flex items-center justify-between">
            <span>{clearSessionsResult}</span>
            <button
              onClick={() => setClearSessionsResult(null)}
              className="text-slate-500 hover:text-slate-300 ml-4"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ── Session Management ─────────────────────────────────────────── */}
        <SectionCard icon={Server} title="Session Management">
          {sessionStatsLoading ? (
            <div className="flex items-center gap-2 text-sm text-slate-500 py-4">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading session data...
            </div>
          ) : sessionStats ? (
            <div className="space-y-0">
              <InfoRow
                label="Active Sessions"
                value={sessionStats.active_sessions}
              />
              <InfoRow
                label="Session TTL"
                value={formatDuration(sessionStats.ttl_seconds)}
                muted="(configured server-side)"
              />
              <InfoRow
                label="Max Sessions"
                value={sessionStats.max_sessions}
                muted="(configured server-side)"
              />
              <InfoRow
                label="Sessions Directory"
                value={
                  <span className="font-mono text-xs text-slate-400 max-w-[300px] truncate inline-block">
                    {sessionStats.sessions_dir}
                  </span>
                }
              />
              <InfoRow
                label="Disk Usage"
                value={formatBytes(sessionStats.disk_usage_bytes)}
              />
              <div className="pt-4">
                <ConfirmButton
                  label="Clear All Sessions"
                  confirmLabel="Yes, clear all"
                  onConfirm={handleClearAllSessions}
                  loading={clearingSessions}
                  variant="danger"
                />
                <p className="text-[11px] text-slate-600 mt-2">
                  Removes all loaded EEG data from memory and disk. Active
                  editor sessions will need to reload their files.
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500 py-2">
              Could not load session stats. Ensure the backend is running.
            </p>
          )}
        </SectionCard>

        {/* ── Application Info ───────────────────────────────────────────── */}
        <SectionCard icon={Info} title="Application Info">
          <div className="space-y-0">
            <div className="flex items-center justify-between py-2.5 border-b border-slate-800/50">
              <span className="text-sm text-slate-400">Backend Status</span>
              <div className="flex items-center gap-2">
                {backendStatus.loading ? (
                  <Loader2 className="w-3.5 h-3.5 text-slate-500 animate-spin" />
                ) : backendStatus.connected ? (
                  <>
                    <Wifi className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-sm text-emerald-400 font-medium">
                      Connected
                    </span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-sm text-red-400 font-medium">
                      Disconnected
                    </span>
                  </>
                )}
              </div>
            </div>
            <InfoRow
              label="MNE Version"
              value={
                backendStatus.mneVersion ? (
                  <span className="flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-slate-500" />
                    {backendStatus.mneVersion}
                  </span>
                ) : (
                  <span className="text-slate-600">--</span>
                )
              }
            />
            <InfoRow
              label="Available Node Types"
              value={
                nodeCount !== null ? (
                  <span className="flex items-center gap-1.5">
                    <LayoutGrid className="w-3.5 h-3.5 text-slate-500" />
                    {nodeCount}
                  </span>
                ) : (
                  <span className="text-slate-600">--</span>
                )
              }
            />
            <InfoRow label="App Version" value="0.1.0-beta" />
          </div>
        </SectionCard>

        {/* ── Storage ────────────────────────────────────────────────────── */}
        <SectionCard icon={Database} title="Storage">
          <div className="space-y-0">
            <InfoRow
              label="Custom Nodes"
              value={
                customNodeCount !== null ? (
                  <span className="flex items-center gap-1.5">
                    <Blocks className="w-3.5 h-3.5 text-slate-500" />
                    {customNodeCount}
                  </span>
                ) : (
                  <span className="text-slate-600">--</span>
                )
              }
            />
            <InfoRow
              label="Custom Nodes Location"
              value={
                <span className="font-mono text-xs text-slate-400">
                  ~/.oscilloom/custom_nodes/
                </span>
              }
            />
            <InfoRow
              label="Saved Workflows"
              value={
                workflowStats ? (
                  <span className="flex items-center gap-1.5">
                    {workflowStats.count} file{workflowStats.count !== 1 ? "s" : ""}
                    <span className="text-slate-500">({formatBytes(workflowStats.disk_usage_bytes)})</span>
                  </span>
                ) : (
                  <span className="text-slate-600">--</span>
                )
              }
            />
            {workflowStats && (
              <InfoRow
                label="Workflows Directory"
                value={
                  <span className="font-mono text-xs text-slate-400 max-w-[300px] truncate inline-block">
                    {workflowStats.workflows_dir}
                  </span>
                }
              />
            )}
            <InfoRow
              label="Run History"
              value={
                historyStats ? (
                  <span className="flex items-center gap-1.5">
                    {historyStats.count} run{historyStats.count !== 1 ? "s" : ""}
                    <span className="text-slate-500">({formatBytes(historyStats.disk_usage_bytes)})</span>
                  </span>
                ) : (
                  <span className="text-slate-600">--</span>
                )
              }
            />
            {historyStats && (
              <InfoRow
                label="History Directory"
                value={
                  <span className="font-mono text-xs text-slate-400 max-w-[300px] truncate inline-block">
                    {historyStats.history_dir}
                  </span>
                }
              />
            )}
            <div className="flex items-center justify-between py-2.5 border-b border-slate-800/50">
              <span className="text-sm text-slate-400">Data Location</span>
              <span className="text-[10px] text-slate-600 italic">
                Workflows and run history stored in ~/.oscilloom/
              </span>
            </div>
            <div className="pt-4 flex flex-wrap gap-3">
              <ConfirmButton
                label="Clear Run History"
                confirmLabel="Yes, clear"
                onConfirm={handleClearRunHistory}
                loading={clearingRuns}
                variant="warning"
              />
              <ConfirmButton
                label="Clear All Workflows"
                confirmLabel="Yes, delete all"
                onConfirm={handleClearWorkflows}
                loading={clearingWorkflows}
                variant="danger"
              />
            </div>
            <p className="text-[11px] text-slate-600 mt-2">
              Clearing workflows removes all saved pipelines from the server.
              This cannot be undone.
            </p>
          </div>
        </SectionCard>

        {/* ── Appearance ─────────────────────────────────────────────────── */}
        <SectionCard icon={Palette} title="Appearance">
          <div className="flex items-center justify-between py-2">
            <div>
              <p className="text-sm text-slate-300 font-medium">Theme</p>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Switch between dark and light interface themes
              </p>
            </div>
            <button
              onClick={handleThemeToggle}
              className="flex items-center gap-2.5 px-4 py-2 rounded-lg border border-slate-700/60 bg-slate-800/60 hover:bg-slate-700/60 transition-colors"
            >
              {theme === "dark" ? (
                <>
                  <Moon className="w-4 h-4 text-slate-400" />
                  <span className="text-sm text-slate-300">Dark</span>
                </>
              ) : (
                <>
                  <Sun className="w-4 h-4 text-amber-400" />
                  <span className="text-sm text-slate-300">Light</span>
                </>
              )}
            </button>
          </div>
        </SectionCard>

        {/* ── About ──────────────────────────────────────────────────────── */}
        <footer className="border-t border-slate-800/60 pt-6 pb-8">
          <div className="flex flex-wrap items-center justify-between gap-4 text-xs text-slate-600">
            <div className="flex items-center gap-2">
              <HardDrive className="w-3.5 h-3.5" />
              <span>All data processed locally -- nothing leaves this machine</span>
            </div>
            <div className="flex items-center gap-2">
              <FolderOpen className="w-3.5 h-3.5" />
              <span>Config: ~/.oscilloom/</span>
            </div>
            <div>Oscilloom v0.1.0-beta</div>
          </div>
        </footer>
      </main>
    </div>
  );
}
