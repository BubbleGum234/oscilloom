import type {
  PipelineGraph, ExecuteResponse, NodeRegistry,
  StagedFile, BatchProgress, BatchFileResult, BatchResults,
  SavedBatchSummary,
} from "../types/pipeline";

const JSON_HEADERS = { "Content-Type": "application/json" } as const;

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function getRegistry(): Promise<{ nodes: NodeRegistry }> {
  const res = await fetch(`${BASE_URL}/registry/nodes`);
  if (!res.ok) throw new Error(`Registry fetch failed: ${res.statusText}`);
  return res.json();
}

export interface PipelineTemplateNode {
  id: string;
  node_type: string;
  label: string;
  params: Record<string, unknown>;
}

export interface PipelineTemplateEdge {
  source: string;
  source_handle: string;
  target: string;
  target_handle: string;
  handle_type: string;
}

export interface PipelineTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  nodes: PipelineTemplateNode[];
  edges: PipelineTemplateEdge[];
}

export async function getTemplates(): Promise<{ templates: PipelineTemplate[]; count: number }> {
  const res = await fetch(`${BASE_URL}/registry/templates`);
  if (!res.ok) throw new Error(`Templates fetch failed: ${res.statusText}`);
  return res.json();
}

export async function loadSession(
  file: File
): Promise<{ session_id: string; info: Record<string, unknown> }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/session/load`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to load file.");
  }
  return res.json();
}

export async function executePipeline(
  sessionId: string,
  pipeline: PipelineGraph
): Promise<ExecuteResponse> {
  const res = await fetch(`${BASE_URL}/pipeline/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, pipeline }),
  });
  if (!res.ok) throw new Error(`Execution failed: ${res.statusText}`);
  return res.json();
}

export async function validatePipeline(
  sessionId: string,
  pipeline: PipelineGraph
): Promise<{ valid: boolean; errors: string[] }> {
  const res = await fetch(`${BASE_URL}/pipeline/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, pipeline }),
  });
  if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
  return res.json();
}

export async function exportPipeline(
  sessionId: string,
  pipeline: PipelineGraph,
  auditLog: object[]
): Promise<{ script: string; filename: string }> {
  const res = await fetch(`${BASE_URL}/pipeline/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, pipeline, audit_log: auditLog }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    const detail = (err as { detail?: string } | null)?.detail ?? res.statusText;
    throw new Error(`Export failed: ${detail}`);
  }
  return res.json();
}

export async function downloadFif(
  sessionId: string,
  pipeline: PipelineGraph
): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/pipeline/download_fif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, pipeline }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "FIF download failed");
  }
  return res.blob();
}

export async function getStatus(): Promise<{ status: string; mne_version: string }> {
  const res = await fetch(`${BASE_URL}/status`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

export interface SessionStats {
  active_sessions: number;
  sessions_dir: string;
  ttl_seconds: number;
  max_sessions: number;
  disk_usage_bytes: number;
}

export async function getSessionStats(): Promise<SessionStats> {
  const res = await fetch(`${BASE_URL}/session/stats`);
  if (!res.ok) throw new Error("Failed to fetch session stats");
  return res.json();
}

export async function clearAllSessions(): Promise<{ status: string; deleted_count: number }> {
  const res = await fetch(`${BASE_URL}/session/clear-all`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear sessions");
  return res.json();
}

export interface ReportSections {
  data_quality: boolean;
  pipeline_config: boolean;
  analysis_results: boolean;
  clinical_interpretation: boolean;
  visualizations: boolean;
  audit_trail: boolean;
  notes: boolean;
}

export interface ReportOptions {
  nodeResults: Record<string, unknown>;
  title: string;
  patientId: string;
  clinicName: string;
  sessionInfo?: Record<string, unknown> | null;
  pipelineConfig?: Array<Record<string, unknown>> | null;
  auditLog?: Array<Record<string, unknown>> | null;
  notes?: string;
  sections?: ReportSections;
  includedNodes?: string[];
}

export async function generateReport(options: ReportOptions): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/pipeline/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      node_results: options.nodeResults,
      title: options.title,
      patient_id: options.patientId,
      clinic_name: options.clinicName,
      session_info: options.sessionInfo ?? null,
      pipeline_config: options.pipelineConfig ?? null,
      audit_log: options.auditLog ?? null,
      notes: options.notes ?? "",
      sections: options.sections ?? undefined,
      included_nodes: options.includedNodes ?? null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Report generation failed");
  }
  return res.blob();
}

// ---------------------------------------------------------------------------
// Batch processing (Tier 4)
// ---------------------------------------------------------------------------

export async function stageFiles(
  files: File[]
): Promise<{ staged_files: StagedFile[]; count: number }> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await fetch(`${BASE_URL}/pipeline/batch/stage`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to stage files."
    );
  }
  return res.json();
}

export async function updateFileMetadata(
  fileId: string,
  metadata: Record<string, string>
): Promise<{ file_id: string; metadata: Record<string, string> }> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/stage/${fileId}/metadata`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ metadata }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to update metadata."
    );
  }
  return res.json();
}

export async function startBatch(
  fileIds: string[],
  pipeline: PipelineGraph
): Promise<{ batch_id: string; total_files: number }> {
  const res = await fetch(`${BASE_URL}/pipeline/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      file_ids: fileIds,
      pipeline,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to start batch."
    );
  }
  return res.json();
}

export async function getBatchProgress(
  batchId: string
): Promise<BatchProgress> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}/progress`);
  if (!res.ok) throw new Error("Failed to get batch progress.");
  return res.json();
}

export async function getBatchResults(
  batchId: string
): Promise<BatchResults> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}/results`);
  if (!res.ok) throw new Error("Failed to get batch results.");
  return res.json();
}

export async function getBatchFileDetail(
  batchId: string,
  fileId: string
): Promise<BatchFileResult> {
  const res = await fetch(
    `${BASE_URL}/pipeline/batch/${batchId}/file/${fileId}`
  );
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to get file detail."
    );
  }
  return res.json();
}

export interface BatchReportConfig {
  clinic_name: string;
  notes: string;
  pipeline_config: Array<Record<string, unknown>> | null;
  sections: ReportSections;
}

export async function generateBatchReports(
  batchId: string,
  config?: BatchReportConfig,
): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}/reports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config ?? {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to generate reports."
    );
  }
  return res.blob();
}

export async function cancelBatch(
  batchId: string
): Promise<{ batch_id: string; status: string }> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}/cancel`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to cancel batch.");
  return res.json();
}

export async function saveBatchResults(
  batchId: string
): Promise<{ batch_id: string; saved: boolean }> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}/save`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to save batch results."
    );
  }
  return res.json();
}

export async function deleteBatchJob(
  batchId: string
): Promise<{ batch_id: string; deleted: boolean }> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/${batchId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to delete batch job."
    );
  }
  return res.json();
}

export async function listSavedBatches(): Promise<{
  saved_batches: SavedBatchSummary[];
  count: number;
}> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/saved`);
  if (!res.ok) throw new Error("Failed to list saved batches.");
  return res.json();
}

export async function loadSavedBatch(
  batchId: string
): Promise<BatchResults> {
  const res = await fetch(`${BASE_URL}/pipeline/batch/saved/${batchId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to load saved batch."
    );
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Compound nodes (Tier 7)
// ---------------------------------------------------------------------------

export async function publishCompound(definition: {
  compound_id: string;
  display_name: string;
  description: string;
  tags: string[];
  sub_graph: Record<string, unknown>;
  entry_node_id: string;
  output_node_id: string;
  exposed_params: Array<{ inner_node_id: string; param_name: string; display_label: string }>;
}): Promise<{ status: string; compound_id: string; display_name: string }> {
  const res = await fetch(`${BASE_URL}/compound/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(definition),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to publish compound node."
    );
  }
  return res.json();
}

export async function deleteCompound(
  compoundId: string
): Promise<{ status: string; compound_id: string }> {
  const res = await fetch(`${BASE_URL}/compound/${compoundId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to delete compound node."
    );
  }
  return res.json();
}

export async function getCompound(
  compoundId: string
): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/compound/${compoundId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to fetch compound definition."
    );
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Research Workbench — Inspector, Browser, Export, Re-run
// ---------------------------------------------------------------------------

export async function openMneBrowser(
  sessionId: string,
  targetNodeId: string,
  nodeLabel?: string,
): Promise<{ status: string; node_id: string }> {
  const res = await fetch(`${BASE_URL}/pipeline/inspect/browser`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      session_id: sessionId,
      target_node_id: targetNodeId,
      node_label: nodeLabel,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to open MNE browser."
    );
  }
  return res.json();
}

export async function checkBrowserStatus(
  sessionId: string,
  nodeId: string,
): Promise<boolean> {
  const res = await fetch(
    `${BASE_URL}/pipeline/inspect/browser/status?session_id=${encodeURIComponent(sessionId)}&node_id=${encodeURIComponent(nodeId)}`
  );
  if (!res.ok) return false;
  const data = await res.json();
  return data.open;
}

export async function exportNodeOutput(
  sessionId: string,
  targetNodeId: string,
  format: string,
  nodeLabel: string,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/pipeline/node/export`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      session_id: sessionId,
      target_node_id: targetNodeId,
      format,
      node_label: nodeLabel,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Export failed."
    );
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition");
  const filename = disposition?.split("filename=")[1] || `export.${format}`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function syncBrowserAnnotations(
  sessionId: string,
  targetNodeId: string,
): Promise<{
  status: string;
  node_id: string;
  n_annotations: number;
  annotations: Array<{ onset: number; duration: number; description: string }>;
  session_info: Record<string, unknown> | null;
}> {
  const res = await fetch(`${BASE_URL}/pipeline/inspect/browser/sync-annotations`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      session_id: sessionId,
      target_node_id: targetNodeId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Failed to sync annotations."
    );
  }
  return res.json();
}

export async function executeFromNode(
  sessionId: string,
  pipeline: PipelineGraph,
  fromNodeId: string,
): Promise<ExecuteResponse> {
  const res = await fetch(`${BASE_URL}/pipeline/inspect/execute_from`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      session_id: sessionId,
      pipeline,
      from_node_id: fromNodeId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error(
      (err as { detail?: string } | null)?.detail ?? "Re-run failed."
    );
  }
  return res.json();
}

/** Fetch the generated MNE code for a node with given parameters. */
export async function getNodeCode(
  nodeType: string,
  params: Record<string, unknown> = {},
): Promise<{ code: string | null; docs_url: string | null; methods: string | null }> {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      searchParams.set(key, String(value));
    }
  }
  const query = searchParams.toString();
  const url = `${BASE_URL}/registry/nodes/${nodeType}/code${query ? `?${query}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch code for ${nodeType}`);
  return res.json();
}

/** Generate an academic Methods section from a pipeline. */
export async function generateMethods(
  sessionId: string,
  pipeline: PipelineGraph,
): Promise<{ methods_section: string; word_count: number; citations: string[] }> {
  const res = await fetch(`${BASE_URL}/pipeline/generate-methods`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ session_id: sessionId, pipeline }),
  });
  if (!res.ok) throw new Error("Failed to generate methods section");
  return res.json();
}

/** Download reproducibility package as a zip blob. */
export async function exportPackage(
  sessionId: string,
  pipeline: PipelineGraph,
  auditLog: unknown[] = [],
): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/pipeline/export-package`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ session_id: sessionId, pipeline, audit_log: auditLog }),
  });
  if (!res.ok) throw new Error("Failed to export package");
  return res.blob();
}

/** Export pipeline data in BIDS format as a zip blob. */
export async function exportBids(
  sessionId: string,
  pipeline: PipelineGraph,
  params: {
    subject_id: string;
    session: string;
    task: string;
    run: string;
    format: string;
  }
): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/pipeline/export-bids`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      pipeline,
      ...params,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

// ---------------------------------------------------------------------------
// Custom Node Presets (B5-B6)
// ---------------------------------------------------------------------------

export interface CustomNodeDefinition {
  slug: string;
  display_name: string;
  description: string;
  code: string;
  timeout_s: number;
  created_at: string;
}

export async function saveCustomNode(
  displayName: string,
  description: string,
  code: string,
  timeoutS: number = 60,
): Promise<{ status: string; node: CustomNodeDefinition }> {
  const res = await fetch(`${BASE_URL}/custom-nodes`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      display_name: displayName,
      description,
      code,
      timeout_s: timeoutS,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to save custom node.");
  }
  return res.json();
}

export async function listCustomNodes(): Promise<{
  custom_nodes: CustomNodeDefinition[];
  count: number;
}> {
  const res = await fetch(`${BASE_URL}/custom-nodes`);
  if (!res.ok) throw new Error("Failed to list custom nodes.");
  return res.json();
}

export async function deleteCustomNode(slug: string): Promise<{ status: string; slug: string }> {
  const res = await fetch(`${BASE_URL}/custom-nodes/${slug}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to delete custom node.");
  }
  return res.json();
}

export async function exportCustomNode(slug: string): Promise<CustomNodeDefinition> {
  const res = await fetch(`${BASE_URL}/custom-nodes/${slug}/export`);
  if (!res.ok) throw new Error("Failed to export custom node.");
  return res.json();
}

export async function importCustomNode(
  definition: { display_name: string; description: string; code: string; timeout_s: number },
): Promise<{ status: string; node: CustomNodeDefinition }> {
  const res = await fetch(`${BASE_URL}/custom-nodes/import`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(definition),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to import custom node.");
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Workflow storage API (backend-persisted)
// ---------------------------------------------------------------------------

import type { SavedWorkflow } from "../store/workflowStore";
import type { RunSnapshot } from "../hooks/useRunHistory";

export async function apiListWorkflows(): Promise<{ workflows: SavedWorkflow[]; count: number }> {
  const res = await fetch(`${BASE_URL}/workflows`);
  if (!res.ok) throw new Error("Failed to list workflows");
  return res.json();
}

export async function apiGetWorkflow(id: string): Promise<SavedWorkflow> {
  const res = await fetch(`${BASE_URL}/workflows/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error("Failed to get workflow");
  return res.json();
}

export async function apiSaveWorkflow(workflow: SavedWorkflow): Promise<{ workflow: SavedWorkflow }> {
  const res = await fetch(`${BASE_URL}/workflows`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(workflow),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to save workflow");
  }
  return res.json();
}

export async function apiUpdateWorkflow(id: string, workflow: SavedWorkflow): Promise<{ workflow: SavedWorkflow }> {
  const res = await fetch(`${BASE_URL}/workflows/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(workflow),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to update workflow");
  }
  return res.json();
}

export async function apiDeleteWorkflow(id: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/workflows/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete workflow");
}

export async function apiDuplicateWorkflow(id: string): Promise<{ workflow: SavedWorkflow }> {
  const res = await fetch(`${BASE_URL}/workflows/${encodeURIComponent(id)}/duplicate`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to duplicate workflow");
  return res.json();
}

export async function apiClearAllWorkflows(): Promise<{ deleted_count: number }> {
  const res = await fetch(`${BASE_URL}/workflows/clear-all`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear workflows");
  const data = await res.json();
  return { deleted_count: data.deleted_count };
}

export async function apiGetWorkflowStats(): Promise<{ count: number; disk_usage_bytes: number; workflows_dir: string }> {
  const res = await fetch(`${BASE_URL}/workflows/stats`);
  if (!res.ok) throw new Error("Failed to fetch workflow stats");
  return res.json();
}

// ---------------------------------------------------------------------------
// Run history API (backend-persisted)
// ---------------------------------------------------------------------------

export async function apiListRuns(): Promise<{ runs: RunSnapshot[]; count: number }> {
  const res = await fetch(`${BASE_URL}/history`);
  if (!res.ok) throw new Error("Failed to list run history");
  return res.json();
}

export async function apiGetRun(id: string): Promise<RunSnapshot> {
  const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error("Failed to get run");
  return res.json();
}

export async function apiSaveRun(run: RunSnapshot): Promise<{ run: RunSnapshot }> {
  const res = await fetch(`${BASE_URL}/history`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(run),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => null);
    throw new Error((err as { detail?: string } | null)?.detail ?? "Failed to save run");
  }
  return res.json();
}

export async function apiRenameRun(id: string, name: string): Promise<{ run: RunSnapshot }> {
  const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(id)}/rename`, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error("Failed to rename run");
  return res.json();
}

export async function apiDeleteRun(id: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error("Failed to delete run");
}

export async function apiClearAllRuns(): Promise<{ deleted_count: number }> {
  const res = await fetch(`${BASE_URL}/history/clear-all`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to clear run history");
  const data = await res.json();
  return { deleted_count: data.deleted_count };
}

export async function apiGetHistoryStats(): Promise<{ count: number; disk_usage_bytes: number; history_dir: string }> {
  const res = await fetch(`${BASE_URL}/history/stats`);
  if (!res.ok) throw new Error("Failed to fetch history stats");
  return res.json();
}
