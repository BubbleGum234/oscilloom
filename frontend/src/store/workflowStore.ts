/**
 * Backend-API-backed workflow storage for saved pipelines.
 *
 * Each workflow stores the full PipelineGraph JSON plus metadata
 * (name, timestamps, node count, thumbnail preview).
 *
 * Previously used IndexedDB -- now delegates to the backend REST API
 * so that data is persisted on the server filesystem (~/.oscilloom/).
 */

import type { PipelineGraph } from "../types/pipeline";
import {
  apiListWorkflows,
  apiGetWorkflow,
  apiSaveWorkflow,
  apiUpdateWorkflow,
  apiDeleteWorkflow,
  apiDuplicateWorkflow,
} from "../api/client";

export interface SavedWorkflow {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  nodeCount: number;
  edgeCount: number;
  pipeline: PipelineGraph;
}

export async function listWorkflows(): Promise<SavedWorkflow[]> {
  const data = await apiListWorkflows();
  return data.workflows;
}

export async function getWorkflow(id: string): Promise<SavedWorkflow | undefined> {
  try {
    const wf = await apiGetWorkflow(id);
    return wf;
  } catch {
    return undefined;
  }
}

export async function saveWorkflow(workflow: SavedWorkflow): Promise<void> {
  if (workflow.id) {
    // Check if the workflow already exists before deciding create vs update.
    try {
      const existing = await apiGetWorkflow(workflow.id);
      if (existing) {
        await apiUpdateWorkflow(workflow.id, workflow);
        return;
      }
    } catch {
      // 404 or not found — fall through to create
    }
  }
  await apiSaveWorkflow(workflow);
}

export async function deleteWorkflow(id: string): Promise<void> {
  await apiDeleteWorkflow(id);
}

export async function duplicateWorkflow(id: string): Promise<SavedWorkflow> {
  const data = await apiDuplicateWorkflow(id);
  return data.workflow;
}
