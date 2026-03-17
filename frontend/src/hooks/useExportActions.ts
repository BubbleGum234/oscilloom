// Extracted from App.tsx — export, download, and report generation actions
import { useState, useCallback } from "react";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData, ExecuteResponse, AuditLogEntry } from "../types/pipeline";
import type { ReportSections } from "../api/client";
import { exportPipeline, downloadFif, generateReport } from "../api/client";
import { serializePipeline } from "../utils/serializePipeline";

export interface UseExportActionsReturn {
  loadingExport: boolean;
  loadingFif: boolean;
  loadingReport: boolean;
  handleExport: () => Promise<void>;
  handleDownloadFif: () => Promise<void>;
  handleGenerateReport: (
    title: string,
    patientId: string,
    clinicName: string,
    notes: string,
    sections: ReportSections,
    includedNodes: string[],
  ) => Promise<void>;
}

export function useExportActions(
  nodes: Node[],
  edges: Edge[],
  sessionId: string | null,
  pipelineName: string,
  auditLog: AuditLogEntry[],
  result: ExecuteResponse | null,
  sessionInfo: Record<string, unknown> | null,
  toast: (msg: string, type?: "success" | "error" | "warning" | "info") => void,
): UseExportActionsReturn {
  const [loadingExport, setLoadingExport] = useState(false);
  const [loadingFif, setLoadingFif] = useState(false);
  const [loadingReport, setLoadingReport] = useState(false);

  // ── Download processed .fif ───────────────────────────────────────────────
  const handleDownloadFif = useCallback(async () => {
    if (!sessionId) return;
    setLoadingFif(true);
    try {
      const pipeline = serializePipeline(nodes as Node[], edges as Edge[], pipelineName);
      const blob = await downloadFif(sessionId, pipeline);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "processed.fif";
      a.click();
      URL.revokeObjectURL(url);
      toast("Downloaded processed.fif", "success");
    } catch (err) {
      toast(`Download failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setLoadingFif(false);
    }
  }, [nodes, edges, sessionId, pipelineName, toast]);

  // ── Export .py ────────────────────────────────────────────────────────────
  const handleExport = useCallback(async () => {
    if (!sessionId) {
      toast("Load an EEG file first.", "warning");
      return;
    }
    setLoadingExport(true);
    try {
      const pipeline = serializePipeline(nodes as Node[], edges as Edge[], pipelineName);
      const { script, filename } = await exportPipeline(sessionId, pipeline, auditLog);
      const blob = new Blob([script], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      toast(`Exported ${filename}`, "success");
    } catch (err) {
      toast(`Export failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setLoadingExport(false);
    }
  }, [nodes, edges, sessionId, auditLog, pipelineName, toast]);

  // ── Generate PDF report ───────────────────────────────────────────────────
  const handleGenerateReport = useCallback(
    async (
      title: string,
      patientId: string,
      clinicName: string,
      notes: string,
      sections: ReportSections,
      includedNodes: string[],
    ) => {
      if (!result) throw new Error("No pipeline results. Run the pipeline first.");

      setLoadingReport(true);
      try {
        // Build pipeline config from current canvas nodes
        const pipelineConfig = (nodes as Node[]).map((n) => {
          const nodeData = n.data as NodeData;
          return {
            node_id: n.id,
            node_type: nodeData.nodeType ?? nodeData.descriptor?.node_type ?? "",
            label: nodeData.label,
            parameters: nodeData.parameters,
          };
        });

        const blob = await generateReport({
          nodeResults: result.node_results,
          title,
          patientId,
          clinicName,
          sessionInfo: sessionInfo,
          pipelineConfig,
          auditLog: auditLog as unknown as Array<Record<string, unknown>>,
          notes,
          sections,
          includedNodes,
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${title.replace(/\s+/g, "_") || "report"}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
        toast("Report generated", "success");
      } finally {
        setLoadingReport(false);
      }
    },
    [result, nodes, sessionInfo, auditLog, toast]
  );

  return {
    loadingExport,
    loadingFif,
    loadingReport,
    handleExport,
    handleDownloadFif,
    handleGenerateReport,
  };
}
