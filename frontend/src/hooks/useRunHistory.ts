import { useState, useCallback, useEffect } from "react";
import type { ExecuteResponse, NodeResult, NodeData } from "../types/pipeline";
import type { Node } from "@xyflow/react";
import {
  apiListRuns,
  apiSaveRun,
  apiRenameRun,
  apiDeleteRun,
} from "../api/client";

export interface RunSnapshot {
  id: string;
  timestamp: string;
  name: string;
  nodeResults: Record<string, NodeResult>;
  paramSnapshot: Record<string, Record<string, unknown>>;
  thumbnails: Record<string, string>;
  nodeCount: number;
  errorCount: number;
}

const MAX_RUNS = 10;

async function resizePlotToThumbnail(
  base64Png: string,
  maxWidth = 400
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, maxWidth / img.width);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/png", 0.7));
    };
    img.onerror = () => resolve("");
    img.src = base64Png;
  });
}

export function useRunHistory() {
  const [history, setHistory] = useState<RunSnapshot[]>([]);

  // Load history from backend API on mount
  useEffect(() => {
    apiListRuns()
      .then((data) => setHistory(data.runs.slice(0, MAX_RUNS)))
      .catch((err) => {
        console.error("Failed to load run history from backend:", err);
      });
  }, []);

  const capture = useCallback(
    async (result: ExecuteResponse, nodes: Node[]) => {
      const id = crypto.randomUUID();
      const timestamp = new Date().toISOString();

      // Build param snapshot
      const paramSnapshot: Record<string, Record<string, unknown>> = {};
      for (const node of nodes) {
        const params = (node.data as NodeData).parameters;
        if (params && typeof params === "object") {
          paramSnapshot[node.id] = { ...params };
        }
      }

      // Build thumbnails from plot nodes
      const thumbnails: Record<string, string> = {};
      for (const [nodeId, nr] of Object.entries(result.node_results)) {
        if (
          nr.status === "success" &&
          nr.data &&
          typeof nr.data === "string" &&
          nr.data.startsWith("data:image")
        ) {
          thumbnails[nodeId] = await resizePlotToThumbnail(nr.data);
        }
      }

      // Clone node results without full-res plot data (save space)
      const nodeResults: Record<string, NodeResult> = {};
      for (const [nodeId, nr] of Object.entries(result.node_results)) {
        nodeResults[nodeId] = {
          ...nr,
          data: thumbnails[nodeId] || undefined,
        };
      }

      const errorCount = Object.values(result.node_results).filter(
        (r) => r.status === "error"
      ).length;

      const run: RunSnapshot = {
        id,
        timestamp,
        name: `Run ${timestamp.split("T")[1].split(".")[0]}`,
        nodeResults,
        paramSnapshot,
        thumbnails,
        nodeCount: Object.keys(result.node_results).length,
        errorCount,
      };

      await apiSaveRun(run);

      // Refresh the list from backend to get accurate state
      try {
        const data = await apiListRuns();
        setHistory(data.runs.slice(0, MAX_RUNS));
      } catch {
        // Optimistic local update as fallback
        setHistory((prev) => [run, ...prev].slice(0, MAX_RUNS));
      }
    },
    []
  );

  const restore = useCallback(
    (runId: string): RunSnapshot | undefined => {
      return history.find((h) => h.id === runId);
    },
    [history]
  );

  const rename = useCallback(
    async (runId: string, name: string) => {
      try {
        const { run: updated } = await apiRenameRun(runId, name);
        setHistory((prev) =>
          prev.map((h) => (h.id === runId ? updated : h))
        );
      } catch {
        // Optimistic local update as fallback
        setHistory((prev) =>
          prev.map((h) => (h.id === runId ? { ...h, name } : h))
        );
      }
    },
    []
  );

  const remove = useCallback(
    async (runId: string) => {
      await apiDeleteRun(runId);
      setHistory((prev) => prev.filter((h) => h.id !== runId));
    },
    []
  );

  return { history, capture, restore, rename, remove };
}
