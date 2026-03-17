import { useState, useCallback } from "react";
import type { Node, Edge } from "@xyflow/react";
import type { ExecuteResponse } from "../types/pipeline";
import { executePipeline } from "../api/client";
import { serializePipeline } from "../utils/serializePipeline";

export function usePipelineRunner() {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ExecuteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(
    async (nodes: Node[], edges: Edge[], sessionId: string) => {
      setRunning(true);
      setError(null);
      setResult(null);
      try {
        const pipeline = serializePipeline(nodes, edges);
        const response = await executePipeline(sessionId, pipeline);
        setResult(response);
        if (response.status === "error") {
          setError(response.error ?? "Pipeline execution failed.");
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        setError(msg);
      } finally {
        setRunning(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // Dependencies intentionally empty: all referenced values are either
    // stable setState functions or top-level imports that never change.
    []
  );

  return { run, running, result, error };
}
