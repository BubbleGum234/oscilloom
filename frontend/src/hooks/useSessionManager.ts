// Extracted from App.tsx — EEG file session management
import { useState, useCallback } from "react";
import type { Node } from "@xyflow/react";

import { loadSession } from "../api/client";
import type { NodeData } from "../types/pipeline";

export interface UseSessionManagerReturn {
  sessionId: string | null;
  sessionInfo: Record<string, unknown> | null;
  loadingFile: boolean;
  handleFileLoad: (file: File, nodeId: string) => Promise<void>;
  setSessionInfo: (info: Record<string, unknown>) => void;
}

export function useSessionManager(
  setNodes: React.Dispatch<React.SetStateAction<Node[]>>,
  markDirty: () => void,
  toast: (msg: string, type?: "success" | "error" | "warning" | "info") => void,
): UseSessionManagerReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<Record<string, unknown> | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);

  const handleFileLoad = useCallback(
    async (file: File, nodeId: string) => {
      setLoadingFile(true);
      try {
        const { session_id, info } = await loadSession(file);
        setSessionId(session_id);
        setSessionInfo(info as Record<string, unknown>);
        setNodes((nds) =>
          nds.map((n) =>
            n.id === nodeId
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    parameters: {
                      ...((n.data as NodeData).parameters),
                      file_path: file.name,
                    },
                    sessionInfo: info,
                  },
                }
              : n
          )
        );
        toast(`Loaded ${file.name}`, "success");
        markDirty();
      } catch (err) {
        toast(
          `Failed to load file: ${err instanceof Error ? err.message : String(err)}`,
          "error"
        );
      } finally {
        setLoadingFile(false);
      }
    },
    [setNodes, toast, markDirty]
  );

  return {
    sessionId,
    sessionInfo,
    loadingFile,
    handleFileLoad,
    setSessionInfo,
  };
}
