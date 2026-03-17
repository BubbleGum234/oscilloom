import { useCallback, useRef, useState } from "react";
import type { Node, Edge } from "@xyflow/react";

type Snapshot = { nodes: Node[]; edges: Edge[] };

/**
 * Lightweight undo stack for canvas node/edge state.
 *
 * push()  — saves a snapshot before a destructive change (deletion).
 * pop()   — restores the most recent snapshot synchronously via ref.
 * canUndo — true when there is at least one snapshot to restore.
 *
 * History is capped at 50 snapshots (enough for any realistic session).
 */
export function useHistory() {
  const stackRef = useRef<Snapshot[]>([]);
  const [canUndo, setCanUndo] = useState(false);

  const push = useCallback((nodes: Node[], edges: Edge[]) => {
    stackRef.current = [...stackRef.current.slice(-49), { nodes, edges }];
    setCanUndo(true);
  }, []);

  const pop = useCallback((): Snapshot | null => {
    if (stackRef.current.length === 0) return null;
    const snapshot = stackRef.current[stackRef.current.length - 1];
    stackRef.current = stackRef.current.slice(0, -1);
    setCanUndo(stackRef.current.length > 0);
    return snapshot;
  }, []);

  return { push, pop, canUndo };
}
