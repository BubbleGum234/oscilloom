import { useState, useEffect, useCallback } from "react";
import type { NodeRegistry } from "../types/pipeline";
import { getRegistry } from "../api/client";

let _fetchPromise: Promise<NodeRegistry> | null = null;

export function useNodeRegistry() {
  const [registry, setRegistry] = useState<NodeRegistry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRegistry = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (!_fetchPromise) {
        _fetchPromise = getRegistry().then((data) => data.nodes);
      }
      const result = await _fetchPromise;
      setRegistry(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      _fetchPromise = null; // allow retry on error
    } finally {
      setLoading(false);
    }
  }, []);

  const refresh = useCallback(async () => {
    _fetchPromise = null; // force re-fetch
    await fetchRegistry();
  }, [fetchRegistry]);

  useEffect(() => {
    fetchRegistry();
  }, [fetchRegistry]);

  return { registry, loading, error, refresh };
}
