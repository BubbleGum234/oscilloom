import { useState, useCallback } from "react";
import type { AuditLogEntry } from "../types/pipeline";

const SESSION_STORAGE_KEY = "oscilloom_audit_log";

function loadFromStorage(): AuditLogEntry[] {
  try {
    const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function useAuditLog() {
  const [log, setLog] = useState<AuditLogEntry[]>(loadFromStorage);

  const append = useCallback((entry: Omit<AuditLogEntry, "timestamp">) => {
    const now = new Date();
    const timestamp = now.toTimeString().split(" ")[0]; // "HH:MM:SS"
    const fullEntry: AuditLogEntry = { ...entry, timestamp };

    setLog((prev) => {
      const updated = [...prev, fullEntry];
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const clear = useCallback(() => {
    setLog([]);
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
  }, []);

  return { log, append, clear };
}
