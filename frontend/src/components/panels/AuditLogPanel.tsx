import type { AuditLogEntry } from "../../types/pipeline";

interface AuditLogPanelProps {
  log: AuditLogEntry[];
  onClear: () => void;
}

export function AuditLogPanel({ log, onClear }: AuditLogPanelProps) {
  return (
    <div className="flex flex-col flex-1 min-h-0 bg-slate-900">
      <div className="px-3 py-2 border-b border-slate-700 flex items-center justify-between flex-shrink-0">
        <h2 className="text-slate-200 font-semibold text-sm">Audit Log</h2>
        {log.length > 0 && (
          <button
            onClick={onClear}
            className="text-slate-500 hover:text-slate-300 text-[10px] transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {log.length === 0 ? (
          <p className="text-slate-600 text-[10px] text-center mt-4">
            No parameter changes yet.
          </p>
        ) : (
          [...log].reverse().map((entry, i) => (
            <div
              key={i}
              className="text-[10px] border border-slate-800 rounded px-2 py-1 bg-slate-950"
            >
              <div className="flex items-center justify-between gap-1">
                <span className="text-slate-400 truncate font-medium">
                  {entry.nodeDisplayName}
                </span>
                <span className="text-slate-600 font-mono flex-shrink-0">
                  {entry.timestamp}
                </span>
              </div>
              <div className="text-slate-400 truncate">{entry.paramLabel}</div>
              <div className="font-mono mt-0.5">
                <span className="text-red-400">{String(entry.oldValue)}</span>
                <span className="text-slate-600"> → </span>
                <span className="text-cyan-400">{String(entry.newValue)}</span>
                {entry.unit && (
                  <span className="text-slate-600"> {entry.unit}</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
