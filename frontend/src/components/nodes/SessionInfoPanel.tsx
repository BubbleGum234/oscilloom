import type { SessionInfo } from "../../types/pipeline";

export function SessionInfoPanel({ info }: { info: SessionInfo }) {
  return (
    <div className="text-[10px] space-y-0.5 text-slate-300">
      {/* Core recording stats */}
      <div className="flex justify-between">
        <span className="text-slate-400">Channels</span>
        <span className="font-mono">{info.nchan}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Rate</span>
        <span className="font-mono">{info.sfreq} Hz</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Duration</span>
        <span className="font-mono">{info.duration_s.toFixed(1)} s</span>
      </div>

      {/* Hardware filter — affects downstream bandpass choices */}
      <div className="flex justify-between">
        <span className="text-slate-400">HW filter</span>
        <span className="font-mono">{info.highpass}–{info.lowpass} Hz</span>
      </div>

      {/* Measurement date — identifies the recording */}
      {info.meas_date && (
        <div className="flex justify-between">
          <span className="text-slate-400">Recorded</span>
          <span className="font-mono">{info.meas_date.slice(0, 10)}</span>
        </div>
      )}

      {/* Channel type breakdown — tells researcher if Pick Channels is needed */}
      {Object.keys(info.ch_types).length > 0 && (
        <div className="flex justify-between items-start">
          <span className="text-slate-400 shrink-0">Types</span>
          <span className="font-mono text-right ml-2">
            {Object.entries(info.ch_types).map(([t, n]) => `${t}:${n}`).join("  ")}
          </span>
        </div>
      )}

      {/* Events / annotations — critical: researcher needs these to set Event ID */}
      {info.n_annotations > 0 && (
        <div className="mt-1 pt-1 border-t border-slate-700">
          <div className="flex justify-between mb-0.5">
            <span className="text-slate-400">Events</span>
            <span className="font-mono">{info.n_annotations}</span>
          </div>
          <div className="flex flex-wrap gap-0.5">
            {info.annotation_labels.map((lbl) => (
              <span
                key={lbl}
                className={`px-1 rounded font-mono ${
                  lbl.startsWith("BAD_")
                    ? "bg-red-900/40 text-red-400"
                    : "bg-slate-700 text-cyan-300"
                }`}
                style={{ fontSize: "9px" }}
              >
                {lbl}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Pre-marked bad channels from file header */}
      {info.bads.length > 0 && (
        <div className="mt-1 pt-1 border-t border-slate-700">
          <div className="text-amber-400 font-medium mb-0.5" style={{ fontSize: "9px" }}>
            Pre-marked bads:
          </div>
          <div className="text-amber-300 font-mono break-all" style={{ fontSize: "9px" }}>
            {info.bads.join(", ")}
          </div>
        </div>
      )}

      {/* Channel name list — researcher needs these to fill Bad Channels / Reference params */}
      {info.ch_names.length > 0 && (
        <div className="mt-1 pt-1 border-t border-slate-700">
          <div className="text-slate-400 mb-0.5" style={{ fontSize: "9px" }}>
            Channel names{info.ch_names_truncated ? ` (first 20 of ${info.nchan})` : ""}:
          </div>
          <div
            className="text-slate-300 font-mono break-all leading-tight"
            style={{ fontSize: "9px" }}
          >
            {info.ch_names.slice(0, 20).join(", ")}
            {info.ch_names_truncated && "\u2026"}
          </div>
        </div>
      )}

      {/* P1: Auto-detected naming convention suggestion */}
      {info.naming_hints?.rename_suggestion && (
        <div className="mt-1 pt-1 border-t border-slate-700">
          <div
            className="bg-amber-950 border border-amber-800 rounded px-1.5 py-1 text-amber-200 leading-tight"
            style={{ fontSize: "9px" }}
          >
            <span className="font-medium text-amber-300">Naming hint: </span>
            {info.naming_hints.rename_suggestion}
            {info.naming_hints.rename_params && (
              <div className="mt-0.5 text-amber-400 font-mono">
                {info.naming_hints.rename_params.strip_prefix && (
                  <span>prefix: &quot;{info.naming_hints.rename_params.strip_prefix}&quot; </span>
                )}
                {info.naming_hints.rename_params.strip_suffix && (
                  <span>suffix: &quot;{info.naming_hints.rename_params.strip_suffix}&quot;</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
