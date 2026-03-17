import { useState, useRef, useCallback } from "react";
import type { Node } from "@xyflow/react";
import type { NodeTypeDescriptor, ParameterSchema, SessionInfo, NodeData } from "../../types/pipeline";
import { SURFACE } from "../../constants/theme";

interface NodeParameterPanelProps {
  node: Node | null;
  onParamChange: (nodeId: string, paramName: string, value: unknown) => void;
  onFileLoad: (file: File, nodeId: string) => void;
  // TASK-22: shown for save_to_fif nodes so user can download processed EEG
  onDownloadFif?: () => void;
  selectionCount?: number;
  sessionInfo?: SessionInfo | null;
}

// ---------------------------------------------------------------------------
// Channel Multi-Select — tag-style picker for channel_hint="multi"
// ---------------------------------------------------------------------------

function ChannelMultiSelect({
  channelNames,
  value,
  onChange,
}: {
  channelNames: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);

  const selected = value
    ? value.split(",").map((s) => s.trim()).filter(Boolean)
    : [];

  const filtered = channelNames.filter(
    (ch) =>
      ch.toLowerCase().includes(search.toLowerCase()) &&
      !selected.includes(ch)
  );

  const toggle = (ch: string) => {
    const next = selected.includes(ch)
      ? selected.filter((s) => s !== ch)
      : [...selected, ch];
    onChange(next.join(", "));
  };

  const remove = (ch: string) => {
    onChange(selected.filter((s) => s !== ch).join(", "));
  };

  return (
    <div className="relative">
      {/* Selected tags */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-0.5 mb-1">
          {selected.map((ch) => (
            <span
              key={ch}
              className="inline-flex items-center gap-0.5 bg-cyan-900 text-cyan-200 text-[10px] font-mono px-1.5 py-0.5 rounded"
            >
              {ch}
              <button
                onClick={() => remove(ch)}
                className="text-cyan-400 hover:text-white ml-0.5"
              >
                x
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Search input */}
      <input
        type="text"
        value={search}
        onChange={(e) => { setSearch(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        placeholder="Search channels..."
        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs font-mono focus:outline-none focus:border-slate-400"
      />

      {/* Dropdown */}
      {open && filtered.length > 0 && (
        <div className="absolute z-20 mt-0.5 w-full max-h-32 overflow-y-auto bg-slate-800 border border-slate-600 rounded shadow-lg">
          {filtered.slice(0, 50).map((ch) => (
            <button
              key={ch}
              onMouseDown={(e) => { e.preventDefault(); toggle(ch); }}
              className="w-full text-left px-2 py-0.5 text-xs font-mono text-slate-200 hover:bg-slate-700"
            >
              {ch}
            </button>
          ))}
          {filtered.length > 50 && (
            <div className="px-2 py-0.5 text-[10px] text-slate-500">
              +{filtered.length - 50} more...
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Channel Single-Select — searchable dropdown for channel_hint="single"
// ---------------------------------------------------------------------------

function ChannelSingleSelect({
  channelNames,
  value,
  onChange,
}: {
  channelNames: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);

  const filtered = channelNames.filter((ch) =>
    ch.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="relative">
      <input
        type="text"
        value={open ? search : (value || "")}
        onChange={(e) => { setSearch(e.target.value); setOpen(true); }}
        onFocus={() => { setSearch(value || ""); setOpen(true); }}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
        placeholder="Select channel..."
        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs font-mono focus:outline-none focus:border-slate-400"
      />

      {open && filtered.length > 0 && (
        <div className="absolute z-20 mt-0.5 w-full max-h-32 overflow-y-auto bg-slate-800 border border-slate-600 rounded shadow-lg">
          {/* Show an empty option for optional channel params */}
          <button
            onMouseDown={(e) => { e.preventDefault(); onChange(""); setOpen(false); setSearch(""); }}
            className="w-full text-left px-2 py-0.5 text-xs text-slate-500 italic hover:bg-slate-700"
          >
            (none)
          </button>
          {filtered.slice(0, 50).map((ch) => (
            <button
              key={ch}
              onMouseDown={(e) => { e.preventDefault(); onChange(ch); setOpen(false); setSearch(""); }}
              className={`w-full text-left px-2 py-0.5 text-xs font-mono hover:bg-slate-700 ${ch === value ? "text-cyan-300 bg-slate-750" : "text-slate-200"}`}
            >
              {ch}
            </button>
          ))}
          {filtered.length > 50 && (
            <div className="px-2 py-0.5 text-[10px] text-slate-500">
              +{filtered.length - 50} more...
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Channel Type Editor — table view for set_channel_types node
// ---------------------------------------------------------------------------

const TYPE_HINTS: Record<string, string> = {
  EOG: "eog", eog: "eog",
  ECG: "ecg", ecg: "ecg", EKG: "ecg",
  EMG: "emg", emg: "emg",
  STI: "stim", Status: "stim", stim: "stim",
  EXG: "eog",  // BioSemi external channels, commonly EOG
};

const VALID_CH_TYPES = ["eeg", "eog", "ecg", "emg", "misc", "stim", "bio", "resp"];

function detectMismatch(name: string, currentType: string): string | null {
  for (const [hint, expectedType] of Object.entries(TYPE_HINTS)) {
    if (name.includes(hint) && currentType !== expectedType) {
      return expectedType;
    }
  }
  return null;
}

function ChannelTypeEditor({
  channels,
  value,
  onChange,
}: {
  channels: Array<{ name: string; type: string }>;
  value: string;
  onChange: (v: string) => void;
}) {
  const [search, setSearch] = useState("");

  // Parse current mapping from string
  const overrides: Record<string, string> = {};
  if (value) {
    for (const pair of value.split(",")) {
      const trimmed = pair.trim();
      if (trimmed.includes("=")) {
        const [ch, t] = trimmed.split("=", 2);
        overrides[ch.trim()] = t.trim().toLowerCase();
      }
    }
  }

  const setOverride = (chName: string, newType: string, currentType: string) => {
    const next = { ...overrides };
    if (newType === currentType) {
      delete next[chName]; // Reset to original
    } else {
      next[chName] = newType;
    }
    const mappingStr = Object.entries(next)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    onChange(mappingStr);
  };

  const filtered = search
    ? channels.filter((ch) => ch.name.toLowerCase().includes(search.toLowerCase()))
    : channels;

  const mismatchCount = channels.filter((ch) => detectMismatch(ch.name, ch.type)).length;

  return (
    <div className="space-y-1.5">
      <div className="relative">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search channels..."
          className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs placeholder-slate-500 focus:outline-none focus:border-slate-400"
        />
      </div>

      {mismatchCount > 0 && !search && (
        <div className="text-[10px] bg-amber-950 border border-amber-800 rounded px-2 py-1 text-amber-200">
          {mismatchCount} channel{mismatchCount > 1 ? "s" : ""} may be mislabeled
        </div>
      )}

      <div className="max-h-48 overflow-y-auto border border-slate-700 rounded">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-slate-800">
            <tr className="text-slate-400">
              <th className="text-left px-1.5 py-1 font-medium">Channel</th>
              <th className="text-left px-1.5 py-1 font-medium">Current</th>
              <th className="text-left px-1.5 py-1 font-medium">New Type</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 100).map((ch) => {
              const mismatch = detectMismatch(ch.name, ch.type);
              const override = overrides[ch.name];
              const isChanged = !!override;
              return (
                <tr
                  key={ch.name}
                  className={
                    isChanged
                      ? "bg-cyan-950/30"
                      : mismatch
                        ? "bg-amber-950/20"
                        : "hover:bg-slate-800/50"
                  }
                >
                  <td className="px-1.5 py-0.5 font-mono text-slate-200">
                    {ch.name}
                    {mismatch && !isChanged && (
                      <span className="ml-1 text-amber-400" title={`Name suggests ${mismatch}`}>!</span>
                    )}
                  </td>
                  <td className="px-1.5 py-0.5 text-slate-400 font-mono">{ch.type}</td>
                  <td className="px-1.5 py-0.5">
                    <select
                      value={override || ch.type}
                      onChange={(e) => setOverride(ch.name, e.target.value, ch.type)}
                      className={`bg-transparent border rounded px-1 py-0 text-[10px] font-mono focus:outline-none ${
                        isChanged
                          ? "border-cyan-600 text-cyan-300"
                          : "border-transparent text-slate-400"
                      }`}
                    >
                      {VALID_CH_TYPES.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {filtered.length > 100 && (
          <div className="text-[10px] text-slate-500 text-center py-1">
            Showing first 100 of {filtered.length} — use search to filter
          </div>
        )}
      </div>

      {Object.keys(overrides).length > 0 && (
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-cyan-400">
            {Object.keys(overrides).length} change{Object.keys(overrides).length > 1 ? "s" : ""}
          </span>
          <button
            onClick={() => onChange("")}
            className="text-[10px] text-slate-500 hover:text-slate-300"
          >
            Reset all
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ParamControl — renders the correct widget for each parameter type
// ---------------------------------------------------------------------------

function ParamControl({
  param,
  value,
  onChange,
  channelNames,
  nodeType,
  sessionInfo,
}: {
  param: ParameterSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  channelNames?: string[];
  nodeType?: string;
  sessionInfo?: SessionInfo | null;
}) {
  // Set Channel Types — interactive table editor
  if (
    nodeType === "set_channel_types" &&
    param.name === "mapping" &&
    sessionInfo?.ch_name_type_list &&
    sessionInfo.ch_name_type_list.length > 0
  ) {
    return (
      <ChannelTypeEditor
        channels={sessionInfo.ch_name_type_list}
        value={String(value ?? param.default)}
        onChange={(v) => onChange(v)}
      />
    );
  }

  // P2: Channel dropdown when channel_hint is set and channels are available
  if (param.channel_hint && channelNames && channelNames.length > 0) {
    if (param.channel_hint === "multi") {
      return (
        <ChannelMultiSelect
          channelNames={channelNames}
          value={String(value ?? param.default)}
          onChange={(v) => onChange(v)}
        />
      );
    }
    // "single"
    return (
      <ChannelSingleSelect
        channelNames={channelNames}
        value={String(value ?? param.default)}
        onChange={(v) => onChange(v)}
      />
    );
  }

  if (param.type === "select") {
    return (
      <select
        value={String(value ?? param.default)}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
      >
        {param.options?.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    );
  }

  if (param.type === "bool") {
    return (
      <input
        type="checkbox"
        checked={Boolean(value ?? param.default)}
        onChange={(e) => onChange(e.target.checked)}
        className="w-4 h-4 accent-cyan-500"
      />
    );
  }

  if (param.type === "string" && param.name === "code") {
    return (
      <textarea
        value={String(value ?? param.default)}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        rows={12}
        className="w-full bg-gray-900 border border-gray-700 rounded p-3 text-blue-400 text-xs font-mono resize-y focus:outline-none focus:border-slate-400 leading-relaxed"
        placeholder="# Write your MNE-Python code here..."
      />
    );
  }

  if (param.type === "string") {
    return (
      <input
        type="text"
        value={String(value ?? param.default)}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs font-mono focus:outline-none focus:border-slate-400"
      />
    );
  }

  // float or int
  return (
    <input
      type="number"
      value={String(value ?? param.default)}
      min={param.min}
      max={param.max}
      step={param.step ?? "any"}
      onChange={(e) => {
        const v =
          param.type === "int"
            ? parseInt(e.target.value, 10)
            : parseFloat(e.target.value);
        onChange(isNaN(v) ? param.default : v);
      }}
      className="w-24 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-slate-200 text-xs font-mono text-right focus:outline-none focus:border-slate-400"
    />
  );
}

export function NodeParameterPanel({
  node,
  onParamChange,
  onFileLoad,
  onDownloadFif,
  selectionCount = 0,
  sessionInfo,
}: NodeParameterPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const nodeData = node?.data as NodeData | undefined;
  const descriptor = nodeData?.descriptor;
  const parameters = nodeData?.parameters;

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && node) onFileLoad(file, node.id);
      // Reset so same file can be re-selected if needed
      e.target.value = "";
    },
    [node, onFileLoad]
  );

  const isSaveToFif = descriptor?.node_type === "save_to_fif";

  // P2: Channel names from session info for dropdown population
  const channelNames = sessionInfo?.ch_names;

  return (
    <div className="flex-1 overflow-y-auto min-h-0" style={{ background: SURFACE.panel }}>
      <div className="px-3 py-2.5 border-b border-slate-700 flex-shrink-0 sticky top-0 z-10" style={{ background: SURFACE.panelHeader }}>
        <h2 className="text-slate-400 font-semibold text-[11px] uppercase tracking-wide">Parameters</h2>
        <p className="text-slate-500 text-[11px] mt-0.5">
          {selectionCount > 1
            ? `${selectionCount} nodes selected`
            : descriptor
              ? descriptor.display_name
              : "Select a node"}
        </p>
      </div>

      {!node && (
        <div className="p-4 text-slate-600 text-xs text-center mt-6 leading-relaxed">
          Click a node on the canvas
          <br />
          to edit its parameters.
        </div>
      )}

      {descriptor && node && (
        <div className="p-3 space-y-4">
          {/* File picker — shown for source nodes (no inputs) */}
          {descriptor.inputs.length === 0 && (
            <div className="space-y-1.5">
              <input
                ref={fileInputRef}
                type="file"
                accept=".edf,.fif,.bdf,.set,.vhdr,.cnt"
                className="hidden"
                onChange={handleFileChange}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full bg-blue-700 hover:bg-blue-600 text-white text-xs rounded px-3 py-2 transition-colors"
              >
                Browse for EEG file…
              </button>
              {!!parameters?.file_path && (
                <div className="text-slate-400 text-[10px] break-all px-1">
                  {String(parameters.file_path).split("/").pop()}
                </div>
              )}
            </div>
          )}

          {/* TASK-22: Download .fif button for save_to_fif nodes */}
          {isSaveToFif && onDownloadFif && (
            <div className="space-y-1.5">
              <button
                onClick={onDownloadFif}
                className="w-full bg-teal-700 hover:bg-teal-600 text-white text-xs rounded px-3 py-2 transition-colors"
                title="Download the processed EEG as a .fif file via your browser"
              >
                ↓ Download .fif
              </button>
              <p className="text-slate-600 text-[10px] leading-tight px-1">
                Downloads processed EEG directly to your browser. Leave Output
                Path blank to use this button instead of saving to disk.
              </p>
            </div>
          )}

          {/* Parameter controls — hidden params (e.g. file_path) are excluded */}
          {descriptor.parameters.filter((param) => !param.hidden).map((param) => (
            <div key={param.name}>
              <div className="flex items-center justify-between mb-1">
                <label
                  className="text-slate-300 text-[12px] font-medium"
                  title={param.description}
                >
                  {param.label}
                  {param.unit && (
                    <span className="ml-1 text-slate-600 font-normal">
                      ({param.unit})
                    </span>
                  )}
                </label>
              </div>
              <ParamControl
                param={param}
                value={parameters?.[param.name]}
                onChange={(v) => onParamChange(node.id, param.name, v)}
                channelNames={channelNames}
                nodeType={descriptor?.node_type}
                sessionInfo={sessionInfo}
              />
              {param.description && (
                <p className="text-slate-600 text-[10px] mt-1 leading-tight">
                  {param.description}
                </p>
              )}
            </div>
          ))}

          {descriptor.parameters.length === 0 && descriptor.inputs.length > 0 && (
            <p className="text-slate-600 text-xs text-center mt-2">
              No configurable parameters.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
