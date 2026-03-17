import { useState, useMemo } from "react";
import type { ReportSections } from "../../api/client";

/** Describes one selectable output from the pipeline run. */
export interface NodeOutput {
  nodeId: string;
  label: string;        // User-friendly label (node display name)
  nodeType: string;     // e.g. "compute_asymmetry", "plot_psd"
  kind: "metrics" | "plot";
}

interface ReportModalProps {
  onClose: () => void;
  onGenerate: (
    title: string,
    patientId: string,
    clinicName: string,
    notes: string,
    sections: ReportSections,
    includedNodes: string[],
  ) => Promise<void>;
  hasSessionInfo: boolean;
  hasPipelineConfig: boolean;
  hasAuditLog: boolean;
  /** All reportable outputs from the last pipeline run. */
  availableOutputs: NodeOutput[];
}

const DEFAULT_SECTIONS: ReportSections = {
  data_quality: true,
  pipeline_config: true,
  analysis_results: true,
  clinical_interpretation: true,
  visualizations: true,
  audit_trail: true,
  notes: true,
};

const SECTION_LABELS: Record<keyof ReportSections, { label: string; description: string }> = {
  data_quality: {
    label: "Data Quality Summary",
    description: "Recording metadata: sampling rate, channels, duration, bad channels",
  },
  pipeline_config: {
    label: "Pipeline Configuration",
    description: "Processing steps with parameter values",
  },
  analysis_results: {
    label: "Analysis Results",
    description: "Metrics tables from selected analysis nodes",
  },
  clinical_interpretation: {
    label: "Clinical Interpretation",
    description: "Reference ranges and status flags for known metrics",
  },
  visualizations: {
    label: "Visualizations",
    description: "Plots from selected visualization nodes",
  },
  audit_trail: {
    label: "Audit Trail",
    description: "Parameter change history during this session",
  },
  notes: {
    label: "Clinician Notes",
    description: "Free-text observations included in the report",
  },
};

const SECTION_ORDER: (keyof ReportSections)[] = [
  "data_quality",
  "pipeline_config",
  "analysis_results",
  "clinical_interpretation",
  "visualizations",
  "audit_trail",
  "notes",
];

export function ReportModal({
  onClose,
  onGenerate,
  hasSessionInfo,
  hasPipelineConfig,
  hasAuditLog,
  availableOutputs,
}: ReportModalProps) {
  const [title, setTitle] = useState("Oscilloom EEG Report");
  const [patientId, setPatientId] = useState("");
  const [clinicName, setClinicName] = useState("");
  const [notes, setNotes] = useState("");
  const [sections, setSections] = useState<ReportSections>(DEFAULT_SECTIONS);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // All outputs start selected
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(
    () => new Set(availableOutputs.map((o) => o.nodeId))
  );

  const metricsOutputs = useMemo(
    () => availableOutputs.filter((o) => o.kind === "metrics"),
    [availableOutputs]
  );
  const plotOutputs = useMemo(
    () => availableOutputs.filter((o) => o.kind === "plot"),
    [availableOutputs]
  );

  const selectedMetrics = metricsOutputs.filter((o) => selectedNodes.has(o.nodeId));
  const selectedPlots = plotOutputs.filter((o) => selectedNodes.has(o.nodeId));

  const toggleSection = (key: keyof ReportSections) => {
    setSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const toggleNode = (nodeId: string) => {
    setSelectedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  };

  const selectAllOutputs = () =>
    setSelectedNodes(new Set(availableOutputs.map((o) => o.nodeId)));
  const selectNoneOutputs = () => setSelectedNodes(new Set());

  // Section availability is now based on what the user selected
  const sectionAvailability: Record<keyof ReportSections, boolean> = {
    data_quality: hasSessionInfo,
    pipeline_config: hasPipelineConfig,
    analysis_results: selectedMetrics.length > 0,
    clinical_interpretation: selectedMetrics.length > 0,
    visualizations: selectedPlots.length > 0,
    audit_trail: hasAuditLog,
    notes: true,
  };

  const enabledCount = SECTION_ORDER.filter(
    (k) => sections[k] && sectionAvailability[k]
  ).length;

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    try {
      await onGenerate(
        title, patientId, clinicName, notes, sections,
        Array.from(selectedNodes),
      );
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Report generation failed.");
    } finally {
      setGenerating(false);
    }
  }

  function renderOutputGroup(label: string, outputs: NodeOutput[], icon: string) {
    if (outputs.length === 0) return null;
    return (
      <div>
        <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1">
          <span>{icon}</span> {label} ({outputs.filter((o) => selectedNodes.has(o.nodeId)).length}/{outputs.length})
        </div>
        <div className="space-y-0.5">
          {outputs.map((o) => (
            <label
              key={o.nodeId}
              className="flex items-center gap-2 px-2 py-1 rounded cursor-pointer hover:bg-slate-800 transition-colors"
            >
              <input
                type="checkbox"
                checked={selectedNodes.has(o.nodeId)}
                onChange={() => toggleNode(o.nodeId)}
                className="rounded border-slate-600"
              />
              <span className="text-xs text-slate-200 truncate">{o.label}</span>
              <span className="text-[10px] text-slate-600 ml-auto flex-shrink-0">
                {o.nodeType.replace(/_/g, " ")}
              </span>
            </label>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-slate-800 border border-slate-600 rounded-lg shadow-2xl w-[560px] max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
          <h2 className="text-slate-100 font-semibold text-sm">Generate PDF Report</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-lg leading-none"
          >
            x
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {/* Report metadata */}
          <div className="space-y-3">
            <div>
              <label className="block text-slate-400 text-xs mb-1">Report Title</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
              />
            </div>

            <div className="flex gap-3">
              <div className="flex-1">
                <label className="block text-slate-400 text-xs mb-1">Patient ID</label>
                <input
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g. PT-001"
                  className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
                />
              </div>
              <div className="flex-1">
                <label className="block text-slate-400 text-xs mb-1">Clinic / Lab</label>
                <input
                  type="text"
                  value={clinicName}
                  onChange={(e) => setClinicName(e.target.value)}
                  placeholder="e.g. Neurology Dept."
                  className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400"
                />
              </div>
            </div>
          </div>

          {/* ── Output Selector ──────────────────────────────────── */}
          {availableOutputs.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-slate-400 text-xs">
                  Include in Report ({selectedNodes.size}/{availableOutputs.length} outputs)
                </label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={selectAllOutputs}
                    className="text-[10px] text-purple-400 hover:text-purple-300"
                  >
                    All
                  </button>
                  <button
                    type="button"
                    onClick={selectNoneOutputs}
                    className="text-[10px] text-slate-500 hover:text-slate-400"
                  >
                    None
                  </button>
                </div>
              </div>
              <div className="bg-slate-900 border border-slate-700 rounded p-2 space-y-3 max-h-48 overflow-y-auto">
                {renderOutputGroup("Metrics", metricsOutputs, "\u2630")}
                {renderOutputGroup("Visualizations", plotOutputs, "\u25A3")}
              </div>
            </div>
          )}

          {/* ── Section toggles ──────────────────────────────────── */}
          <div>
            <label className="block text-slate-400 text-xs mb-2">
              Report Sections ({enabledCount} enabled)
            </label>
            <div className="bg-slate-900 border border-slate-700 rounded p-2 space-y-1">
              {SECTION_ORDER.map((key) => {
                const available = sectionAvailability[key];
                const enabled = sections[key];
                const info = SECTION_LABELS[key];

                return (
                  <label
                    key={key}
                    className={`flex items-start gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                      available
                        ? "hover:bg-slate-800"
                        : "opacity-40 cursor-not-allowed"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={enabled && available}
                      disabled={!available}
                      onChange={() => toggleSection(key)}
                      className="mt-0.5 rounded border-slate-600"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-slate-200 flex items-center gap-2">
                        {info.label}
                        {!available && (
                          <span className="text-[10px] text-slate-500 bg-slate-800 px-1.5 rounded">
                            no data
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        {info.description}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Report preview summary */}
          <div className="bg-slate-900/50 border border-slate-700 rounded px-3 py-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
              Preview
            </div>
            <div className="text-xs text-slate-400 space-y-0.5">
              {selectedMetrics.length > 0 && sections.analysis_results && (
                <div>{selectedMetrics.length} metrics table{selectedMetrics.length > 1 ? "s" : ""}: {selectedMetrics.map((o) => o.label).join(", ")}</div>
              )}
              {selectedMetrics.length > 0 && sections.clinical_interpretation && (
                <div>Clinical interpretation flags</div>
              )}
              {selectedPlots.length > 0 && sections.visualizations && (
                <div>{selectedPlots.length} visualization{selectedPlots.length > 1 ? "s" : ""}: {selectedPlots.map((o) => o.label).join(", ")}</div>
              )}
              {hasSessionInfo && sections.data_quality && (
                <div>Recording quality metadata</div>
              )}
              {hasPipelineConfig && sections.pipeline_config && (
                <div>Pipeline step documentation</div>
              )}
              {hasAuditLog && sections.audit_trail && (
                <div>Parameter change log</div>
              )}
              {enabledCount === 0 && (
                <div className="text-slate-600 italic">No sections selected</div>
              )}
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-slate-400 text-xs mb-1">
              Clinician Notes (optional)
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. Patient was drowsy during recording. Eyes-closed resting state."
              rows={3}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-slate-200 text-xs focus:outline-none focus:border-slate-400 resize-none"
            />
          </div>

          {error && (
            <div className="text-red-400 text-xs bg-red-950 border border-red-800 rounded px-2 py-1.5">
              {error}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex gap-2 px-5 py-3 border-t border-slate-700">
          <button
            onClick={handleGenerate}
            disabled={generating || enabledCount === 0}
            className="flex-1 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-700 disabled:text-slate-500 text-white text-xs rounded px-3 py-2 transition-colors font-medium"
          >
            {generating ? "Generating..." : "Download PDF"}
          </button>
          <button
            onClick={onClose}
            className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded px-3 py-2 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
