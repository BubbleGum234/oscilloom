export type HandleType =
  | "raw_eeg"
  | "filtered_eeg"
  | "epochs"
  | "evoked"
  | "psd"
  | "tfr"
  | "plot"
  | "array"
  | "scalar"
  | "metrics"
  | "connectivity"
  | "raw_fnirs"
  | "features";

export interface ParameterSchema {
  name: string;
  label: string;
  type: "float" | "int" | "bool" | "string" | "select";
  default: number | string | boolean;
  description: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  unit?: string;
  hidden?: boolean;
  exposed?: boolean;
  channel_hint?: "single" | "multi";
}

export interface HandleSchema {
  id: string;
  type: HandleType;
  label: string;
  required: boolean;
}

export interface NodeTypeDescriptor {
  node_type: string;
  display_name: string;
  category: string;
  description: string;
  inputs: HandleSchema[];
  outputs: HandleSchema[];
  parameters: ParameterSchema[];
  tags: string[];
}

export type NodeRegistry = Record<string, NodeTypeDescriptor>;

export interface PipelineNode {
  id: string;
  node_type: string;
  label: string;
  parameters: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface PipelineEdge {
  id: string;
  source_node_id: string;
  source_handle_id: string;
  source_handle_type: string;
  target_node_id: string;
  target_handle_id: string;
  target_handle_type: string;
}

export interface PipelineMetadata {
  name: string;
  description: string;
  created_by: "human" | "ai";
  schema_version: string;
}

export interface PipelineGraph {
  metadata: PipelineMetadata;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}

export interface AuditLogEntry {
  timestamp: string;
  nodeId: string;
  nodeDisplayName: string;
  paramLabel: string;
  oldValue: unknown;
  newValue: unknown;
  unit?: string;
}

export interface ExecuteResponse {
  status: "success" | "error";
  node_results: Record<string, NodeResult>;
  error?: string;
  failed_node_id?: string;
}

export interface NodeResult {
  node_type: string;
  status: "success" | "error";
  output_type: string;
  data?: string;
  preview?: string;
  error?: string;
  metrics?: Record<string, unknown>;
  summary?: NodeOutputSummary;
  rerun?: boolean;
  cache_hit?: boolean;
  execution_time_ms?: number;
}

export interface NodeOutputSummary {
  kind: string;
  python_type: string;
  // Raw
  n_channels?: number;
  sfreq?: number;
  duration_s?: number;
  shape?: number[];
  ch_names_preview?: string[];
  bads?: string[];
  highpass?: number;
  lowpass?: number;
  // Epochs
  n_epochs?: number;
  n_dropped?: number;
  tmin?: number;
  tmax?: number;
  event_counts?: Record<string, number>;
  // Evoked
  nave?: number;
  comment?: string;
  peak_channel?: string;
  peak_latency_s?: number;
  peak_amplitude?: number;
  // PSD / Spectrum
  freq_min?: number;
  freq_max?: number;
  n_freqs?: number;
  method?: string;
  // Array
  dtype?: string;
  min?: number;
  max?: number;
  mean?: number;
  // Metrics
  metrics?: Record<string, unknown>;
  // Scalar
  value?: number;
  // Connectivity
  n_connections?: number;
  // Error
  message?: string;
  traceback_preview?: string;
  // Unknown
  repr?: string;
  // Re-run flag
  rerun?: boolean;
}

export interface NamingHints {
  detected_prefix: string | null;
  detected_suffix: string | null;
  standard_match_pct: number;
  rename_suggestion: string | null;
  rename_params: { strip_prefix?: string; strip_suffix?: string } | null;
}

export interface SessionInfo {
  nchan: number;
  sfreq: number;
  ch_names: string[];
  ch_names_truncated: boolean;
  duration_s: number;
  highpass: number;
  lowpass: number;
  meas_date: string | null;
  ch_types: Record<string, number>;  // e.g. { eeg: 62, eog: 2 }
  bads: string[];                    // pre-marked bad channels from file header
  n_annotations: number;
  annotation_labels: string[];       // unique event labels, e.g. ["T0","T1","T2"]
  naming_hints?: NamingHints;        // auto-detected naming convention
  ch_name_type_list?: Array<{ name: string; type: string }>; // per-channel type mapping
}

// ---------------------------------------------------------------------------
// Batch processing (Tier 4)
// ---------------------------------------------------------------------------

export interface StagedFile {
  file_id: string;
  filename: string;
  metadata?: Record<string, string>;
}

export interface BatchProgress {
  batch_id: string;
  status: "running" | "complete" | "failed" | "cancelled";
  total: number;
  completed: number;
  failed: number;
  current_file: string | null;
}

export interface BatchFileResult {
  file_id: string;
  filename: string;
  status: "success" | "error";
  node_results: Record<string, NodeResult>;
  metrics: Record<string, unknown>;
  metadata?: Record<string, string>;
  file_info?: { n_channels: number; sfreq: number; duration_s: number };
  processing_time_s?: number;
  error: string | null;
}

export interface MetricStats {
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
}

export interface AggregateStatistics {
  overall: Record<string, MetricStats>;
  by_group: Record<string, Record<string, MetricStats>>;
}

export interface BatchResults {
  batch_id: string;
  status: string;
  file_results: BatchFileResult[];
  failed_files: Array<{ file_id: string; filename: string; error: string }>;
  metrics_csv: string;
  statistics?: AggregateStatistics;
  summary: {
    total: number;
    completed: number;
    failed: number;
    runtime_s: number;
  };
}

export interface SavedBatchSummary {
  batch_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  saved_at: number;
}

// ---------------------------------------------------------------------------
// Canvas node data — typed payload for every React Flow node
// ---------------------------------------------------------------------------

/** Data payload for every node on the React Flow canvas. */
export interface NodeData {
  descriptor: NodeTypeDescriptor;
  parameters: Record<string, unknown>;
  label: string;
  customLabel?: string;
  nodeResult?: NodeResult | null;
  sessionInfo?: SessionInfo | null;
  isRunning?: boolean;
  batchMode?: boolean;
  onFileLoad?: (file: File) => void;
  onRename?: (nodeId: string, newLabel: string) => void;
  nodeType?: string; // = descriptor.node_type, used by some serialization paths
}

/** Typed React Flow node for Oscilloom. */
export type OscilloomNode = import("@xyflow/react").Node<NodeData>;
