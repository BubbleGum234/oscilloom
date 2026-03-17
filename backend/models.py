"""
backend/models.py

Pydantic data models for the Oscilloom API.

These models define the JSON schema for all API requests and responses
involving pipeline graphs. They are the contract between frontend and backend.

The TypeScript types in frontend/src/types/pipeline.ts mirror these models
exactly. When you change a model here, update the TypeScript types too.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class PipelineNode(BaseModel):
    """
    A single node in the pipeline graph.

    `node_type` must match a key in NODE_REGISTRY. `parameters` keys must
    match the ParameterSchema.name fields for that node type. Missing
    parameter keys fall back to the schema's default values in engine.py.
    """
    id: str
    # Unique within the pipeline. E.g., "node_001".

    node_type: str
    # Foreign key into NODE_REGISTRY. E.g., "bandpass_filter".

    label: str
    # User-editable display label. E.g., "Remove Line Noise".

    parameters: dict[str, Any]
    # Parameter values. Keys must match ParameterSchema.name for this node_type.

    position: dict[str, float]
    # Canvas position for rendering. {"x": 300.0, "y": 200.0}


class PipelineEdge(BaseModel):
    """
    A directed connection between two node handles.

    source_handle_type and target_handle_type must be equal for a valid
    connection. Duplicating the type on the edge allows validation to be
    a pure data operation — no registry lookup required.
    """
    id: str

    source_node_id: str
    source_handle_id: str
    # Must match a handle ID in the source node descriptor's outputs list.
    source_handle_type: str
    # The handle type of the source. Must match target_handle_type.

    target_node_id: str
    target_handle_id: str
    # Must match a handle ID in the target node descriptor's inputs list.
    target_handle_type: str
    # Must equal source_handle_type for a valid edge.


class PipelineMetadata(BaseModel):
    name: str
    description: str
    created_by: str        # "human" | "ai"
    schema_version: str = "1.0"
    # Reserved for future migration. Not read in MVP. Always write "1.0".


class PipelineGraph(BaseModel):
    """
    A complete pipeline: metadata + list of nodes + list of edges.

    This is the universal format for:
      - Frontend → backend API calls (execution, validation, export)
      - Pipeline save/load files (.json)
    """
    metadata: PipelineMetadata
    nodes: list[PipelineNode]
    edges: list[PipelineEdge]


class ExecuteRequest(BaseModel):
    session_id: str
    pipeline: PipelineGraph


class ExecuteResponse(BaseModel):
    status: str                           # "success" | "error"
    node_results: dict[str, Any]          # node_id → result dict
    error: Optional[str] = None           # Set when status == "error"
    failed_node_id: Optional[str] = None  # Set when status == "error"


class ExportRequest(BaseModel):
    session_id: str
    pipeline: PipelineGraph
    audit_log: list[dict[str, Any]]
    # List of AuditLogEntry objects from the frontend.
    # Included as comments at the top of the generated script.


class ExportResponse(BaseModel):
    script: str       # Complete .py file content as a string
    filename: str     # Suggested download filename (e.g., "My Pipeline.py")


class ReportSections(BaseModel):
    """Toggles for which sections appear in the PDF report."""
    data_quality: bool = True
    pipeline_config: bool = True
    analysis_results: bool = True
    clinical_interpretation: bool = True
    visualizations: bool = True
    audit_trail: bool = True
    notes: bool = True


class ReportRequest(BaseModel):
    node_results: dict[str, Any]
    # node_id → result dict as returned by POST /pipeline/execute.
    # The report endpoint scans this for metrics (output_type=="dict")
    # and plot (data starts with "data:image/png;base64,") entries.

    title: str = "Oscilloom EEG Report"
    patient_id: str = ""
    clinic_name: str = ""

    # Enhanced report fields (Tier A)
    session_info: Optional[dict[str, Any]] = None
    # Session metadata (sfreq, nchan, duration_s, ch_names, bads, etc.)
    # Passed from the frontend's sessionInfo state.

    pipeline_config: Optional[list[dict[str, Any]]] = None
    # List of {node_id, node_type, label, parameters} for each pipeline node.
    # Used to document the processing steps in the report.

    audit_log: Optional[list[dict[str, Any]]] = None
    # Parameter change history entries from the frontend audit log.

    notes: str = ""
    # Free-text clinician notes to include in the report.

    sections: ReportSections = ReportSections()

    included_nodes: Optional[list[str]] = None
    # If set, only include these node IDs in the report (for both metrics and
    # visualizations).  If None (default), all successful nodes are included.


# ---------------------------------------------------------------------------
# Batch processing (Tier 4)
# ---------------------------------------------------------------------------

class BidsExportRequest(BaseModel):
    """Request body for POST /pipeline/export-bids."""
    session_id: str
    pipeline: PipelineGraph
    subject_id: str = "01"
    session: str = ""
    task: str = "rest"
    run: str = "01"
    format: str = "BrainVision"


class BatchRequest(BaseModel):
    """Request body for POST /pipeline/batch."""
    file_ids: list[str]
    # List of staged file IDs returned by POST /pipeline/batch/stage.
    pipeline: PipelineGraph
    # Files are always processed sequentially to keep peak memory at one Raw object.
