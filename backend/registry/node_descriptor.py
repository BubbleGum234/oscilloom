"""
backend/registry/node_descriptor.py

Core data model for the Oscilloom node registry.

Every node type in Oscilloom is fully described by a NodeDescriptor instance.
This is the single source of truth used by:
  - engine.py      — calls execute_fn to run the node
  - registry_routes.py — serializes to JSON for the frontend palette
  - validation.py  — checks handle type compatibility on edges
  - script_exporter.py — knows parameter names for template rendering

To add a new node type:
  1. Create a NodeDescriptor in backend/registry/nodes/<category>.py
  2. Add one import + one list entry in backend/registry/__init__.py
  3. Restart the server. No other files change.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal, Optional


# ---------------------------------------------------------------------------
# Handle Types
# ---------------------------------------------------------------------------
# The "type system" for node connections. A connection (edge) is valid only
# when source_handle_type == target_handle_type.
#
# Add new types here when a new category of node produces a genuinely
# different data shape. Do NOT create sub-types for minor variations
# (e.g., don't create "notch_filtered_eeg" — use "filtered_eeg").
# ---------------------------------------------------------------------------

HandleType = Literal[
    "raw_eeg",       # mne.io.BaseRaw — continuous, unprocessed EEG signal
    "filtered_eeg",  # mne.io.BaseRaw — continuous, filtered/conditioned signal
    "epochs",        # mne.Epochs     — EEG segmented into trials
    "evoked",        # mne.Evoked     — averaged epochs (ERP)
    "psd",           # mne.time_frequency.Spectrum — power spectral density
    "tfr",           # mne.time_frequency.AverageTFR — time-frequency representation
    "plot",          # str (base64 PNG data URI) — terminal; no downstream
    "array",         # numpy.ndarray  — generic numerical array
    "scalar",        # float          — single numerical value
    "metrics",       # dict[str, Any] — named clinical/analysis metrics (e.g. ERP peaks,
                     #   band powers, asymmetry index). Multiple values in one edge,
                     #   avoiding the need for many scalar outputs per analysis node.
    "connectivity",  # mne_connectivity.SpectralConnectivity — frequency-resolved
                     #   pairwise connectivity matrix (coherence, PLV, PLI, etc.)
                     #   Added in Tier 2.
    "raw_fnirs",     # mne.io.BaseRaw with fNIRS channels (optical density, HbO, HbR).
                     #   Added in Tier 5.
    "features",      # dict with "X" (list), "labels" (list), "label_names" (dict).
                     #   Carries feature matrix + class labels for BCI classification.
                     #   Added in Tier 5.
]

# Set of valid handle types for runtime validation checks
VALID_HANDLE_TYPES: frozenset[str] = frozenset(
    [
        "raw_eeg", "filtered_eeg", "epochs", "evoked", "psd", "tfr",
        "plot", "array", "scalar", "metrics",
        "connectivity",   # Tier 2
        "raw_fnirs",      # Tier 5
        "features",       # Tier 5
    ]
)


# ---------------------------------------------------------------------------
# ParameterSchema
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ParameterSchema:
    """
    Describes one configurable parameter on a node.

    The frontend uses this to auto-render the correct UI control.
    The AI system prompt uses `description` and `default` to generate values.

    All fields must be filled in — they are all consumed by at least one
    downstream system. Never add a parameter with an empty description.
    """

    name: str
    # Internal key. Used as the dict key in pipeline JSON `parameters`.
    # Use snake_case. Must be unique within a node type.
    # Once set and a pipeline has been saved, do NOT rename — it breaks
    # saved pipeline files.

    label: str
    # Human-readable. Shown in the parameter panel. Title Case.
    # Include unit in parentheses if applicable: "High Cutoff (Hz)".

    type: Literal["float", "int", "bool", "string", "select"]
    # Controls which UI widget is rendered:
    #   float/int  → number input with min/max/step
    #   bool       → checkbox
    #   string     → text input
    #   select     → dropdown (options must be provided)

    default: Any
    # Must match `type`. Used as fallback when a parameter is missing
    # from a pipeline JSON (e.g., old saved file missing a new param).

    description: str
    # Plain English explanation. Included in the AI system prompt and
    # shown as a tooltip in the UI. Write for a non-programmer researcher.

    min: Optional[float] = None
    # For float/int only. Minimum allowed value. Enforced client-side.

    max: Optional[float] = None
    # For float/int only. Maximum allowed value. Enforced client-side.

    step: Optional[float] = None
    # For float/int only. Step size for the number input widget.

    options: Optional[list[str]] = None
    # For select type only. List of allowed string values.

    unit: Optional[str] = None
    # Displayed next to the value in the UI. E.g., "Hz", "s", "samples".
    # Do not include the unit in `label` if it is provided here.

    hidden: bool = False
    # When True, this parameter is excluded from the UI parameter panel.
    # Use for internal/computed values (e.g., file_path set by the file picker)
    # that should not be hand-edited but must still be serialized in the pipeline JSON.

    exposed: bool = False
    # When True, this parameter is shown to end-users in simplified Workflow App /
    # compound node interfaces. When False (default), the parameter uses the locked
    # default set by the workflow creator. Ignored in full Canvas mode where all
    # non-hidden params are always visible.

    channel_hint: Optional[Literal["single", "multi"]] = None
    # When set, the frontend renders a channel-name dropdown instead of a plain
    # text input, populated from the loaded file's channel list (sessionInfo).
    #   "single" — one channel (e.g., ERP peak detection channel)
    #   "multi"  — comma-separated multi-select (e.g., bad channels list)


# ---------------------------------------------------------------------------
# HandleSchema
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HandleSchema:
    """
    Describes one input or output port (handle) on a node.

    `type` is used for edge validation. A connection is valid only when
    the source output type equals the target input type.
    """

    id: str
    # Unique within the node. Used in pipeline JSON edge definitions.
    # Convention: inputs end in "_in", outputs end in "_out".
    # E.g., "eeg_in", "psd_out".

    type: HandleType
    # The data type flowing through this handle. Must be a valid HandleType.

    label: str
    # Short human-readable name. Shown as a tooltip on the handle in the UI.

    required: bool = True
    # If True, the node cannot execute without data arriving at this handle.
    # Source nodes (loaders) have no inputs so this field is irrelevant for them.


# ---------------------------------------------------------------------------
# NodeDescriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NodeDescriptor:
    """
    Complete, self-contained description of one node type.

    This is the single source of truth. Every system that needs to know
    anything about a node type reads it from here.

    SERIALIZATION NOTE:
    `execute_fn` is a Python callable and is intentionally excluded from
    JSON serialization in registry_routes.py. Never attempt to serialize it.
    The serializer calls dataclasses.asdict(descriptor) and then pops
    "execute_fn" before returning the JSON response.
    """

    node_type: str
    # Unique identifier. snake_case, all lowercase.
    # PERMANENT: do not rename after any pipeline file references this type.
    # Renaming breaks saved pipeline JSON files without a migration step.

    display_name: str
    # Human-readable. Shown in the node palette and node header. Title Case.

    category: str
    # Groups nodes in the sidebar palette. Use existing categories:
    #   "I/O", "Preprocessing", "Epoching", "Analysis", "Visualization", "Utility"
    # Create a new category only if none of the above fits.

    description: str
    # Full plain-English description. Included in the AI system prompt.
    # Write for a non-programmer researcher. Explain what the node does
    # and when to use it. At least 2 sentences.

    inputs: list[HandleSchema]
    # Input handles (left side of the node). Empty list for source nodes (loaders).

    outputs: list[HandleSchema]
    # Output handles (right side of the node). Empty list for terminal nodes.

    parameters: list[ParameterSchema]
    # Configurable parameters. Empty list if the node has no user-facing settings.

    execute_fn: Callable[[Any, dict[str, Any]], Any]
    # The function that implements this node's computation.
    #
    # Signature: fn(input_data: Any, params: dict[str, Any]) -> Any
    #
    # CONTRACT:
    #   - input_data: the runtime output of the upstream node. None for source nodes.
    #   - params: dict with parameter values. Keys match ParameterSchema.name.
    #             Missing keys have already been filled with schema defaults
    #             by engine.py before this function is called.
    #   - Returns: a NEW object. NEVER mutate input_data.
    #   - Must be synchronous (runs in ThreadPoolExecutor, not async).
    #   - Must not suppress exceptions — let them propagate to engine.py.
    #   - Must use verbose=False for all MNE calls.

    tags: list[str] = dataclasses.field(default_factory=list)
    # Lowercase keywords used by the AI for semantic retrieval.
    # E.g., ["filter", "frequency", "60hz"]. No spaces; use hyphens if needed.

    code_template: Callable[[dict[str, Any]], str] | None = None
    # Optional function that returns the MNE-Python code this node executes.
    # Signature: fn(params: dict) -> str
    # Used by the script exporter and the frontend "View Code" feature.
    # The generated code should be valid Python that a researcher can run
    # standalone in a Jupyter notebook or .py script.

    methods_template: Callable[[dict[str, Any]], str] | None = None
    # Optional function that returns one sentence for a journal Methods section.
    # Signature: fn(params: dict) -> str
    # Used by the "Generate Methods" feature. Write in academic prose,
    # past tense, citing MNE-Python (Gramfort et al., 2013) where appropriate.

    docs_url: str | None = None
    # Optional URL to the relevant MNE-Python documentation page.
    # Shown in the frontend code viewer as a clickable link.
