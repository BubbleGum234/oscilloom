"""
backend/registry/__init__.py

The Node Registry — the single source of truth for all available node types.

HOW TO ADD A NEW NODE TYPE:
  1. Create (or add to) a file in backend/registry/nodes/<category>.py
  2. Define the NodeDescriptor constant following existing patterns.
  3. Import it below and add it to the list in NODE_REGISTRY.
  4. Restart the server (or wait for --reload to pick up the change).
  5. Done. No other files need to change.

NODE_REGISTRY is imported by:
  - engine.py          — to look up execute_fn for each node in a pipeline
  - registry_routes.py — to serve GET /registry/nodes to the frontend
  - validation.py      — to check node types and handle types on edges
  - script_exporter.py — to know parameter names for template rendering
"""

from __future__ import annotations

from typing import Dict

from backend.registry.node_descriptor import NodeDescriptor

# ---------------------------------------------------------------------------
# Import all node descriptors
# ---------------------------------------------------------------------------
# Uncomment each import as the corresponding file is implemented.
# If a descriptor import fails, the server will fail to start — this is
# intentional. A broken descriptor must be fixed, not silently skipped.

from backend.registry.nodes.io import EDF_LOADER, SAVE_TO_FIF, BIDS_EXPORT
from backend.registry.nodes.io_extended import (
    ANT_LOADER,
    FIF_LOADER,
    BRAINVISION_LOADER,
    BDF_LOADER,
)
from backend.registry.nodes.preprocessing import (
    BANDPASS_FILTER,
    NOTCH_FILTER,
    RESAMPLE,
    SET_EEG_REFERENCE,
    ICA_DECOMPOSITION,
    MARK_BAD_CHANNELS,
    PICK_CHANNELS,
    SET_CHANNEL_TYPES,
    CROP,
    SET_MONTAGE,
    INTERPOLATE_BAD_CHANNELS,
    ANNOTATE_ARTIFACTS,
    RENAME_CHANNELS,
    DETECT_BAD_SEGMENTS,
    DETECT_FLATLINE,
    DETECT_BAD_GRADIENT,
    FILTER_ANNOTATIONS,
)
from backend.registry.nodes.epoching import (
    EPOCH_BY_EVENTS,
    BASELINE_CORRECTION,
    REJECT_EPOCHS,
    FILTER_EPOCHS,
    APPLY_AUTOREJECT,
    EPOCH_BY_TIME,
    EQUALIZE_EVENT_COUNTS,
)
from backend.registry.nodes.analysis import (
    COMPUTE_PSD,
    COMPUTE_EVOKED,
    TIME_FREQUENCY_MORLET,
    COMPUTE_BANDPOWER,
    SUMMARIZE_ANNOTATIONS,
)
from backend.registry.nodes.visualization import (
    PLOT_PSD,
    PLOT_RAW,
    PLOT_TOPOMAP,
    PLOT_EVOKED,
    PLOT_EPOCHS_IMAGE,
    PLOT_TFR,
    PLOT_EVOKED_TOPOMAP,
    PLOT_ICA_COMPONENTS,
    PLOT_EVOKED_JOINT,
)
from backend.registry.nodes.erp import (
    COMPUTE_GFP,
    PLOT_GFP,
    DETECT_ERP_PEAK,
    COMPUTE_DIFFERENCE_WAVE,
    PLOT_COMPARISON_EVOKED,
)
from backend.registry.nodes.connectivity import (
    COMPUTE_COHERENCE,
    COMPUTE_PLV,
    COMPUTE_PLI,
    COMPUTE_ENVELOPE_CORRELATION,
    PLOT_CONNECTIVITY_CIRCLE,
    PLOT_CONNECTIVITY_MATRIX,
)
from backend.registry.nodes.statistics import (
    CLUSTER_PERMUTATION_TEST,
    COMPUTE_T_TEST,
    APPLY_FDR_CORRECTION,
    COMPUTE_NOISE_FLOOR,
)
from backend.registry.nodes.clinical import (
    COMPUTE_ALPHA_PEAK,
    COMPUTE_ASYMMETRY,
    COMPUTE_BAND_RATIO,
    Z_SCORE_NORMALIZE,
    DETECT_SPIKES,
)
from backend.registry.nodes.fnirs import (
    SNIRF_LOADER,
    COMPUTE_OPTICAL_DENSITY,
    BEER_LAMBERT_TRANSFORM,
    COMPUTE_HRF,
    PLOT_FNIRS_SIGNAL,
)
from backend.registry.nodes.meg import (
    MAXWELL_FILTER,
    APPLY_SSP,
)
from backend.registry.nodes.bci import (
    COMPUTE_CSP,
    EXTRACT_EPOCH_FEATURES,
    CLASSIFY_LDA,
    PLOT_ROC_CURVE,
)
from backend.registry.nodes.sleep import (
    COMPUTE_SLEEP_STAGES,
    COMPUTE_SLEEP_ARCHITECTURE,
    DETECT_SPINDLES,
    DETECT_SLOW_OSCILLATIONS,
    PLOT_HYPNOGRAM,
)
from backend.registry.nodes.custom import CUSTOM_PYTHON

# ---------------------------------------------------------------------------
# NODE_REGISTRY
# ---------------------------------------------------------------------------
# Dict key MUST equal descriptor.node_type. The dict comprehension enforces
# this — if they diverge, the wrong descriptor will be returned for a type.

NODE_REGISTRY: Dict[str, NodeDescriptor] = {
    descriptor.node_type: descriptor
    for descriptor in [
        # I/O
        ANT_LOADER,
        EDF_LOADER,
        SAVE_TO_FIF,
        BIDS_EXPORT,
        FIF_LOADER,
        BRAINVISION_LOADER,
        BDF_LOADER,
        # Preprocessing
        BANDPASS_FILTER,
        NOTCH_FILTER,
        RESAMPLE,
        SET_EEG_REFERENCE,
        ICA_DECOMPOSITION,
        MARK_BAD_CHANNELS,
        PICK_CHANNELS,
        SET_CHANNEL_TYPES,
        CROP,
        SET_MONTAGE,
        INTERPOLATE_BAD_CHANNELS,
        ANNOTATE_ARTIFACTS,
        RENAME_CHANNELS,
        DETECT_BAD_SEGMENTS,
        DETECT_FLATLINE,
        DETECT_BAD_GRADIENT,
        FILTER_ANNOTATIONS,
        # Epoching
        EPOCH_BY_EVENTS,
        BASELINE_CORRECTION,
        REJECT_EPOCHS,
        FILTER_EPOCHS,
        APPLY_AUTOREJECT,
        EPOCH_BY_TIME,
        EQUALIZE_EVENT_COUNTS,
        # Analysis
        COMPUTE_PSD,
        COMPUTE_EVOKED,
        TIME_FREQUENCY_MORLET,
        COMPUTE_BANDPOWER,
        SUMMARIZE_ANNOTATIONS,
        COMPUTE_GFP,
        DETECT_ERP_PEAK,
        COMPUTE_DIFFERENCE_WAVE,
        # Connectivity (Tier 2)
        COMPUTE_COHERENCE,
        COMPUTE_PLV,
        COMPUTE_PLI,
        COMPUTE_ENVELOPE_CORRELATION,
        # Statistics (Tier 2)
        CLUSTER_PERMUTATION_TEST,
        COMPUTE_T_TEST,
        APPLY_FDR_CORRECTION,
        COMPUTE_NOISE_FLOOR,
        # Clinical / qEEG (Tier 3)
        COMPUTE_ALPHA_PEAK,
        COMPUTE_ASYMMETRY,
        COMPUTE_BAND_RATIO,
        Z_SCORE_NORMALIZE,
        DETECT_SPIKES,
        # Visualization
        PLOT_PSD,
        PLOT_RAW,
        PLOT_TOPOMAP,
        PLOT_EVOKED,
        PLOT_EPOCHS_IMAGE,
        PLOT_TFR,
        PLOT_EVOKED_TOPOMAP,
        PLOT_ICA_COMPONENTS,
        PLOT_EVOKED_JOINT,
        PLOT_GFP,
        PLOT_COMPARISON_EVOKED,
        PLOT_CONNECTIVITY_CIRCLE,
        PLOT_CONNECTIVITY_MATRIX,
        # fNIRS (Tier 5)
        SNIRF_LOADER,
        COMPUTE_OPTICAL_DENSITY,
        BEER_LAMBERT_TRANSFORM,
        COMPUTE_HRF,
        PLOT_FNIRS_SIGNAL,
        # MEG (Tier 5)
        MAXWELL_FILTER,
        APPLY_SSP,
        # BCI (Tier 5)
        COMPUTE_CSP,
        EXTRACT_EPOCH_FEATURES,
        CLASSIFY_LDA,
        PLOT_ROC_CURVE,
        # Sleep (Tier 6)
        COMPUTE_SLEEP_STAGES,
        COMPUTE_SLEEP_ARCHITECTURE,
        DETECT_SPINDLES,
        DETECT_SLOW_OSCILLATIONS,
        PLOT_HYPNOGRAM,
        # Custom
        CUSTOM_PYTHON,
    ]
}

# Sanity check at import time: verify no duplicate node_type values in the list.
# A duplicate would silently overwrite the first with the second.
_all_types = [d.node_type for d in NODE_REGISTRY.values()]
assert len(_all_types) == len(set(_all_types)), (
    f"Duplicate node_type values detected in NODE_REGISTRY: "
    f"{[t for t in _all_types if _all_types.count(t) > 1]}"
)
