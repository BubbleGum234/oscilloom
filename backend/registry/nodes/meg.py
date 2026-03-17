"""
backend/registry/nodes/meg.py

MEG-specific node descriptors — Tier 5.

Nodes:
  maxwell_filter      — Signal Space Separation for MEG interference suppression
  apply_ssp           — Compute and apply SSP projectors for artifact removal
MEG data is loaded via fif_loader (FIF is the native MEG format) and flows
through the existing raw_eeg/filtered_eeg handle types since both MEG and
EEG data are mne.io.BaseRaw internally.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _figure_to_base64_png(fig: plt.Figure) -> str:
    """Converts a Matplotlib Figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Node 1: maxwell_filter
# ---------------------------------------------------------------------------

def _execute_maxwell_filter(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> "mne.io.BaseRaw":
    """Apply Maxwell filtering (SSS/tSSS) for MEG interference suppression."""
    raw_copy = raw.copy()

    # Guard: MEG channels required
    picks_meg = mne.pick_types(raw_copy.info, meg=True, eeg=False)
    if len(picks_meg) == 0:
        raise ValueError(
            "Maxwell Filter requires MEG channels. Your recording contains "
            f"{', '.join(sorted(set(raw_copy.get_channel_types())))} channels only. "
            "This node is for Elekta/MEGIN MEG data loaded from FIF files. "
            "For EEG artifact removal, use the ICA Decomposition node instead."
        )

    st_duration = params.get("st_duration", 0)
    st_duration = float(st_duration) if st_duration else 0
    # st_duration=0 means basic SSS; >0 enables temporal SSS (tSSS)
    kwargs = {}
    if st_duration > 0:
        kwargs["st_duration"] = st_duration

    result = mne.preprocessing.maxwell_filter(
        raw_copy,
        verbose=False,
        **kwargs,
    )
    return result


MAXWELL_FILTER = NodeDescriptor(
    node_type="maxwell_filter",
    display_name="Maxwell Filter (SSS)",
    category="Preprocessing",
    description=(
        "Applies Signal Space Separation (SSS) or temporal SSS (tSSS) to "
        "suppress external magnetic interference in MEG data. This is the "
        "standard first preprocessing step for Elekta/MEGIN MEG recordings. "
        "Requires MEG data loaded from a FIF file with device geometry info. "
        "Will raise an error if applied to pure EEG data."
    ),
    tags=["maxwell", "sss", "tsss", "meg", "interference", "preprocessing", "neuromag"],
    inputs=[
        HandleSchema(id="eeg_in", type="raw_eeg", label="Raw MEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="raw_eeg", label="Maxwell-Filtered"),
    ],
    parameters=[
        ParameterSchema(
            name="st_duration",
            label="tSSS Duration",
            type="float",
            default=0,
            min=0,
            max=60.0,
            step=1.0,
            unit="s",
            description=(
                "Duration for temporal SSS (tSSS) in seconds. Set to 0 for "
                "basic SSS only. Typical values: 10-30s. tSSS provides "
                "additional suppression of nearby interference sources."
            ),
        ),
    ],
    execute_fn=_execute_maxwell_filter,
    code_template=lambda p: f'raw = mne.preprocessing.maxwell_filter(raw.copy(){", st_duration=" + str(p["st_duration"]) if p.get("st_duration", 0) > 0 else ""}, verbose=False)',
    methods_template=lambda p: f'Signal Space Separation {"(tSSS, duration=" + str(p["st_duration"]) + "s)" if p.get("st_duration", 0) > 0 else "(SSS)"} was applied to suppress external magnetic interference (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html",
)


# ---------------------------------------------------------------------------
# Node 2: apply_ssp (merged compute_ssp + apply_ssp)
# ---------------------------------------------------------------------------

def _execute_apply_ssp(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> "mne.io.BaseRaw":
    """Compute and apply SSP projectors for artifact suppression."""
    raw_copy = raw.copy()

    # Guard: MEG channels required
    picks_meg = mne.pick_types(raw_copy.info, meg=True, eeg=False)
    if len(picks_meg) == 0:
        raise ValueError(
            "SSP Projectors require MEG channels. Your recording contains "
            f"{', '.join(sorted(set(raw_copy.get_channel_types())))} channels only. "
            "For EEG artifact removal, use the ICA Decomposition node instead."
        )

    n_eeg = int(params.get("n_eeg", 2))
    n_mag = int(params.get("n_mag", 2))
    n_grad = int(params.get("n_grad", 2))

    projs = mne.compute_proj_raw(
        raw_copy,
        n_eeg=n_eeg,
        n_mag=n_mag,
        n_grad=n_grad,
        verbose=False,
    )
    raw_copy.add_proj(projs)
    raw_copy.apply_proj(verbose=False)
    return raw_copy


APPLY_SSP = NodeDescriptor(
    node_type="apply_ssp",
    display_name="Apply SSP",
    category="Preprocessing",
    description=(
        "Computes and applies Signal Space Projection (SSP) vectors for "
        "artifact suppression. SSP identifies spatial patterns of artifacts "
        "(heartbeat, eye blinks, environmental noise) and projects them out. "
        "Works on both EEG and MEG data. The number of projectors per channel "
        "type controls how many artifact components are removed."
    ),
    tags=["ssp", "projection", "artifact", "meg", "eeg", "preprocessing"],
    inputs=[
        HandleSchema(id="eeg_in", type="filtered_eeg", label="Filtered EEG/MEG"),
    ],
    outputs=[
        HandleSchema(id="eeg_out", type="filtered_eeg", label="SSP-Cleaned"),
    ],
    parameters=[
        ParameterSchema(
            name="n_eeg",
            label="EEG Projectors",
            type="int",
            default=2,
            min=0,
            max=10,
            description="Number of SSP projectors for EEG channels.",
        ),
        ParameterSchema(
            name="n_mag",
            label="Magnetometer Projectors",
            type="int",
            default=2,
            min=0,
            max=10,
            description="Number of SSP projectors for MEG magnetometer channels.",
        ),
        ParameterSchema(
            name="n_grad",
            label="Gradiometer Projectors",
            type="int",
            default=2,
            min=0,
            max=10,
            description="Number of SSP projectors for MEG gradiometer channels.",
        ),
    ],
    execute_fn=_execute_apply_ssp,
    code_template=lambda p: f'projs = mne.compute_proj_raw(raw.copy(), n_eeg={p.get("n_eeg", 2)}, n_mag={p.get("n_mag", 2)}, n_grad={p.get("n_grad", 2)}, verbose=False)\nraw.add_proj(projs)\nraw.apply_proj(verbose=False)',
    methods_template=lambda p: f'Signal Space Projection (SSP) was applied with {p.get("n_eeg", 2)} EEG, {p.get("n_mag", 2)} magnetometer, and {p.get("n_grad", 2)} gradiometer projectors to suppress artifacts (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.compute_proj_raw.html",
)
