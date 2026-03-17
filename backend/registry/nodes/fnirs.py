"""
backend/registry/nodes/fnirs.py

fNIRS (functional Near-Infrared Spectroscopy) node descriptors — Tier 5.

Pipeline flow:
  snirf_loader → compute_optical_density → beer_lambert_transform → compute_hrf
                                                                  → plot_fnirs_signal

All fNIRS preprocessing uses MNE's built-in mne.preprocessing.nirs module.
No additional dependencies required beyond MNE 1.6+.
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
# Shared utility (duplicated from visualization.py to avoid circular imports)
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
# Node 1: snirf_loader
# ---------------------------------------------------------------------------

def _execute_snirf_loader(
    input_data: "mne.io.BaseRaw | None",
    params: dict,
) -> "mne.io.BaseRaw":
    """Load a SNIRF file. Passes through if input_data is already a BaseRaw."""
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data

    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "SNIRF Loader requires a file. Use the Browse button or upload "
            "a .snirf file to load fNIRS data."
        )

    raw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
    return raw


SNIRF_LOADER = NodeDescriptor(
    node_type="snirf_loader",
    display_name="SNIRF Loader",
    category="I/O",
    description=(
        "Loads fNIRS data from a SNIRF (.snirf) file. SNIRF is the standard "
        "format for fNIRS data, containing raw optical intensity measurements "
        "from source-detector pairs at multiple wavelengths. Output flows into "
        "Compute Optical Density as the first preprocessing step."
    ),
    tags=["load", "snirf", "fnirs", "nirs", "io", "source", "optical"],
    inputs=[],
    outputs=[
        HandleSchema(id="fnirs_out", type="raw_fnirs", label="fNIRS Raw"),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path",
            type="string",
            default="",
            description="Path to the .snirf file (set automatically by file picker).",
            hidden=True,
        ),
    ],
    execute_fn=_execute_snirf_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_snirf("{p.get("file_path", "recording.snirf")}", preload=True, verbose=False)',
    methods_template=lambda p: "fNIRS data were loaded from SNIRF format using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_snirf.html",
)


# ---------------------------------------------------------------------------
# Node 2: compute_optical_density
# ---------------------------------------------------------------------------

def _execute_compute_optical_density(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> "mne.io.BaseRaw":
    """Convert raw fNIRS intensity to optical density (OD)."""
    from mne.preprocessing.nirs import optical_density

    # Guard: fNIRS channels required
    fnirs_picks = mne.pick_types(raw.info, fnirs=True)
    if len(fnirs_picks) == 0:
        raise ValueError(
            "Optical Density conversion requires fNIRS channels. "
            "Your recording has no fNIRS channels. "
            "Load a .snirf file using the SNIRF Loader node first."
        )

    raw_copy = raw.copy()
    return optical_density(raw_copy, verbose=False)


COMPUTE_OPTICAL_DENSITY = NodeDescriptor(
    node_type="compute_optical_density",
    display_name="Optical Density",
    category="fNIRS",
    description=(
        "Converts raw fNIRS intensity measurements to optical density (OD) "
        "using the modified Beer-Lambert law: OD = -log(I / I0). This is the "
        "mandatory first preprocessing step for fNIRS data. Input must be raw "
        "fNIRS intensity data (from SNIRF Loader). Output feeds into "
        "Beer-Lambert Transform."
    ),
    tags=["optical-density", "fnirs", "nirs", "preprocessing", "od"],
    inputs=[
        HandleSchema(id="fnirs_in", type="raw_fnirs", label="fNIRS Raw"),
    ],
    outputs=[
        HandleSchema(id="fnirs_out", type="raw_fnirs", label="Optical Density"),
    ],
    parameters=[],
    execute_fn=_execute_compute_optical_density,
    code_template=lambda p: 'from mne.preprocessing.nirs import optical_density\nraw_od = optical_density(raw.copy(), verbose=False)',
    methods_template=lambda p: "Raw fNIRS intensity data were converted to optical density using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.preprocessing.nirs.optical_density.html",
)


# ---------------------------------------------------------------------------
# Node 3: beer_lambert_transform
# ---------------------------------------------------------------------------

def _execute_beer_lambert_transform(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> "mne.io.BaseRaw":
    """Convert optical density to hemoglobin concentration (HbO/HbR)."""
    from mne.preprocessing.nirs import beer_lambert_law

    # Guard: fNIRS channels required
    fnirs_picks = mne.pick_types(raw.info, fnirs=True)
    if len(fnirs_picks) == 0:
        raise ValueError(
            "Beer-Lambert Transform requires fNIRS optical density channels. "
            "Your recording has no fNIRS channels. "
            "Connect this node after the Optical Density node."
        )

    raw_copy = raw.copy()
    ppf = float(params.get("ppf", 6.0))
    return beer_lambert_law(raw_copy, ppf=ppf)


BEER_LAMBERT_TRANSFORM = NodeDescriptor(
    node_type="beer_lambert_transform",
    display_name="Beer-Lambert Transform",
    category="fNIRS",
    description=(
        "Applies the modified Beer-Lambert Law to convert optical density "
        "into oxygenated (HbO) and deoxygenated (HbR) hemoglobin "
        "concentration changes. The partial pathlength factor (PPF) accounts "
        "for photon scattering in tissue. Input must be optical density data "
        "(from Optical Density node). Output is ready for HRF analysis or "
        "visualization."
    ),
    tags=["beer-lambert", "hbo", "hbr", "hemoglobin", "fnirs", "nirs", "concentration"],
    inputs=[
        HandleSchema(id="fnirs_in", type="raw_fnirs", label="Optical Density"),
    ],
    outputs=[
        HandleSchema(id="fnirs_out", type="raw_fnirs", label="HbO/HbR"),
    ],
    parameters=[
        ParameterSchema(
            name="ppf",
            label="Partial Pathlength Factor",
            type="float",
            default=6.0,
            min=0.1,
            max=20.0,
            step=0.1,
            description=(
                "Partial pathlength factor for the Beer-Lambert law. "
                "Accounts for photon scattering in tissue. Typical values: "
                "0.1 for adults, varies by age and wavelength."
            ),
        ),
    ],
    execute_fn=_execute_beer_lambert_transform,
    code_template=lambda p: f'from mne.preprocessing.nirs import beer_lambert_law\nraw_hb = beer_lambert_law(raw_od.copy(), ppf={p.get("ppf", 6.0)})',
    methods_template=lambda p: f'Optical density was converted to oxygenated (HbO) and deoxygenated (HbR) hemoglobin concentrations using the modified Beer-Lambert Law with a partial pathlength factor of {p.get("ppf", 6.0)} (MNE-Python; Gramfort et al., 2013).',
    docs_url="https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html",
)


# ---------------------------------------------------------------------------
# Node 4: compute_hrf
# ---------------------------------------------------------------------------

def _execute_compute_hrf(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> "mne.Evoked":
    """Epoch fNIRS data around events and average to estimate the HRF."""
    raw_copy = raw.copy()
    tmin = float(params.get("tmin", -5.0))
    tmax = float(params.get("tmax", 15.0))

    events, event_id = mne.events_from_annotations(raw_copy, verbose=False)
    if len(events) == 0:
        raise ValueError(
            "No events or annotations found in the fNIRS data. "
            "compute_hrf requires stimulus event markers to epoch around. "
            "Add annotations to your data or use the Annotate Artifacts node."
        )

    epochs = mne.Epochs(
        raw_copy, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=(None, 0),
        preload=True, verbose=False,
    )

    return epochs.average()


COMPUTE_HRF = NodeDescriptor(
    node_type="compute_hrf",
    display_name="Compute HRF",
    category="fNIRS",
    description=(
        "Computes the Hemodynamic Response Function (HRF) by epoching fNIRS "
        "data around stimulus events and averaging across trials. The result "
        "is an Evoked object showing the mean HbO/HbR response time-locked "
        "to stimuli. This is the fNIRS equivalent of an ERP. Requires event "
        "annotations in the data."
    ),
    tags=["hrf", "hemodynamic", "fnirs", "nirs", "evoked", "analysis", "epoch"],
    inputs=[
        HandleSchema(id="fnirs_in", type="raw_fnirs", label="fNIRS (HbO/HbR)"),
    ],
    outputs=[
        HandleSchema(id="evoked_out", type="evoked", label="HRF (Evoked)"),
    ],
    parameters=[
        ParameterSchema(
            name="tmin",
            label="Epoch Start",
            type="float",
            default=-5.0,
            min=-30.0,
            max=0.0,
            step=0.5,
            unit="s",
            description="Start time relative to event onset (negative = before event).",
        ),
        ParameterSchema(
            name="tmax",
            label="Epoch End",
            type="float",
            default=15.0,
            min=1.0,
            max=60.0,
            step=0.5,
            unit="s",
            description=(
                "End time relative to event onset. The HRF typically peaks "
                "at ~6s and returns to baseline by ~15s."
            ),
        ),
    ],
    execute_fn=_execute_compute_hrf,
    code_template=lambda p: f'events, event_id = mne.events_from_annotations(raw, verbose=False)\nepochs = mne.Epochs(raw, events, event_id=event_id, tmin={p.get("tmin", -5.0)}, tmax={p.get("tmax", 15.0)}, baseline=(None, 0), preload=True, verbose=False)\nhrf = epochs.average()',
    methods_template=lambda p: f'The hemodynamic response function (HRF) was estimated by epoching fNIRS data from {p.get("tmin", -5.0)} to {p.get("tmax", 15.0)} s relative to stimulus onset and averaging across trials.',
    docs_url="https://mne.tools/stable/auto_tutorials/fnirs/index.html",
)


# ---------------------------------------------------------------------------
# Node 5: plot_fnirs_signal
# ---------------------------------------------------------------------------

def _execute_plot_fnirs_signal(
    raw: "mne.io.BaseRaw",
    params: dict,
) -> str:
    """Plot fNIRS HbO/HbR channel timecourses."""
    duration = float(params.get("duration_s", 30.0))
    n_channels = int(params.get("n_channels", 10))

    fig = raw.plot(
        duration=duration,
        n_channels=n_channels,
        show=False,
        scalings="auto",
        verbose=False,
    )
    return _figure_to_base64_png(fig)


PLOT_FNIRS_SIGNAL = NodeDescriptor(
    node_type="plot_fnirs_signal",
    display_name="Plot fNIRS Signal",
    category="Visualization",
    description=(
        "Plots fNIRS channel timecourses showing HbO (oxygenated) and HbR "
        "(deoxygenated) hemoglobin concentration changes over time. Useful "
        "for visual inspection of signal quality, motion artifacts, and "
        "stimulus-evoked responses."
    ),
    tags=["fnirs", "nirs", "hbo", "hbr", "plot", "visualization", "timecourse"],
    inputs=[
        HandleSchema(id="fnirs_in", type="raw_fnirs", label="fNIRS Data"),
    ],
    outputs=[
        HandleSchema(id="plot_out", type="plot", label="fNIRS Plot"),
    ],
    parameters=[
        ParameterSchema(
            name="duration_s",
            label="Duration",
            type="float",
            default=30.0,
            min=1.0,
            max=300.0,
            step=1.0,
            unit="s",
            description="Duration of signal to display in the plot.",
        ),
        ParameterSchema(
            name="n_channels",
            label="Channels to Show",
            type="int",
            default=10,
            min=1,
            max=50,
            description="Number of channels to display simultaneously.",
        ),
    ],
    execute_fn=_execute_plot_fnirs_signal,
    code_template=lambda p: f'fig = raw.plot(duration={p.get("duration_s", 30.0)}, n_channels={p.get("n_channels", 10)}, show=False, scalings="auto")',
    methods_template=None,
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot",
)
