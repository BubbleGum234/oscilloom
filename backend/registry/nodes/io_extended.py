"""
backend/registry/nodes/io_extended.py

Extended I/O node types: additional file format loaders beyond the original
EDF loader. All follow the same source-node passthrough pattern as edf_loader:
if input_data is already a BaseRaw (injected by the session or batch processor),
return it unchanged; otherwise load from the file_path parameter.

Supported formats added here:
  - FIF  (.fif, .fif.gz) — MNE's native format; output of all MNE pipelines + MEG data
  - BrainVision (.vhdr)  — Brain Products; dominant in European and many North American labs
  - BDF  (.bdf)          — BioSemi ActiveTwo; dominant in many EEG research labs

All loaders use the same hidden file_path parameter pattern as edf_loader so the
existing file-picker widget in NodeParameterPanel.tsx works without modification.
"""

from __future__ import annotations

import mne

from backend.path_security import validate_read_path
from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)


# ---------------------------------------------------------------------------
# FIF Loader
# ---------------------------------------------------------------------------

def _execute_fif_loader(input_data: mne.io.BaseRaw | None, params: dict) -> mne.io.BaseRaw:
    """
    Loads a FIF file (.fif or .fif.gz) — MNE's native binary format.

    FIF files are the output of MNE pipelines (save_to_fif node) and the
    standard format for MEG recordings from Elekta/Neuromag/MEGIN systems.
    Accepts both raw FIF and processed FIF files.

    If input_data is already a BaseRaw (injected by session or batch processor),
    returns it unchanged — this is the standard source-node passthrough pattern.
    """
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data
    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "FIF Loader requires a file path. "
            "Upload a .fif file using the file picker."
        )
    validate_read_path(file_path)
    return mne.io.read_raw_fif(file_path, preload=True, verbose=False)


FIF_LOADER = NodeDescriptor(
    node_type="fif_loader",
    display_name="Load EEG/MEG (FIF)",
    category="I/O",
    description=(
        "Loads a recording from a FIF file (.fif or .fif.gz) — MNE's native format. "
        "Use this for files previously saved by Oscilloom's 'Save to FIF' node, or for "
        "MEG recordings from Elekta/Neuromag/MEGIN systems. FIF files preserve all "
        "MNE metadata: channel positions, events, bad channels, and projectors."
    ),
    tags=["load", "input", "fif", "file", "source", "io", "meg", "mne"],
    inputs=[],
    outputs=[
        HandleSchema(id="eeg_out", type="raw_eeg", label="Raw Signal"),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path",
            type="string",
            default="",
            description="Path to the .fif or .fif.gz file to load.",
            hidden=True,
        ),
    ],
    execute_fn=_execute_fif_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_fif("{p.get("file_path", "recording.fif")}", preload=True, verbose=False)',
    methods_template=lambda p: "EEG/MEG data were loaded from FIF format using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_fif.html",
)


# ---------------------------------------------------------------------------
# BrainVision Loader
# ---------------------------------------------------------------------------

def _execute_brainvision_loader(input_data: mne.io.BaseRaw | None, params: dict) -> mne.io.BaseRaw:
    """
    Loads a BrainProducts BrainVision file (.vhdr header file).

    BrainVision format consists of three files that must be in the same directory:
      - .vhdr  — header (text, describes channels, sfreq, etc.)
      - .vmrk  — markers (events/triggers)
      - .eeg   — binary data

    The file_path must point to the .vhdr file. MNE reads all three automatically.
    This format is dominant in BrainProducts amplifier setups (actiCAP, BrainAmp, etc.)
    which are widely used in European and North American EEG labs.
    """
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data
    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "BrainVision Loader requires a .vhdr file path. "
            "Ensure the .vhdr, .vmrk, and .eeg files are all in the same directory."
        )
    validate_read_path(file_path)
    return mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)


BRAINVISION_LOADER = NodeDescriptor(
    node_type="brainvision_loader",
    display_name="Load EEG (BrainVision)",
    category="I/O",
    description=(
        "Loads a BrainProducts BrainVision recording (.vhdr format). "
        "Point to the .vhdr header file — MNE automatically reads the paired "
        ".vmrk (markers/events) and .eeg (binary data) files from the same directory. "
        "Used with BrainProducts amplifiers: BrainAmp, actiCHamp, LiveAmp, etc. "
        "Dominant format in European EEG labs."
    ),
    tags=["load", "input", "brainvision", "vhdr", "brainproducts", "file", "source", "io"],
    inputs=[],
    outputs=[
        HandleSchema(id="eeg_out", type="raw_eeg", label="Raw EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path (.vhdr)",
            type="string",
            default="",
            description="Path to the .vhdr header file. The .vmrk and .eeg files must be in the same directory.",
            hidden=True,
        ),
    ],
    execute_fn=_execute_brainvision_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_brainvision("{p.get("file_path", "recording.vhdr")}", preload=True, verbose=False)',
    methods_template=lambda p: "EEG data were loaded from BrainVision format using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_brainvision.html",
)


# ---------------------------------------------------------------------------
# BDF Loader
# ---------------------------------------------------------------------------

def _execute_bdf_loader(input_data: mne.io.BaseRaw | None, params: dict) -> mne.io.BaseRaw:
    """
    Loads a BioSemi BDF file (.bdf) — BDF (BioSemi Data Format).

    BDF is a 24-bit extension of EDF used exclusively by BioSemi ActiveTwo systems.
    BioSemi amplifiers are widely used in cognitive neuroscience and ERP research.

    BDF files often contain external (EX) channels (8 extra general-purpose inputs,
    commonly used for EOG, EMG, or reference electrodes) that are loaded as
    channel type 'misc' by MNE. Use a Pick Channels node downstream to select
    only EEG channels if needed.
    """
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data
    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "BDF Loader requires a file path. "
            "Upload a .bdf file (BioSemi ActiveTwo format)."
        )
    validate_read_path(file_path)
    return mne.io.read_raw_bdf(file_path, preload=True, verbose=False)


BDF_LOADER = NodeDescriptor(
    node_type="bdf_loader",
    display_name="Load EEG (BDF / BioSemi)",
    category="I/O",
    description=(
        "Loads a BioSemi BDF file (.bdf) — the native format of BioSemi ActiveTwo EEG systems. "
        "BDF is a 24-bit extension of EDF used by BioSemi amplifiers (common in cognitive "
        "neuroscience and ERP research). BDF files often include extra EX channels for EOG "
        "and reference electrodes — use the Pick Channels node to isolate EEG channels if needed."
    ),
    tags=["load", "input", "bdf", "biosemi", "activetwo", "file", "source", "io"],
    inputs=[],
    outputs=[
        HandleSchema(id="eeg_out", type="raw_eeg", label="Raw EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path (.bdf)",
            type="string",
            default="",
            description="Path to the .bdf BioSemi data file to load.",
            hidden=True,
        ),
    ],
    execute_fn=_execute_bdf_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_bdf("{p.get("file_path", "recording.bdf")}", preload=True, verbose=False)',
    methods_template=lambda p: "EEG data were loaded from BioSemi BDF format using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_bdf.html",
)


# ---------------------------------------------------------------------------
# ANT Neuro Loader
# ---------------------------------------------------------------------------

def _execute_ant_loader(input_data: mne.io.BaseRaw | None, params: dict) -> mne.io.BaseRaw:
    """
    Loads an ANT Neuro .cnt file — the native format of ANT Neuro (eego) systems.

    ANT Neuro's CNT format is used by eego mylab and eego sports amplifiers.
    MNE supports this format via mne.io.read_raw_ant() (added in MNE 1.7).

    If input_data is already a BaseRaw (injected by session or batch processor),
    returns it unchanged — this is the standard source-node passthrough pattern.
    """
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data
    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "ANT Neuro Loader requires a file path. "
            "Upload a .cnt file (ANT Neuro format)."
        )
    validate_read_path(file_path)
    return mne.io.read_raw_ant(file_path, preload=True, verbose=False)


ANT_LOADER = NodeDescriptor(
    node_type="ant_loader",
    display_name="ANT Neuro Loader",
    category="I/O",
    description=(
        "Loads a raw EEG recording from an ANT Neuro .cnt file. "
        "This format is the native output of ANT Neuro (eego) amplifier systems, "
        "including eego mylab and eego sports. Use this node as the starting point "
        "for pipelines processing data recorded with ANT Neuro hardware."
    ),
    tags=["ant", "neuro", "cnt", "eeg", "loader", "io", "import"],
    inputs=[],
    outputs=[
        HandleSchema(id="raw_out", type="raw_eeg", label="Raw EEG"),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path (.cnt)",
            type="string",
            default="",
            description="Path to the .cnt ANT Neuro data file to load.",
            hidden=True,
        ),
    ],
    execute_fn=_execute_ant_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_ant("{p.get("file_path", "your_file.cnt")}", preload=True, verbose=False)',
    methods_template=lambda p: "EEG data were loaded from ANT Neuro CNT format using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_ant.html",
)
