"""
backend/registry/nodes/io.py

I/O node types: loading EEG files from disk into memory.

To add a new loader (e.g., FIF, BDF):
  1. Write _execute_<format>_loader following the EDF pattern below.
  2. Define the NodeDescriptor constant.
  3. Import it in backend/registry/__init__.py and add to NODE_REGISTRY.
"""

from __future__ import annotations

import mne

from backend.path_security import validate_read_path, validate_write_path
from backend.registry.node_descriptor import (
    HandleSchema,
    NodeDescriptor,
    ParameterSchema,
)

# Lazy import for mne_bids — only needed when BIDS export actually writes.
# Keeps server startup fast and avoids hard failure if mne-bids is not installed
# in a dev environment that doesn't need the BIDS node.



# ---------------------------------------------------------------------------
# EDF Loader
# ---------------------------------------------------------------------------

def _execute_edf_loader(input_data: mne.io.BaseRaw | None, params: dict) -> mne.io.BaseRaw:
    """
    Loads an EDF file from disk into an MNE Raw object.

    For source nodes the engine passes raw_copy as input_data (the initial
    Raw object from session_store). If input_data is already a BaseRaw,
    it is returned directly — this is the case both in production (where
    session_store always pre-loads the file) and in tests (where a Raw is
    injected directly). If input_data is None, the file_path parameter is
    used to load from disk (legacy / standalone-script path).

    Raises:
        ValueError: if input_data is None and file_path is empty.
        FileNotFoundError: if the file does not exist at the given path.
        RuntimeError: if MNE cannot parse the file format.
    """
    # If the engine already resolved a Raw (from session_store), pass it through.
    if isinstance(input_data, mne.io.BaseRaw):
        return input_data

    file_path: str = params.get("file_path", "")
    if not file_path:
        raise ValueError(
            "The EDF Loader node requires a file path. "
            "Use the file picker button inside the node to select a .edf file."
        )
    validate_read_path(file_path)

    raw = mne.io.read_raw_edf(
        input_fname=file_path,
        preload=True,    # Always preload — see ARCHITECTURE.md Decision 4
        verbose=False,   # Suppress MNE's verbose output in API context
    )
    return raw


EDF_LOADER = NodeDescriptor(
    node_type="edf_loader",
    display_name="Load EEG (EDF)",
    category="I/O",
    description=(
        "Loads a raw EEG recording from an EDF (European Data Format) file. "
        "This is the standard starting node for any pipeline. The file is read "
        "entirely into memory, so ensure sufficient RAM for large recordings "
        "(typically 50–500 MB for single-session clinical EEG). "
        "Supports both EDF and EDF+ formats."
    ),
    tags=["load", "input", "edf", "file", "source", "io"],
    inputs=[],  # Source node: no upstream connections
    outputs=[
        HandleSchema(
            id="eeg_out",
            type="raw_eeg",
            label="Raw EEG",
        ),
    ],
    parameters=[
        ParameterSchema(
            name="file_path",
            label="File Path",
            type="string",
            default="",
            description=(
                "Absolute path to the EEG file on your local machine. "
                "This is set automatically when you use the file picker "
                "button inside the node. Do not edit this field manually."
            ),
            hidden=True,
        ),
    ],
    execute_fn=_execute_edf_loader,
    code_template=lambda p: f'raw = mne.io.read_raw_edf("{p.get("file_path", "your_file.edf")}", preload=True, verbose=False)',
    methods_template=lambda p: "EEG data were loaded from European Data Format (EDF) files using MNE-Python (Gramfort et al., 2013).",
    docs_url="https://mne.tools/stable/generated/mne.io.read_raw_edf.html",
)


# ---------------------------------------------------------------------------
# Save to FIF
# ---------------------------------------------------------------------------

def _execute_save_to_fif(raw: mne.io.BaseRaw, params: dict) -> mne.io.BaseRaw:
    """
    Saves the processed EEG recording to a .fif file on disk and returns
    the raw object unchanged (passthrough).

    When output_path is non-empty, saves to that path (overwriting any existing
    file). When output_path is blank, the node is a pure passthrough — use the
    toolbar Download .fif button to export processed data via the browser.

    Raises:
        OSError: if the directory does not exist or is not writable.
    """
    output_path = str(params.get("output_path", "")).strip()
    if output_path:
        validate_write_path(output_path, allowed_extensions=[".fif"])
        raw.save(output_path, overwrite=True, verbose=False)
    return raw


SAVE_TO_FIF = NodeDescriptor(
    node_type="save_to_fif",
    display_name="Save to FIF",
    category="I/O",
    description=(
        "Saves the processed EEG data to a .fif file (MNE's native format). "
        "FIF files preserve all processing metadata — channel positions, filter "
        "history, event markers, and annotations — making them ideal for "
        "handoff to other tools (EEGLAB via eeglab2fiff, FieldTrip, or custom "
        "Python scripts). "
        "This node passes data through unchanged, so you can continue connecting "
        "downstream analysis nodes after saving."
    ),
    tags=["save", "export", "fif", "output", "io", "file"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG",      required=False),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG", required=False),
    ],
    outputs=[
        HandleSchema(id="raw_out",      type="raw_eeg",      label="Raw EEG (passthrough)"),
        HandleSchema(id="filtered_out", type="filtered_eeg", label="Filtered EEG (passthrough)"),
    ],
    parameters=[
        ParameterSchema(
            name="output_path",
            label="Output Path",
            type="string",
            default="",
            description=(
                "Absolute path for the output .fif file. "
                "Example: /home/user/data/S049_cleaned.fif. "
                "The file will be created or overwritten (parent directory must exist). "
                "Leave blank to skip saving to disk — use the Download .fif button "
                "in the node panel to export the processed data via your browser."
            ),
        ),
    ],
    execute_fn=_execute_save_to_fif,
    code_template=lambda p: f'raw.save("{p.get("output_path", "output.fif")}", overwrite=True, verbose=False)',
    methods_template=lambda p: "Processed EEG data were saved in FIF format using MNE-Python (Gramfort et al., 2013) to preserve full processing provenance.",
    docs_url="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.save",
)


# ---------------------------------------------------------------------------
# BIDS Export
# ---------------------------------------------------------------------------

def _execute_bids_export(input_data, params: dict):
    """
    Export EEG data in BIDS format. Passthrough — returns raw unchanged.

    When output_dir is non-empty, writes the data into a BIDS-compliant
    directory structure using mne-bids. When output_dir is blank, the node
    acts as a pure passthrough (no files written).
    """
    raw = input_data.copy()

    output_dir = str(params.get("output_dir", "")).strip()
    if not output_dir:
        # No output dir = passthrough only (actual export happens via endpoint)
        return raw

    validate_write_path(output_dir, allowed_extensions=None)

    import mne_bids

    session = str(params.get("session_id", "")).strip() or None
    run = str(params.get("run", "")).strip() or None

    bids_path = mne_bids.BIDSPath(
        subject=str(params.get("subject_id", "01")),
        session=session,
        task=str(params.get("task", "rest")),
        run=run,
        datatype="eeg",
        root=output_dir,
    )

    mne_bids.write_raw_bids(
        raw,
        bids_path,
        overwrite=True,
        allow_preload=True,
        verbose=False,
        format=str(params.get("format", "BrainVision")),
    )

    return raw


BIDS_EXPORT = NodeDescriptor(
    node_type="bids_export",
    display_name="Export to BIDS",
    category="I/O",
    description=(
        "Exports EEG data in Brain Imaging Data Structure (BIDS) format, the "
        "community standard for organizing neuroimaging datasets. BIDS creates a "
        "standardized directory structure with JSON sidecar files that capture "
        "recording metadata (channel types, sampling rate, electrode positions, task "
        "description), making your data immediately shareable, reproducible, and "
        "compatible with BIDS-Apps analysis pipelines."
    ),
    tags=["bids", "export", "save", "standardize", "metadata"],
    inputs=[
        HandleSchema(id="raw_in",      type="raw_eeg",      label="Raw EEG",      required=False),
        HandleSchema(id="filtered_in", type="filtered_eeg", label="Filtered EEG", required=False),
    ],
    outputs=[
        HandleSchema(id="raw_out",      type="raw_eeg",      label="Raw (passthrough)"),
        HandleSchema(id="filtered_out", type="filtered_eeg", label="Filtered (passthrough)"),
    ],
    parameters=[
        ParameterSchema(
            name="output_dir",
            label="Output Directory",
            type="string",
            default="",
            description=(
                "Absolute path to the root directory where the BIDS dataset will be "
                "created. Example: /home/user/my_bids_dataset. The directory will be "
                "created if it does not exist. Leave blank to skip writing to disk."
            ),
        ),
        ParameterSchema(
            name="subject_id",
            label="Subject ID",
            type="string",
            default="01",
            description=(
                "Subject identifier used in the BIDS file naming convention "
                "(e.g., '01' becomes sub-01). Must contain only alphanumeric characters."
            ),
        ),
        ParameterSchema(
            name="session_id",
            label="Session ID",
            type="string",
            default="",
            description=(
                "Optional session identifier (e.g., '01' becomes ses-01). "
                "Leave blank if the study has only one session per subject."
            ),
        ),
        ParameterSchema(
            name="task",
            label="Task Name",
            type="string",
            default="rest",
            description=(
                "Name of the experimental task (e.g., 'rest', 'oddball', 'n-back'). "
                "Used in the BIDS filename (task-rest). Must contain only letters and numbers."
            ),
        ),
        ParameterSchema(
            name="run",
            label="Run Number",
            type="string",
            default="01",
            description=(
                "Run number for repeated recordings of the same task and session "
                "(e.g., '01' becomes run-01). Leave blank if there is only one run."
            ),
        ),
        ParameterSchema(
            name="format",
            label="Output Format",
            type="select",
            default="BrainVision",
            options=["BrainVision", "EDF"],
            description=(
                "File format for the exported EEG data within the BIDS directory. "
                "BrainVision (.vhdr/.vmrk/.eeg) is the recommended BIDS-EEG format. "
                "EDF is an alternative if your downstream tools require it."
            ),
        ),
    ],
    execute_fn=_execute_bids_export,
    code_template=lambda p: (
        f'import mne_bids\n'
        f'bids_path = mne_bids.BIDSPath(\n'
        f'    subject="{p.get("subject_id", "01")}",\n'
        f'    session={repr(p.get("session_id", "") or None)},\n'
        f'    task="{p.get("task", "rest")}",\n'
        f'    run={repr(p.get("run", "01") or None)},\n'
        f'    datatype="eeg",\n'
        f'    root="{p.get("output_dir", "bids_dataset")}",\n'
        f')\n'
        f'mne_bids.write_raw_bids(raw, bids_path, overwrite=True, verbose=False, '
        f'format="{p.get("format", "BrainVision")}")'
    ),
    methods_template=lambda p: (
        "EEG data were exported in Brain Imaging Data Structure (BIDS) format "
        "(Pernet et al., 2019) using MNE-BIDS (Appelhoff et al., 2019)."
    ),
    docs_url="https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html",
)
