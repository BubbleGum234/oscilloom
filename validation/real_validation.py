"""
REAL VALIDATION — Three-Way Comparison
=======================================

Path A: Oscilloom Engine (the actual DAG executor, session store, execute_fn chain)
Path B: Oscilloom Export (the code_template-generated script, run independently)
Path C: Independent MNE (hand-written, no Oscilloom code)

This test exercises the REAL code paths, not copy-pasted MNE calls.
"""

import os
import sys
import json
import tempfile
import textwrap
import subprocess

import numpy as np
import mne

# ---------------------------------------------------------------------------
# Add project root to path so we can import backend modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

from backend.engine import execute_pipeline, topological_sort
from backend.models import PipelineGraph, PipelineNode, PipelineEdge, PipelineMetadata
from backend.registry import NODE_REGISTRY
from backend.script_exporter import export as export_script
from backend.session_store import create_session, get_raw_copy

EDF_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "sample_data", "eegmmidb", "S001", "S001R01.edf",
))
OUT_DIR = os.path.join(os.path.dirname(__file__), "real_validation_output")
os.makedirs(OUT_DIR, exist_ok=True)

POS = {"x": 0.0, "y": 0.0}  # Dummy canvas position

# ======================================================================
# Build Pipeline B as a PipelineGraph (exactly what the frontend sends)
# ======================================================================

pipeline = PipelineGraph(
    metadata=PipelineMetadata(
        name="Pipeline B Validation",
        description="Load → Notch 60Hz → Bandpass 1-40Hz → Resample 128Hz → ICA → PSD → Topomap",
        created_by="human",
    ),
    nodes=[
        PipelineNode(
            id="n1", node_type="edf_loader", label="Load EEG",
            parameters={"file_path": EDF_PATH}, position=POS,
        ),
        PipelineNode(
            id="n2", node_type="notch_filter", label="Notch 60Hz",
            parameters={"notch_freq_hz": 60.0}, position=POS,
        ),
        PipelineNode(
            id="n3", node_type="bandpass_filter", label="Bandpass 1-40Hz",
            parameters={"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"},
            position=POS,
        ),
        PipelineNode(
            id="n4", node_type="resample", label="Resample 128Hz",
            parameters={"target_sfreq": 128.0}, position=POS,
        ),
        PipelineNode(
            id="n5", node_type="ica_decomposition", label="ICA",
            parameters={"n_components": 20, "method": "fastica", "exclude_components": ""},
            position=POS,
        ),
        PipelineNode(
            id="n6", node_type="compute_psd", label="Compute PSD",
            parameters={"method": "welch", "fmin": 0.5, "fmax": 60.0, "n_fft": 2048, "n_overlap": 1024},
            position=POS,
        ),
        PipelineNode(
            id="n7", node_type="plot_topomap", label="Topomap",
            parameters={"bands": "alpha"}, position=POS,
        ),
    ],
    edges=[
        PipelineEdge(id="e1", source_node_id="n1", target_node_id="n2",
                     source_handle_id="eeg_out", target_handle_id="eeg_in",
                     source_handle_type="raw_eeg", target_handle_type="raw_eeg"),
        PipelineEdge(id="e2", source_node_id="n2", target_node_id="n3",
                     source_handle_id="eeg_out", target_handle_id="filtered_in",
                     source_handle_type="filtered_eeg", target_handle_type="filtered_eeg"),
        PipelineEdge(id="e3", source_node_id="n3", target_node_id="n4",
                     source_handle_id="eeg_out", target_handle_id="filtered_in",
                     source_handle_type="filtered_eeg", target_handle_type="filtered_eeg"),
        PipelineEdge(id="e4", source_node_id="n4", target_node_id="n5",
                     source_handle_id="eeg_out", target_handle_id="filtered_in",
                     source_handle_type="filtered_eeg", target_handle_type="filtered_eeg"),
        PipelineEdge(id="e5", source_node_id="n5", target_node_id="n6",
                     source_handle_id="eeg_out", target_handle_id="eeg_in",
                     source_handle_type="filtered_eeg", target_handle_type="filtered_eeg"),
        PipelineEdge(id="e6", source_node_id="n6", target_node_id="n7",
                     source_handle_id="psd_out", target_handle_id="psd_in",
                     source_handle_type="psd", target_handle_type="psd"),
    ],
)


# ======================================================================
# PATH A: Run through REAL Oscilloom Engine
# ======================================================================
print("=" * 70)
print("  PATH A: OSCILLOOM ENGINE (real session_store + engine.py)")
print("=" * 70)

# 1. Create a real session (exactly what POST /session/load does)
session_id, info = create_session(EDF_PATH)
print(f"  Session created: {session_id[:8]}...")
print(f"  Session info: {info['nchan']} ch, {info['sfreq']} Hz, {info['duration_s']}s")

# 2. Get a raw copy (exactly what pipeline_routes.py does before execution)
raw_copy = get_raw_copy(session_id)
print(f"  Raw copy obtained: {type(raw_copy).__name__}")

# 3. Execute the pipeline through the REAL engine
results, node_outputs = execute_pipeline(raw_copy, pipeline, cache=None, generate_previews=False)
print(f"  Engine executed: {len(results)} nodes")

for nid in topological_sort(pipeline):
    r = results[nid]
    print(f"    {nid} ({r['node_type']}): {r['status']} → {r['output_type']}")

# 4. Extract the REAL numerical data from engine outputs
engine_ica_raw = node_outputs["n5"]  # Raw after ICA
engine_psd = node_outputs["n6"]      # Spectrum object

engine_raw_data = engine_ica_raw.get_data()
engine_psd_data = engine_psd.get_data()
engine_psd_freqs = engine_psd.freqs
engine_alpha = engine_psd.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)

print(f"\n  Engine preprocessed data: {engine_raw_data.shape}")
print(f"  Engine PSD: {engine_psd_data.shape}")
print(f"  Engine alpha power mean: {engine_alpha.mean():.10e}")

np.save(os.path.join(OUT_DIR, "engine_raw_data.npy"), engine_raw_data)
np.save(os.path.join(OUT_DIR, "engine_psd_data.npy"), engine_psd_data)
np.save(os.path.join(OUT_DIR, "engine_psd_freqs.npy"), engine_psd_freqs)


# ======================================================================
# PATH B: Run the EXPORTED script (what code_template generates)
# ======================================================================
print("\n" + "=" * 70)
print("  PATH B: OSCILLOOM EXPORT (code_template-generated script)")
print("=" * 70)

# 1. Generate the export script (exactly what POST /pipeline/export does)
exported_script = export_script(pipeline, audit_log=[])
print(f"  Exported script: {len(exported_script)} chars, {len(exported_script.splitlines())} lines")

# 2. Save the raw exported script for inspection
export_script_path = os.path.join(OUT_DIR, "exported_pipeline.py")
with open(export_script_path, "w") as f:
    f.write(exported_script)
print(f"  Saved to: {export_script_path}")

# 3. We need to inject data extraction BEFORE the topomap node (which may crash
#    due to code_template not handling PhysioNet channel name dots).
#    Split the exported script: extract data after compute_psd, before plot_topomap.

lines = exported_script.splitlines()
injection_point = None
for i, line in enumerate(lines):
    if 'plot_topomap complete' in line or 'plot_topomap' in line.lower():
        # Find the start of the topomap section (the comment block before it)
        for j in range(i - 1, max(0, i - 10), -1):
            if lines[j].strip().startswith("# ---") and "plot_topomap" in lines[j + 1].lower():
                injection_point = j
                break
        if injection_point is None:
            injection_point = i
        break

if injection_point is None:
    # Fallback: inject at the end
    injection_point = len(lines)

extraction_code = f'''
# --- VALIDATION: Extract numerical results BEFORE topomap ---
import numpy as np
raw_data = raw.get_data()
psd_data = spectrum.get_data()
psd_freqs = spectrum.freqs
alpha_power = spectrum.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)
np.save("{os.path.join(OUT_DIR, "export_raw_data.npy")}", raw_data)
np.save("{os.path.join(OUT_DIR, "export_psd_data.npy")}", psd_data)
np.save("{os.path.join(OUT_DIR, "export_psd_freqs.npy")}", psd_freqs)
print(f"EXPORT_SHAPE={{raw_data.shape}}")
print(f"EXPORT_PSD_SHAPE={{psd_data.shape}}")
print(f"EXPORT_ALPHA_MEAN={{alpha_power.mean():.10e}}")
print("EXPORT_DATA_SAVED")
'''

modified_lines = lines[:injection_point] + extraction_code.splitlines() + lines[injection_point:]
runnable_script = "\n".join(modified_lines)

runnable_script_path = os.path.join(OUT_DIR, "exported_pipeline_runnable.py")
with open(runnable_script_path, "w") as f:
    f.write(runnable_script)

# 4. Run the exported script in a SEPARATE Python process (clean environment)
print("  Running exported script in subprocess...")
result = subprocess.run(
    [sys.executable, runnable_script_path],
    capture_output=True, text=True, timeout=120,
    cwd=PROJECT_ROOT,
)
export_topomap_crashed = False
if result.returncode != 0:
    # Check if data was saved before the crash
    if "EXPORT_DATA_SAVED" in result.stdout:
        print(f"  Data extracted successfully, but script crashed AFTER (likely topomap)")
        export_topomap_crashed = True
        stderr_last = result.stderr.strip().split("\n")[-3:]
        print(f"  Crash: {stderr_last[-1]}")
    else:
        print(f"  EXPORT SCRIPT COMPLETELY FAILED!")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
else:
    for line in result.stdout.strip().split("\n"):
        if line.startswith("EXPORT_"):
            print(f"  {line}")

for line in result.stdout.strip().split("\n"):
    if line.startswith("EXPORT_"):
        print(f"  {line}")

# 5. Load the export outputs
export_raw_data = np.load(os.path.join(OUT_DIR, "export_raw_data.npy"))
export_psd_data = np.load(os.path.join(OUT_DIR, "export_psd_data.npy"))
export_psd_freqs = np.load(os.path.join(OUT_DIR, "export_psd_freqs.npy"))


# ======================================================================
# PATH C: Independent MNE (no Oscilloom imports)
# ======================================================================
print("\n" + "=" * 70)
print("  PATH C: INDEPENDENT MNE (hand-written, zero Oscilloom code)")
print("=" * 70)

# Write a completely independent MNE script
# NOTE: I'm intentionally NOT looking at execute_fn or code_template.
# I'm writing what a researcher would write from MNE docs.
independent_script = f'''
import mne
import numpy as np

import matplotlib
matplotlib.use("Agg")

# Step 1: Load
raw = mne.io.read_raw_edf("{EDF_PATH}", preload=True, verbose=False)

# Step 2: Notch filter 60 Hz
raw = raw.copy().notch_filter(freqs=60.0, verbose=False)

# Step 3: Bandpass 1-40 Hz, FIR hamming
raw = raw.copy().filter(l_freq=1.0, h_freq=40.0, method="fir",
                        fir_window="hamming", verbose=False)

# Step 4: Resample to 128 Hz
raw = raw.copy().resample(sfreq=128.0, verbose=False)

# Step 5: ICA — 20 components, fastica, seed=42, no exclusions
ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                            random_state=42, verbose=False)
ica.fit(raw, verbose=False)
raw = ica.apply(raw.copy(), exclude=[], verbose=False)

# Step 6: PSD — welch, 0.5-60 Hz, n_fft=2048, n_overlap=1024
spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=60.0,
                           n_fft=2048, n_overlap=1024, verbose=False)

# Save results
raw_data = raw.get_data()
psd_data = spectrum.get_data()
psd_freqs = spectrum.freqs

np.save("{os.path.join(OUT_DIR, "independent_raw_data.npy")}", raw_data)
np.save("{os.path.join(OUT_DIR, "independent_psd_data.npy")}", psd_data)
np.save("{os.path.join(OUT_DIR, "independent_psd_freqs.npy")}", psd_freqs)

print(f"INDEPENDENT_SHAPE={{raw_data.shape}}")
print(f"INDEPENDENT_PSD_SHAPE={{psd_data.shape}}")
alpha = spectrum.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)
print(f"INDEPENDENT_ALPHA_MEAN={{alpha.mean():.10e}}")
print("INDEPENDENT_DONE")
'''

independent_path = os.path.join(OUT_DIR, "independent_mne.py")
with open(independent_path, "w") as f:
    f.write(independent_script)

print("  Running independent MNE script in subprocess...")
result = subprocess.run(
    [sys.executable, independent_path],
    capture_output=True, text=True, timeout=120,
)
if result.returncode != 0:
    print(f"  INDEPENDENT SCRIPT FAILED!")
    print(f"  stdout: {result.stdout[-500:]}")
    print(f"  stderr: {result.stderr[-500:]}")
else:
    for line in result.stdout.strip().split("\n"):
        if line.startswith("INDEPENDENT_"):
            print(f"  {line}")

independent_raw_data = np.load(os.path.join(OUT_DIR, "independent_raw_data.npy"))
independent_psd_data = np.load(os.path.join(OUT_DIR, "independent_psd_data.npy"))
independent_psd_freqs = np.load(os.path.join(OUT_DIR, "independent_psd_freqs.npy"))


# ======================================================================
# THREE-WAY COMPARISON
# ======================================================================
print("\n" + "=" * 70)
print("  THREE-WAY COMPARISON")
print("=" * 70)

all_pass = True
findings = []

def check(name, passed, detail=""):
    global all_pass
    icon = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
        findings.append((name, detail))
    print(f"  [{icon}] {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")


# --- Preprocessed Data (after ICA, before PSD) ---
print("\n--- PREPROCESSED DATA (64 channels x ~7808 samples) ---")

check("Shape: Engine vs Export",
      engine_raw_data.shape == export_raw_data.shape,
      f"Engine={engine_raw_data.shape}, Export={export_raw_data.shape}")

check("Shape: Engine vs Independent",
      engine_raw_data.shape == independent_raw_data.shape,
      f"Engine={engine_raw_data.shape}, Independent={independent_raw_data.shape}")

# Engine vs Export (tests execute_fn vs code_template)
ae_exact = np.array_equal(engine_raw_data, export_raw_data)
check("Engine vs Export: bit-for-bit", ae_exact)
if not ae_exact:
    ae_diff = np.max(np.abs(engine_raw_data - export_raw_data))
    ae_close = np.allclose(engine_raw_data, export_raw_data)
    check("Engine vs Export: np.allclose", ae_close, f"max_abs_diff={ae_diff:.2e}")

# Engine vs Independent (tests full Oscilloom vs hand-written MNE)
ai_exact = np.array_equal(engine_raw_data, independent_raw_data)
check("Engine vs Independent: bit-for-bit", ai_exact)
if not ai_exact:
    ai_diff = np.max(np.abs(engine_raw_data - independent_raw_data))
    ai_close = np.allclose(engine_raw_data, independent_raw_data)
    check("Engine vs Independent: np.allclose", ai_close, f"max_abs_diff={ai_diff:.2e}")

# Export vs Independent (tests code_template vs hand-written)
ei_exact = np.array_equal(export_raw_data, independent_raw_data)
check("Export vs Independent: bit-for-bit", ei_exact)
if not ei_exact:
    ei_diff = np.max(np.abs(export_raw_data - independent_raw_data))
    ei_close = np.allclose(export_raw_data, independent_raw_data)
    check("Export vs Independent: np.allclose", ei_close, f"max_abs_diff={ei_diff:.2e}")


# --- PSD Data ---
print("\n--- PSD DATA ---")

check("PSD shape: Engine vs Export",
      engine_psd_data.shape == export_psd_data.shape,
      f"Engine={engine_psd_data.shape}, Export={export_psd_data.shape}")

check("PSD shape: Engine vs Independent",
      engine_psd_data.shape == independent_psd_data.shape,
      f"Engine={engine_psd_data.shape}, Independent={independent_psd_data.shape}")

psd_ae_exact = np.array_equal(engine_psd_data, export_psd_data)
check("PSD Engine vs Export: bit-for-bit", psd_ae_exact)
if not psd_ae_exact:
    psd_ae_diff = np.max(np.abs(engine_psd_data - export_psd_data))
    psd_ae_close = np.allclose(engine_psd_data, export_psd_data)
    psd_ae_rel = np.max(np.abs(engine_psd_data - export_psd_data) / (np.abs(engine_psd_data) + 1e-30))
    check("PSD Engine vs Export: np.allclose", psd_ae_close,
          f"max_abs_diff={psd_ae_diff:.2e}, max_rel_diff={psd_ae_rel:.2e}")

psd_ai_exact = np.array_equal(engine_psd_data, independent_psd_data)
check("PSD Engine vs Independent: bit-for-bit", psd_ai_exact)
if not psd_ai_exact:
    psd_ai_diff = np.max(np.abs(engine_psd_data - independent_psd_data))
    psd_ai_close = np.allclose(engine_psd_data, independent_psd_data)
    check("PSD Engine vs Independent: np.allclose", psd_ai_close,
          f"max_abs_diff={psd_ai_diff:.2e}")

psd_ei_exact = np.array_equal(export_psd_data, independent_psd_data)
check("PSD Export vs Independent: bit-for-bit", psd_ei_exact)
if not psd_ei_exact:
    psd_ei_diff = np.max(np.abs(export_psd_data - independent_psd_data))
    psd_ei_close = np.allclose(export_psd_data, independent_psd_data)
    check("PSD Export vs Independent: np.allclose", psd_ei_close,
          f"max_abs_diff={psd_ei_diff:.2e}")

# --- Frequency bins ---
print("\n--- FREQUENCY BINS ---")
check("Freq bins: Engine vs Export", np.array_equal(engine_psd_freqs, export_psd_freqs))
check("Freq bins: Engine vs Independent", np.array_equal(engine_psd_freqs, independent_psd_freqs))


# --- Session Store behavior ---
print("\n--- SESSION STORE INTEGRITY ---")
# Re-fetch the raw to check session store wasn't mutated
raw_check = get_raw_copy(session_id)
original_data = raw_check.get_data()
# Original should be UNFILTERED (160 Hz, 64 ch, full duration)
check("Session store not mutated (sfreq still 160 Hz)",
      raw_check.info["sfreq"] == 160.0,
      f"Got sfreq={raw_check.info['sfreq']}")
check("Session store not mutated (n_times unchanged)",
      raw_check.n_times == 9760,  # 160 Hz * 61s
      f"Got n_times={raw_check.n_times}")


# ======================================================================
# CODE TEMPLATE AUDIT
# ======================================================================
print("\n" + "=" * 70)
print("  CODE TEMPLATE AUDIT")
print("=" * 70)

# Check what the export script actually contains
print(f"\n  Exported script ({len(exported_script.splitlines())} lines):")
print("  " + "-" * 50)
for i, line in enumerate(exported_script.splitlines(), 1):
    # Show only the MNE calls, not comments/headers
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and not stripped.startswith("print"):
        print(f"  {i:3d}: {line}")
print("  " + "-" * 50)

# Check for missing n_overlap in export
if "n_overlap" not in exported_script:
    print("\n  [FINDING-1] Export script does NOT include n_overlap parameter")
    print("             execute_fn passes n_overlap=1024 explicitly")
    print("             MNE defaults to n_fft//2=1024, so results happen to match")
    print("             BUT violates export's promise: 'No hidden defaults'")
    findings.append(("code_template omits n_overlap",
                     "execute_fn explicit, export relies on MNE default"))
else:
    print("\n  [OK] Export script includes n_overlap")

# Check for topomap crash
if export_topomap_crashed:
    print("\n  [FINDING-2] Exported plot_topomap CRASHES on PhysioNet EDF files")
    print("             execute_fn strips trailing dots from channel names (Fc5. → FC5)")
    print("             code_template does NOT — montage assignment fails")
    print("             Engine succeeds; exported script crashes")
    findings.append(("plot_topomap code_template crashes on PhysioNet EDF",
                     "execute_fn strips trailing dots, code_template doesn't"))

# Check ICA code path divergence
ica_lines = [l for l in exported_script.splitlines() if "ica" in l.lower() and not l.strip().startswith("#")]
print("\n  ICA export code:")
for l in ica_lines:
    print(f"    {l.strip()}")
print("  ICA execute_fn: raw_copy = raw.copy(); ica.fit(raw_copy); ica.apply(raw_copy)")
print("  ICA code_template: ica.fit(raw); raw = ica.apply(raw.copy())")
print("  [FINDING-3] Different objects: execute_fn fits+applies on SAME copy;")
print("             code_template fits on original, applies on a NEW copy")
findings.append(("ICA code_template diverges from execute_fn",
                 "fit/apply targets differ (same copy vs separate copies)"))


# ======================================================================
# FINAL VERDICT
# ======================================================================
print("\n" + "=" * 70)
if all_pass:
    print("  VERDICT: ALL THREE PATHS PRODUCE IDENTICAL RESULTS")
else:
    print("  VERDICT: DIVERGENCES FOUND — SEE FINDINGS BELOW")
    for name, detail in findings:
        print(f"\n  FINDING: {name}")
        if detail:
            print(f"  {detail}")
print("=" * 70)

# Cleanup session
from backend.session_store import delete_session
delete_session(session_id)
