#!/usr/bin/env python3
"""
Oscilloom End-to-End Pipeline Test Script
==========================================
Uploads an EEG file, runs a comprehensive analysis pipeline,
and generates a PDF report.

Usage:
    python test_pipeline.py /path/to/your/eeg_file.edf

The server must be running:
    uvicorn backend.main:app --reload --port 8000
"""
import sys
import json
import requests

BASE_URL = "http://localhost:8000"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py /path/to/your/eeg_file.edf")
        print("Supported formats: .edf, .fif, .bdf, .vhdr, .cnt")
        sys.exit(1)

    eeg_path = sys.argv[1]

    # ── Step 1: Upload EEG file ──────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Uploading EEG file...")
    print("=" * 60)
    with open(eeg_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/session/load", files={"file": f})
    resp.raise_for_status()
    session_data = resp.json()
    session_id = session_data["session_id"]
    info = session_data["info"]

    print(f"  Session ID : {session_id}")
    print(f"  Channels   : {info['nchan']}")
    print(f"  Sfreq      : {info['sfreq']} Hz")
    print(f"  Duration   : {info['duration_s']:.1f} s")
    print(f"  Channels   : {', '.join(info['ch_names'][:10])}...")
    print()

    # ── Step 2: Build pipeline ───────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Building pipeline...")
    print("=" * 60)

    pipeline = {
        "metadata": {
            "name": "Full EEG Analysis",
            "description": "Bandpass + Notch filter, PSD, clinical metrics, visualizations",
            "created_by": "human",
            "schema_version": "1.0",
        },
        "nodes": [
            # ── Preprocessing ──
            {
                "id": "n1_bandpass",
                "node_type": "bandpass_filter",
                "label": "Bandpass 1-40 Hz",
                "parameters": {
                    "low_cutoff_hz": 1.0,
                    "high_cutoff_hz": 40.0,
                    "method": "fir",
                },
                "position": {"x": 0, "y": 200},
            },
            {
                "id": "n2_notch",
                "node_type": "notch_filter",
                "label": "Notch 60 Hz",
                "parameters": {"notch_freq_hz": 60.0},
                "position": {"x": 250, "y": 200},
            },
            # ── Analysis ──
            {
                "id": "n3_psd",
                "node_type": "compute_psd",
                "label": "Compute PSD (Welch)",
                "parameters": {
                    "method": "welch",
                    "fmin": 0.5,
                    "fmax": 50.0,
                    "n_fft": 2048,
                },
                "position": {"x": 500, "y": 100},
            },
            # ── Clinical metrics ──
            {
                "id": "n4_alpha_peak",
                "node_type": "compute_alpha_peak",
                "label": "Alpha Peak (IAF)",
                "parameters": {
                    "fmin": 7.0,
                    "fmax": 13.0,
                    "method": "cog",
                },
                "position": {"x": 750, "y": 0},
            },
            {
                "id": "n5_band_ratio",
                "node_type": "compute_band_ratio",
                "label": "Theta/Beta Ratio",
                "parameters": {
                    "numerator_fmin": 4.0,
                    "numerator_fmax": 8.0,
                    "denominator_fmin": 13.0,
                    "denominator_fmax": 30.0,
                    "log_scale": True,
                },
                "position": {"x": 750, "y": 100},
            },
            {
                "id": "n6_detect_spikes",
                "node_type": "detect_spikes",
                "label": "Spike Detection",
                "parameters": {
                    "threshold_std": 4.0,
                    "min_duration_ms": 10.0,
                },
                "position": {"x": 500, "y": 350},
            },
            # ── Visualization ──
            {
                "id": "n7_plot_psd",
                "node_type": "plot_psd",
                "label": "PSD Plot",
                "parameters": {"dB": True, "show_average": True},
                "position": {"x": 750, "y": 200},
            },
            {
                "id": "n8_plot_raw",
                "node_type": "plot_raw",
                "label": "Raw Trace",
                "parameters": {
                    "n_channels": 10,
                    "start_time_s": 0.0,
                    "duration_s": 10.0,
                },
                "position": {"x": 500, "y": 450},
            },
        ],
        "edges": [
            # bandpass → notch
            {
                "id": "e1",
                "source_node_id": "n1_bandpass",
                "source_handle_id": "eeg_out",
                "source_handle_type": "filtered_eeg",
                "target_node_id": "n2_notch",
                "target_handle_id": "filtered_in",
                "target_handle_type": "filtered_eeg",
            },
            # notch → compute_psd
            {
                "id": "e2",
                "source_node_id": "n2_notch",
                "source_handle_id": "eeg_out",
                "source_handle_type": "filtered_eeg",
                "target_node_id": "n3_psd",
                "target_handle_id": "eeg_in",
                "target_handle_type": "filtered_eeg",
            },
            # compute_psd → alpha_peak
            {
                "id": "e3",
                "source_node_id": "n3_psd",
                "source_handle_id": "psd_out",
                "source_handle_type": "psd",
                "target_node_id": "n4_alpha_peak",
                "target_handle_id": "psd_in",
                "target_handle_type": "psd",
            },
            # compute_psd → band_ratio
            {
                "id": "e4",
                "source_node_id": "n3_psd",
                "source_handle_id": "psd_out",
                "source_handle_type": "psd",
                "target_node_id": "n5_band_ratio",
                "target_handle_id": "psd_in",
                "target_handle_type": "psd",
            },
            # notch → detect_spikes
            {
                "id": "e5",
                "source_node_id": "n2_notch",
                "source_handle_id": "eeg_out",
                "source_handle_type": "filtered_eeg",
                "target_node_id": "n6_detect_spikes",
                "target_handle_id": "filtered_in",
                "target_handle_type": "filtered_eeg",
            },
            # compute_psd → plot_psd
            {
                "id": "e6",
                "source_node_id": "n3_psd",
                "source_handle_id": "psd_out",
                "source_handle_type": "psd",
                "target_node_id": "n7_plot_psd",
                "target_handle_id": "psd_in",
                "target_handle_type": "psd",
            },
            # notch → plot_raw
            {
                "id": "e7",
                "source_node_id": "n2_notch",
                "source_handle_id": "eeg_out",
                "source_handle_type": "filtered_eeg",
                "target_node_id": "n8_plot_raw",
                "target_handle_id": "filtered_in",
                "target_handle_type": "filtered_eeg",
            },
        ],
    }

    node_names = [n["label"] for n in pipeline["nodes"]]
    print(f"  Pipeline: {pipeline['metadata']['name']}")
    print(f"  Nodes ({len(pipeline['nodes'])}): {', '.join(node_names)}")
    print(f"  Edges: {len(pipeline['edges'])}")
    print()

    # ── Step 3: Validate ─────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Validating pipeline...")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/pipeline/validate",
        json={"session_id": session_id, "pipeline": pipeline},
    )
    resp.raise_for_status()
    val = resp.json()
    print(f"  Valid: {val['valid']}")
    if val["errors"]:
        print(f"  Errors: {val['errors']}")
        sys.exit(1)
    print()

    # ── Step 4: Execute ──────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Executing pipeline (this may take a moment)...")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/pipeline/execute",
        json={"session_id": session_id, "pipeline": pipeline},
    )
    resp.raise_for_status()
    result = resp.json()

    print(f"  Status: {result['status']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")
        sys.exit(1)

    node_results = result["node_results"]
    for nid, nr in node_results.items():
        label = next(
            (n["label"] for n in pipeline["nodes"] if n["id"] == nid), nid
        )
        status = nr["status"]
        out_type = nr.get("output_type", "?")
        extras = []
        if nr.get("data") and isinstance(nr["data"], str):
            extras.append(f"plot={len(nr['data'])} bytes")
        if nr.get("metrics"):
            extras.append(f"metrics={json.dumps(nr['metrics'], indent=None)}")
        extra_str = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  [{status}] {label} -> {out_type}{extra_str}")
    print()

    # ── Step 5: Generate PDF report ──────────────────────────────────────
    print("=" * 60)
    print("STEP 5: Generating PDF report...")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/pipeline/report",
        json={
            "node_results": node_results,
            "title": "Oscilloom EEG Analysis Report",
            "patient_id": "TEST-001",
            "clinic_name": "Oscilloom Lab",
        },
    )
    resp.raise_for_status()

    report_path = "eeg_report.pdf"
    with open(report_path, "wb") as f:
        f.write(resp.content)
    print(f"  Report saved to: {report_path} ({len(resp.content)} bytes)")
    print()

    # ── Step 6: Export as Python script ──────────────────────────────────
    print("=" * 60)
    print("STEP 6: Exporting pipeline as Python script...")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/pipeline/export",
        json={
            "session_id": session_id,
            "pipeline": pipeline,
            "audit_log": [],
        },
    )
    resp.raise_for_status()
    export = resp.json()
    script_path = export.get("filename", "pipeline_export.py")
    with open(script_path, "w") as f:
        f.write(export["script"])
    print(f"  Script saved to: {script_path}")
    print()

    # ── Done ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"  - PDF report  : {report_path}")
    print(f"  - Python script: {script_path}")
    print(f"  - Session ID   : {session_id}")
    print()
    print("You can also test via the frontend at http://localhost:5173")


if __name__ == "__main__":
    main()
