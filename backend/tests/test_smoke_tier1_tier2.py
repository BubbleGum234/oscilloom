#!/usr/bin/env python3
"""
smoke_test_tier1_tier2.py
─────────────────────────
End-to-end smoke test for all Tier 1 and Tier 2 nodes added in
EXPANSION_PLAN.md. No live server required — uses FastAPI's TestClient
and injects synthetic EEG directly into the session store.

Run from the Oscilloom project root (with venv active):
    cd /Users/prafulmanikbhujbal/Documents/NeuroFlow
    source .venv/bin/activate
    python smoke_test_tier1_tier2.py

What it tests
─────────────
Section 1  Server health + node count
Section 2  Tier 1 preprocessing: set_montage, annotate_artifacts,
            interpolate_bad_channels
Section 3  Tier 1 epoching:   epoch_by_time, equalize_event_counts
Section 4  Tier 1 ERP analysis:  compute_gfp, detect_erp_peak,
            compute_difference_wave, plot_gfp, plot_comparison_evoked
Section 5  Tier 2 connectivity: compute_coherence, compute_plv,
            compute_pli, plot_connectivity_circle, plot_connectivity_matrix
Section 6  Tier 2 statistics: cluster_permutation_test, compute_t_test,
            apply_fdr_correction
"""

from __future__ import annotations

import sys
import uuid

import numpy as np
import mne
from fastapi.testclient import TestClient

from backend.main import app
import backend.session_store as _store


# ────────────────────────────────────────────────────────────────────────────
# Terminal colour helpers
# ────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

_pass_count = 0
_fail_count = 0


def ok(msg: str) -> None:
    global _pass_count
    _pass_count += 1
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str) -> None:
    global _fail_count
    _fail_count += 1
    print(f"  {RED}✗{RESET}  {msg}")
    sys.exit(1)


def info(msg: str) -> None:
    print(f"  {CYAN}→{RESET}  {msg}")


def section(title: str) -> None:
    bar = "─" * 62
    print(f"\n{BOLD}{YELLOW}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{bar}{RESET}")


# ────────────────────────────────────────────────────────────────────────────
# Synthetic EEG factory
# ────────────────────────────────────────────────────────────────────────────

_EEG_CH_NAMES = ["Fz", "Cz", "Pz", "Oz", "O1", "O2"]
_SFREQ = 250.0


def make_raw() -> mne.io.RawArray:
    """
    60-second synthetic EEG with six standard-10-20 channels plus an STI
    channel carrying two event types (type 1 and type 2).

    Channel names are in the standard_1020 montage so set_montage tests work.
    """
    n_ch  = len(_EEG_CH_NAMES)
    n_s   = int(60 * _SFREQ)
    rng   = np.random.default_rng(42)

    eeg   = rng.standard_normal((n_ch, n_s)) * 5e-6
    stim  = np.zeros((1, n_s))

    # 9 events of type 1 and 9 of type 2, well-separated
    for t in [3, 6, 9, 12, 15, 18, 21, 24, 27]:
        stim[0, int(t * _SFREQ)] = 1
    for t in [33, 36, 39, 42, 45, 48, 51, 54, 57]:
        stim[0, int(t * _SFREQ)] = 2

    data  = np.vstack([eeg, stim])
    names = _EEG_CH_NAMES + ["STI 014"]
    types = ["eeg"] * n_ch + ["stim"]
    info  = mne.create_info(ch_names=names, sfreq=_SFREQ, ch_types=types)
    return mne.io.RawArray(data, info, verbose=False)


def make_multi_condition_epochs() -> mne.Epochs:
    """
    Minimal multi-condition Epochs for difference wave / comparison tests.
    Uses EpochsArray so no raw injection is needed.
    """
    rng     = np.random.default_rng(7)
    n_each  = 8
    n_ch    = 4
    n_times = 125
    ch_names = ["Fz", "Cz", "Pz", "Oz"]

    d1 = rng.standard_normal((n_each, n_ch, n_times)) * 1e-6
    d2 = rng.standard_normal((n_each, n_ch, n_times)) * 1e-6
    data = np.vstack([d1, d2])
    events = np.array(
        [[i,          0, 1] for i in range(n_each)] +
        [[i + n_each, 0, 2] for i in range(n_each)]
    )
    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ, ch_types="eeg")
    return mne.EpochsArray(
        data, info, events=events, event_id={"1": 1, "2": 2}, verbose=False
    )


# ────────────────────────────────────────────────────────────────────────────
# Pipeline builder helpers
# ────────────────────────────────────────────────────────────────────────────

def _node(nid: str, node_type: str, params: dict | None = None) -> dict:
    return {
        "id": nid,
        "node_type": node_type,
        "label": node_type,
        "parameters": params or {},
        "position": {"x": 0.0, "y": 0.0},
    }


def _edge(eid: str,
          src: str, src_h: str, src_t: str,
          tgt: str, tgt_h: str, tgt_t: str) -> dict:
    return {
        "id": eid,
        "source_node_id": src, "source_handle_id": src_h, "source_handle_type": src_t,
        "target_node_id": tgt, "target_handle_id": tgt_h, "target_handle_type": tgt_t,
    }


def _run(client: TestClient, session_id: str, name: str,
         nodes: list[dict], edges: list[dict]) -> dict:
    """POST /pipeline/execute and return the parsed JSON."""
    payload = {
        "session_id": session_id,
        "pipeline": {
            "metadata": {
                "name": name, "description": "",
                "created_by": "smoke_test", "schema_version": "1.0",
            },
            "nodes": nodes,
            "edges": edges,
        },
    }
    resp = client.post("/pipeline/execute", json=payload)
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text}"
    return resp.json()


def _check(result: dict, node_id: str, label: str = "",
           expect_type: str | None = None,
           expect_plot: bool = False) -> dict:
    """
    Assert a node result is success and (optionally) matches expected output.
    Returns the result entry dict.
    """
    if result["status"] != "success":
        fail(f"{label}: pipeline-level error — {result.get('error')}")

    nr = result["node_results"].get(node_id)
    if nr is None:
        fail(f"{label}: node '{node_id}' missing from results "
             f"({list(result['node_results'].keys())})")
    if nr.get("status") != "success":
        fail(f"{label}: node error — {nr}")

    if expect_type and nr["output_type"] != expect_type:
        fail(f"{label}: expected output_type='{expect_type}', "
             f"got '{nr['output_type']}'")

    if expect_plot:
        data = nr.get("data", "")
        if not (isinstance(data, str) and
                data.startswith("data:image/png;base64,")):
            fail(f"{label}: expected PNG data URI, got: {str(data)[:60]}")

    return nr


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    client = TestClient(app)

    # ── Section 1: health + registry ──────────────────────────────────────
    section("1 · Server health + node registry")

    resp = client.get("/status")
    assert resp.status_code == 200
    status = resp.json()
    ok(f"GET /status → MNE {status['mne_version']}")

    resp = client.get("/registry/nodes")
    count = resp.json()["count"]
    assert count >= 49, f"Expected ≥49 nodes, got {count}"
    ok(f"GET /registry/nodes → {count} nodes registered (expected ≥49)")

    # ── Session setup ─────────────────────────────────────────────────────
    raw = make_raw()
    sid = str(uuid.uuid4())
    _store._sessions[sid] = raw          # inject directly — no file upload needed
    info(f"Session {sid[:8]}… | {len(_EEG_CH_NAMES)} EEG ch + STI | "
         f"{_SFREQ:.0f} Hz | 60 s")

    # ── Section 2: Tier 1 Preprocessing ───────────────────────────────────
    section("2 · Tier 1 — Preprocessing nodes")

    # set_montage
    r = _run(client, sid, "set_montage",
             nodes=[_node("n1","edf_loader"),
                    _node("n2","set_montage",{"montage":"standard_1020"})],
             edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg")])
    _check(r,"n2","set_montage", expect_type="RawArray")
    ok("set_montage: standard_1020 assigned  →  output_type=RawArray")

    # annotate_artifacts
    r = _run(client, sid, "annotate_artifacts",
             nodes=[_node("n1","edf_loader"),
                    _node("n2","annotate_artifacts",{
                        "onsets_s":"5.0,15.0","durations_s":"2.0",
                        "description":"BAD_muscle"})],
             edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg")])
    _check(r,"n2","annotate_artifacts", expect_type="RawArray")
    ok("annotate_artifacts: BAD_muscle annotations added  →  output_type=RawArray")

    # interpolate_bad_channels — no bads in synthetic data, so passthrough
    r = _run(client, sid, "interp_bads",
             nodes=[_node("n1","edf_loader"),
                    _node("n2","set_montage",{"montage":"standard_1020"}),
                    _node("n3","interpolate_bad_channels",{"reset_bads":True})],
             edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg"),
                    _edge("e2","n2","eeg_out","filtered_eeg","n3","filtered_in","filtered_eeg")])
    _check(r,"n3","interpolate_bad_channels", expect_type="RawArray")
    ok("interpolate_bad_channels: no bads → passthrough  →  output_type=RawArray")

    # fif_loader / brainvision_loader / bdf_loader passthrough
    for loader in ("fif_loader","brainvision_loader","bdf_loader"):
        r = _run(client, sid, f"{loader}_passthrough",
                 nodes=[_node("n1", loader)],
                 edges=[])
        _check(r,"n1",loader)
        ok(f"{loader}: BaseRaw passthrough  →  output_type={r['node_results']['n1']['output_type']}")

    # ── Section 3: Tier 1 Epoching ────────────────────────────────────────
    section("3 · Tier 1 — Epoching nodes")

    # epoch_by_time
    r = _run(client, sid, "epoch_by_time",
             nodes=[_node("n1","edf_loader"),
                    _node("n2","epoch_by_time",{"duration_s":2.0,"overlap_s":0.0})],
             edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg")])
    nr = _check(r,"n2","epoch_by_time", expect_type="Epochs")
    ok(f"epoch_by_time: 2 s windows  →  output_type={nr['output_type']}")

    # epoch_by_time with overlap
    r_ov = _run(client, sid, "epoch_by_time_overlap",
                nodes=[_node("n1","edf_loader"),
                       _node("n2","epoch_by_time",{"duration_s":2.0,"overlap_s":1.0})],
                edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg")])
    _check(r_ov,"n2","epoch_by_time_overlap")
    ok("epoch_by_time: overlap=1 s  →  pipeline succeeded")

    # equalize_event_counts (single-condition epochs → no-op)
    r = _run(client, sid, "equalize",
             nodes=[_node("n1","edf_loader"),
                    _node("n2","epoch_by_time",{"duration_s":2.0}),
                    _node("n3","equalize_event_counts",{"method":"mintime"})],
             edges=[_edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg"),
                    _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs")])
    _check(r,"n3","equalize_event_counts", expect_type="Epochs")
    ok("equalize_event_counts: method=mintime  →  output_type=Epochs")

    # ── Section 4: Tier 1 ERP nodes ──────────────────────────────────────
    section("4 · Tier 1 — ERP analysis + visualization")

    # Shared chain: edf_loader → epoch_by_time → compute_evoked
    _chain_evoked = [
        _node("n1","edf_loader"),
        _node("n2","epoch_by_time",{"duration_s":2.0}),
        _node("n3","compute_evoked"),
    ]
    _chain_edges = [
        _edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg"),
        _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
    ]

    # compute_gfp
    r = _run(client, sid, "compute_gfp",
             nodes=_chain_evoked + [_node("n4","compute_gfp")],
             edges=_chain_edges + [
                 _edge("e3","n3","evoked_out","evoked","n4","evoked_in","evoked")])
    _check(r,"n4","compute_gfp", expect_type="ndarray")
    ok("compute_gfp: GFP array  →  output_type=ndarray")

    # detect_erp_peak
    r = _run(client, sid, "detect_erp_peak",
             nodes=_chain_evoked + [_node("n4","detect_erp_peak",{
                 "channel":"Cz","tmin_ms":0.0,"tmax_ms":500.0,"polarity":"positive"})],
             edges=_chain_edges + [
                 _edge("e3","n3","evoked_out","evoked","n4","evoked_in","evoked")])
    _check(r,"n4","detect_erp_peak", expect_type="dict")
    ok("detect_erp_peak: P300 window on Cz  →  output_type=dict (metrics)")

    # plot_gfp
    r = _run(client, sid, "plot_gfp",
             nodes=_chain_evoked + [_node("n4","plot_gfp",{"highlight_peaks":True})],
             edges=_chain_edges + [
                 _edge("e3","n3","evoked_out","evoked","n4","evoked_in","evoked")])
    nr = _check(r,"n4","plot_gfp", expect_plot=True)
    kb = len(nr["data"]) // 1024
    ok(f"plot_gfp: GFP timecourse PNG  →  {kb} KB")

    # compute_difference_wave + plot_comparison_evoked — direct Python call
    # (multi-condition epochs can't be created via the current linear API chain)
    info("compute_difference_wave / plot_comparison_evoked — "
         "direct execute_fn call (multi-condition epochs required)")

    from backend.registry.nodes.erp import (
        _execute_compute_difference_wave,
        _execute_plot_comparison_evoked,
    )
    mc = make_multi_condition_epochs()
    diff = _execute_compute_difference_wave(mc, {"condition_a":"1","condition_b":"2"})
    assert hasattr(diff, "data"), "Expected mne.Evoked"
    ok(f"compute_difference_wave: Evoked shape {diff.data.shape}  →  A−B waveform")

    png = _execute_plot_comparison_evoked(mc, {"conditions":"1,2","channel":"Cz"})
    assert png.startswith("data:image/png;base64,")
    ok(f"plot_comparison_evoked: conditions 1 vs 2 on Cz  →  {len(png)//1024} KB PNG")

    # ── Section 5: Tier 2 Connectivity ────────────────────────────────────
    section("5 · Tier 2 — Connectivity nodes")

    # Shared epoch chain (short 2 s windows, 60 s data → ~30 epochs)
    _epo_nodes = [
        _node("n1","edf_loader"),
        _node("n2","epoch_by_time",{"duration_s":2.0,"overlap_s":0.0}),
    ]
    _epo_edges = [
        _edge("e1","n1","eeg_out","raw_eeg","n2","raw_in","raw_eeg"),
    ]

    # compute_coherence → plot_connectivity_circle
    r = _run(client, sid, "coherence_circle",
             nodes=_epo_nodes + [
                 _node("n3","compute_coherence",{"fmin_hz":4.0,"fmax_hz":30.0}),
                 _node("n4","plot_connectivity_circle",{"n_lines":8,"colormap":"hot"}),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
                 _edge("e3","n3","conn_out","connectivity","n4","conn_in","connectivity"),
             ])
    _check(r,"n3","compute_coherence", expect_type="SpectralConnectivity")
    nr = _check(r,"n4","plot_connectivity_circle", expect_plot=True)
    kb = len(nr["data"]) // 1024
    ok(f"compute_coherence (4–30 Hz)  →  SpectralConnectivity")
    ok(f"plot_connectivity_circle (top 8 connections)  →  {kb} KB PNG")

    # compute_plv → plot_connectivity_matrix
    r = _run(client, sid, "plv_matrix",
             nodes=_epo_nodes + [
                 _node("n3","compute_plv",{"fmin_hz":8.0,"fmax_hz":13.0}),
                 _node("n4","plot_connectivity_matrix",{"colormap":"RdYlBu_r"}),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
                 _edge("e3","n3","conn_out","connectivity","n4","conn_in","connectivity"),
             ])
    _check(r,"n3","compute_plv", expect_type="SpectralConnectivity")
    nr = _check(r,"n4","plot_connectivity_matrix", expect_plot=True)
    kb = len(nr["data"]) // 1024
    ok(f"compute_plv (alpha band 8–13 Hz)  →  SpectralConnectivity")
    ok(f"plot_connectivity_matrix (N×N heatmap)  →  {kb} KB PNG")

    # compute_pli (standalone — no downstream visualization)
    r = _run(client, sid, "pli",
             nodes=_epo_nodes + [
                 _node("n3","compute_pli",{"fmin_hz":1.0,"fmax_hz":40.0}),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
             ])
    _check(r,"n3","compute_pli", expect_type="SpectralConnectivity")
    ok("compute_pli (1–40 Hz, volume-conduction-robust)  →  SpectralConnectivity")

    # ── Section 6: Tier 2 Statistics ──────────────────────────────────────
    section("6 · Tier 2 — Statistics nodes")

    # cluster_permutation_test
    r = _run(client, sid, "cluster_test",
             nodes=_epo_nodes + [
                 _node("n3","cluster_permutation_test",{
                     "tmin_ms":0.0,"tmax_ms":500.0,
                     "n_permutations":200,   # smaller for speed
                     "alpha":0.05,
                 }),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
             ])
    _check(r,"n3","cluster_permutation_test", expect_type="dict")
    ok("cluster_permutation_test (200 perms, 0–500 ms)  →  dict (metrics)")

    # compute_t_test
    r = _run(client, sid, "ttest",
             nodes=_epo_nodes + [
                 _node("n3","compute_t_test",{
                     "channel":"Cz","tmin_ms":0.0,"tmax_ms":500.0,"popmean":0.0,
                 }),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
             ])
    _check(r,"n3","compute_t_test", expect_type="dict")
    ok("compute_t_test (Cz, 0–500 ms)  →  dict (t, p, df, CI, mean µV)")

    # compute_t_test → apply_fdr_correction (chained)
    r = _run(client, sid, "ttest_fdr",
             nodes=_epo_nodes + [
                 _node("n3","compute_t_test",{
                     "channel":"Cz","tmin_ms":0.0,"tmax_ms":500.0,"popmean":0.0,
                 }),
                 _node("n4","apply_fdr_correction",{"alpha":0.05}),
             ],
             edges=_epo_edges + [
                 _edge("e2","n2","epochs_out","epochs","n3","epochs_in","epochs"),
                 _edge("e3","n3","metrics_out","metrics","n4","metrics_in","metrics"),
             ])
    _check(r,"n4","apply_fdr_correction", expect_type="dict")
    ok("compute_t_test → apply_fdr_correction (BH, α=0.05)  →  dict (corrected p)")

    # ── Cleanup + summary ─────────────────────────────────────────────────
    _store.delete_session(sid)

    section("SUMMARY")
    print(f"\n  {GREEN}{BOLD}All {_pass_count} smoke tests passed  ✓{RESET}\n")
    print(f"  {'Tier 1 nodes exercised:':30s} set_montage, annotate_artifacts,")
    print(f"  {'':30s} interpolate_bad_channels, fif/brainvision/bdf loaders,")
    print(f"  {'':30s} epoch_by_time, equalize_event_counts,")
    print(f"  {'':30s} compute_gfp, detect_erp_peak, plot_gfp,")
    print(f"  {'':30s} compute_difference_wave, plot_comparison_evoked")
    print()
    print(f"  {'Tier 2 nodes exercised:':30s} compute_coherence, compute_plv, compute_pli,")
    print(f"  {'':30s} plot_connectivity_circle, plot_connectivity_matrix,")
    print(f"  {'':30s} cluster_permutation_test, compute_t_test,")
    print(f"  {'':30s} apply_fdr_correction")
    print()


if __name__ == "__main__":
    main()
