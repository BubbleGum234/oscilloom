"""
COMPARISON SCRIPT — Oscilloom vs Manual MNE
Loads both outputs and compares numerically.
Reports: exact match, floating-point tolerance, or divergence.
"""

import json
import os
import numpy as np

BASE = os.path.dirname(__file__)
OSC_DIR = os.path.join(BASE, "oscilloom_output")
MNE_DIR = os.path.join(BASE, "manual_mne_output")


def load_results(directory, prefix):
    results = json.load(open(os.path.join(directory, f"{prefix}_results.json")))
    raw_data = np.load(os.path.join(directory, f"{prefix}_raw_data.npy"))
    psd_data = np.load(os.path.join(directory, f"{prefix}_psd_data.npy"))
    psd_freqs = np.load(os.path.join(directory, f"{prefix}_psd_freqs.npy"))
    return results, raw_data, psd_data, psd_freqs


print("=" * 70)
print("  OSCILLOOM vs MANUAL MNE — NUMERICAL COMPARISON")
print("=" * 70)

osc_res, osc_raw, osc_psd, osc_freqs = load_results(OSC_DIR, "oscilloom")
mne_res, mne_raw, mne_psd, mne_freqs = load_results(MNE_DIR, "manual_mne")

all_pass = True

def check(name, passed, detail=""):
    global all_pass
    icon = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{icon}] {name}")
    if detail:
        print(f"         {detail}")


print("\n--- 1. METADATA ---")
check("Channel count",
      osc_res["n_channels"] == mne_res["n_channels"],
      f"Oscilloom={osc_res['n_channels']}, MNE={mne_res['n_channels']}")

check("Sampling rate",
      osc_res["sfreq"] == mne_res["sfreq"],
      f"Oscilloom={osc_res['sfreq']}, MNE={mne_res['sfreq']}")

check("Channel names",
      osc_res["ch_names"] == mne_res["ch_names"],
      f"First 3: {osc_res['ch_names'][:3]} vs {mne_res['ch_names'][:3]}")


print("\n--- 2. PREPROCESSED DATA (after ICA) ---")
check("Data shape",
      osc_res["raw_shape"] == mne_res["raw_shape"],
      f"Oscilloom={osc_res['raw_shape']}, MNE={mne_res['raw_shape']}")

# Exact binary match
exact = np.array_equal(osc_raw, mne_raw)
check("Exact binary match (bit-for-bit)", exact)

if not exact:
    max_abs_diff = np.max(np.abs(osc_raw - mne_raw))
    mean_abs_diff = np.mean(np.abs(osc_raw - mne_raw))
    rel_diff = np.max(np.abs(osc_raw - mne_raw) / (np.abs(mne_raw) + 1e-30))

    # Use allclose with standard float64 tolerance
    close_default = np.allclose(osc_raw, mne_raw)
    close_tight = np.allclose(osc_raw, mne_raw, atol=1e-12, rtol=1e-10)
    close_loose = np.allclose(osc_raw, mne_raw, atol=1e-6, rtol=1e-4)

    check("np.allclose (default: atol=1e-8, rtol=1e-5)", close_default,
          f"max_abs_diff={max_abs_diff:.2e}, mean_abs_diff={mean_abs_diff:.2e}")
    check("np.allclose (tight: atol=1e-12, rtol=1e-10)", close_tight)
    check("np.allclose (loose: atol=1e-6, rtol=1e-4)", close_loose,
          f"max_relative_diff={rel_diff:.2e}")

    # Per-channel comparison
    per_ch_max_diff = np.max(np.abs(osc_raw - mne_raw), axis=1)
    worst_ch = np.argmax(per_ch_max_diff)
    print(f"         Worst channel: #{worst_ch} (max diff={per_ch_max_diff[worst_ch]:.2e})")

# First 10 samples comparison
first10_osc = np.array(osc_res["raw_first_10_samples_ch0"])
first10_mne = np.array(mne_res["raw_first_10_samples_ch0"])
check("First 10 samples (ch0) match",
      np.allclose(first10_osc, first10_mne, atol=1e-12),
      f"Oscilloom: {first10_osc[:5]}\n         MNE:      {first10_mne[:5]}")


print("\n--- 3. SUMMARY STATISTICS ---")
for stat in ["raw_mean", "raw_std", "raw_min", "raw_max"]:
    osc_val = osc_res[stat]
    mne_val = mne_res[stat]
    diff = abs(osc_val - mne_val)
    rel = diff / (abs(mne_val) + 1e-30)
    check(f"{stat}",
          rel < 1e-10,
          f"Oscilloom={osc_val:.12e}, MNE={mne_val:.12e}, rel_diff={rel:.2e}")


print("\n--- 4. PSD DATA ---")
check("PSD shape",
      osc_res["psd_shape"] == mne_res["psd_shape"],
      f"Oscilloom={osc_res['psd_shape']}, MNE={mne_res['psd_shape']}")

psd_exact = np.array_equal(osc_psd, mne_psd)
check("PSD exact binary match", psd_exact)

if not psd_exact:
    psd_max_diff = np.max(np.abs(osc_psd - mne_psd))
    psd_close = np.allclose(osc_psd, mne_psd)
    check("PSD np.allclose (default)", psd_close,
          f"max_abs_diff={psd_max_diff:.2e}")

# Frequency axis
freq_exact = np.array_equal(osc_freqs, mne_freqs)
check("Frequency bins exact match", freq_exact,
      f"Oscilloom range: {osc_freqs[0]:.2f}–{osc_freqs[-1]:.2f} Hz ({len(osc_freqs)} bins)\n"
      f"         MNE range:      {mne_freqs[0]:.2f}–{mne_freqs[-1]:.2f} Hz ({len(mne_freqs)} bins)")


print("\n--- 5. ALPHA BAND POWER (8-13 Hz) ---")
osc_alpha = np.array(osc_res["alpha_power_per_channel"])
mne_alpha = np.array(mne_res["alpha_power_per_channel"])

alpha_exact = np.array_equal(osc_alpha, mne_alpha)
check("Alpha power exact match", alpha_exact)

if not alpha_exact:
    alpha_close = np.allclose(osc_alpha, mne_alpha)
    alpha_max_diff = np.max(np.abs(osc_alpha - mne_alpha))
    check("Alpha power np.allclose (default)", alpha_close,
          f"max_abs_diff={alpha_max_diff:.2e}")

check("Alpha mean",
      abs(osc_res["alpha_power_per_channel"][0] - mne_res["alpha_power_per_channel"][0]) < 1e-20,
      f"Oscilloom ch0={osc_res['alpha_power_per_channel'][0]:.12e}\n"
      f"         MNE ch0=     {mne_res['alpha_power_per_channel'][0]:.12e}")


print("\n" + "=" * 70)
if all_pass:
    print("  VERDICT: ALL CHECKS PASSED")
    print("  Oscilloom produces IDENTICAL results to manual MNE-Python.")
else:
    print("  VERDICT: SOME CHECKS FAILED — SEE DETAILS ABOVE")
    print("  Review the failing checks to understand the divergence.")
print("=" * 70)
