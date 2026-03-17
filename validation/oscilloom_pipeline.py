"""
OSCILLOOM EXPORT — Pipeline B
Generated to match exactly what Oscilloom's execute_fn chain does.
Each step mirrors the exact code path in backend/registry/nodes/*.py.

Pipeline: Load EDF → Notch 60Hz → Bandpass 1-40Hz → Resample 128Hz
          → ICA (20, fastica) → Compute PSD (welch) → Plot Topomap (alpha)
"""

import os
import sys
import json
import numpy as np
import mne

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EDF_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "sample_data", "eegmmidb", "S001", "S001R01.edf",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "oscilloom_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# Step 1: Load EDF (mirrors _execute_edf_loader)
# ======================================================================
raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
print(f"[1] Loaded: {len(raw.ch_names)} ch, {raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s")


# ======================================================================
# Step 2: Notch Filter 60 Hz (mirrors _execute_notch_filter)
# ======================================================================
# Oscilloom: raw.copy().notch_filter(freqs=60.0, verbose=False)
raw = raw.copy().notch_filter(freqs=60.0, verbose=False)
print(f"[2] Notch filter applied: 60 Hz")


# ======================================================================
# Step 3: Bandpass Filter 1-40 Hz (mirrors _execute_bandpass_filter)
# ======================================================================
# Oscilloom: raw.copy().filter(l_freq=1.0, h_freq=40.0, method="fir",
#            fir_window="hamming", verbose=False)
raw = raw.copy().filter(
    l_freq=1.0, h_freq=40.0,
    method="fir", fir_window="hamming",
    verbose=False,
)
print(f"[3] Bandpass filter applied: 1.0–40.0 Hz FIR hamming")


# ======================================================================
# Step 4: Resample to 128 Hz (mirrors _execute_resample)
# ======================================================================
# Oscilloom: raw.copy().resample(sfreq=128.0, verbose=False)
raw = raw.copy().resample(sfreq=128.0, verbose=False)
print(f"[4] Resampled to {raw.info['sfreq']} Hz, {raw.n_times} samples")


# ======================================================================
# Step 5: ICA Decomposition (mirrors _execute_ica_decomposition)
# ======================================================================
# Oscilloom: ICA(n_components=20, method="fastica", random_state=42)
#            ica.fit(raw_copy), ica.apply(raw_copy, exclude=[])
raw_copy = raw.copy()
ica = mne.preprocessing.ICA(
    n_components=20,
    method="fastica",
    random_state=42,
    verbose=False,
)
ica.fit(raw_copy, verbose=False)
ica.apply(raw_copy, exclude=[], verbose=False)
raw = raw_copy
print(f"[5] ICA applied: 20 components, fastica, 0 excluded")


# ======================================================================
# Step 6: Compute PSD (mirrors _execute_compute_psd, welch defaults)
# ======================================================================
# Oscilloom defaults: method="welch", fmin=0.5, fmax=60.0, n_fft=2048, n_overlap=1024
# BUT sfreq is now 128 Hz → n_fft=2048 > n_times? Let's check.
# After resample: 128 Hz * 61s ≈ 7808 samples. 2048 < 7808, so it's fine.
# However fmax=60.0 but Nyquist is 64 Hz, so MNE will clamp internally.
spectrum = raw.compute_psd(
    method="welch",
    fmin=0.5,
    fmax=60.0,
    n_fft=2048,
    n_overlap=1024,
    verbose=False,
)
print(f"[6] PSD computed: welch, {spectrum.freqs[0]:.1f}–{spectrum.freqs[-1]:.1f} Hz, {len(spectrum.freqs)} bins")


# ======================================================================
# Step 7: Extract numerical results for comparison
# ======================================================================

# 7a: Raw channel data after all preprocessing (before PSD)
raw_data = raw.get_data()  # shape: (n_channels, n_times)
print(f"[7a] Preprocessed data shape: {raw_data.shape}")
print(f"     Mean: {raw_data.mean():.10e}")
print(f"     Std:  {raw_data.std():.10e}")
print(f"     Min:  {raw_data.min():.10e}")
print(f"     Max:  {raw_data.max():.10e}")

# 7b: PSD data
psd_data = spectrum.get_data()  # shape: (n_channels, n_freqs)
psd_freqs = spectrum.freqs
print(f"[7b] PSD shape: {psd_data.shape}")
print(f"     PSD mean: {psd_data.mean():.10e}")
print(f"     PSD std:  {psd_data.std():.10e}")

# 7c: Alpha band power (8-13 Hz) per channel
alpha_power = spectrum.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)
print(f"[7c] Alpha power shape: {alpha_power.shape}")
print(f"     Alpha mean: {alpha_power.mean():.10e}")

# Save numerical results
results = {
    "raw_shape": list(raw_data.shape),
    "raw_mean": float(raw_data.mean()),
    "raw_std": float(raw_data.std()),
    "raw_min": float(raw_data.min()),
    "raw_max": float(raw_data.max()),
    "raw_first_10_samples_ch0": raw_data[0, :10].tolist(),
    "psd_shape": list(psd_data.shape),
    "psd_mean": float(psd_data.mean()),
    "psd_std": float(psd_data.std()),
    "psd_freqs_first_5": psd_freqs[:5].tolist(),
    "psd_freqs_last_5": psd_freqs[-5:].tolist(),
    "alpha_power_per_channel": alpha_power.tolist(),
    "sfreq": float(raw.info["sfreq"]),
    "n_channels": len(raw.ch_names),
    "ch_names": raw.ch_names,
}

with open(os.path.join(OUT_DIR, "oscilloom_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save raw numerical arrays for binary comparison
np.save(os.path.join(OUT_DIR, "oscilloom_raw_data.npy"), raw_data)
np.save(os.path.join(OUT_DIR, "oscilloom_psd_data.npy"), psd_data)
np.save(os.path.join(OUT_DIR, "oscilloom_psd_freqs.npy"), psd_freqs)

print(f"\n[DONE] Results saved to {OUT_DIR}/")
