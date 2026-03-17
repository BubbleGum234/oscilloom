# =============================================================================
# Oscilloom Pipeline Export
# Generated : 2026-03-15 14:19:01
# Pipeline  : Pipeline B Validation
# Version   : 1.0
# Created by: human
#
# Load → Notch 60Hz → Bandpass 1-40Hz → Resample 128Hz → ICA → PSD → Topomap
#
# To reproduce this pipeline:
#   pip install mne matplotlib
#   python pipeline_b_validation.py
#
# All MNE parameters are explicit below. No hidden defaults.
# =============================================================================

import mne
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Load EEG [edf_loader]
# Loads a raw EEG recording from an EDF (European Data Format) file. This is
# the standard starting node for any pipeline. The file is read entirely into
# memory, so ensure sufficient RAM for large recordings (typically 50–500 MB
# for single-session clinical EEG). Supports both EDF and EDF+ formats.
# -----------------------------------------------------------------------------
raw = mne.io.read_raw_edf("/Users/prafulmanikbhujbal/Documents/NeuroFlow/sample_data/eegmmidb/S001/S001R01.edf", preload=True, verbose=False)
print(f"edf_loader complete.")
# -----------------------------------------------------------------------------
# 2. Notch 60Hz [notch_filter]
# Removes power line interference (electrical noise) at a specific frequency
# and its harmonics. Use 60 Hz for equipment in North America, 50 Hz for
# Europe and most of Asia. This node should typically be applied before
# bandpass filtering in the pipeline.
# -----------------------------------------------------------------------------
raw = raw.copy().notch_filter(freqs=60.0, verbose=False)
print(f"notch_filter complete.")
# -----------------------------------------------------------------------------
# 3. Bandpass 1-40Hz [bandpass_filter]
# Applies a bandpass filter to remove frequencies outside a specified range.
# The low cutoff removes slow signal drifts (high-pass component); the high
# cutoff removes high-frequency noise (low-pass component). Typical EEG
# research range: 1–40 Hz. FIR method is recommended because it has linear
# phase and does not introduce temporal distortion. Typically connected after
# a Notch Filter — apply notch filtering first to remove power line
# interference, then use this node to define the frequency band of interest.
# -----------------------------------------------------------------------------
raw = raw.copy().filter(l_freq=1.0, h_freq=40.0, method="fir", fir_window="hamming", verbose=False)
print(f"bandpass_filter complete.")
# -----------------------------------------------------------------------------
# 4. Resample 128Hz [resample]
# Changes the sampling rate of the EEG data. Downsampling (e.g., from 1000 Hz
# to 256 Hz) reduces data size and speeds up subsequent analysis. MNE applies
# an automatic anti-aliasing filter before downsampling. Apply a bandpass
# filter upstream before resampling to avoid aliasing.
# -----------------------------------------------------------------------------
raw = raw.copy().resample(sfreq=128.0, verbose=False)
print(f"resample complete.")
# -----------------------------------------------------------------------------
# 5. ICA [ica_decomposition]
# Fits an Independent Component Analysis (ICA) model to separate EEG signals
# into statistically independent components. ICA is the standard method for
# removing eye blink, eye movement, and muscle artifacts. The current node
# fits and applies ICA without removing components — use the output to inspect
# the ICA-decomposed signal. Requires bandpass-filtered data (0.5–40 Hz
# recommended) as input.
# -----------------------------------------------------------------------------
ica = mne.preprocessing.ICA(n_components=20, method='fastica', random_state=42, verbose=False)
ica.fit(raw, verbose=False)
raw = ica.apply(raw.copy(), exclude=[], verbose=False)
print(f"ica_decomposition complete.")
# -----------------------------------------------------------------------------
# 6. Compute PSD [compute_psd]
# Computes the Power Spectral Density (PSD) of the EEG signal. Welch's method
# (default) segments the signal, computes FFT on each segment, and averages —
# reducing noise compared to a single FFT. Multitaper (DPSS) provides better
# frequency concentration and is preferred for short recordings. Connect the
# output to a Plot PSD node to visualize.
# -----------------------------------------------------------------------------
spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=60.0, n_fft=2048, verbose=False)
print(f"compute_psd complete.")

# --- VALIDATION: Extract numerical results BEFORE topomap ---
import numpy as np
raw_data = raw.get_data()
psd_data = spectrum.get_data()
psd_freqs = spectrum.freqs
alpha_power = spectrum.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)
np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/export_raw_data.npy", raw_data)
np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/export_psd_data.npy", psd_data)
np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/export_psd_freqs.npy", psd_freqs)
print(f"EXPORT_SHAPE={raw_data.shape}")
print(f"EXPORT_PSD_SHAPE={psd_data.shape}")
print(f"EXPORT_ALPHA_MEAN={alpha_power.mean():.10e}")
print("EXPORT_DATA_SAVED")
# -----------------------------------------------------------------------------
# 7. Topomap [plot_topomap]
# Renders a scalp topography (topomap) showing the spatial distribution of EEG
# power across electrodes for a selected frequency band. Warmer colours
# indicate higher power; cooler colours indicate lower power. Requires
# standard 10-20 electrode names (e.g. Fp1, Fz, Cz, Pz, Oz). Non-standard
# channel names are excluded from the map. This is a terminal node — its
# output cannot connect to other nodes.
# -----------------------------------------------------------------------------
montage = mne.channels.make_standard_montage("standard_1020")
spectrum.info.set_montage(montage, on_missing="ignore", match_case=False, verbose=False)
fig = spectrum.plot_topomap(bands="alpha", show=False)
print(f"plot_topomap complete.")
print("Pipeline complete.")