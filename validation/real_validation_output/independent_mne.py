
import mne
import numpy as np

import matplotlib
matplotlib.use("Agg")

# Step 1: Load
raw = mne.io.read_raw_edf("/Users/prafulmanikbhujbal/Documents/NeuroFlow/sample_data/eegmmidb/S001/S001R01.edf", preload=True, verbose=False)

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

np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/independent_raw_data.npy", raw_data)
np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/independent_psd_data.npy", psd_data)
np.save("/Users/prafulmanikbhujbal/Documents/NeuroFlow/validation/real_validation_output/independent_psd_freqs.npy", psd_freqs)

print(f"INDEPENDENT_SHAPE={raw_data.shape}")
print(f"INDEPENDENT_PSD_SHAPE={psd_data.shape}")
alpha = spectrum.get_data(fmin=8.0, fmax=13.0).mean(axis=-1)
print(f"INDEPENDENT_ALPHA_MEAN={alpha.mean():.10e}")
print("INDEPENDENT_DONE")
