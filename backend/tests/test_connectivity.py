"""
backend/tests/test_connectivity.py

Unit tests for all Tier 2 nodes: connectivity analysis + statistics.

Tests call execute_fn directly (not via the pipeline engine) for speed and
isolation.  All fixtures use in-memory synthetic data.

Coverage:
  Connectivity (5 nodes):
    - test_compute_coherence_returns_spectral_connectivity
    - test_compute_coherence_shape
    - test_compute_plv_returns_spectral_connectivity
    - test_compute_pli_returns_spectral_connectivity
    - test_connectivity_values_in_range        (0 ≤ COH/PLV/PLI ≤ 1)
    - test_plot_connectivity_circle_returns_png
    - test_plot_connectivity_matrix_returns_png

  Statistics (6 nodes):
    - test_cluster_permutation_test_returns_metrics
    - test_cluster_permutation_test_keys_present
    - test_compute_t_test_returns_metrics
    - test_compute_t_test_keys_present
    - test_compute_t_test_invalid_channel_raises
    - test_apply_fdr_single_p_value
    - test_apply_fdr_p_values_list
    - test_apply_fdr_cluster_p_values
    - test_apply_fdr_invalid_input_raises

  Handle type:
    - test_connectivity_in_valid_handle_types
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from mne_connectivity import SpectralConnectivity

from backend.registry.node_descriptor import VALID_HANDLE_TYPES
from backend.registry.nodes.connectivity import (
    ConnectivityMatrix,
    _execute_compute_coherence,
    _execute_compute_plv,
    _execute_compute_pli,
    _execute_compute_envelope_correlation,
    _execute_plot_connectivity_circle,
    _execute_plot_connectivity_matrix,
)
from backend.registry.nodes.statistics import (
    _execute_cluster_permutation_test,
    _execute_compute_t_test,
    _execute_apply_fdr_correction,
    _execute_compute_noise_floor,
)


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

_SFREQ = 250.0
_CH_NAMES = ["Fz", "Cz", "Pz", "Oz"]
_N_EPOCHS = 20   # minimum for stable connectivity estimates
_EPOCH_DURATION = 2.0  # seconds


@pytest.fixture(scope="module")
def synthetic_epochs() -> mne.Epochs:
    """
    20 epochs × 4 EEG channels × 500 samples at 250 Hz.
    Data is independent Gaussian noise — connectivity values should be near 0.
    """
    rng = np.random.default_rng(42)
    n_ch = len(_CH_NAMES)
    n_times = int(_EPOCH_DURATION * _SFREQ)
    data = rng.standard_normal((_N_EPOCHS, n_ch, n_times)) * 1e-6
    info = mne.create_info(
        ch_names=_CH_NAMES, sfreq=_SFREQ, ch_types=["eeg"] * n_ch
    )
    return mne.EpochsArray(data, info, verbose=False)


@pytest.fixture(scope="module")
def coherence_con(synthetic_epochs) -> SpectralConnectivity:
    """Pre-computed coherence for downstream visualization tests."""
    return _execute_compute_coherence(
        synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0}
    )


# ---------------------------------------------------------------------------
# Connectivity Compute Tests
# ---------------------------------------------------------------------------


def test_compute_coherence_returns_spectral_connectivity(synthetic_epochs):
    """compute_coherence must return a SpectralConnectivity object."""
    con = _execute_compute_coherence(
        synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0}
    )
    assert isinstance(con, SpectralConnectivity), (
        f"Expected SpectralConnectivity, got {type(con)}"
    )


def test_compute_coherence_shape(synthetic_epochs):
    """
    Dense connectivity matrix must have shape (n_ch, n_ch, n_freqs).
    n_freqs depends on the frequency resolution of the epoch length.
    """
    con = _execute_compute_coherence(
        synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0}
    )
    dense = con.get_data(output="dense")
    n_ch = len(_CH_NAMES)
    assert dense.shape[0] == n_ch, f"Expected {n_ch} rows, got {dense.shape[0]}"
    assert dense.shape[1] == n_ch, f"Expected {n_ch} cols, got {dense.shape[1]}"
    assert dense.shape[2] > 0, "Expected at least one frequency bin"


def test_compute_plv_returns_spectral_connectivity(synthetic_epochs):
    """compute_plv must return a SpectralConnectivity object."""
    con = _execute_compute_plv(
        synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0}
    )
    assert isinstance(con, SpectralConnectivity), (
        f"Expected SpectralConnectivity, got {type(con)}"
    )


def test_compute_pli_returns_spectral_connectivity(synthetic_epochs):
    """compute_pli must return a SpectralConnectivity object."""
    con = _execute_compute_pli(
        synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0}
    )
    assert isinstance(con, SpectralConnectivity), (
        f"Expected SpectralConnectivity, got {type(con)}"
    )


def test_connectivity_values_in_range(synthetic_epochs):
    """Coherence, PLV, and PLI values must all be in [0, 1]."""
    for method_fn, name in [
        (_execute_compute_coherence, "coherence"),
        (_execute_compute_plv, "PLV"),
        (_execute_compute_pli, "PLI"),
    ]:
        con = method_fn(synthetic_epochs, {"fmin_hz": 4.0, "fmax_hz": 30.0})
        vals = con.get_data()  # (n_connections, n_freqs)
        assert np.all(vals >= -1e-6), (
            f"{name}: values below 0 found: min={vals.min():.6f}"
        )
        assert np.all(vals <= 1.0 + 1e-6), (
            f"{name}: values above 1 found: max={vals.max():.6f}"
        )


# ---------------------------------------------------------------------------
# Connectivity Visualization Tests
# ---------------------------------------------------------------------------


def test_plot_connectivity_circle_returns_png(coherence_con):
    """plot_connectivity_circle must return a base64 PNG data URI."""
    result = _execute_plot_connectivity_circle(
        coherence_con, {"n_lines": 6, "colormap": "hot"}
    )
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,"), (
        f"Output not a PNG URI: {result[:60]}"
    )
    assert len(result) > 1000, f"PNG suspiciously short: {len(result)} chars"


def test_plot_connectivity_matrix_returns_png(coherence_con):
    """plot_connectivity_matrix must return a base64 PNG data URI."""
    result = _execute_plot_connectivity_matrix(
        coherence_con, {"colormap": "RdYlBu_r"}
    )
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,"), (
        f"Output not a PNG URI: {result[:60]}"
    )
    assert len(result) > 1000, f"PNG suspiciously short: {len(result)} chars"


# ---------------------------------------------------------------------------
# Statistics Tests
# ---------------------------------------------------------------------------


def test_cluster_permutation_test_returns_metrics(synthetic_epochs):
    """cluster_permutation_test must return a dict."""
    result = _execute_cluster_permutation_test(
        synthetic_epochs,
        {
            "tmin_ms": 0.0,
            "tmax_ms": 500.0,
            "n_permutations": 200,   # small for speed in tests
            "alpha": 0.05,
        },
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_cluster_permutation_test_keys_present(synthetic_epochs):
    """cluster_permutation_test result must contain the required keys."""
    result = _execute_cluster_permutation_test(
        synthetic_epochs,
        {
            "tmin_ms": 0.0,
            "tmax_ms": 500.0,
            "n_permutations": 200,
            "alpha": 0.05,
        },
    )
    required_keys = {
        "n_clusters",
        "n_significant_clusters",
        "cluster_p_values",
        "alpha",
        "n_permutations",
        "test_type",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in cluster test result: {missing}"

    # Type checks
    assert isinstance(result["n_clusters"], int)
    assert isinstance(result["cluster_p_values"], list)
    assert result["test_type"] == "permutation_cluster_1samp"


def test_compute_t_test_returns_metrics(synthetic_epochs):
    """compute_t_test must return a dict."""
    result = _execute_compute_t_test(
        synthetic_epochs,
        {
            "channel": "Cz",
            "tmin_ms": 0.0,
            "tmax_ms": 500.0,
            "popmean": 0.0,
        },
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_compute_t_test_keys_present(synthetic_epochs):
    """compute_t_test result must contain the required keys."""
    result = _execute_compute_t_test(
        synthetic_epochs,
        {
            "channel": "Cz",
            "tmin_ms": 0.0,
            "tmax_ms": 500.0,
            "popmean": 0.0,
        },
    )
    required_keys = {
        "t_statistic",
        "p_value",
        "degrees_of_freedom",
        "n_epochs",
        "mean_amplitude_uv",
        "channel",
        "test_type",
    }
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in t-test result: {missing}"

    assert result["channel"] == "Cz"
    assert result["n_epochs"] == _N_EPOCHS
    assert result["degrees_of_freedom"] == _N_EPOCHS - 1
    # p-value must be in (0, 1]
    assert 0.0 < result["p_value"] <= 1.0, (
        f"p-value out of range: {result['p_value']}"
    )


def test_compute_t_test_invalid_channel_raises(synthetic_epochs):
    """A channel not in the epochs must raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        _execute_compute_t_test(
            synthetic_epochs,
            {
                "channel": "NONEXISTENT",
                "tmin_ms": 0.0,
                "tmax_ms": 500.0,
                "popmean": 0.0,
            },
        )


def test_apply_fdr_single_p_value():
    """apply_fdr_correction accepts a single p_value key from compute_t_test."""
    metrics = {
        "t_statistic": 2.5,
        "p_value": 0.02,
        "test_type": "ttest_1samp",
    }
    result = _execute_apply_fdr_correction(metrics, {"alpha": 0.05})
    assert "corrected_p_values" in result
    assert "reject_h0" in result
    assert "fdr_alpha" in result
    assert result["fdr_alpha"] == 0.05
    assert result["n_tested"] == 1
    # Original keys must be preserved
    assert result["t_statistic"] == 2.5
    assert result["test_type"] == "ttest_1samp"


def test_apply_fdr_p_values_list():
    """apply_fdr_correction accepts a p_values list (multiple tests)."""
    metrics = {
        "p_values": [0.001, 0.04, 0.3, 0.8, 0.02],
        "test_type": "multi",
    }
    result = _execute_apply_fdr_correction(metrics, {"alpha": 0.05})
    assert len(result["corrected_p_values"]) == 5
    assert len(result["reject_h0"]) == 5
    assert all(isinstance(r, bool) for r in result["reject_h0"])
    # The two smallest p-values (0.001, 0.02) should be rejected at FDR 0.05
    assert result["n_rejected"] >= 1


def test_apply_fdr_cluster_p_values():
    """apply_fdr_correction accepts cluster_p_values from cluster permutation test."""
    metrics = {
        "n_clusters": 3,
        "cluster_p_values": [0.01, 0.04, 0.5],
        "test_type": "permutation_cluster_1samp",
    }
    result = _execute_apply_fdr_correction(metrics, {"alpha": 0.05})
    assert len(result["corrected_p_values"]) == 3
    assert result["n_clusters"] == 3  # original keys preserved


def test_apply_fdr_invalid_input_raises():
    """apply_fdr_correction must raise ValueError if no p-value keys found."""
    metrics = {"mean_amplitude_uv": 1.5, "channel": "Cz"}
    with pytest.raises(ValueError, match="p_value"):
        _execute_apply_fdr_correction(metrics, {"alpha": 0.05})


# ---------------------------------------------------------------------------
# Handle Type Test
# ---------------------------------------------------------------------------


def test_connectivity_in_valid_handle_types():
    """'connectivity' must be registered in VALID_HANDLE_TYPES (added in Tier 2)."""
    assert "connectivity" in VALID_HANDLE_TYPES, (
        f"'connectivity' not found in VALID_HANDLE_TYPES: {sorted(VALID_HANDLE_TYPES)}"
    )


# ---------------------------------------------------------------------------
# Fixtures for new Tier 2 straggler nodes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_evoked() -> mne.Evoked:
    """
    Synthetic Evoked with a simple sine-wave signal at Pz (simulated ERP bump)
    and noise on other channels.  Baseline is t=-0.2 to t=0.
    """
    rng = np.random.default_rng(7)
    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    n_ch = len(ch_names)
    sfreq = 250.0
    tmin = -0.2
    n_times = int((0.8 - tmin) * sfreq)   # 250 samples covering -200 to +800 ms
    times = np.linspace(tmin, 0.8, n_times)

    data = rng.standard_normal((n_ch, n_times)) * 2e-6   # ~2 µV noise baseline
    # Add a fake P300-like bump on Pz (channel index 2) at ~300 ms
    t_peak = 0.3
    sigma = 0.05
    bump = 10e-6 * np.exp(-0.5 * ((times - t_peak) / sigma) ** 2)
    data[2, :] += bump

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    evoked = mne.EvokedArray(data, info, tmin=tmin, nave=20, verbose=False)
    return evoked


# ---------------------------------------------------------------------------
# compute_envelope_correlation tests
# ---------------------------------------------------------------------------

def test_compute_envelope_correlation_returns_connectivity_matrix(synthetic_epochs):
    """compute_envelope_correlation must return a ConnectivityMatrix instance."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    assert isinstance(result, ConnectivityMatrix), (
        f"Expected ConnectivityMatrix, got {type(result)}"
    )


def test_compute_envelope_correlation_shape(synthetic_epochs):
    """AEC matrix must be square with side == n_channels."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    n_ch = len(_CH_NAMES)
    assert result.matrix.shape == (n_ch, n_ch), (
        f"Expected ({n_ch}, {n_ch}), got {result.matrix.shape}"
    )


def test_compute_envelope_correlation_symmetric(synthetic_epochs):
    """AEC matrix must be symmetric."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    assert np.allclose(result.matrix, result.matrix.T, atol=1e-10), (
        "AEC matrix is not symmetric"
    )


def test_compute_envelope_correlation_diagonal_ones(synthetic_epochs):
    """Diagonal (self-correlation) must be exactly 1."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    diag = np.diag(result.matrix)
    assert np.allclose(diag, 1.0, atol=1e-10), (
        f"Diagonal not all 1.0: {diag}"
    )


def test_compute_envelope_correlation_values_in_range(synthetic_epochs):
    """AEC values must lie in [-1, 1] (Pearson correlation range)."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    assert np.all(result.matrix >= -1.0 - 1e-8), "AEC values below -1"
    assert np.all(result.matrix <= 1.0 + 1e-8), "AEC values above 1"


def test_compute_envelope_correlation_method_label(synthetic_epochs):
    """ConnectivityMatrix.method must be 'aec'."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    assert result.method == "aec"


def test_compute_envelope_correlation_names_match(synthetic_epochs):
    """ConnectivityMatrix.names must match epoch channel names."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    assert result.names == _CH_NAMES


def test_compute_envelope_correlation_get_data_dense(synthetic_epochs):
    """get_data(output='dense') must return shape (n_ch, n_ch, 1)."""
    result = _execute_compute_envelope_correlation(synthetic_epochs, {})
    dense = result.get_data(output="dense")
    n_ch = len(_CH_NAMES)
    assert dense.shape == (n_ch, n_ch, 1), (
        f"Expected ({n_ch}, {n_ch}, 1), got {dense.shape}"
    )


def test_aec_matrix_plot_returns_png(synthetic_epochs):
    """plot_connectivity_matrix must work with ConnectivityMatrix (AEC) input."""
    aec = _execute_compute_envelope_correlation(synthetic_epochs, {})
    result = _execute_plot_connectivity_matrix(aec, {"colormap": "viridis"})
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,"), (
        f"AEC matrix plot not a PNG URI: {result[:60]}"
    )
    assert len(result) > 1000


def test_aec_circle_plot_returns_png(synthetic_epochs):
    """plot_connectivity_circle must work with ConnectivityMatrix (AEC) input."""
    aec = _execute_compute_envelope_correlation(synthetic_epochs, {})
    result = _execute_plot_connectivity_circle(aec, {"n_lines": 4, "colormap": "hot"})
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,"), (
        f"AEC circle plot not a PNG URI: {result[:60]}"
    )
    assert len(result) > 1000


# ---------------------------------------------------------------------------
# compute_noise_floor tests
# ---------------------------------------------------------------------------

def test_compute_noise_floor_returns_dict(synthetic_evoked):
    """compute_noise_floor must return a dict."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


def test_compute_noise_floor_keys_present(synthetic_evoked):
    """Noise floor result must contain all required keys."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    required = {
        "noise_floor_global_uv",
        "noise_floor_max_uv",
        "noise_floor_min_uv",
        "noise_floor_mean_uv",
        "worst_channel",
        "best_channel",
        "n_channels",
        "baseline_tmin_ms",
        "baseline_tmax_ms",
        "measurement",
    }
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"


def test_compute_noise_floor_positive(synthetic_evoked):
    """Noise floor values must be positive."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    assert result["noise_floor_global_uv"] > 0, "Global noise floor must be > 0"
    assert result["noise_floor_max_uv"] >= result["noise_floor_min_uv"], (
        "max noise floor must be >= min"
    )


def test_compute_noise_floor_snr_positive(synthetic_evoked):
    """SNR must be positive (P300 bump should exceed noise floor)."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    assert result["snr_db"] is not None, "SNR should not be None when post-stim data exists"
    assert result["snr_db"] > 0, (
        f"Expected positive SNR for a simulated P300, got {result['snr_db']}"
    )


def test_compute_noise_floor_channel_names_valid(synthetic_evoked):
    """worst_channel and best_channel must be valid channel names."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    valid_channels = synthetic_evoked.ch_names
    assert result["worst_channel"] in valid_channels, (
        f"worst_channel '{result['worst_channel']}' not in {valid_channels}"
    )
    assert result["best_channel"] in valid_channels, (
        f"best_channel '{result['best_channel']}' not in {valid_channels}"
    )


def test_compute_noise_floor_invalid_window_raises(synthetic_evoked):
    """A baseline window outside the Evoked time range must raise ValueError."""
    with pytest.raises(ValueError, match="No time points found"):
        _execute_compute_noise_floor(
            synthetic_evoked, {"tmin_ms": 1000.0, "tmax_ms": 2000.0}
        )


def test_compute_noise_floor_measurement_label(synthetic_evoked):
    """measurement key must equal 'noise_floor'."""
    result = _execute_compute_noise_floor(
        synthetic_evoked, {"tmin_ms": -200.0, "tmax_ms": 0.0}
    )
    assert result["measurement"] == "noise_floor"
