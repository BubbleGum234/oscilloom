"""
backend/tests/test_sleep.py

Tests for Tier 6 — Sleep Analysis nodes (YASA).

Covers:
  - Registry registration (5 nodes, total count 73)
  - compute_sleep_stages: happy path, auto-detect channel, missing channel error
  - compute_sleep_architecture: happy path, missing hypnogram error, NaN handling
  - detect_spindles: happy path, no-spindles case, does not mutate input
  - detect_slow_oscillations: happy path, no-SO case, does not mutate input
  - plot_hypnogram: happy path, missing hypnogram error, show_stats toggle
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

# NumPy 2.0 compat (same as sleep.py)
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from backend.registry import NODE_REGISTRY
from backend.registry.nodes.sleep import (
    _execute_compute_sleep_stages,
    _execute_compute_sleep_architecture,
    _execute_detect_spindles,
    _execute_detect_slow_oscillations,
    _execute_plot_hypnogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_eeg(n_channels: int = 6, sfreq: float = 100.0, duration_s: float = 300.0):
    """Create a test Raw object. 6 ch, 100 Hz, 300 s (5 min = 10 × 30-s epochs)."""
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    n_samples = int(sfreq * duration_s)
    data = np.random.randn(n_channels, n_samples) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def _make_raw_with_slow_oscillations(sfreq: float = 100.0, duration_s: float = 300.0):
    """
    Create a test Raw with embedded slow oscillations (0.75 Hz, ~100 µV).
    Guarantees sw_detect will find events.
    """
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq
    # Large slow oscillation + small noise
    so_signal = np.sin(2 * np.pi * 0.75 * t) * 100e-6
    noise = np.random.randn(n_samples) * 2e-6
    data = (so_signal + noise).reshape(1, -1)
    info = mne.create_info(ch_names=["C3"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _make_hypnogram_metrics() -> dict:
    """Standard test hypnogram metrics dict (mimics compute_sleep_stages output)."""
    # 20 epochs × 30 s = 10 minutes
    # W W N1 N2 N2 N2 N3 N3 N2 R R N2 N2 N3 N3 N2 R R W W
    hypno_int = [0, 0, 1, 2, 2, 2, 3, 3, 2, 4, 4, 2, 2, 3, 3, 2, 4, 4, 0, 0]
    labels = ["W", "W", "N1", "N2", "N2", "N2", "N3", "N3", "N2",
              "R", "R", "N2", "N2", "N3", "N3", "N2", "R", "R", "W", "W"]
    return {
        "hypnogram": hypno_int,
        "hypnogram_labels": labels,
        "stage_counts": {"W": 4, "N1": 1, "N2": 7, "N3": 4, "R": 4},
        "n_epochs": 20,
        "epoch_duration_s": 30.0,
    }


# ===========================================================================
# Registry Tests
# ===========================================================================

class TestTier6Registry:
    TIER6_NODES = [
        "compute_sleep_stages",
        "compute_sleep_architecture",
        "detect_spindles",
        "detect_slow_oscillations",
        "plot_hypnogram",
    ]

    def test_all_tier6_nodes_registered(self):
        for node_type in self.TIER6_NODES:
            assert node_type in NODE_REGISTRY, f"{node_type} missing from registry"

    def test_tier6_node_count(self):
        registered = [n for n in self.TIER6_NODES if n in NODE_REGISTRY]
        assert len(registered) == 5

    def test_total_node_count(self):
        # >= 73 because user-created compound nodes may also be loaded
        assert len(NODE_REGISTRY) >= 73

    def test_all_handle_types_valid(self):
        from backend.registry.node_descriptor import VALID_HANDLE_TYPES
        for nt in self.TIER6_NODES:
            desc = NODE_REGISTRY[nt]
            for h in desc.inputs:
                assert h.type in VALID_HANDLE_TYPES, f"{nt} input {h.id} type '{h.type}' invalid"
            for h in desc.outputs:
                assert h.type in VALID_HANDLE_TYPES, f"{nt} output {h.id} type '{h.type}' invalid"

    def test_sleep_category(self):
        for nt in self.TIER6_NODES:
            assert NODE_REGISTRY[nt].category == "Sleep"


# ===========================================================================
# Compute Sleep Stages Tests
# ===========================================================================

class TestComputeSleepStages:
    def test_returns_metrics_dict(self):
        raw = _make_raw_eeg()
        result = _execute_compute_sleep_stages(raw, {})
        assert isinstance(result, dict)
        assert "hypnogram" in result
        assert "hypnogram_labels" in result
        assert "stage_counts" in result
        assert "n_epochs" in result
        assert result["epoch_duration_s"] == 30.0

    def test_hypnogram_length_matches_epochs(self):
        raw = _make_raw_eeg(duration_s=300.0)  # 10 epochs
        result = _execute_compute_sleep_stages(raw, {})
        assert result["n_epochs"] == len(result["hypnogram"])
        assert result["n_epochs"] == len(result["hypnogram_labels"])
        assert result["n_epochs"] == 10

    def test_valid_stage_values(self):
        raw = _make_raw_eeg()
        result = _execute_compute_sleep_stages(raw, {})
        valid_int = {0, 1, 2, 3, 4}
        valid_str = {"W", "N1", "N2", "N3", "R"}
        assert all(s in valid_int for s in result["hypnogram"])
        assert all(s in valid_str for s in result["hypnogram_labels"])

    def test_stage_counts_sum(self):
        raw = _make_raw_eeg()
        result = _execute_compute_sleep_stages(raw, {})
        total = sum(result["stage_counts"].values())
        assert total == result["n_epochs"]

    def test_auto_detect_eeg_channel(self):
        raw = _make_raw_eeg()
        result = _execute_compute_sleep_stages(raw, {"eeg_channel": ""})
        # Should auto-detect first EEG channel
        assert result["eeg_channel"] == "EEG000"

    def test_specified_eeg_channel(self):
        raw = _make_raw_eeg()
        result = _execute_compute_sleep_stages(raw, {"eeg_channel": "EEG002"})
        assert result["eeg_channel"] == "EEG002"

    def test_missing_channel_raises(self):
        raw = _make_raw_eeg()
        with pytest.raises(ValueError, match="not found"):
            _execute_compute_sleep_stages(raw, {"eeg_channel": "NONEXISTENT"})

    def test_does_not_mutate_input(self):
        raw = _make_raw_eeg()
        original = raw.get_data().copy()
        _execute_compute_sleep_stages(raw, {})
        np.testing.assert_array_equal(raw.get_data(), original)


# ===========================================================================
# Compute Sleep Architecture Tests
# ===========================================================================

class TestComputeSleepArchitecture:
    def test_returns_metrics_dict(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        assert isinstance(result, dict)

    def test_contains_aasm_keys(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        expected_keys = ["TIB", "SPT", "TST", "WASO", "SE", "SME", "SOL"]
        for key in expected_keys:
            assert key in result, f"Missing AASM key: {key}"

    def test_contains_stage_durations(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        for stage in ["WAKE", "N1", "N2", "N3", "REM"]:
            assert stage in result, f"Missing stage duration: {stage}"

    def test_contains_stage_percentages(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        for key in ["%N1", "%N2", "%N3", "%REM"]:
            assert key in result, f"Missing stage percentage: {key}"

    def test_tib_matches_expected(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        # 20 epochs × 30 s = 600 s = 10 min
        assert result["TIB"] == 10.0

    def test_sleep_efficiency_range(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        assert 0 <= result["SE"] <= 100

    def test_missing_hypnogram_raises(self):
        with pytest.raises(ValueError, match="hypnogram"):
            _execute_compute_sleep_architecture({"foo": "bar"}, {})

    def test_nan_values_become_none(self):
        """SOL_5min may be NaN for short recordings; should become None."""
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        # Values should be either numeric or None, never NaN
        for k, v in result.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"Key {k} is NaN — should be None"

    def test_n_epochs_preserved(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_compute_sleep_architecture(metrics, {})
        assert result["n_epochs"] == 20
        assert result["epoch_duration_s"] == 30.0


# ===========================================================================
# Detect Spindles Tests
# ===========================================================================

class TestDetectSpindles:
    def test_returns_metrics_dict(self):
        raw = _make_raw_eeg(n_channels=2, duration_s=300.0)
        result = _execute_detect_spindles(raw, {})
        assert isinstance(result, dict)
        assert "n_spindles" in result
        assert "density_per_min" in result
        assert "spindles" in result
        assert "freq_range_hz" in result

    def test_no_spindles_returns_zero(self):
        """With very short or flat data, spindles may not be found."""
        info = mne.create_info(ch_names=["C3"], sfreq=100, ch_types="eeg")
        # Flat data → no spindles
        data = np.zeros((1, 10000))
        raw = mne.io.RawArray(data, info, verbose=False)
        result = _execute_detect_spindles(raw, {})
        assert result["n_spindles"] == 0
        assert result["spindles"] == []

    def test_spindle_list_capped(self):
        """If more than 50 spindles, list should be capped."""
        raw = _make_raw_eeg(n_channels=2, duration_s=300.0)
        result = _execute_detect_spindles(raw, {})
        assert len(result["spindles"]) <= 50

    def test_custom_frequency_range(self):
        raw = _make_raw_eeg(n_channels=2, duration_s=300.0)
        result = _execute_detect_spindles(raw, {"freq_min": 11.0, "freq_max": 16.0})
        assert result["freq_range_hz"] == "11.0-16.0"

    def test_does_not_mutate_input(self):
        raw = _make_raw_eeg(n_channels=2, duration_s=300.0)
        original = raw.get_data().copy()
        _execute_detect_spindles(raw, {})
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_spindle_entry_fields(self):
        """When spindles are found, each entry should have expected fields."""
        raw = _make_raw_eeg(n_channels=2, duration_s=300.0)
        result = _execute_detect_spindles(raw, {})
        if result["n_spindles"] > 0:
            entry = result["spindles"][0]
            assert "onset_s" in entry
            assert "duration_s" in entry
            assert "channel" in entry
            assert "frequency_hz" in entry


# ===========================================================================
# Detect Slow Oscillations Tests
# ===========================================================================

class TestDetectSlowOscillations:
    def test_returns_metrics_dict(self):
        raw = _make_raw_with_slow_oscillations()
        result = _execute_detect_slow_oscillations(raw, {})
        assert isinstance(result, dict)
        assert "n_slow_oscillations" in result
        assert "density_per_min" in result
        assert "slow_oscillations" in result

    def test_detects_embedded_so(self):
        """With proper slow oscillations in data, should find events."""
        raw = _make_raw_with_slow_oscillations()
        result = _execute_detect_slow_oscillations(raw, {})
        assert result["n_slow_oscillations"] > 0

    def test_no_so_returns_zero(self):
        """Flat data → no slow oscillations."""
        info = mne.create_info(ch_names=["C3"], sfreq=100, ch_types="eeg")
        data = np.zeros((1, 30000))
        raw = mne.io.RawArray(data, info, verbose=False)
        result = _execute_detect_slow_oscillations(raw, {})
        assert result["n_slow_oscillations"] == 0
        assert result["slow_oscillations"] == []

    def test_so_entry_fields(self):
        raw = _make_raw_with_slow_oscillations()
        result = _execute_detect_slow_oscillations(raw, {})
        if result["n_slow_oscillations"] > 0:
            entry = result["slow_oscillations"][0]
            assert "onset_s" in entry
            assert "duration_s" in entry
            assert "channel" in entry
            assert "ptp_uv" in entry
            assert "neg_peak_uv" in entry

    def test_does_not_mutate_input(self):
        raw = _make_raw_with_slow_oscillations()
        original = raw.get_data().copy()
        _execute_detect_slow_oscillations(raw, {})
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_custom_params(self):
        raw = _make_raw_with_slow_oscillations()
        result = _execute_detect_slow_oscillations(raw, {
            "freq_min": 0.3,
            "freq_max": 1.5,
            "amp_neg_min": 30.0,
            "amp_neg_max": 250.0,
            "amp_ptp_min": 60.0,
            "amp_ptp_max": 400.0,
        })
        assert isinstance(result, dict)
        assert result["freq_range_hz"] == "0.3-1.5"


# ===========================================================================
# Plot Hypnogram Tests
# ===========================================================================

class TestPlotHypnogram:
    def test_returns_base64_png(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_plot_hypnogram(metrics, {})
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 1000

    def test_missing_hypnogram_raises(self):
        with pytest.raises(ValueError, match="hypnogram"):
            _execute_plot_hypnogram({"foo": "bar"}, {})

    def test_show_stats_true(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_plot_hypnogram(metrics, {"show_stats": True})
        assert result.startswith("data:image/png;base64,")

    def test_show_stats_false(self):
        metrics = _make_hypnogram_metrics()
        result = _execute_plot_hypnogram(metrics, {"show_stats": False})
        assert result.startswith("data:image/png;base64,")

    def test_single_epoch_hypnogram(self):
        """Edge case: single-epoch hypnogram should not crash."""
        metrics = {
            "hypnogram": [0],
            "hypnogram_labels": ["W"],
            "stage_counts": {"W": 1, "N1": 0, "N2": 0, "N3": 0, "R": 0},
            "n_epochs": 1,
            "epoch_duration_s": 30.0,
        }
        result = _execute_plot_hypnogram(metrics, {})
        assert result.startswith("data:image/png;base64,")

    def test_all_same_stage(self):
        """Hypnogram with all N2 should not crash."""
        metrics = {
            "hypnogram": [2] * 10,
            "hypnogram_labels": ["N2"] * 10,
            "stage_counts": {"W": 0, "N1": 0, "N2": 10, "N3": 0, "R": 0},
            "n_epochs": 10,
            "epoch_duration_s": 30.0,
        }
        result = _execute_plot_hypnogram(metrics, {})
        assert result.startswith("data:image/png;base64,")
