"""
backend/tests/test_tier1.py

Unit tests for all 13 Tier 1 nodes added in EXPANSION_PLAN.md §10.

Tests call execute_fn directly (not via the pipeline engine) to keep them fast,
focused, and independent of the routing layer.  All fixtures use in-memory
synthetic data — no disk I/O, no large dataset downloads.

Coverage map (matches §10 table):
  Loaders (3):        fif/brainvision/bdf passthrough
  Set Montage (2):    standard_1020 assigned, unknown-ch on_missing="warn"
  Interpolate (2):    bads cleared, no-bads passthrough
  Annotate (1):       annotation added with correct onset/duration/description
  Epoch by Time (2):  correct epoch length, overlap > non-overlap
  Equalize (1):       all conditions have same trial count
  Compute GFP (2):    1-D array, n_times long; all values ≥ 0
  ERP Peak (3):       keys present; peak within window; invalid channel raises
  Difference Wave (2): same shape as input; same-condition → near-zero
  Plot Comparison (1): returns PNG data URI
  Plot GFP (1):        returns PNG data URI
  Handle type (1):     "metrics" in VALID_HANDLE_TYPES
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from backend.registry.node_descriptor import VALID_HANDLE_TYPES
from backend.registry.nodes.io_extended import (
    _execute_fif_loader,
    _execute_brainvision_loader,
    _execute_bdf_loader,
    _execute_ant_loader,
)
from backend.registry.nodes.preprocessing import (
    _execute_set_montage,
    _execute_interpolate_bad_channels,
    _execute_annotate_artifacts,
)
from backend.registry.nodes.epoching import (
    _execute_epoch_by_time,
    _execute_equalize_event_counts,
)
from backend.registry.nodes.erp import (
    _execute_compute_gfp,
    _execute_detect_erp_peak,
    _execute_compute_difference_wave,
    _execute_plot_comparison_evoked,
    _execute_plot_gfp,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SFREQ = 250.0
# Channel names that exist in the standard_1020 montage:
_EEG_CH_NAMES = ["Fz", "Cz", "Pz", "O1", "O2", "Oz"]


@pytest.fixture(scope="module")
def raw_eeg() -> mne.io.RawArray:
    """
    30-second synthetic raw EEG with six standard-10-20 electrode names.
    No montage is set — add the montage fixture for tests that require positions.
    """
    rng = np.random.default_rng(0)
    n_ch = len(_EEG_CH_NAMES)
    n_samples = int(30 * _SFREQ)
    data = rng.standard_normal((n_ch, n_samples)) * 1e-6
    info = mne.create_info(
        ch_names=_EEG_CH_NAMES, sfreq=_SFREQ, ch_types=["eeg"] * n_ch
    )
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture(scope="module")
def raw_with_montage(raw_eeg: mne.io.RawArray) -> mne.io.BaseRaw:
    """Raw EEG with the standard_1020 montage already applied."""
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = raw_eeg.copy()
    raw.set_montage(montage, on_missing="warn", verbose=False)
    return raw


@pytest.fixture(scope="module")
def multi_condition_epochs(raw_eeg: mne.io.RawArray) -> mne.Epochs:
    """
    Epochs with two event types (event_id {"1": 1, "2": 2}).
    Condition "1" has 5 trials; condition "2" has 3 trials — unequal on purpose
    for the equalize_event_counts test.
    """
    sfreq = _SFREQ
    events = np.array(
        [
            [int(2 * sfreq), 0, 1],
            [int(5 * sfreq), 0, 1],
            [int(8 * sfreq), 0, 1],
            [int(11 * sfreq), 0, 1],
            [int(14 * sfreq), 0, 1],
            [int(17 * sfreq), 0, 2],
            [int(20 * sfreq), 0, 2],
            [int(23 * sfreq), 0, 2],
        ]
    )
    return mne.Epochs(
        raw_eeg,
        events,
        event_id={"1": 1, "2": 2},
        tmin=-0.1,
        tmax=0.5,
        baseline=None,
        preload=True,
        verbose=False,
    )


@pytest.fixture(scope="module")
def evoked_with_peak() -> mne.EvokedArray:
    """
    Synthetic evoked with a sharp positive peak at 300 ms on channel Cz.
    Times span -200 ms to +800 ms (1001 samples at 1000 Hz for precise control).
    """
    sfreq = 1000.0
    tmin = -0.2
    tmax = 0.8
    times = np.arange(tmin, tmax + 1 / sfreq, 1 / sfreq)
    n_ch = len(_EEG_CH_NAMES)
    data = np.zeros((n_ch, len(times)))

    # Place a strong positive peak at exactly 300 ms on Cz (index 1)
    peak_idx = int((0.3 - tmin) * sfreq)
    data[1, peak_idx] = 10e-6  # 10 µV on Cz

    info = mne.create_info(
        ch_names=_EEG_CH_NAMES, sfreq=sfreq, ch_types=["eeg"] * n_ch
    )
    return mne.EvokedArray(data, info, tmin=tmin, verbose=False)


# ---------------------------------------------------------------------------
# 1. Loader Passthrough Tests
# ---------------------------------------------------------------------------


def test_fif_loader_passthrough(raw_eeg):
    """If input_data is already BaseRaw, fif_loader returns it unchanged."""
    result = _execute_fif_loader(raw_eeg, {})
    assert result is raw_eeg


def test_brainvision_loader_passthrough(raw_eeg):
    """If input_data is already BaseRaw, brainvision_loader returns it unchanged."""
    result = _execute_brainvision_loader(raw_eeg, {})
    assert result is raw_eeg


def test_bdf_loader_passthrough(raw_eeg):
    """If input_data is already BaseRaw, bdf_loader returns it unchanged."""
    result = _execute_bdf_loader(raw_eeg, {})
    assert result is raw_eeg


def test_ant_loader_passthrough(raw_eeg):
    """If input_data is already BaseRaw, ant_loader returns it unchanged."""
    result = _execute_ant_loader(raw_eeg, {})
    assert result is raw_eeg


def test_ant_loader_raises_without_file_path():
    """ant_loader raises ValueError when input_data is None and no file_path."""
    with pytest.raises(ValueError, match="ANT Neuro Loader requires a file path"):
        _execute_ant_loader(None, {})


def test_ant_loader_descriptor_in_registry():
    """ant_loader must be present in NODE_REGISTRY with correct fields."""
    from backend.registry import NODE_REGISTRY
    descriptor = NODE_REGISTRY.get("ant_loader")
    assert descriptor is not None, "ant_loader not found in NODE_REGISTRY"
    assert descriptor.category == "I/O"
    assert descriptor.display_name == "ANT Neuro Loader"
    assert descriptor.inputs == []
    assert len(descriptor.outputs) == 1
    assert descriptor.outputs[0].type == "raw_eeg"
    assert descriptor.outputs[0].id == "raw_out"
    assert callable(descriptor.execute_fn)


# ---------------------------------------------------------------------------
# 2. Set Montage Tests
# ---------------------------------------------------------------------------


def test_set_montage_standard_1020(raw_eeg):
    """
    After set_montage, every channel that is in the montage should have a
    non-None position (loc array with non-zero values).
    """
    result = _execute_set_montage(raw_eeg, {"montage": "standard_1020"})
    # At least one channel should have electrode position data
    positions = [result.info["chs"][i]["loc"][:3] for i in range(result.info["nchan"])]
    has_positions = any(not np.allclose(pos, 0.0) for pos in positions)
    assert has_positions, "No channel positions found after set_montage"


def test_set_montage_unknown_channel_warns(raw_eeg):
    """
    A raw with channel names NOT in the montage should not raise — on_missing="warn".
    """
    rng = np.random.default_rng(1)
    n_ch = 4
    n_samples = int(5 * _SFREQ)
    data = rng.standard_normal((n_ch, n_samples)) * 1e-6
    info = mne.create_info(
        ch_names=["EEG001", "EEG002", "EEG003", "EEG004"],
        sfreq=_SFREQ,
        ch_types=["eeg"] * n_ch,
    )
    raw_unknown = mne.io.RawArray(data, info, verbose=False)
    # Must not raise even though none of these names are in standard_1020
    result = _execute_set_montage(raw_unknown, {"montage": "standard_1020"})
    assert result is not raw_unknown  # returns a copy


# ---------------------------------------------------------------------------
# 3. Interpolate Bad Channels Tests
# ---------------------------------------------------------------------------


def test_interpolate_bad_channels_clears_bads(raw_with_montage):
    """
    With reset_bads=True (default), bads list should be empty after interpolation.
    """
    raw_with_bads = raw_with_montage.copy()
    raw_with_bads.info["bads"] = ["Fz"]  # mark one channel bad
    result = _execute_interpolate_bad_channels(raw_with_bads, {"reset_bads": True})
    assert result.info["bads"] == [], f"Expected empty bads, got {result.info['bads']}"


def test_interpolate_no_bads_passthrough(raw_with_montage):
    """
    If no channels are marked bad, the function returns a copy without
    touching any data — bads list remains empty.
    """
    raw_clean = raw_with_montage.copy()
    assert raw_clean.info["bads"] == []  # precondition
    result = _execute_interpolate_bad_channels(raw_clean, {"reset_bads": True})
    # Must be a different object (copy) but still have no bads
    assert result is not raw_clean
    assert result.info["bads"] == []


# ---------------------------------------------------------------------------
# 4. Annotate Artifacts Tests
# ---------------------------------------------------------------------------


def test_annotate_artifacts_adds_annotation(raw_eeg):
    """
    After annotate_artifacts, the raw should have a new annotation with the
    correct onset, duration, and description.
    """
    onset_s = 5.0
    duration_s = 2.0
    description = "BAD_muscle"
    result = _execute_annotate_artifacts(
        raw_eeg,
        {
            "onsets_s": str(onset_s),
            "durations_s": str(duration_s),
            "description": description,
        },
    )
    # Should have at least one annotation
    assert len(result.annotations) >= 1
    # Find our annotation
    found = any(
        abs(ann["onset"] - onset_s) < 0.01
        and abs(ann["duration"] - duration_s) < 0.01
        and ann["description"] == description
        for ann in result.annotations
    )
    assert found, f"Expected annotation at {onset_s}s not found. Got: {result.annotations}"


# ---------------------------------------------------------------------------
# 5. Epoch by Time Tests
# ---------------------------------------------------------------------------


def test_epoch_by_time_correct_length(raw_eeg):
    """
    With duration_s=2.0 and a 30-second recording, we expect ~15 epochs.
    Each epoch's duration should equal the requested duration.
    """
    duration_s = 2.0
    result = _execute_epoch_by_time(raw_eeg, {"duration_s": duration_s, "overlap_s": 0.0})
    # Epoch duration in samples: result.times covers the epoch window
    epoch_duration = result.times[-1] - result.times[0]
    assert abs(epoch_duration - duration_s) < 0.1, (
        f"Expected epoch duration ~{duration_s}s, got {epoch_duration:.3f}s"
    )
    # Should have roughly 30s / 2s = 15 epochs (±1 for boundary)
    assert 12 <= len(result) <= 16, f"Expected ~15 epochs, got {len(result)}"


def test_epoch_by_time_overlap(raw_eeg):
    """
    Overlapping windows should produce more epochs than non-overlapping windows
    from the same recording.
    """
    no_overlap = _execute_epoch_by_time(
        raw_eeg, {"duration_s": 2.0, "overlap_s": 0.0}
    )
    with_overlap = _execute_epoch_by_time(
        raw_eeg, {"duration_s": 2.0, "overlap_s": 1.0}
    )
    assert len(with_overlap) > len(no_overlap), (
        f"Expected overlap to give more epochs: {len(with_overlap)} vs {len(no_overlap)}"
    )


# ---------------------------------------------------------------------------
# 6. Equalize Event Counts Tests
# ---------------------------------------------------------------------------


def test_equalize_event_counts_balances(multi_condition_epochs):
    """
    After equalize_event_counts, every condition should have the same number
    of trials (equal to the smallest condition count before equalization).
    """
    # Precondition: unequal counts (5 vs 3)
    counts_before = {
        cond: len(multi_condition_epochs[cond])
        for cond in multi_condition_epochs.event_id
    }
    assert len(set(counts_before.values())) > 1, "Precondition: counts should be unequal"

    result = _execute_equalize_event_counts(
        multi_condition_epochs, {"method": "mintime"}
    )

    counts_after = {
        cond: len(result[cond]) for cond in result.event_id
    }
    assert len(set(counts_after.values())) == 1, (
        f"Expected equal counts after equalization, got: {counts_after}"
    )
    # All should equal the minimum of the original counts
    assert list(set(counts_after.values()))[0] == min(counts_before.values())


# ---------------------------------------------------------------------------
# 7. Compute GFP Tests
# ---------------------------------------------------------------------------


def test_compute_gfp_shape(evoked_with_peak):
    """GFP output is a 1-D array with length == evoked.n_times."""
    gfp = _execute_compute_gfp(evoked_with_peak, {})
    assert gfp.ndim == 1, f"Expected 1-D array, got shape {gfp.shape}"
    assert len(gfp) == evoked_with_peak.data.shape[1], (
        f"Expected {evoked_with_peak.data.shape[1]} samples, got {len(gfp)}"
    )


def test_compute_gfp_positive(evoked_with_peak):
    """All GFP values must be ≥ 0 (std is always non-negative)."""
    gfp = _execute_compute_gfp(evoked_with_peak, {})
    assert np.all(gfp >= 0), f"Negative GFP value found: min={gfp.min()}"


# ---------------------------------------------------------------------------
# 8. Detect ERP Peak Tests
# ---------------------------------------------------------------------------


def test_detect_erp_peak_returns_metrics_dict(evoked_with_peak):
    """Output must contain peak_latency_ms and peak_amplitude_uv keys."""
    params = {
        "channel": "Cz",
        "tmin_ms": 250.0,
        "tmax_ms": 500.0,
        "polarity": "positive",
    }
    result = _execute_detect_erp_peak(evoked_with_peak, params)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "peak_latency_ms" in result, f"Missing 'peak_latency_ms' key: {result}"
    assert "peak_amplitude_uv" in result, f"Missing 'peak_amplitude_uv' key: {result}"


def test_detect_erp_peak_in_window(evoked_with_peak):
    """The detected peak latency must fall within the specified search window."""
    tmin_ms, tmax_ms = 250.0, 500.0
    params = {
        "channel": "Cz",
        "tmin_ms": tmin_ms,
        "tmax_ms": tmax_ms,
        "polarity": "positive",
    }
    result = _execute_detect_erp_peak(evoked_with_peak, params)
    latency = result["peak_latency_ms"]
    assert tmin_ms <= latency <= tmax_ms, (
        f"Peak latency {latency:.1f} ms is outside window {tmin_ms}–{tmax_ms} ms"
    )
    # Synthetic peak was placed at 300 ms — should be near 300
    assert abs(latency - 300.0) < 5.0, (
        f"Expected peak near 300 ms, got {latency:.1f} ms"
    )


def test_detect_erp_peak_invalid_channel_raises(evoked_with_peak):
    """A channel name not present in the evoked must raise ValueError."""
    params = {
        "channel": "NONEXISTENT_CH",
        "tmin_ms": 250.0,
        "tmax_ms": 500.0,
        "polarity": "positive",
    }
    with pytest.raises(ValueError, match="not found"):
        _execute_detect_erp_peak(evoked_with_peak, params)


# ---------------------------------------------------------------------------
# 9. Compute Difference Wave Tests
# ---------------------------------------------------------------------------


def test_compute_difference_wave_shape(multi_condition_epochs):
    """The difference wave must have the same channel/time shape as one evoked."""
    params = {"condition_a": "1", "condition_b": "2"}
    result = _execute_compute_difference_wave(multi_condition_epochs, params)
    assert isinstance(result, mne.Evoked), f"Expected mne.Evoked, got {type(result)}"
    ref = multi_condition_epochs["1"].average()
    assert result.data.shape == ref.data.shape, (
        f"Shape mismatch: diff={result.data.shape} vs ref={ref.data.shape}"
    )


def test_compute_difference_wave_is_zero_same_conditions(multi_condition_epochs):
    """Subtracting a condition from itself (A − A) produces a near-zero evoked."""
    # Use both condition_a and condition_b as "1"
    params = {"condition_a": "1", "condition_b": "1"}
    result = _execute_compute_difference_wave(multi_condition_epochs, params)
    assert np.allclose(result.data, 0.0, atol=1e-15), (
        f"Expected near-zero difference wave, max abs = {np.abs(result.data).max():.2e}"
    )


# ---------------------------------------------------------------------------
# 10. Plot Comparison Evoked
# ---------------------------------------------------------------------------


def test_plot_comparison_evoked_returns_png(multi_condition_epochs):
    """plot_comparison_evoked must return a base64 PNG data URI."""
    params = {"conditions": "1,2", "channel": "Cz"}
    result = _execute_plot_comparison_evoked(multi_condition_epochs, params)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert result.startswith("data:image/png;base64,"), (
        f"Output is not a PNG data URI: {result[:60]}"
    )
    # Sanity check: non-trivial image (>1 KB of base64)
    assert len(result) > 1000, f"PNG data URI suspiciously short: {len(result)} chars"


# ---------------------------------------------------------------------------
# 11. Plot GFP
# ---------------------------------------------------------------------------


def test_plot_gfp_returns_png(evoked_with_peak):
    """plot_gfp must return a base64 PNG data URI."""
    result = _execute_plot_gfp(evoked_with_peak, {"highlight_peaks": True})
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert result.startswith("data:image/png;base64,"), (
        f"Output is not a PNG data URI: {result[:60]}"
    )
    assert len(result) > 1000, f"PNG data URI suspiciously short: {len(result)} chars"


# ---------------------------------------------------------------------------
# 12. Handle Type Registry
# ---------------------------------------------------------------------------


def test_metrics_in_valid_handle_types():
    """'metrics' must be registered in VALID_HANDLE_TYPES (added in Tier 1)."""
    assert "metrics" in VALID_HANDLE_TYPES, (
        f"'metrics' not found in VALID_HANDLE_TYPES: {sorted(VALID_HANDLE_TYPES)}"
    )
