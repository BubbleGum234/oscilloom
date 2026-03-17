"""
backend/tests/test_multimodal.py

Tests for Tier 5 — Multimodal (MEG, fNIRS, BCI) nodes.
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from backend.registry import NODE_REGISTRY
from backend.registry.node_descriptor import VALID_HANDLE_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FNIRS_SFREQ = 10.0
_FNIRS_DURATION = 60.0


def _make_fnirs_raw(
    n_sources: int = 2,
    sfreq: float = _FNIRS_SFREQ,
    duration: float = _FNIRS_DURATION,
) -> mne.io.RawArray:
    """
    Create synthetic fNIRS raw data with proper wavelength info.

    Channel naming: S{n}_D{n} {wavelength} (760nm and 850nm per pair).
    Source-detector distance ~3cm (realistic for fNIRS).
    """
    ch_names = []
    ch_types = []
    for s in range(1, n_sources + 1):
        ch_names.extend([f"S{s}_D{s} 760", f"S{s}_D{s} 850"])
        ch_types.extend(["fnirs_cw_amplitude", "fnirs_cw_amplitude"])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set source/detector positions + wavelength in loc array
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        pair = i // 2
        # Source position (meters) — spread along x-axis
        loc[0:3] = [0.05 * pair, 0.0, 0.1]
        # Detector position — 3cm away (realistic fNIRS distance)
        loc[3:6] = [0.05 * pair + 0.03, 0.0, 0.1]
        # Wavelength at index 9
        loc[9] = 760.0 if (i % 2 == 0) else 850.0
        ch["loc"] = loc

    n_times = int(sfreq * duration)
    rng = np.random.default_rng(42)
    # Positive values required for optical_density (log transform)
    data = rng.uniform(0.5, 1.5, (len(ch_names), n_times))
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _make_fnirs_with_events(
    n_sources: int = 2,
    sfreq: float = _FNIRS_SFREQ,
    duration: float = _FNIRS_DURATION,
) -> mne.io.RawArray:
    """fNIRS raw with stimulus annotations (needed for compute_hrf)."""
    raw = _make_fnirs_raw(n_sources, sfreq, duration)
    # Add 3 stimulus events at 10s, 25s, 40s
    raw.annotations.append(onset=[10.0, 25.0, 40.0], duration=[0.0] * 3,
                           description=["stimulus"] * 3)
    return raw


_BCI_SFREQ = 250.0
_BCI_N_CHANNELS = 5
_BCI_N_EPOCHS = 40
_BCI_EPOCH_DURATION = 2.0


def _make_bci_epochs(
    n_epochs: int = _BCI_N_EPOCHS,
    n_channels: int = _BCI_N_CHANNELS,
    sfreq: float = _BCI_SFREQ,
    duration: float = _BCI_EPOCH_DURATION,
) -> mne.EpochsArray:
    """
    Create synthetic 2-class BCI epochs (motor imagery style).

    Class 1 ("left") has amplified channel 0 — gives CSP/LDA something to
    discriminate, so accuracy is above chance.
    """
    info = mne.create_info(
        ch_names=[f"C{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    rng = np.random.default_rng(42)
    n_times = int(sfreq * duration)
    data = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6

    # Make class 1 distinguishable from class 2
    half = n_epochs // 2
    data[:half, 0, :] *= 3.0  # amplify channel 0 for class 1

    events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.array([1] * half + [2] * (n_epochs - half)),
    ])
    epochs = mne.EpochsArray(
        data, info, events=events,
        event_id={"left": 1, "right": 2},
        verbose=False,
    )
    return epochs


@pytest.fixture(scope="module")
def fnirs_raw() -> mne.io.RawArray:
    return _make_fnirs_raw()


@pytest.fixture(scope="module")
def fnirs_with_events() -> mne.io.RawArray:
    return _make_fnirs_with_events()


@pytest.fixture(scope="module")
def bci_epochs() -> mne.EpochsArray:
    return _make_bci_epochs()


# ---------------------------------------------------------------------------
# Handle type validation
# ---------------------------------------------------------------------------

class TestHandleTypes:

    def test_raw_fnirs_in_valid_handle_types(self):
        assert "raw_fnirs" in VALID_HANDLE_TYPES

    def test_features_in_valid_handle_types(self):
        assert "features" in VALID_HANDLE_TYPES


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

class TestTier5Registry:

    TIER5_NODES = [
        "snirf_loader", "compute_optical_density", "beer_lambert_transform",
        "compute_hrf", "plot_fnirs_signal",
        "maxwell_filter", "apply_ssp",
        "compute_csp", "extract_epoch_features", "classify_lda",
        "plot_roc_curve",
    ]

    def test_all_tier5_nodes_registered(self):
        for node_type in self.TIER5_NODES:
            assert node_type in NODE_REGISTRY, f"{node_type} not in NODE_REGISTRY"

    def test_tier5_node_count(self):
        registered = [n for n in self.TIER5_NODES if n in NODE_REGISTRY]
        assert len(registered) == 11

    def test_total_node_count(self):
        # >= 72 because user-created compound nodes may also be loaded
        assert len(NODE_REGISTRY) >= 72

    def test_all_handle_types_valid(self):
        for node_type, desc in NODE_REGISTRY.items():
            for h in desc.inputs + desc.outputs:
                assert h.type in VALID_HANDLE_TYPES, (
                    f"{node_type}: handle '{h.id}' has invalid type '{h.type}'"
                )


# ---------------------------------------------------------------------------
# fNIRS nodes
# ---------------------------------------------------------------------------

class TestSnirfLoader:

    def test_passthrough(self, fnirs_raw):
        from backend.registry.nodes.fnirs import _execute_snirf_loader

        result = _execute_snirf_loader(fnirs_raw, {})
        assert result is fnirs_raw

    def test_empty_path_raises(self):
        from backend.registry.nodes.fnirs import _execute_snirf_loader

        with pytest.raises(ValueError, match="SNIRF Loader requires a file"):
            _execute_snirf_loader(None, {"file_path": ""})


class TestComputeOpticalDensity:

    def test_returns_raw(self, fnirs_raw):
        from backend.registry.nodes.fnirs import _execute_compute_optical_density

        result = _execute_compute_optical_density(fnirs_raw, {})
        assert isinstance(result, mne.io.BaseRaw)

    def test_channel_type_changes(self, fnirs_raw):
        from backend.registry.nodes.fnirs import _execute_compute_optical_density

        result = _execute_compute_optical_density(fnirs_raw, {})
        ch_types = result.get_channel_types()
        assert all(t == "fnirs_od" for t in ch_types)

    def test_does_not_mutate_input(self, fnirs_raw):
        from backend.registry.nodes.fnirs import _execute_compute_optical_density

        original_types = fnirs_raw.get_channel_types()
        _execute_compute_optical_density(fnirs_raw, {})
        assert fnirs_raw.get_channel_types() == original_types


class TestBeerLambertTransform:

    def test_returns_raw(self, fnirs_raw):
        from backend.registry.nodes.fnirs import (
            _execute_compute_optical_density,
            _execute_beer_lambert_transform,
        )

        od = _execute_compute_optical_density(fnirs_raw, {})
        result = _execute_beer_lambert_transform(od, {"ppf": 6.0})
        assert isinstance(result, mne.io.BaseRaw)

    def test_channel_types_hbo_hbr(self, fnirs_raw):
        from backend.registry.nodes.fnirs import (
            _execute_compute_optical_density,
            _execute_beer_lambert_transform,
        )

        od = _execute_compute_optical_density(fnirs_raw, {})
        result = _execute_beer_lambert_transform(od, {"ppf": 6.0})
        ch_types = result.get_channel_types()
        assert set(ch_types) == {"hbo", "hbr"}

    def test_does_not_mutate_input(self, fnirs_raw):
        from backend.registry.nodes.fnirs import (
            _execute_compute_optical_density,
            _execute_beer_lambert_transform,
        )

        od = _execute_compute_optical_density(fnirs_raw, {})
        original_types = od.get_channel_types()
        _execute_beer_lambert_transform(od, {"ppf": 6.0})
        assert od.get_channel_types() == original_types


class TestComputeHrf:

    def test_returns_evoked(self, fnirs_with_events):
        from backend.registry.nodes.fnirs import (
            _execute_compute_optical_density,
            _execute_beer_lambert_transform,
            _execute_compute_hrf,
        )

        od = _execute_compute_optical_density(fnirs_with_events, {})
        haemo = _execute_beer_lambert_transform(od, {"ppf": 6.0})
        result = _execute_compute_hrf(haemo, {"tmin": -2.0, "tmax": 5.0})
        assert isinstance(result, mne.Evoked)

    def test_raises_without_events(self, fnirs_raw):
        from backend.registry.nodes.fnirs import (
            _execute_compute_optical_density,
            _execute_beer_lambert_transform,
            _execute_compute_hrf,
        )

        od = _execute_compute_optical_density(fnirs_raw, {})
        haemo = _execute_beer_lambert_transform(od, {"ppf": 6.0})
        with pytest.raises(ValueError, match="No events"):
            _execute_compute_hrf(haemo, {"tmin": -2.0, "tmax": 5.0})


class TestPlotFnirsSignal:

    def test_returns_base64_png(self, fnirs_raw):
        from backend.registry.nodes.fnirs import _execute_plot_fnirs_signal

        result = _execute_plot_fnirs_signal(fnirs_raw, {
            "duration_s": 10.0, "n_channels": 4,
        })
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 1000  # Non-trivial image


# ---------------------------------------------------------------------------
# MEG nodes
# ---------------------------------------------------------------------------

class TestMaxwellFilter:

    def test_raises_on_pure_eeg(self):
        """maxwell_filter requires MEG data with device info."""
        from backend.registry.nodes.meg import _execute_maxwell_filter

        # Create pure EEG data — maxwell_filter should raise
        info = mne.create_info(["EEG1", "EEG2"], 256.0, ch_types="eeg")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((2, 512)) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)

        with pytest.raises((ValueError, RuntimeError)):
            _execute_maxwell_filter(raw, {"st_duration": 0})


class TestApplySSP:

    def test_rejects_eeg_only_data(self):
        """SSP requires MEG channels — EEG-only should raise ValueError."""
        from backend.registry.nodes.meg import _execute_apply_ssp

        info = mne.create_info(
            [f"EEG{i:03d}" for i in range(5)], 256.0, ch_types="eeg",
        )
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 2560)) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)

        with pytest.raises(ValueError, match="SSP Projectors require MEG"):
            _execute_apply_ssp(raw, {"n_eeg": 2, "n_mag": 0, "n_grad": 0})


# ---------------------------------------------------------------------------
# BCI nodes
# ---------------------------------------------------------------------------

class TestComputeCSP:

    def test_returns_features_dict(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_compute_csp

        result = _execute_compute_csp(bci_epochs, {"n_components": 4})
        assert isinstance(result, dict)
        assert "X" in result
        assert "labels" in result
        assert "label_names" in result

    def test_feature_matrix_shape(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_compute_csp

        result = _execute_compute_csp(bci_epochs, {"n_components": 4})
        X = np.array(result["X"])
        assert X.shape == (_BCI_N_EPOCHS, 4)

    def test_labels_length(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_compute_csp

        result = _execute_compute_csp(bci_epochs, {"n_components": 4})
        assert len(result["labels"]) == _BCI_N_EPOCHS

    def test_label_names_present(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_compute_csp

        result = _execute_compute_csp(bci_epochs, {"n_components": 4})
        assert 1 in result["label_names"] or "1" in result["label_names"]

    def test_requires_two_classes(self):
        from backend.registry.nodes.bci import _execute_compute_csp

        # Single-class epochs
        info = mne.create_info(["C0", "C1"], 250.0, ch_types="eeg")
        data = np.random.default_rng(42).standard_normal((10, 2, 500)) * 1e-6
        events = np.column_stack([
            np.arange(10) * 500, np.zeros(10, dtype=int), np.ones(10, dtype=int),
        ])
        epochs = mne.EpochsArray(data, info, events=events, verbose=False)

        with pytest.raises(ValueError, match="at least 2 event classes"):
            _execute_compute_csp(epochs, {"n_components": 2})


class TestExtractEpochFeatures:

    def test_returns_features_dict(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_extract_epoch_features

        result = _execute_extract_epoch_features(bci_epochs, {})
        assert isinstance(result, dict)
        assert "X" in result
        assert "labels" in result

    def test_feature_matrix_shape(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_extract_epoch_features

        result = _execute_extract_epoch_features(bci_epochs, {})
        X = np.array(result["X"])
        # 3 features per channel: variance, mobility, complexity
        assert X.shape == (_BCI_N_EPOCHS, _BCI_N_CHANNELS * 3)

    def test_n_features_matches(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_extract_epoch_features

        result = _execute_extract_epoch_features(bci_epochs, {})
        assert result["n_features"] == _BCI_N_CHANNELS * 3

    def test_no_nan_values(self, bci_epochs):
        from backend.registry.nodes.bci import _execute_extract_epoch_features

        result = _execute_extract_epoch_features(bci_epochs, {})
        X = np.array(result["X"])
        assert not np.any(np.isnan(X))


class TestClassifyLDA:

    def test_returns_metrics_dict(self, bci_epochs):
        from backend.registry.nodes.bci import (
            _execute_extract_epoch_features,
            _execute_classify_lda,
        )

        features = _execute_extract_epoch_features(bci_epochs, {})
        result = _execute_classify_lda(features, {"n_folds": 5})
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "n_folds" in result
        assert "n_classes" in result

    def test_accuracy_range(self, bci_epochs):
        from backend.registry.nodes.bci import (
            _execute_extract_epoch_features,
            _execute_classify_lda,
        )

        features = _execute_extract_epoch_features(bci_epochs, {})
        result = _execute_classify_lda(features, {"n_folds": 5})
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_roc_data_present_binary(self, bci_epochs):
        from backend.registry.nodes.bci import (
            _execute_extract_epoch_features,
            _execute_classify_lda,
        )

        features = _execute_extract_epoch_features(bci_epochs, {})
        result = _execute_classify_lda(features, {"n_folds": 5})
        assert result["roc"] is not None
        assert "fpr" in result["roc"]
        assert "tpr" in result["roc"]
        assert "auc" in result["roc"]
        assert 0.0 <= result["roc"]["auc"] <= 1.0

    def test_class_labels(self, bci_epochs):
        from backend.registry.nodes.bci import (
            _execute_extract_epoch_features,
            _execute_classify_lda,
        )

        features = _execute_extract_epoch_features(bci_epochs, {})
        result = _execute_classify_lda(features, {"n_folds": 5})
        assert result["n_classes"] == 2
        assert len(result["class_labels"]) == 2

    def test_too_few_samples_raises(self):
        from backend.registry.nodes.bci import _execute_classify_lda

        features = {
            "X": [[1.0], [2.0]],
            "labels": [1, 2],
            "label_names": {},
        }
        with pytest.raises(ValueError, match="at least"):
            _execute_classify_lda(features, {"n_folds": 5})

    def test_csp_to_lda_pipeline(self, bci_epochs):
        """Full CSP → LDA pipeline."""
        from backend.registry.nodes.bci import (
            _execute_compute_csp,
            _execute_classify_lda,
        )

        csp_features = _execute_compute_csp(bci_epochs, {"n_components": 4})
        result = _execute_classify_lda(csp_features, {"n_folds": 5})
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["roc"] is not None


class TestPlotROCCurve:

    def test_returns_base64_png(self, bci_epochs):
        from backend.registry.nodes.bci import (
            _execute_extract_epoch_features,
            _execute_classify_lda,
            _execute_plot_roc_curve,
        )

        features = _execute_extract_epoch_features(bci_epochs, {})
        metrics = _execute_classify_lda(features, {"n_folds": 5})
        result = _execute_plot_roc_curve(metrics, {})
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 1000

    def test_raises_without_roc_data(self):
        from backend.registry.nodes.bci import _execute_plot_roc_curve

        metrics = {"accuracy": 0.8, "roc": None}
        with pytest.raises(ValueError, match="ROC data not available"):
            _execute_plot_roc_curve(metrics, {})
