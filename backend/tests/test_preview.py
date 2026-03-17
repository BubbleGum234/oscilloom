"""
backend/tests/test_preview.py

Tests for the inline signal preview generator (backend.preview).

The generate_preview() function accepts an arbitrary pipeline node output
and returns a base64 PNG data URI for supported types, or None for
unsupported / already-visual outputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from backend.preview import generate_preview


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw() -> mne.io.BaseRaw:
    """
    Synthetic 256 Hz, 10-channel, 10-second MNE Raw object.
    Matches the shared fixture described in conftest.py.
    """
    rng = np.random.default_rng(0)
    sfreq = 256.0
    n_channels = 10
    n_seconds = 10
    n_samples = int(sfreq * n_seconds)

    data = rng.standard_normal((n_channels, n_samples)) * 1e-6
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_BASE64_PREFIX = "data:image/png;base64,"


def test_preview_raw_returns_base64(raw: mne.io.BaseRaw) -> None:
    result = generate_preview(raw)
    assert result is not None
    assert result.startswith(_BASE64_PREFIX)
    assert len(result) > 100


def test_preview_spectrum_returns_base64(raw: mne.io.BaseRaw) -> None:
    spectrum = raw.compute_psd(method="welch", verbose=False)
    result = generate_preview(spectrum)
    assert result is not None
    assert result.startswith(_BASE64_PREFIX)


def test_preview_ndarray_1d_returns_base64() -> None:
    arr = np.random.randn(100)
    result = generate_preview(arr)
    assert result is not None
    assert result.startswith(_BASE64_PREFIX)


def test_preview_ndarray_2d_returns_base64() -> None:
    arr = np.random.randn(5, 100)
    result = generate_preview(arr)
    assert result is not None
    assert result.startswith(_BASE64_PREFIX)


def test_preview_plot_string_returns_none() -> None:
    result = generate_preview("data:image/png;base64,abc123")
    assert result is None


def test_preview_unknown_type_returns_none() -> None:
    result = generate_preview({"key": "value"})
    assert result is None


def test_preview_none_input_returns_none() -> None:
    result = generate_preview(None)
    assert result is None


def test_preview_never_raises() -> None:
    """generate_preview must never propagate exceptions — it returns None on failure."""
    class Bomb:
        def __getattr__(self, name: str):
            raise RuntimeError("boom")

    bomb = Bomb()

    result = generate_preview(bomb)
    assert result is None
