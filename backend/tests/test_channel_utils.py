"""
backend/tests/test_channel_utils.py

Tests for:
  - _channel_utils.resolve_channel / resolve_channel_optional
  - rename_channels node
  - Fuzzy channel matching in compute_asymmetry, detect_erp_peak,
    compute_t_test, compute_sleep_stages
  - match_case=False in set_montage
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from backend.registry.nodes._channel_utils import (
    resolve_channel,
    resolve_channel_optional,
    detect_naming_convention,
)


# ---------------------------------------------------------------------------
# resolve_channel unit tests
# ---------------------------------------------------------------------------

class TestResolveChannel:

    def test_exact_match(self):
        assert resolve_channel("F3", ["F3", "F4", "Cz"]) == "F3"

    def test_case_insensitive(self):
        assert resolve_channel("f3", ["F3", "F4", "Cz"]) == "F3"

    def test_case_insensitive_upper(self):
        assert resolve_channel("CZ", ["F3", "F4", "Cz"]) == "Cz"

    def test_substring_match_prefixed(self):
        """'F3' should match 'EEG F3' when it's the only substring match."""
        assert resolve_channel("F3", ["EEG F3", "EEG F4", "EEG Cz"]) == "EEG F3"

    def test_substring_match_with_suffix(self):
        """'F3' should match 'EEG F3-Ref' when it's the only substring match."""
        assert resolve_channel("F3", ["EEG F3-Ref", "EEG F4-Ref"]) == "EEG F3-Ref"

    def test_ambiguous_raises(self):
        """'F' matches both 'EEG F3' and 'EEG F4' — should raise."""
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_channel("F", ["EEG F3", "EEG F4", "EEG Cz"])

    def test_not_found_raises(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_channel("Pz", ["F3", "F4", "Cz"])

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_channel("", ["F3", "F4"])

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve_channel("   ", ["F3", "F4"])

    def test_whitespace_stripped(self):
        assert resolve_channel("  F3  ", ["F3", "F4"]) == "F3"

    def test_case_insensitive_substring(self):
        """'f3' should match 'EEG F3' via case-insensitive substring."""
        assert resolve_channel("f3", ["EEG F3", "EEG F4"]) == "EEG F3"

    def test_trailing_dot_physionet(self):
        """PhysioNet-style 'F3.' should match if user queries 'F3'."""
        assert resolve_channel("F3", ["F3.", "F4.", "Cz."]) == "F3."

    def test_exact_match_preferred_over_substring(self):
        """If exact match exists, prefer it even if substring also matches."""
        # "F3" is an exact match; "EEG F3" is a substring match — exact wins
        assert resolve_channel("F3", ["F3", "EEG F3", "F4"]) == "F3"

    # --- Normalized match tests (step 3) ---

    def test_normalized_strips_trailing_dots(self):
        """'C3' should match 'C3..' but NOT 'Fc3.' via normalized match."""
        assert resolve_channel("C3", ["Fc3.", "C3.."]) == "C3.."

    def test_normalized_strips_prefix_and_suffix(self):
        """'F3' should match 'EEG F3-Ref' via normalization."""
        assert resolve_channel("F3", ["EEG F3-Ref", "EEG F4-Ref", "EEG Cz-Ref"]) == "EEG F3-Ref"

    def test_normalized_physionet_multiple_dots(self):
        """PhysioNet files often have 'C3..' — should resolve cleanly."""
        channels = ["Fc3.", "C3..", "Fc4.", "C4.."]
        assert resolve_channel("C3", channels) == "C3.."
        assert resolve_channel("C4", channels) == "C4.."
        assert resolve_channel("Fc3", channels) == "Fc3."
        assert resolve_channel("Fc4", channels) == "Fc4."

    # --- Word-boundary match tests (step 4) ---

    def test_boundary_prevents_partial_electrode_match(self):
        """'C3' should NOT match 'FC3' via word-boundary — different electrode."""
        # FC3 and C3 are different channels; only FC3 is present
        with pytest.raises(ValueError, match="not found"):
            resolve_channel("C3", ["FC3", "FC4", "Cz"])

    def test_boundary_matches_with_separator(self):
        """'C3' should match 'EEG-C3' via word boundary (hyphen is a boundary)."""
        assert resolve_channel("C3", ["EEG-FC3", "EEG-C3", "EEG-C4"]) == "EEG-C3"


class TestResolveChannelOptional:

    def test_returns_none_on_miss(self):
        assert resolve_channel_optional("Pz", ["F3", "F4"]) is None

    def test_returns_none_on_ambiguous(self):
        assert resolve_channel_optional("F", ["EEG F3", "EEG F4"]) is None

    def test_returns_match(self):
        assert resolve_channel_optional("f3", ["F3", "F4"]) == "F3"


# ---------------------------------------------------------------------------
# rename_channels node tests
# ---------------------------------------------------------------------------

def _make_raw(ch_names: list[str], sfreq: float = 256.0) -> mne.io.RawArray:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    n_times = int(sfreq * 2)
    data = np.zeros((len(ch_names), n_times))
    return mne.io.RawArray(data, info, verbose=False)


class TestRenameChannels:

    def test_strip_eeg_prefix(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3", "EEG F4", "EEG Cz"])
        result = _execute_rename_channels(raw, {"strip_prefix": "EEG ", "strip_suffix": "", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4", "Cz"]

    def test_strip_ref_suffix(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["F3-Ref", "F4-Ref", "Cz-Ref"])
        result = _execute_rename_channels(raw, {"strip_prefix": "", "strip_suffix": "-Ref", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4", "Cz"]

    def test_combined_prefix_and_suffix(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3-Ref", "EEG F4-Ref"])
        result = _execute_rename_channels(raw, {"strip_prefix": "EEG ", "strip_suffix": "-Ref", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4"]

    def test_regex_strip_trailing_dot(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["F3.", "F4.", "Cz."])
        result = _execute_rename_channels(raw, {"strip_prefix": "", "strip_suffix": "", "regex_pattern": r"\.$", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4", "Cz"]

    def test_no_op_when_nothing_to_strip(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["F3", "F4"])
        result = _execute_rename_channels(raw, {"strip_prefix": "", "strip_suffix": "", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4"]

    def test_does_not_mutate_input(self):
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3", "EEG F4"])
        _execute_rename_channels(raw, {"strip_prefix": "EEG ", "strip_suffix": "", "regex_pattern": "", "regex_replacement": ""})
        assert raw.ch_names == ["EEG F3", "EEG F4"]

    def test_partial_prefix_match(self):
        """Only channels with the prefix get it stripped."""
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3", "EOG1"])
        result = _execute_rename_channels(raw, {"strip_prefix": "EEG ", "strip_suffix": "", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "EOG1"]

    def test_case_insensitive_prefix(self):
        """'eeg ' (lowercase) should still strip 'EEG ' prefix."""
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3", "EEG F4"])
        result = _execute_rename_channels(raw, {"strip_prefix": "eeg ", "strip_suffix": "", "regex_pattern": "", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4"]

    def test_regex_to_empty_leaves_channel_unchanged(self):
        """Regex that wipes entire name should leave channel as-is."""
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["F3", "F4"])
        result = _execute_rename_channels(raw, {"strip_prefix": "", "strip_suffix": "", "regex_pattern": ".*", "regex_replacement": ""})
        assert list(result.ch_names) == ["F3", "F4"]

    def test_duplicate_target_names_raises(self):
        """Renaming that produces duplicates should raise ValueError."""
        from backend.registry.nodes.preprocessing import _execute_rename_channels
        raw = _make_raw(["EEG F3", "EMG F3"])
        with pytest.raises(ValueError, match="duplicate"):
            _execute_rename_channels(raw, {"strip_prefix": "", "strip_suffix": "", "regex_pattern": r"^(EEG|EMG)\s+", "regex_replacement": ""})

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "rename_channels" in NODE_REGISTRY

    def test_descriptor_output_type(self):
        from backend.registry.nodes.preprocessing import RENAME_CHANNELS
        assert RENAME_CHANNELS.outputs[0].type == "filtered_eeg"


# ---------------------------------------------------------------------------
# Integration: fuzzy matching in affected nodes
# ---------------------------------------------------------------------------

def _make_spectrum(ch_names: list[str]) -> "mne.time_frequency.Spectrum":
    """Create a test spectrum with named channels."""
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    n_times = int(256.0 * 10)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(ch_names), n_times)) * 10e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw.compute_psd(method="welch", fmin=1.0, fmax=40.0, verbose=False)


class TestAsymmetryFuzzyMatch:
    """compute_asymmetry should resolve channels case-insensitively and via substring."""

    def test_case_insensitive(self):
        from backend.registry.nodes.clinical import _execute_compute_asymmetry
        spectrum = _make_spectrum(["F3", "F4", "Cz"])
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "f3", "right_channel": "f4"}
        )
        assert "asymmetry_index" in result

    def test_prefixed_channel(self):
        from backend.registry.nodes.clinical import _execute_compute_asymmetry
        spectrum = _make_spectrum(["EEG F3", "EEG F4", "EEG Cz"])
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "F4"}
        )
        assert "asymmetry_index" in result
        assert result["channel_left"] == "EEG F3"


class TestERPPeakFuzzyMatch:
    """detect_erp_peak should resolve channels fuzzy."""

    def test_case_insensitive(self):
        from backend.registry.nodes.erp import _execute_detect_erp_peak
        info = mne.create_info(ch_names=["Cz", "Fz"], sfreq=256.0, ch_types="eeg")
        rng = np.random.default_rng(0)
        n_times = int(256.0 * 0.6)
        data = rng.standard_normal((2, n_times)) * 1e-6
        evoked = mne.EvokedArray(data, info, tmin=-0.1, verbose=False)
        result = _execute_detect_erp_peak(
            evoked, {"channel": "cz", "tmin_ms": 0, "tmax_ms": 400, "polarity": "positive"}
        )
        assert result["channel"] == "Cz"


class TestTTestFuzzyMatch:
    """compute_t_test should resolve channels fuzzy."""

    def test_case_insensitive(self):
        from backend.registry.nodes.statistics import _execute_compute_t_test
        info = mne.create_info(ch_names=["Cz", "Fz"], sfreq=256.0, ch_types="eeg")
        rng = np.random.default_rng(0)
        n_epochs = 10
        n_times = int(256.0 * 0.5)
        data = rng.standard_normal((n_epochs, 2, n_times)) * 1e-6
        events = np.column_stack([
            np.arange(0, n_epochs * n_times, n_times),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ])
        epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, verbose=False)
        result = _execute_compute_t_test(
            epochs, {"channel": "cz", "tmin_ms": 0, "tmax_ms": 200, "popmean": 0.0}
        )
        assert result["channel"] == "Cz"


# ---------------------------------------------------------------------------
# set_montage match_case=False
# ---------------------------------------------------------------------------

class TestSetMontageCaseInsensitive:
    """set_montage should work even with non-standard channel casing."""

    def test_lowercase_channels(self):
        from backend.registry.nodes.preprocessing import _execute_set_montage
        raw = _make_raw(["fp1", "fp2", "f3", "f4", "cz"])
        result = _execute_set_montage(raw, {"montage": "standard_1020"})
        # With match_case=False, lowercase names should still get positions
        montage = result.get_montage()
        assert montage is not None


# ---------------------------------------------------------------------------
# detect_naming_convention tests (P1)
# ---------------------------------------------------------------------------

class TestDetectNamingConvention:

    def test_standard_names_no_suggestion(self):
        """Standard 10-20 names should produce no rename suggestion."""
        result = detect_naming_convention(["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2"])
        assert result["rename_suggestion"] is None
        assert result["detected_prefix"] is None
        assert result["detected_suffix"] is None
        assert result["standard_match_pct"] >= 80

    def test_eeg_prefix_detected(self):
        """'EEG F3', 'EEG F4' ... should detect 'EEG ' prefix."""
        names = [f"EEG {ch}" for ch in ["F3", "F4", "Cz", "Pz", "O1", "O2", "Fp1", "Fp2"]]
        result = detect_naming_convention(names)
        assert result["detected_prefix"] == "EEG "
        assert result["rename_suggestion"] is not None
        assert "prefix" in result["rename_suggestion"].lower()
        assert result["rename_params"] is not None
        assert result["rename_params"]["strip_prefix"] == "EEG "

    def test_ref_suffix_detected(self):
        """'F3-Ref', 'F4-Ref' ... should detect '-Ref' suffix."""
        names = [f"{ch}-Ref" for ch in ["F3", "F4", "Cz", "Pz", "O1", "O2", "Fp1", "Fp2"]]
        result = detect_naming_convention(names)
        assert result["detected_suffix"] == "-Ref"
        assert result["rename_suggestion"] is not None
        assert "suffix" in result["rename_suggestion"].lower()
        assert result["rename_params"] is not None
        assert result["rename_params"]["strip_suffix"] == "-Ref"

    def test_combined_prefix_and_suffix(self):
        """'EEG F3-Ref' should detect both prefix and suffix."""
        names = [f"EEG {ch}-Ref" for ch in ["F3", "F4", "Cz", "Pz", "O1", "O2", "Fp1", "Fp2"]]
        result = detect_naming_convention(names)
        assert result["detected_prefix"] == "EEG "
        assert result["detected_suffix"] == "-Ref"
        assert result["rename_params"] is not None
        assert result["rename_params"]["strip_prefix"] == "EEG "
        assert result["rename_params"]["strip_suffix"] == "-Ref"

    def test_empty_names(self):
        """Empty list should return safe defaults."""
        result = detect_naming_convention([])
        assert result["rename_suggestion"] is None
        assert result["standard_match_pct"] == 100

    def test_single_channel(self):
        """Single channel should not crash."""
        result = detect_naming_convention(["EEG F3"])
        # Single channel — can't determine a common prefix reliably
        assert isinstance(result, dict)

    def test_mixed_channels_no_common_prefix(self):
        """Mixed naming with no dominant prefix should not suggest."""
        names = ["EEG F3", "EOG1", "ECG", "EMG1", "STIM1"]
        result = detect_naming_convention(names)
        # "EEG " appears in only 1/5 channels — below 60% threshold
        assert result["detected_prefix"] is None

    def test_underscore_prefix(self):
        """'EEG_F3' style should detect 'EEG_' prefix."""
        names = [f"EEG_{ch}" for ch in ["F3", "F4", "Cz", "Pz", "O1", "O2", "Fp1", "Fp2"]]
        result = detect_naming_convention(names)
        assert result["detected_prefix"] == "EEG_"

    def test_non_standard_no_affix(self):
        """Names like '1', '2', '3' — non-standard but no prefix/suffix."""
        names = [str(i) for i in range(10)]
        result = detect_naming_convention(names)
        assert result["standard_match_pct"] == 0
        assert result["detected_prefix"] is None
        assert result["detected_suffix"] is None

    def test_standard_match_percentage(self):
        """Half standard, half non-standard should give ~50% match."""
        names = ["F3", "F4", "Cz", "Pz", "CH5", "CH6", "CH7", "CH8"]
        result = detect_naming_convention(names)
        assert result["standard_match_pct"] == 50


# ---------------------------------------------------------------------------
# channel_hint field on ParameterSchema (P2)
# ---------------------------------------------------------------------------

class TestChannelHintField:

    def test_channel_hint_default_none(self):
        """channel_hint should default to None."""
        from backend.registry.node_descriptor import ParameterSchema
        ps = ParameterSchema(name="x", label="X", type="string", default="", description="test")
        assert ps.channel_hint is None

    def test_channel_hint_single(self):
        from backend.registry.node_descriptor import ParameterSchema
        ps = ParameterSchema(name="x", label="X", type="string", default="", description="test", channel_hint="single")
        assert ps.channel_hint == "single"

    def test_channel_hint_multi(self):
        from backend.registry.node_descriptor import ParameterSchema
        ps = ParameterSchema(name="x", label="X", type="string", default="", description="test", channel_hint="multi")
        assert ps.channel_hint == "multi"

    def test_mark_bad_channels_has_multi_hint(self):
        """mark_bad_channels node should have channel_hint='multi'."""
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["mark_bad_channels"]
        bad_param = next(p for p in desc.parameters if p.name == "bad_channels")
        assert bad_param.channel_hint == "multi"

    def test_asymmetry_has_single_hints(self):
        """compute_asymmetry left/right channels should have channel_hint='single'."""
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["compute_asymmetry"]
        left = next(p for p in desc.parameters if p.name == "left_channel")
        right = next(p for p in desc.parameters if p.name == "right_channel")
        assert left.channel_hint == "single"
        assert right.channel_hint == "single"

    def test_erp_peak_has_single_hint(self):
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["detect_erp_peak"]
        ch = next(p for p in desc.parameters if p.name == "channel")
        assert ch.channel_hint == "single"

    def test_t_test_has_single_hint(self):
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["compute_t_test"]
        ch = next(p for p in desc.parameters if p.name == "channel")
        assert ch.channel_hint == "single"

    def test_sleep_channels_have_single_hint(self):
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["compute_sleep_stages"]
        for pname in ["eeg_channel", "eog_channel", "emg_channel"]:
            p = next(p for p in desc.parameters if p.name == pname)
            assert p.channel_hint == "single", f"{pname} should have single hint"

    def test_plot_epochs_image_channel_has_single_hint(self):
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["plot_epochs_image"]
        ch = next(p for p in desc.parameters if p.name == "channel_name")
        assert ch.channel_hint == "single"

    def test_channel_hint_serialized_in_registry(self):
        """channel_hint should be present in the serialized registry JSON."""
        import dataclasses
        from backend.registry import NODE_REGISTRY
        desc = NODE_REGISTRY["mark_bad_channels"]
        as_dict = dataclasses.asdict(desc)
        bad_param = next(p for p in as_dict["parameters"] if p["name"] == "bad_channels")
        assert bad_param["channel_hint"] == "multi"


# ---------------------------------------------------------------------------
# session_store naming hints integration (P1)
# ---------------------------------------------------------------------------

class TestSessionStoreNamingHints:

    def test_build_info_dict_includes_naming_hints(self):
        from backend.session_store import _build_info_dict
        raw = _make_raw(["EEG F3", "EEG F4", "EEG Cz", "EEG Pz", "EEG O1", "EEG O2", "EEG Fp1", "EEG Fp2"])
        info = _build_info_dict(raw)
        assert "naming_hints" in info
        assert info["naming_hints"]["detected_prefix"] == "EEG "
        assert info["naming_hints"]["rename_suggestion"] is not None

    def test_build_info_dict_standard_names_no_hint(self):
        from backend.session_store import _build_info_dict
        raw = _make_raw(["F3", "F4", "Cz", "Pz", "O1", "O2", "Fp1", "Fp2"])
        info = _build_info_dict(raw)
        assert "naming_hints" in info
        assert info["naming_hints"]["rename_suggestion"] is None

    def test_build_info_dict_all_ch_names(self):
        """_build_info_dict should return ALL channel names, not just first 20."""
        from backend.session_store import _build_info_dict
        ch_names = [f"Ch{i}" for i in range(30)]
        raw = _make_raw(ch_names)
        info = _build_info_dict(raw)
        assert len(info["ch_names"]) == 30
        assert info["ch_names_truncated"] is True
