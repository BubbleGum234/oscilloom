"""
backend/tests/test_clinical.py

Tests for Tier 3 clinical + qEEG nodes:
  - compute_alpha_peak
  - compute_asymmetry
  - compute_band_ratio
  - z_score_normalize
  - detect_spikes

Also tests:
  - engine.py metrics dict capture (result_entry["metrics"])
  - POST /pipeline/report endpoint (PDF generation)
"""

from __future__ import annotations

import numpy as np
import pytest
import mne

from backend.registry.nodes.clinical import (
    _execute_compute_alpha_peak,
    _execute_compute_asymmetry,
    _execute_compute_band_ratio,
    _execute_z_score_normalize,
    _execute_detect_spikes,
    COMPUTE_ALPHA_PEAK,
    COMPUTE_ASYMMETRY,
    COMPUTE_BAND_RATIO,
    Z_SCORE_NORMALIZE,
    DETECT_SPIKES,
)
from backend.registry.node_descriptor import VALID_HANDLE_TYPES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_test_raw(n_channels: int = 10, sfreq: float = 256.0, duration: float = 10.0) -> mne.io.RawArray:
    """Standard EEG test fixture."""
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    n_times = int(sfreq * duration)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_channels, n_times)) * 10e-6
    return mne.io.RawArray(data, info, verbose=False)


def make_test_raw_named(ch_names: list[str], sfreq: float = 256.0, duration: float = 10.0) -> mne.io.RawArray:
    """Test fixture with specific channel names (e.g. F3, F4 for asymmetry tests)."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    n_times = int(sfreq * duration)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(ch_names), n_times)) * 10e-6
    return mne.io.RawArray(data, info, verbose=False)


def make_test_spectrum(raw: mne.io.RawArray) -> "mne.time_frequency.Spectrum":
    """Compute a PSD Spectrum from a test raw."""
    return raw.compute_psd(method="welch", fmin=1.0, fmax=40.0, verbose=False)


# ---------------------------------------------------------------------------
# compute_alpha_peak
# ---------------------------------------------------------------------------

class TestComputeAlphaPeak:

    def test_cog_method_returns_dict(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_alpha_peak(spectrum, {"fmin": 7.0, "fmax": 13.0, "method": "cog"})
        assert isinstance(result, dict)

    def test_cog_method_keys(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_alpha_peak(spectrum, {"fmin": 7.0, "fmax": 13.0, "method": "cog"})
        assert "iaf_hz" in result
        assert "method" in result
        assert "alpha_range_hz" in result
        assert "n_channels_averaged" in result

    def test_cog_iaf_in_range(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_alpha_peak(spectrum, {"fmin": 7.0, "fmax": 13.0, "method": "cog"})
        assert 7.0 <= result["iaf_hz"] <= 13.0

    def test_peak_method_returns_dict(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_alpha_peak(spectrum, {"fmin": 8.0, "fmax": 12.0, "method": "peak"})
        assert isinstance(result, dict)
        assert 8.0 <= result["iaf_hz"] <= 12.0

    def test_peak_method_label(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_alpha_peak(spectrum, {"fmin": 8.0, "fmax": 12.0, "method": "peak"})
        assert result["method"] == "peak"

    def test_invalid_range_raises(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        with pytest.raises(ValueError, match="No frequency bins"):
            _execute_compute_alpha_peak(spectrum, {"fmin": 100.0, "fmax": 200.0, "method": "cog"})

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "compute_alpha_peak" in NODE_REGISTRY

    def test_descriptor_output_type(self):
        assert COMPUTE_ALPHA_PEAK.outputs[0].type == "metrics"

    def test_descriptor_input_type(self):
        assert COMPUTE_ALPHA_PEAK.inputs[0].type == "psd"


# ---------------------------------------------------------------------------
# compute_asymmetry
# ---------------------------------------------------------------------------

class TestComputeAsymmetry:

    def test_returns_dict(self):
        raw = make_test_raw_named(["F3", "F4", "Fz"])
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "F4"}
        )
        assert isinstance(result, dict)

    def test_keys_present(self):
        raw = make_test_raw_named(["F3", "F4", "Fz"])
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "F4"}
        )
        assert "asymmetry_index" in result
        assert "channel_left" in result
        assert "channel_right" in result
        assert "interpretation" in result

    def test_asymmetry_is_float(self):
        raw = make_test_raw_named(["F3", "F4"])
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "F4"}
        )
        assert isinstance(result["asymmetry_index"], float)

    def test_same_channel_gives_near_zero_asymmetry(self):
        raw = make_test_raw_named(["F3", "F4"])
        spectrum = make_test_spectrum(raw)
        # Using same channel for both sides → asymmetry should be exactly 0
        result = _execute_compute_asymmetry(
            spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "F3"}
        )
        assert abs(result["asymmetry_index"]) < 1e-6

    def test_missing_left_channel_raises(self):
        raw = make_test_raw_named(["F3", "F4"])
        spectrum = make_test_spectrum(raw)
        with pytest.raises(ValueError, match="not found"):
            _execute_compute_asymmetry(
                spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "Fp1", "right_channel": "F4"}
            )

    def test_missing_right_channel_raises(self):
        raw = make_test_raw_named(["F3", "F4"])
        spectrum = make_test_spectrum(raw)
        with pytest.raises(ValueError, match="not found"):
            _execute_compute_asymmetry(
                spectrum, {"fmin": 8.0, "fmax": 13.0, "left_channel": "F3", "right_channel": "Fp2"}
            )

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "compute_asymmetry" in NODE_REGISTRY

    def test_descriptor_output_type(self):
        assert COMPUTE_ASYMMETRY.outputs[0].type == "metrics"


# ---------------------------------------------------------------------------
# compute_band_ratio
# ---------------------------------------------------------------------------

class TestComputeBandRatio:

    def test_returns_dict(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_band_ratio(
            spectrum,
            {"numerator_fmin": 4.0, "numerator_fmax": 8.0,
             "denominator_fmin": 13.0, "denominator_fmax": 30.0,
             "log_scale": True},
        )
        assert isinstance(result, dict)

    def test_keys_present(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_band_ratio(
            spectrum,
            {"numerator_fmin": 4.0, "numerator_fmax": 8.0,
             "denominator_fmin": 13.0, "denominator_fmax": 30.0,
             "log_scale": True},
        )
        assert "band_ratio" in result
        assert "numerator_band_hz" in result
        assert "denominator_band_hz" in result
        assert "scale" in result

    def test_log_scale_label(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_band_ratio(
            spectrum,
            {"numerator_fmin": 4.0, "numerator_fmax": 8.0,
             "denominator_fmin": 13.0, "denominator_fmax": 30.0,
             "log_scale": True},
        )
        assert result["scale"] == "log10"

    def test_linear_scale(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        result = _execute_compute_band_ratio(
            spectrum,
            {"numerator_fmin": 4.0, "numerator_fmax": 8.0,
             "denominator_fmin": 13.0, "denominator_fmax": 30.0,
             "log_scale": False},
        )
        assert result["scale"] == "linear"
        # Linear ratio is always positive
        assert result["band_ratio"] > 0

    def test_invalid_numerator_band_raises(self):
        raw = make_test_raw()
        spectrum = make_test_spectrum(raw)
        with pytest.raises(ValueError, match="numerator band"):
            _execute_compute_band_ratio(
                spectrum,
                {"numerator_fmin": 200.0, "numerator_fmax": 300.0,
                 "denominator_fmin": 13.0, "denominator_fmax": 30.0,
                 "log_scale": True},
            )

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "compute_band_ratio" in NODE_REGISTRY

    def test_descriptor_output_type(self):
        assert COMPUTE_BAND_RATIO.outputs[0].type == "metrics"


# ---------------------------------------------------------------------------
# z_score_normalize
# ---------------------------------------------------------------------------

class TestZScoreNormalize:

    def test_returns_dict(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        assert isinstance(result, dict)

    def test_keys_present(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        assert "z_scores" in result
        assert "mean_used" in result
        assert "std_used" in result
        assert "n_values" in result

    def test_self_normalized_mean_is_zero(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        z = np.array(result["z_scores"])
        assert abs(z.mean()) < 1e-6

    def test_self_normalized_std_is_one(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        z = np.array(result["z_scores"])
        # z-scores are rounded to 4 decimal places, so allow tolerance of 1e-3
        assert abs(z.std() - 1.0) < 1e-3

    def test_normative_stats(self):
        arr = np.array([100.0, 110.0, 120.0])
        result = _execute_z_score_normalize(
            arr, {"use_data_stats": False, "norm_mean": 100.0, "norm_std": 10.0}
        )
        assert result["mean_used"] == pytest.approx(100.0)
        assert result["std_used"] == pytest.approx(10.0)
        # z-score of 120 with mean=100, std=10 should be 2.0
        assert result["z_scores"][2] == pytest.approx(2.0, abs=1e-3)

    def test_n_values_correct(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        assert result["n_values"] == 3

    def test_zero_std_does_not_crash(self):
        # Constant array → std=0, should not raise ZeroDivisionError
        arr = np.array([5.0, 5.0, 5.0])
        result = _execute_z_score_normalize(arr, {"use_data_stats": True, "norm_mean": 0.0, "norm_std": 1.0})
        # z-scores of a constant array with clamped std=1.0 should all be same value
        assert isinstance(result["z_scores"], list)

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "z_score_normalize" in NODE_REGISTRY

    def test_descriptor_input_type(self):
        assert Z_SCORE_NORMALIZE.inputs[0].type == "array"

    def test_descriptor_output_type(self):
        assert Z_SCORE_NORMALIZE.outputs[0].type == "metrics"


# ---------------------------------------------------------------------------
# detect_spikes
# ---------------------------------------------------------------------------

class TestDetectSpikes:

    def test_returns_dict(self):
        raw = make_test_raw()
        result = _execute_detect_spikes(
            raw, {"threshold_uv": 100.0, "min_duration_ms": 20.0, "max_duration_ms": 100.0}
        )
        assert isinstance(result, dict)

    def test_keys_present(self):
        raw = make_test_raw()
        result = _execute_detect_spikes(
            raw, {"threshold_uv": 100.0, "min_duration_ms": 20.0, "max_duration_ms": 100.0}
        )
        assert "n_spikes" in result
        assert "spike_times_s" in result
        assert "spike_channels" in result
        assert "threshold_uv" in result

    def test_no_spikes_below_threshold(self):
        """Very high threshold → no spikes in random low-amplitude signal."""
        raw = make_test_raw()
        result = _execute_detect_spikes(
            raw, {"threshold_uv": 1e6, "min_duration_ms": 20.0, "max_duration_ms": 100.0}
        )
        assert result["n_spikes"] == 0
        assert result["spike_times_s"] == []

    def test_spikes_detected_with_injected_artifact(self):
        """Inject a known super-threshold artifact and verify it is detected."""
        sfreq = 256.0
        raw = make_test_raw(sfreq=sfreq)
        raw_copy = raw.copy()
        data = raw_copy.get_data()
        # Place a 500 µV artifact at sample 1000 for 10 samples on channel 0
        data[0, 1000:1010] = 500e-6
        raw_copy._data = data

        result = _execute_detect_spikes(
            raw_copy,
            {"threshold_uv": 200.0, "min_duration_ms": 10.0, "max_duration_ms": 200.0},
        )
        assert result["n_spikes"] >= 1
        # The spike at sample 1000 → t ≈ 1000/256 ≈ 3.9 s
        assert any(abs(t - 1000 / sfreq) < 0.1 for t in result["spike_times_s"])

    def test_does_not_mutate_input(self):
        """execute_fn must not modify the original Raw object."""
        raw = make_test_raw()
        original_data = raw.get_data().copy()
        _execute_detect_spikes(
            raw, {"threshold_uv": 100.0, "min_duration_ms": 20.0, "max_duration_ms": 100.0}
        )
        np.testing.assert_array_equal(raw.get_data(), original_data)

    def test_spike_times_list_of_floats(self):
        raw = make_test_raw()
        result = _execute_detect_spikes(
            raw, {"threshold_uv": 30.0, "min_duration_ms": 5.0, "max_duration_ms": 200.0}
        )
        for t in result["spike_times_s"]:
            assert isinstance(t, float)

    def test_descriptor_registered(self):
        from backend.registry import NODE_REGISTRY
        assert "detect_spikes" in NODE_REGISTRY

    def test_descriptor_output_type(self):
        assert DETECT_SPIKES.outputs[0].type == "metrics"

    def test_descriptor_has_two_inputs(self):
        # Should accept both raw_eeg and filtered_eeg
        input_types = [h.type for h in DETECT_SPIKES.inputs]
        assert "raw_eeg" in input_types
        assert "filtered_eeg" in input_types


# ---------------------------------------------------------------------------
# Engine: metrics dict capture
# ---------------------------------------------------------------------------

class TestEngineMetricsCapture:

    def test_metrics_stored_in_result_entry(self):
        """
        engine.py must store dict outputs in result_entry["metrics"]
        so the /pipeline/report endpoint can collect them.
        """
        import mne
        from backend.engine import execute_pipeline
        from backend.models import PipelineGraph, PipelineMetadata, PipelineNode, PipelineEdge

        # Build a minimal pipeline: edf_loader → compute_psd → compute_alpha_peak
        raw = make_test_raw(n_channels=10, sfreq=256.0, duration=10.0)

        pipeline = PipelineGraph(
            metadata=PipelineMetadata(
                name="test", description="", created_by="human", schema_version="1.0"
            ),
            nodes=[
                PipelineNode(id="n1", node_type="edf_loader", label="Loader",
                             parameters={}, position={"x": 0, "y": 0}),
                PipelineNode(id="n2", node_type="compute_psd", label="PSD",
                             parameters={"method": "welch", "fmin": 1.0, "fmax": 40.0,
                                         "n_fft": 256, "n_overlap": 0},
                             position={"x": 200, "y": 0}),
                PipelineNode(id="n3", node_type="compute_alpha_peak", label="Alpha",
                             parameters={"fmin": 7.0, "fmax": 13.0, "method": "cog"},
                             position={"x": 400, "y": 0}),
            ],
            edges=[
                PipelineEdge(id="e1", source_node_id="n1", source_handle_id="raw_out",
                             source_handle_type="raw_eeg",
                             target_node_id="n2", target_handle_id="raw_in",
                             target_handle_type="raw_eeg"),
                PipelineEdge(id="e2", source_node_id="n2", source_handle_id="psd_out",
                             source_handle_type="psd",
                             target_node_id="n3", target_handle_id="psd_in",
                             target_handle_type="psd"),
            ],
        )

        results, _ = execute_pipeline(raw, pipeline)

        # The alpha peak node (n3) output is a dict → must have "metrics" key
        assert "n3" in results
        assert "metrics" in results["n3"], (
            "engine.py must store dict outputs in result_entry['metrics']"
        )
        assert "iaf_hz" in results["n3"]["metrics"]

    def test_metrics_output_type_is_dict(self):
        """output_type for a metrics node should be 'dict' (Python class name)."""
        from backend.engine import execute_pipeline
        from backend.models import PipelineGraph, PipelineMetadata, PipelineNode, PipelineEdge

        raw = make_test_raw(n_channels=10, sfreq=256.0, duration=10.0)

        pipeline = PipelineGraph(
            metadata=PipelineMetadata(name="t", description="", created_by="human"),
            nodes=[
                PipelineNode(id="n1", node_type="edf_loader", label="L",
                             parameters={}, position={"x": 0, "y": 0}),
                PipelineNode(id="n2", node_type="compute_psd", label="P",
                             parameters={"method": "welch", "fmin": 1.0, "fmax": 40.0,
                                         "n_fft": 256, "n_overlap": 0},
                             position={"x": 200, "y": 0}),
                PipelineNode(id="n3", node_type="compute_alpha_peak", label="A",
                             parameters={"fmin": 7.0, "fmax": 13.0, "method": "cog"},
                             position={"x": 400, "y": 0}),
            ],
            edges=[
                PipelineEdge(id="e1", source_node_id="n1", source_handle_id="raw_out",
                             source_handle_type="raw_eeg",
                             target_node_id="n2", target_handle_id="raw_in",
                             target_handle_type="raw_eeg"),
                PipelineEdge(id="e2", source_node_id="n2", source_handle_id="psd_out",
                             source_handle_type="psd",
                             target_node_id="n3", target_handle_id="psd_in",
                             target_handle_type="psd"),
            ],
        )

        results, _ = execute_pipeline(raw, pipeline)
        assert results["n3"]["output_type"] == "dict"


# ---------------------------------------------------------------------------
# POST /pipeline/report
# ---------------------------------------------------------------------------

class TestReportEndpoint:

    def test_report_returns_pdf_bytes(self):
        """report_routes._generate_pdf returns valid PDF bytes."""
        from backend.api.report_routes import _generate_pdf
        pdf_bytes = _generate_pdf(
            node_results={},
            title="Test Report",
            patient_id="PT-001",
            clinic_name="Test Clinic",
        )
        assert isinstance(pdf_bytes, bytes)
        # PDF magic bytes: %PDF
        assert pdf_bytes[:4] == b"%PDF"

    def test_report_with_metrics(self):
        """Report includes metrics section when metrics are present."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "node_alpha": {
                "node_type": "compute_alpha_peak",
                "status": "success",
                "output_type": "dict",
                "data": None,
                "metrics": {"iaf_hz": 10.2, "method": "cog"},
            }
        }
        pdf_bytes = _generate_pdf(node_results, "Test", "", "")
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000  # non-trivial PDF

    def test_report_with_plot(self):
        """Report includes plot section when a base64 PNG is present."""
        import base64, io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from backend.api.report_routes import _generate_pdf

        # Create a tiny PNG
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.plot([0, 1], [0, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=50)
        plt.close(fig)
        buf.seek(0)
        b64_png = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        node_results = {
            "node_plot": {
                "node_type": "plot_psd",
                "status": "success",
                "output_type": "str",
                "data": b64_png,
                "metrics": None,
            }
        }
        pdf_bytes = _generate_pdf(node_results, "Visual Report", "", "")
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 2000

    def test_report_empty_results(self):
        """Empty node_results still produces a valid PDF (shows no-data message)."""
        from backend.api.report_routes import _generate_pdf
        pdf_bytes = _generate_pdf({}, "Empty", "", "")
        assert pdf_bytes[:4] == b"%PDF"

    def test_report_skips_failed_nodes(self):
        """Nodes with status != 'success' must not appear in the PDF."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "node_bad": {
                "node_type": "compute_alpha_peak",
                "status": "error",
                "output_type": "dict",
                "data": None,
                "metrics": {"iaf_hz": 10.0},
            }
        }
        # Should not raise; failed nodes are silently excluded
        pdf_bytes = _generate_pdf(node_results, "Test", "", "")
        assert isinstance(pdf_bytes, bytes)

    # --- Enhanced report tests (Tier A) ---

    def test_report_with_session_info(self):
        """Data Quality section renders when session_info is provided."""
        from backend.api.report_routes import _generate_pdf
        session_info = {
            "sfreq": 256.0,
            "nchan": 64,
            "duration_s": 120.5,
            "ch_types": {"eeg": 62, "eog": 2},
            "bads": ["Fp1", "Fp2"],
            "highpass": 0.1,
            "lowpass": 100.0,
            "meas_date": "2024-01-15 10:30:00",
            "n_annotations": 5,
            "annotation_labels": ["T0", "T1"],
        }
        pdf_bytes = _generate_pdf(
            node_results={}, title="DQ Test", patient_id="", clinic_name="",
            session_info=session_info,
        )
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 1000

    def test_report_with_pipeline_config(self):
        """Pipeline Configuration section renders processing steps."""
        from backend.api.report_routes import _generate_pdf
        config = [
            {"node_id": "n1", "node_type": "bandpass_filter", "label": "BP 1-40",
             "parameters": {"low_cutoff_hz": 1, "high_cutoff_hz": 40}},
            {"node_id": "n2", "node_type": "compute_psd", "label": "PSD",
             "parameters": {"method": "welch", "fmin": 1, "fmax": 40}},
        ]
        pdf_bytes = _generate_pdf(
            node_results={}, title="Config Test", patient_id="", clinic_name="",
            pipeline_config=config,
        )
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 1000

    def test_report_with_audit_log(self):
        """Audit Trail section renders parameter change history."""
        from backend.api.report_routes import _generate_pdf
        audit = [
            {"timestamp": "10:30:15", "nodeId": "n1", "nodeDisplayName": "Bandpass Filter",
             "paramLabel": "Low Cutoff", "oldValue": 0.5, "newValue": 1.0, "unit": "Hz"},
            {"timestamp": "10:31:00", "nodeId": "n1", "nodeDisplayName": "Bandpass Filter",
             "paramLabel": "High Cutoff", "oldValue": 45, "newValue": 40, "unit": "Hz"},
        ]
        pdf_bytes = _generate_pdf(
            node_results={}, title="Audit Test", patient_id="", clinic_name="",
            audit_log=audit,
        )
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 1000

    def test_report_with_clinician_notes(self):
        """Clinician Notes section renders free-text."""
        from backend.api.report_routes import _generate_pdf
        pdf_bytes = _generate_pdf(
            node_results={}, title="Notes Test", patient_id="", clinic_name="",
            notes="Patient was drowsy. Eyes-closed resting state for 2 minutes.",
        )
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 1000

    def test_report_clinical_interpretation_alpha(self):
        """Clinical interpretation flags appear for alpha peak metrics."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "n1": {
                "node_type": "compute_alpha_peak",
                "status": "success",
                "output_type": "dict",
                "data": None,
                "metrics": {"iaf_hz": 10.5, "method": "cog"},
            }
        }
        pdf_bytes = _generate_pdf(node_results, "Interp Test", "", "")
        assert pdf_bytes[:4] == b"%PDF"
        # Should be larger than a basic report due to interpretation content
        assert len(pdf_bytes) > 1200

    def test_report_clinical_interpretation_abnormal(self):
        """Abnormal alpha peak triggers appropriate interpretation."""
        from backend.api.report_routes import _generate_pdf, _CLINICAL_REFERENCES
        interpret_fn = _CLINICAL_REFERENCES["compute_alpha_peak"]["iaf_hz"][2]
        status, explanation = interpret_fn(6.0)
        assert status == "abnormal"

    def test_report_clinical_interpretation_normal(self):
        """Normal alpha peak triggers normal interpretation."""
        from backend.api.report_routes import _CLINICAL_REFERENCES
        interpret_fn = _CLINICAL_REFERENCES["compute_alpha_peak"]["iaf_hz"][2]
        status, explanation = interpret_fn(10.0)
        assert status == "normal"

    def test_report_clinical_interpretation_spikes(self):
        """Spike detection clinical interpretation works."""
        from backend.api.report_routes import _CLINICAL_REFERENCES
        interpret_fn = _CLINICAL_REFERENCES["detect_spikes"]["n_spikes"][2]
        status, _ = interpret_fn(0)
        assert status == "normal"
        status, _ = interpret_fn(5)
        assert status == "borderline"
        status, _ = interpret_fn(50)
        assert status == "abnormal"

    def test_report_clinical_interpretation_asymmetry(self):
        """Asymmetry clinical interpretation works."""
        from backend.api.report_routes import _CLINICAL_REFERENCES
        interpret_fn = _CLINICAL_REFERENCES["compute_asymmetry"]["asymmetry_index"][2]
        status, _ = interpret_fn(0.1)
        assert status == "normal"
        status, _ = interpret_fn(0.8)
        assert status == "borderline"
        status, _ = interpret_fn(1.5)
        assert status == "abnormal"

    def test_report_section_toggles(self):
        """Sections can be toggled off."""
        from backend.api.report_routes import _generate_pdf
        from backend.models import ReportSections
        # All sections off
        sections = ReportSections(
            data_quality=False, pipeline_config=False,
            analysis_results=False, clinical_interpretation=False,
            visualizations=False, audit_trail=False, notes=False,
        )
        pdf_bytes = _generate_pdf(
            node_results={"n1": {"node_type": "compute_alpha_peak", "status": "success",
                                 "output_type": "dict", "data": None,
                                 "metrics": {"iaf_hz": 10.0}}},
            title="Toggle Test", patient_id="", clinic_name="",
            session_info={"sfreq": 256, "nchan": 10, "duration_s": 60},
            pipeline_config=[{"node_type": "bandpass_filter", "label": "BP",
                              "parameters": {}}],
            audit_log=[{"timestamp": "10:00", "nodeId": "n1",
                        "nodeDisplayName": "X", "paramLabel": "Y",
                        "oldValue": 1, "newValue": 2}],
            notes="Some notes",
            sections=sections,
        )
        assert pdf_bytes[:4] == b"%PDF"

    def test_report_full_enhanced(self):
        """Full enhanced report with all sections produces valid PDF."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "n1": {
                "node_type": "compute_alpha_peak",
                "status": "success",
                "output_type": "dict",
                "data": None,
                "metrics": {"iaf_hz": 10.2, "method": "cog",
                            "alpha_range_hz": "7-13", "n_channels_averaged": 10},
            },
            "n2": {
                "node_type": "compute_asymmetry",
                "status": "success",
                "output_type": "dict",
                "data": None,
                "metrics": {"asymmetry_index": 0.15, "channel_left": "F3",
                            "channel_right": "F4", "band_hz": "8-13",
                            "interpretation": "right-dominant (approach)"},
            },
        }
        pdf_bytes = _generate_pdf(
            node_results=node_results,
            title="Full Enhanced Report",
            patient_id="PT-042",
            clinic_name="Neuroscience Lab",
            session_info={"sfreq": 512.0, "nchan": 64, "duration_s": 300.0,
                          "ch_types": {"eeg": 64}, "bads": [],
                          "highpass": 0.1, "lowpass": 200.0,
                          "meas_date": "2024-03-15", "n_annotations": 0,
                          "annotation_labels": []},
            pipeline_config=[
                {"node_type": "edf_loader", "label": "Load EEG",
                 "parameters": {"file_path": "recording.edf"}},
                {"node_type": "bandpass_filter", "label": "Bandpass 1-40 Hz",
                 "parameters": {"low_cutoff_hz": 1, "high_cutoff_hz": 40}},
                {"node_type": "compute_psd", "label": "PSD (Welch)",
                 "parameters": {"method": "welch"}},
                {"node_type": "compute_alpha_peak", "label": "Alpha Peak",
                 "parameters": {"fmin": 7, "fmax": 13, "method": "cog"}},
            ],
            audit_log=[
                {"timestamp": "14:30:15", "nodeId": "n_bp",
                 "nodeDisplayName": "Bandpass Filter",
                 "paramLabel": "Low Cutoff", "oldValue": 0.5,
                 "newValue": 1.0, "unit": "Hz"},
            ],
            notes="Resting state recording, eyes closed. Patient reported feeling alert.",
        )
        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 3000  # Full report should be substantial

    def test_report_included_nodes_filters_metrics(self):
        """Including one node produces a smaller PDF than including both."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "n1": {"node_type": "compute_alpha_peak", "status": "success",
                   "output_type": "dict", "data": None,
                   "metrics": {"iaf_hz": 10.2, "method": "cog",
                               "alpha_range": "7-13"}},
            "n2": {"node_type": "compute_asymmetry", "status": "success",
                   "output_type": "dict", "data": None,
                   "metrics": {"asymmetry_index": 0.1, "channel_left": "F3",
                               "channel_right": "F4", "interpretation": "right"}},
        }
        pdf_both = _generate_pdf(
            node_results=node_results,
            title="Both", patient_id="", clinic_name="",
            included_nodes=None,
        )
        pdf_one = _generate_pdf(
            node_results=node_results,
            title="One", patient_id="", clinic_name="",
            included_nodes=["n1"],
        )
        assert pdf_both[:4] == b"%PDF"
        assert pdf_one[:4] == b"%PDF"
        # Report with both nodes should be larger than report with one
        assert len(pdf_both) > len(pdf_one)

    def test_report_included_nodes_empty_list(self):
        """Empty included_nodes list produces a minimal report (no metrics)."""
        from backend.api.report_routes import _generate_pdf
        node_results = {
            "n1": {"node_type": "compute_alpha_peak", "status": "success",
                   "output_type": "dict", "data": None,
                   "metrics": {"iaf_hz": 10.2}},
        }
        pdf_empty = _generate_pdf(
            node_results=node_results,
            title="Empty", patient_id="", clinic_name="",
            included_nodes=[],
        )
        pdf_with = _generate_pdf(
            node_results=node_results,
            title="With", patient_id="", clinic_name="",
            included_nodes=["n1"],
        )
        assert pdf_empty[:4] == b"%PDF"
        # Empty filter = no content, so smaller than with content
        assert len(pdf_with) > len(pdf_empty)

    def test_report_included_nodes_filters_plots(self):
        """included_nodes also filters visualization outputs."""
        import base64, io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from backend.api.report_routes import _generate_pdf
        # Generate a real PNG via matplotlib
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        node_results = {
            "p1": {"node_type": "plot_psd", "status": "success",
                   "output_type": "str", "data": f"data:image/png;base64,{b64}"},
            "p2": {"node_type": "plot_topomap", "status": "success",
                   "output_type": "str", "data": f"data:image/png;base64,{b64}"},
        }
        pdf_both = _generate_pdf(
            node_results=node_results,
            title="Both Plots", patient_id="", clinic_name="",
            included_nodes=None,
        )
        pdf_one = _generate_pdf(
            node_results=node_results,
            title="One Plot", patient_id="", clinic_name="",
            included_nodes=["p1"],
        )
        assert len(pdf_both) > len(pdf_one)

    def test_report_via_api_with_included_nodes(self):
        """POST /pipeline/report respects included_nodes parameter."""
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        body = {
            "node_results": {
                "n1": {"node_type": "compute_alpha_peak", "status": "success",
                       "output_type": "dict", "data": None,
                       "metrics": {"iaf_hz": 10.0}},
                "n2": {"node_type": "compute_asymmetry", "status": "success",
                       "output_type": "dict", "data": None,
                       "metrics": {"asymmetry_index": 0.1}},
            },
            "title": "API Filter Test",
            "included_nodes": ["n1"],
        }
        resp = client.post("/pipeline/report", json=body)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/pdf"


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------

class TestClinicalRegistry:

    def test_all_clinical_nodes_in_registry(self):
        from backend.registry import NODE_REGISTRY
        expected = [
            "compute_alpha_peak",
            "compute_asymmetry",
            "compute_band_ratio",
            "z_score_normalize",
            "detect_spikes",
        ]
        for node_type in expected:
            assert node_type in NODE_REGISTRY, f"{node_type} missing from NODE_REGISTRY"

    def test_all_clinical_nodes_have_metrics_output(self):
        from backend.registry import NODE_REGISTRY
        clinical_nodes = [
            "compute_alpha_peak",
            "compute_asymmetry",
            "compute_band_ratio",
            "z_score_normalize",
            "detect_spikes",
        ]
        for node_type in clinical_nodes:
            descriptor = NODE_REGISTRY[node_type]
            output_types = [h.type for h in descriptor.outputs]
            assert "metrics" in output_types, (
                f"{node_type} should output 'metrics', got {output_types}"
            )

    def test_all_clinical_nodes_have_category_clinical(self):
        from backend.registry import NODE_REGISTRY
        clinical_nodes = [
            "compute_alpha_peak", "compute_asymmetry",
            "compute_band_ratio", "z_score_normalize", "detect_spikes",
        ]
        for node_type in clinical_nodes:
            assert NODE_REGISTRY[node_type].category == "Clinical"

    def test_metrics_in_valid_handle_types(self):
        assert "metrics" in VALID_HANDLE_TYPES

    def test_no_duplicate_node_types(self):
        from backend.registry import NODE_REGISTRY
        types = list(NODE_REGISTRY.keys())
        assert len(types) == len(set(types))

    def test_total_node_count_increased(self):
        """We should now have 56 nodes (51 Tier 1+2 + 5 Tier 3 clinical)."""
        from backend.registry import NODE_REGISTRY
        assert len(NODE_REGISTRY) >= 56
