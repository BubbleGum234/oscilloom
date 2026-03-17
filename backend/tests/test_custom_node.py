"""Tests for the custom_python node (Feature B)."""

import numpy as np
import pytest
import mne

from backend.registry.nodes.custom import _execute_custom_python


def _make_test_raw():
    """Create a minimal test Raw object."""
    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(5)],
        sfreq=256.0,
        ch_types="eeg",
    )
    data = np.random.randn(5, 1280) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


class TestCustomPythonNode:
    def test_valid_code_executes(self):
        """Simple filter code runs and returns Raw."""
        raw = _make_test_raw()
        result = _execute_custom_python(raw, {
            "code": "data = data.copy().filter(l_freq=1, h_freq=40, verbose=False)",
            "timeout_s": 30,
        })
        assert isinstance(result, mne.io.BaseRaw)

    def test_empty_code_raises(self):
        """Empty code param raises ValueError."""
        raw = _make_test_raw()
        with pytest.raises(ValueError, match="No code provided"):
            _execute_custom_python(raw, {"code": "", "timeout_s": 10})

    def test_whitespace_only_code_raises(self):
        """Whitespace-only code raises ValueError."""
        raw = _make_test_raw()
        with pytest.raises(ValueError, match="No code provided"):
            _execute_custom_python(raw, {"code": "   \n  ", "timeout_s": 10})

    def test_timeout_kills_infinite_loop(self):
        """Code with infinite loop is killed after timeout."""
        raw = _make_test_raw()
        with pytest.raises(TimeoutError, match="timed out"):
            _execute_custom_python(raw, {
                "code": "while True: pass",
                "timeout_s": 2,
            })

    def test_error_returns_clean_message(self):
        """Code that raises returns readable error."""
        raw = _make_test_raw()
        with pytest.raises(RuntimeError, match="Custom node error"):
            _execute_custom_python(raw, {
                "code": "raise ValueError('test error message')",
                "timeout_s": 10,
            })

    def test_restricted_builtins_block_os(self):
        """Code that tries import os fails."""
        raw = _make_test_raw()
        with pytest.raises(RuntimeError, match="Custom node error"):
            _execute_custom_python(raw, {
                "code": "import os; os.listdir('/')",
                "timeout_s": 10,
            })

    def test_restricted_builtins_block_open(self):
        """Code that tries open() fails."""
        raw = _make_test_raw()
        with pytest.raises(RuntimeError, match="Custom node error"):
            _execute_custom_python(raw, {
                "code": "f = open('/tmp/test.txt', 'w')",
                "timeout_s": 10,
            })

    def test_data_reassignment_returns_new_value(self):
        """Code that reassigns data returns the new value."""
        raw = _make_test_raw()
        result = _execute_custom_python(raw, {
            "code": "data = data.copy().pick([0, 1], verbose=False)",
            "timeout_s": 30,
        })
        assert isinstance(result, mne.io.BaseRaw)
        assert result.info["nchan"] == 2

    def test_data_not_reassigned_returns_original(self):
        """Code that doesn't reassign data returns the original."""
        raw = _make_test_raw()
        result = _execute_custom_python(raw, {
            "code": "x = 42\nprint(x)",
            "timeout_s": 30,
        })
        # Should return the original data unchanged
        assert isinstance(result, mne.io.BaseRaw)
        assert result.info["nchan"] == 5

    def test_numpy_available(self):
        """np (numpy) is available in the sandbox."""
        raw = _make_test_raw()
        result = _execute_custom_python(raw, {
            "code": "arr = np.zeros(10)\ndata = data.copy()",
            "timeout_s": 30,
        })
        assert isinstance(result, mne.io.BaseRaw)

    def test_mne_available(self):
        """mne is available in the sandbox."""
        raw = _make_test_raw()
        result = _execute_custom_python(raw, {
            "code": "info = mne.create_info(['ch1'], sfreq=100, ch_types='eeg')\ndata = data.copy()",
            "timeout_s": 30,
        })
        assert isinstance(result, mne.io.BaseRaw)

    def test_custom_node_in_registry(self):
        """custom_python node appears in NODE_REGISTRY."""
        from backend.registry import NODE_REGISTRY
        assert "custom_python" in NODE_REGISTRY
        desc = NODE_REGISTRY["custom_python"]
        assert desc.display_name == "Custom Python"
        assert desc.category == "Custom"
