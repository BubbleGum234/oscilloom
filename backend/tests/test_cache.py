"""
backend/tests/test_cache.py

Tests for the Phase 5 ExecutionCache feature.

Unit tests cover the ExecutionCache class in isolation (put/get, LRU eviction,
hashing, stats). Integration tests verify cache behaviour when passed to
engine.execute_pipeline().
"""

from __future__ import annotations

import pytest
import mne

from backend.execution_cache import ExecutionCache
from backend.engine import execute_pipeline
from backend.models import (
    PipelineEdge,
    PipelineGraph,
    PipelineMetadata,
    PipelineNode,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors test_engine.py conventions)
# ---------------------------------------------------------------------------

def _make_metadata() -> PipelineMetadata:
    return PipelineMetadata(
        name="Cache Test Pipeline",
        description="Test",
        created_by="test",
        schema_version="1.0",
    )


def _make_node(node_id: str, node_type: str, params: dict) -> PipelineNode:
    return PipelineNode(
        id=node_id,
        node_type=node_type,
        label=node_type,
        parameters=params,
        position={"x": 0.0, "y": 0.0},
    )


def _make_edge(
    edge_id: str,
    src_node: str, src_handle: str, src_type: str,
    tgt_node: str, tgt_handle: str, tgt_type: str,
) -> PipelineEdge:
    return PipelineEdge(
        id=edge_id,
        source_node_id=src_node, source_handle_id=src_handle, source_handle_type=src_type,
        target_node_id=tgt_node, target_handle_id=tgt_handle, target_handle_type=tgt_type,
    )


def _two_node_graph(filter_params: dict | None = None) -> PipelineGraph:
    """edf_loader → bandpass_filter pipeline."""
    if filter_params is None:
        filter_params = {
            "low_cutoff_hz": 1.0,
            "high_cutoff_hz": 40.0,
            "method": "fir",
        }
    nodes = [
        _make_node("n1", "edf_loader", {"file_path": "", "preload": True}),
        _make_node("n2", "bandpass_filter", filter_params),
    ]
    edges = [
        _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
    ]
    return PipelineGraph(metadata=_make_metadata(), nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_raw() -> mne.io.BaseRaw:
    """
    Loads the MNE sample dataset (EEG channels only, first 30 seconds).
    Scoped to module so it is loaded once per test file.
    """
    data_path = mne.datasets.sample.data_path()
    raw_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(str(raw_path), preload=True, verbose=False)
    raw.pick("eeg")
    raw.crop(tmax=30.0)
    return raw


# ===========================================================================
# Unit tests — ExecutionCache class
# ===========================================================================

class TestExecutionCacheUnit:
    """Pure unit tests for ExecutionCache (no MNE / engine dependency)."""

    def test_cache_put_and_get(self):
        """Store a value and retrieve it by key."""
        cache = ExecutionCache(max_entries=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss_returns_none(self):
        """get() on an unknown key returns None."""
        cache = ExecutionCache(max_entries=10)
        assert cache.get("nonexistent") is None

    def test_cache_lru_eviction(self):
        """When cache exceeds max_entries, the oldest entry is evicted."""
        cache = ExecutionCache(max_entries=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        # Cache is full (3/3). Adding a fourth should evict "a".
        cache.put("d", 4)

        assert cache.get("a") is None, "oldest entry 'a' should be evicted"
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4
        assert len(cache) == 3

    def test_cache_access_updates_lru_order(self):
        """Accessing an item via get() moves it to most-recent, protecting it from eviction."""
        cache = ExecutionCache(max_entries=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        # Access "a" — it becomes most-recent
        cache.get("a")

        # Add "d" — now "b" is the oldest and should be evicted
        cache.put("d", 4)

        assert cache.get("b") is None, "'b' should be evicted (oldest after 'a' was accessed)"
        assert cache.get("a") == 1, "'a' should survive (was recently accessed)"
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_cache_clear(self):
        """clear() empties the cache and resets counters."""
        cache = ExecutionCache(max_entries=10)
        cache.put("x", 42)
        cache.put("y", 99)
        cache.get("x")       # hit
        cache.get("missing")  # miss

        cache.clear()

        assert len(cache) == 0
        assert cache.get("x") is None  # this counts as a miss after clear
        assert cache.stats["hit_count"] == 0
        assert cache.stats["miss_count"] == 1  # the get("x") after clear

    def test_cache_compute_hash_deterministic(self):
        """Same inputs must always produce the same hash."""
        h1 = ExecutionCache.compute_hash("bandpass_filter", {"low": 1.0, "high": 40.0}, "abc")
        h2 = ExecutionCache.compute_hash("bandpass_filter", {"low": 1.0, "high": 40.0}, "abc")
        assert h1 == h2

    def test_cache_compute_hash_different_params(self):
        """Different params must produce different hashes."""
        h1 = ExecutionCache.compute_hash("bandpass_filter", {"low": 1.0, "high": 40.0}, "abc")
        h2 = ExecutionCache.compute_hash("bandpass_filter", {"low": 2.0, "high": 50.0}, "abc")
        assert h1 != h2

    def test_cache_stats(self):
        """stats property returns correct size, max_entries, hit_count, miss_count."""
        cache = ExecutionCache(max_entries=5)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")          # hit
        cache.get("a")          # hit
        cache.get("nonexistent")  # miss

        stats = cache.stats
        assert stats["size"] == 2
        assert stats["max_entries"] == 5
        assert stats["hit_count"] == 2
        assert stats["miss_count"] == 1


# ===========================================================================
# Integration tests — ExecutionCache + engine.execute_pipeline
# ===========================================================================

class TestExecutionCacheIntegration:
    """Integration tests: cache behaviour through engine.execute_pipeline()."""

    def test_pipeline_cache_hit(self, sample_raw):
        """Run pipeline twice with same params — second run should have cache_hit=True."""
        graph = _two_node_graph()
        cache = ExecutionCache(max_entries=50)

        # First run — all misses
        results1, _ = execute_pipeline(sample_raw.copy(), graph, cache=cache)
        assert results1["n1"]["status"] == "success"
        assert results1["n2"]["status"] == "success"

        # Second run — same graph, same cache → expect hits
        results2, _ = execute_pipeline(sample_raw.copy(), graph, cache=cache)
        assert results2["n1"].get("cache_hit") is True
        assert results2["n2"].get("cache_hit") is True

    def test_pipeline_cache_miss_on_param_change(self, sample_raw):
        """Changing a param on a node causes cache_hit=False for that node."""
        graph1 = _two_node_graph({"low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir"})
        cache = ExecutionCache(max_entries=50)

        # Prime the cache
        execute_pipeline(sample_raw.copy(), graph1, cache=cache)

        # Change bandpass params
        graph2 = _two_node_graph({"low_cutoff_hz": 2.0, "high_cutoff_hz": 30.0, "method": "fir"})
        results2, _ = execute_pipeline(sample_raw.copy(), graph2, cache=cache)

        # edf_loader (n1) params unchanged → hit; bandpass (n2) params changed → miss
        assert results2["n1"].get("cache_hit") is True
        assert results2["n2"].get("cache_hit") is False

    def test_pipeline_cache_upstream_invalidation(self, sample_raw):
        """Changing an upstream node invalidates all downstream nodes."""
        # Three-node pipeline: edf_loader → bandpass → notch
        nodes = [
            _make_node("n1", "edf_loader", {"file_path": "", "preload": True}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 1.0, "high_cutoff_hz": 40.0, "method": "fir",
            }),
            _make_node("n3", "notch_filter", {"notch_freq_hz": 60.0}),
        ]
        edges = [
            _make_edge("e1", "n1", "eeg_out", "raw_eeg", "n2", "eeg_in", "raw_eeg"),
            _make_edge("e2", "n2", "eeg_out", "filtered_eeg", "n3", "eeg_in", "filtered_eeg"),
        ]
        graph1 = PipelineGraph(metadata=_make_metadata(), nodes=nodes, edges=edges)
        cache = ExecutionCache(max_entries=50)

        # Prime
        execute_pipeline(sample_raw.copy(), graph1, cache=cache)

        # Change n2 (bandpass) params — n3 (downstream) should also miss
        nodes_v2 = [
            _make_node("n1", "edf_loader", {"file_path": "", "preload": True}),
            _make_node("n2", "bandpass_filter", {
                "low_cutoff_hz": 2.0, "high_cutoff_hz": 35.0, "method": "fir",
            }),
            _make_node("n3", "notch_filter", {"notch_freq_hz": 60.0}),
        ]
        graph2 = PipelineGraph(metadata=_make_metadata(), nodes=nodes_v2, edges=edges)
        results2, _ = execute_pipeline(sample_raw.copy(), graph2, cache=cache)

        assert results2["n1"].get("cache_hit") is True, "edf_loader unchanged"
        assert results2["n2"].get("cache_hit") is False, "bandpass params changed"
        assert results2["n3"].get("cache_hit") is False, "downstream of changed node"

    def test_pipeline_without_cache_backward_compat(self, sample_raw):
        """execute_pipeline without cache arg still works (backward compatibility)."""
        graph = _two_node_graph()
        results, node_outputs = execute_pipeline(sample_raw.copy(), graph)

        assert results["n1"]["status"] == "success"
        assert results["n2"]["status"] == "success"
        # cache_hit field should not be present (or be absent) when no cache is used
        assert "cache_hit" not in results["n1"] or results["n1"]["cache_hit"] is False

    def test_cache_hit_field_in_results(self, sample_raw):
        """Verify cache_hit field exists in node results when cache is provided."""
        graph = _two_node_graph()
        cache = ExecutionCache(max_entries=50)

        results, _ = execute_pipeline(sample_raw.copy(), graph, cache=cache)

        for node_id in ("n1", "n2"):
            assert "cache_hit" in results[node_id], (
                f"cache_hit field missing from results['{node_id}']"
            )
            # First run — should all be misses
            assert results[node_id]["cache_hit"] is False
