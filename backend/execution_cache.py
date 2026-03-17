"""LRU execution cache: content hash → node output."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any


class ExecutionCache:
    """LRU cache keyed by a content hash of node_type + params + upstream hash."""

    def __init__(self, max_entries: int = 50) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_entries = max_entries
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(node_type: str, params: dict, upstream_hash: str) -> str:
        """SHA-256 of node_type + sorted params + upstream hash, truncated to 16 hex chars."""
        payload = json.dumps(
            {"node_type": node_type, "params": params, "upstream_hash": upstream_hash},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return cached value or None. Moves entry to end on hit (LRU)."""
        if key in self._cache:
            self._hit_count += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._miss_count += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """Store a value, evicting the oldest entry when over *max_entries*."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            self._cache[key] = value
            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached entries and reset hit/miss counters."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "max_entries": self._max_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
        }
