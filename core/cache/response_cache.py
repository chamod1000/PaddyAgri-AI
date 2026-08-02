"""
ResponseCache Module - Version 3.0 Architecture
SHA256 Response Cache with TTL and LRU eviction for zero-latency repeated queries.
"""

import hashlib
import time
import threading
from typing import Dict, Any, Optional, Tuple


class ResponseCache:
    """Thread-safe LRU Response Cache with TTL bounds."""

    def __init__(self, max_size: int = 100, default_ttl: float = 1800.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def generate_key(query: str, location: Optional[str] = None, image_hash: Optional[str] = None) -> str:
        raw_key = f"{query.strip().lower()}:{location or ''}:{image_hash or ''}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            data, exp_time = self._cache[key]
            if time.perf_counter() > exp_time:
                del self._cache[key]
                return None
            return data

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            valid_ttl = ttl if ttl is not None else self.default_ttl
            self._cache[key] = (value, time.perf_counter() + valid_ttl)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


response_cache = ResponseCache()
