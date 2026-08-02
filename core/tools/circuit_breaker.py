"""
CircuitBreaker Module - Version 3.0 Architecture
Implements CLOSED, OPEN, HALF_OPEN state machine for tool resilience.
"""

import time
import threading
from enum import Enum
from typing import Dict


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Per-tool circuit breaker managing automatic failover and cooldowns."""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._states: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_state(self, tool_id: str) -> CircuitState:
        with self._lock:
            state = self._states.get(tool_id, CircuitState.CLOSED)
            if state == CircuitState.OPEN:
                last_fail = self._last_failure_times.get(tool_id, 0.0)
                if time.perf_counter() - last_fail > self.cooldown_seconds:
                    self._states[tool_id] = CircuitState.HALF_OPEN
                    return CircuitState.HALF_OPEN
            return state

    def record_success(self, tool_id: str) -> None:
        with self._lock:
            self._states[tool_id] = CircuitState.CLOSED
            self._failure_counts[tool_id] = 0

    def record_failure(self, tool_id: str) -> None:
        with self._lock:
            cnt = self._failure_counts.get(tool_id, 0) + 1
            self._failure_counts[tool_id] = cnt
            self._last_failure_times[tool_id] = time.perf_counter()
            if cnt >= self.failure_threshold:
                self._states[tool_id] = CircuitState.OPEN


circuit_breaker = CircuitBreaker()
