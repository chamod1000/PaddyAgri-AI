"""
TelemetryBus Module - Version 3.0 Architecture
In-memory event bus for developer telemetry logs capped at 50 records (LRU).
"""

import threading
from datetime import datetime
from typing import List, Dict, Any


class TelemetryBus:
    """Thread-safe bounded in-memory telemetry log buffer."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryBus, cls).__new__(cls)
                cls._instance._logs = []
                cls._instance._max_capacity = 50
        return cls._instance

    def emit(self, event_type: str, details: Dict[str, Any], trace_id: str = "") -> None:
        with self._lock:
            record = {
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "event_type": event_type,
                "trace_id": trace_id,
                "details": details
            }
            self._logs.append(record)
            if len(self._logs) > self._max_capacity:
                self._logs.pop(0)

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._logs[-limit:])

    def clear(self) -> None:
        with self._lock:
            self._logs.clear()


telemetry_bus = TelemetryBus()
