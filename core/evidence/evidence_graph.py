"""
EvidenceGraph Module - Version 3.0 Architecture
Thread-safe blackboard container for structured tool artifacts and execution metadata.
"""

import threading
from typing import Dict, Any, List, Optional


class EvidenceGraph:
    """Thread-safe state graph aggregating execution artifacts."""

    def __init__(self, session_id: str = "", user_query: str = ""):
        self.session_id = session_id
        self.user_query = user_query
        self.artifacts: Dict[str, Dict[str, Any]] = {}
        self.rag_chunks: List[Dict[str, Any]] = []
        self.execution_metrics: Dict[str, float] = {}
        self.errors_encountered: List[Dict[str, str]] = []
        self._lock = threading.Lock()

    def deposit_artifact(self, capability_id: str, artifact: Dict[str, Any], duration_ms: float = 0.0) -> None:
        with self._lock:
            self.artifacts[capability_id] = artifact
            if duration_ms > 0:
                self.execution_metrics[capability_id] = duration_ms

    def record_error(self, tool_id: str, error_msg: str) -> None:
        with self._lock:
            self.errors_encountered.append({"tool_id": tool_id, "error": error_msg})

    def get_artifact(self, capability_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.artifacts.get(capability_id)

    def is_purely_structured(self) -> bool:
        """True if all artifacts in graph are purely structured data (no complex LLM CoT needed)."""
        with self._lock:
            if not self.artifacts:
                return False
            # If pathology diagnosis is present, qualitative LLM synthesis is required
            if "pathology_diagnosis" in self.artifacts:
                return False
            return True
