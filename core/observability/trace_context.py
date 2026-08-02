"""
TraceContext Module - Version 3.0 Architecture
Generates correlated TraceID, RequestID, and SessionID headers for production tracing.
"""

import uuid
from typing import Dict
from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    """Correlation IDs for end-to-end telemetry tracing."""
    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[12:]}")

    def to_dict(self) -> Dict[str, str]:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id
        }
