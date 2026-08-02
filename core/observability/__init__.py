"""
Observability Package Initialization - Version 3.0 Architecture
Exports V3.0 TraceContext & TelemetryBus along with backward-compatible RequestTrace models.
"""

from core.observability.trace_context import TraceContext
from core.observability.telemetry_bus import telemetry_bus, TelemetryBus
from core.observability_legacy import (
    VisionTrace,
    RAGTrace,
    AgentExecutionTrace,
    PerformanceTrace,
    EvaluationTrace,
    RequestTrace
)

__all__ = [
    "TraceContext",
    "telemetry_bus",
    "TelemetryBus",
    "VisionTrace",
    "RAGTrace",
    "AgentExecutionTrace",
    "PerformanceTrace",
    "EvaluationTrace",
    "RequestTrace"
]
