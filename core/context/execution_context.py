"""
ExecutionContext Module - Version 3.0 Architecture
Tracks task execution states, DAG execution metrics, and task correlation data.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ExecutionContext(BaseModel):
    """Runtime context passed to tools during execution."""
    trace_id: str = Field(..., description="Trace correlation ID")
    request_id: str = Field(..., description="Request correlation ID")
    task_id: str = Field(..., description="Active task identifier")
    completed_tasks: List[str] = Field(default_factory=list, description="List of completed task IDs in DAG")
    execution_metrics: Dict[str, float] = Field(default_factory=dict, description="Per-task latency measurements in ms")
    errors_encountered: List[Dict[str, str]] = Field(default_factory=list, description="Non-fatal tool exceptions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom execution metadata")
