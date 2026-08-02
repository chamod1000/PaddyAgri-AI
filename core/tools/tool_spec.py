"""
ToolSpec Module - Version 3.0 Architecture
Defines tool implementation contract metadata and execution signature.
"""

from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """Concrete tool implementation specification contract."""
    tool_id: str = Field(..., description="Unique tool version string (e.g. 'openmeteo_weather_v2')")
    capability_id: str = Field(..., description="Bound capability identifier")
    provider_name: str = Field(default="Internal", description="Upstream service provider name")
    version: str = Field(default="1.0.0", description="Semantic version string")
    dependencies: List[str] = Field(default_factory=list, description="Capabilities/tools that must execute prior")
    parallel_safe: bool = Field(default=True, description="True if thread-safe for parallel execution")
    estimated_latency_ms: float = Field(default=50.0, description="Estimated execution latency in ms")
    estimated_cost_tier: str = Field(default="FREE", description="Cost tier: FREE, LOW, HIGH")
    requires_llm: bool = Field(default=False, description="True if tool invokes upstream LLM API")
    requires_image: bool = Field(default=False, description="True if tool requires attached image bytes")
    requires_location: bool = Field(default=False, description="True if tool requires location metadata")
    execute_fn: Optional[Any] = Field(default=None, description="Execution callable signature (args, context) -> dict")

    model_config = ConfigDict(arbitrary_types_allowed=True)
