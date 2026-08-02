"""
ProcessingContext Module - Version 3.0 Architecture
Stores domain context, conversation memory, weather context, and vision analysis.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ProcessingContext(BaseModel):
    """Domain context passed through planning and execution pipeline."""
    user_query: str = Field(default="", description="User query text")
    request_context: Optional[Any] = Field(default=None, description="Bound RequestContext reference")
    recent_history: List[Dict[str, str]] = Field(default_factory=list, description="Sliding window memory (last 4 turns)")
    weather_context: Optional[Dict[str, Any]] = Field(default=None, description="Ingested weather data")
    seasonal_advisory: Optional[str] = Field(default=None, description="Ingested seasonal advisory text")
    vision_analysis: Optional[Any] = Field(default=None, description="Extracted vision observations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom processing metadata")
