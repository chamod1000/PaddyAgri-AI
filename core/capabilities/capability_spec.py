"""
CapabilitySpec Module - Version 3.0 Architecture
Defines abstract system capability specifications.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class CapabilitySpec(BaseModel):
    """Declarative specification of an abstract system capability."""
    capability_id: str = Field(..., description="Unique capability identifier (e.g. 'weather_intelligence')")
    display_name: str = Field(..., description="Human readable display title")
    description: str = Field(..., description="Semantic description used by Planner")
    required_inputs: List[str] = Field(default_factory=list, description="Mandatory input parameters")
    output_type: str = Field(default="STRUCTURED_JSON", description="Format of returned output")
    default_tool_id: str = Field(..., description="Primary tool implementation ID")
    fallback_tool_id: str = Field(default="", description="Fallback tool implementation ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional capability metadata")
