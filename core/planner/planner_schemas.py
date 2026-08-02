"""
PlannerSchemas Module - Version 3.0 Architecture
Defines PlannerOutputV3 and intent enumerations for JSON compilation.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class PlanningIntent(str, Enum):
    CROP_DIAGNOSIS = "CROP_DIAGNOSIS"
    FERTILIZER_ADVISORY = "FERTILIZER_ADVISORY"
    ENVIRONMENTAL_RISK = "ENVIRONMENTAL_RISK"
    MARKET_INQUIRY = "MARKET_INQUIRY"
    VARIETY_SELECTION = "VARIETY_SELECTION"
    POLICY_LEGAL_INQUIRY = "POLICY_LEGAL_INQUIRY"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"


class PlannerOutputV3(BaseModel):
    """Schema enforced output of the lightweight JSON Planner Compiler."""
    intent: PlanningIntent = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Planner confidence score")
    missing_information: List[str] = Field(default_factory=list, description="Missing input parameters required")
    tasks: List[str] = Field(default_factory=list, description="Abstract capability IDs requested")
    clarification_prompt: str = Field(default="", description="Prompt presented if clarification needed")
