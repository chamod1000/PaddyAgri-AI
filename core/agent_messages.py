"""
Agent-to-Agent Communication Schema Module
Implements Mandatory Requirement 4b: Agent-to-Agent Structured Message Exchange Protocol
Defines typed message payloads passed between RouterAgent, DiagnosticAgent, and FertilizerAgent.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class QueryIntent(str, Enum):
    DISEASE_DIAGNOSIS = "DISEASE_DIAGNOSIS"
    FERTILIZER_RECOMMENDATION = "FERTILIZER_RECOMMENDATION"
    BOTH = "BOTH"
    GENERAL = "GENERAL"


class AgentMessage(BaseModel):
    """
    Structured message payload exchanged between agents.
    Inspired by Agent Communication Protocols (A2A/MCP).
    """
    message_id: str = Field(..., description="Unique message tracking identifier")
    sender: str = Field(..., description="Name of the sending agent")
    receiver: str = Field(..., description="Name of the receiving agent")
    intent: QueryIntent = Field(..., description="Classified intent of the request")
    user_query: str = Field(..., description="Original user query")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Custom payload metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class RAGContextChunk(BaseModel):
    """Retrieved document metadata chunk from RAG vector store."""
    content: str
    filename: str
    category: str
    page: int
    score: float


class DiagnosticResult(BaseModel):
    """Output produced by the Diagnostic Agent."""
    thought_process: str = Field(default="Chain of thought reasoning...", description="Step-by-step reasoning")
    suspected_disease: str
    symptoms_identified: List[str]
    treatment_recommended: List[str]
    confidence_level: str
    rag_sources: List[RAGContextChunk] = []


class FertilizerRecommendation(BaseModel):
    """Output produced by the Fertilizer Agent."""
    thought_process: str = Field(default="Chain of thought reasoning...", description="Step-by-step reasoning")
    season: str  # Yala or Maha
    district_zone: str
    urea_dosage_per_acre_kg: float
    tsp_dosage_per_acre_kg: float
    mop_dosage_per_acre_kg: float
    application_schedule: List[str]
    rag_sources: List[RAGContextChunk] = []


class SafetyVerdict(BaseModel):
    """Safety check result from ReflectionAgent for a single recommendation."""
    check_name: str = Field(..., description="Name of safety check (dosage_validation, allergen_screen, regulatory_citation)")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(default="", description="Human-readable message explaining the verdict")
    severity: str = Field(default="info", description="Severity level: info, warning, critical")


class ReflectionResult(BaseModel):
    """Output produced by the ReflectionAgent after safety & quality verification."""
    recommendation_id: str = Field(default="refl_verified", description="Unique ID linking back to parent recommendation")
    all_checks_passed: bool = Field(default=False, description="Global pass/fail if ALL checks passed")
    verdicts: List[SafetyVerdict] = Field(default_factory=list, description="List of individual safety check results")
    warnings: List[str] = Field(default_factory=list, description="Aggregated warning messages")
    regulatory_citations: List[str] = Field(default_factory=list, description="Applicable Sri Lankan DoA regulatory citations")
    biosecurity_alerts: List[str] = Field(default_factory=list, description="Any biosecurity / invasive species concerns")
    critique_payload: Optional[str] = Field(default=None, description="Structured critique text for self-correction re-synthesis loop")


class AgentResponse(BaseModel):
    """Final synthesized response returned to the user."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    intent: QueryIntent
    diagnostic_info: Optional[Any] = None
    fertilizer_info: Optional[Any] = None
    general_info: Optional[Any] = None
    reflection_result: Optional[Any] = None
    final_synthesis: str
    message_trace: List[AgentMessage] = []
