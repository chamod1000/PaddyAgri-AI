"""
Agent-to-Agent Communication Schema Module
Implements Requirement 4b: Agent-to-Agent Structured Message Exchange Protocol & Case Intelligence Memory Model.
Defines typed message payloads passed between RouterAgent, DiagnosticAgent, FertilizerAgent, and Multimodal Context.
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


class VisionAnalysisResult(BaseModel):
    """Structured observations extracted by the Vision Processing Layer."""
    image_quality: str = Field(default="Good", description="Image quality assessment")
    leaf_color: str = Field(default="Green with lesions", description="Color description of paddy leaf")
    visible_symptoms: List[str] = Field(default_factory=list, description="Visible pathological symptoms")
    spot_characteristics: str = Field(default="", description="Shape, size, margin of leaf spots")
    pattern_distribution: str = Field(default="", description="Distribution pattern on leaf blade/sheath")
    confidence_estimate: str = Field(default="HIGH", description="Confidence in visual feature extraction")
    raw_observations: str = Field(default="", description="Summary of visual findings for upstream agents")


class RAGContextChunk(BaseModel):
    """Retrieved document metadata chunk from RAG vector store."""
    content: str
    filename: str
    category: str
    page: int
    score: float


# ══════════════════════════════════════════════
# TYPED MEMORY & CASE INTELLIGENCE MODELS
# ══════════════════════════════════════════════
class ConversationTurn(BaseModel):
    """Single turn in a conversation trace."""
    role: str = Field(..., description="Role of speaker: user or assistant")
    content: str = Field(..., description="Text content of turn")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    has_image: bool = Field(default=False, description="Whether turn included an image attachment")
    vision_summary: Optional[str] = Field(default=None, description="Visual observations if image present")


class CaseMemory(BaseModel):
    """Structured crop pathology case tracking model."""
    case_id: str = Field(default_factory=lambda: f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    crop_type: str = Field(default="Paddy (Rice)", description="Target crop classification")
    previous_diagnoses: List[str] = Field(default_factory=list, description="Historical diagnoses in this case")
    uploaded_images_count: int = Field(default=0, description="Total images uploaded during case")
    recommendations_given: List[str] = Field(default_factory=list, description="Historical treatments/NPK recommendations")
    follow_up_notes: List[str] = Field(default_factory=list, description="Follow-up observations or progression notes")


class MemoryMetadata(BaseModel):
    """Tracking metadata for conversation memory session."""
    session_id: str = Field(..., description="Unique session identifier")
    request_id: str = Field(..., description="Active request tracking ID")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory(BaseModel):
    """Structured Conversation Memory & Case Intelligence container."""
    turns: List[ConversationTurn] = Field(default_factory=list, description="Chronological turn history")
    case_memory: CaseMemory = Field(default_factory=CaseMemory, description="Crop pathology case memory")
    metadata: MemoryMetadata = Field(..., description="Session tracking metadata")


# ══════════════════════════════════════════════
# MULTIMODAL PROCESSING CONTEXT MODEL
# ══════════════════════════════════════════════
class ProcessingContext(BaseModel):
    """
    Multimodal Context Model encapsulating isolated modalities, memory, observability, weather, & XAI explanation:
      - user_query: Original text input string
      - vision_analysis: Independent VisionAnalysisResult object
      - rag_context: List of retrieved RAGContextChunk objects
      - conversation_memory: Typed ConversationMemory & Case Intelligence object
      - request_trace: Typed RequestTrace telemetry object
      - weather_context: Typed WeatherContext object
      - seasonal_advisory: Typed SeasonalAdvisory object
      - explanation: Typed DiagnosisExplanation object
      - metadata: Session and routing metadata
      - attachment_metadata: File metadata (MIME, size, resolution)
    """
    user_query: str = Field(default="", description="Original user text query")
    vision_analysis: Optional[VisionAnalysisResult] = Field(default=None, description="Structured visual observations")
    rag_context: List[RAGContextChunk] = Field(default_factory=list, description="Retrieved vector store documents")
    conversation_memory: Optional[ConversationMemory] = Field(default=None, description="Typed Conversation Memory & Case Intelligence")
    request_trace: Optional[Any] = Field(default=None, description="Typed RequestTrace telemetry object")
    weather_context: Optional[Any] = Field(default=None, description="Environmental WeatherContext object")
    seasonal_advisory: Optional[Any] = Field(default=None, description="Deterministic SeasonalAdvisory object")
    explanation: Optional[Any] = Field(default=None, description="Deterministic DiagnosisExplanation object")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session and routing metadata")
    attachment_metadata: Dict[str, Any] = Field(default_factory=dict, description="Attachment properties")


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
    context: Optional[ProcessingContext] = Field(default=None, description="Structured Multimodal ProcessingContext")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


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
    intent: Optional[Any] = QueryIntent.GENERAL
    diagnostic_info: Optional[Any] = None
    fertilizer_info: Optional[Any] = None
    general_info: Optional[Any] = None
    reflection_result: Optional[Any] = None
    vision_info: Optional[Any] = None
    processing_context: Optional[Any] = None
    request_trace: Optional[Any] = None
    evaluation_result: Optional[Any] = None
    weather_info: Optional[Any] = None
    seasonal_advisory: Optional[Any] = None
    explanation: Optional[Any] = None
    final_synthesis: str
    message_trace: List[AgentMessage] = []

    def __str__(self) -> str:
        return self.final_synthesis or ""

    def __contains__(self, item: str) -> bool:
        return item in (self.final_synthesis or "")

    def __len__(self) -> int:
        return len(self.final_synthesis or "")

    def startswith(self, prefix: str) -> bool:
        return (self.final_synthesis or "").startswith(prefix)

    def __getitem__(self, item: Any) -> Any:
        return (self.final_synthesis or "")[item]
