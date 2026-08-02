"""
Internal AI Observability & Evaluation Tracing Legacy Module (core/observability_legacy.py)

Defines strongly typed observability models for enterprise tracing:
  - VisionTrace
  - RAGTrace
  - AgentExecutionTrace
  - PerformanceTrace
  - EvaluationTrace
  - RequestTrace
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VisionTrace(BaseModel):
    """Observability trace for the Vision Processing Layer."""
    image_validated: bool = Field(default=True, description="Whether image passed validation")
    mime_type: str = Field(default="image/jpeg", description="MIME type of uploaded image")
    resolution: str = Field(default="Unknown", description="Resolution dimensions")
    extraction_latency_ms: float = Field(default=0.0, description="Vision LLM extraction time in ms")
    vision_provider: str = Field(default="Google Gemini Vision", description="Provider used for vision model")
    confidence_estimate: str = Field(default="HIGH", description="Confidence rating")


class RAGTrace(BaseModel):
    """Observability trace for RAG Vector Search."""
    query: str = Field(..., description="Query passed to FAISS retriever")
    documents_retrieved_count: int = Field(default=0, description="Number of context chunks returned")
    search_latency_ms: float = Field(default=0.0, description="FAISS search execution time in ms")
    top_document_scores: List[float] = Field(default_factory=list, description="Top document similarity scores")
    sources: List[str] = Field(default_factory=list, description="Source filenames referenced")


class AgentExecutionTrace(BaseModel):
    """Observability trace for individual Agent execution."""
    agent_name: str = Field(..., description="Name of executed agent")
    status: str = Field(default="COMPLETED", description="Status: COMPLETED, FAILED, SKIPPED")
    execution_time_ms: float = Field(default=0.0, description="Agent processing latency in ms")
    llm_provider: str = Field(default="Gemini / Groq", description="LLM provider invoked")
    fallback_triggered: bool = Field(default=False, description="Whether fallback model was triggered")
    warnings: List[str] = Field(default_factory=list, description="Warnings or critiques recorded")


class PerformanceTrace(BaseModel):
    """Comprehensive request latency trace breakdown."""
    intent_latency_ms: float = Field(default=0.0, description="Intent routing latency in ms")
    rag_latency_ms: float = Field(default=0.0, description="RAG search latency in ms")
    parallel_agents_latency_ms: float = Field(default=0.0, description="Reasoning agents execution latency in ms")
    reflection_latency_ms: float = Field(default=0.0, description="Reflection safety check latency in ms")
    total_latency_ms: float = Field(default=0.0, description="End-to-end processing latency in ms")


class EvaluationTrace(BaseModel):
    """Internal evaluation and benchmarking metrics trace."""
    intent_classification_correct: bool = Field(default=True, description="Evaluated intent accuracy")
    rag_relevance_score: float = Field(default=1.0, description="Relevance score of retrieved chunks")
    reflection_passed: bool = Field(default=True, description="Whether safety reflection checks passed")
    confidence_score: float = Field(default=1.0, description="Aggregated confidence metric")


class RequestTrace(BaseModel):
    """
    Root Observability Request Trace.
    Encapsulates full end-to-end system telemetry for a single user request.
    """
    request_id: str = Field(..., description="Unique request tracking ID")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    intent: str = Field(..., description="Classified intent value")
    selected_agents: List[str] = Field(default_factory=list, description="List of agents dispatched")
    vision_trace: Optional[VisionTrace] = Field(default=None, description="Vision layer telemetry")
    rag_trace: Optional[RAGTrace] = Field(default=None, description="RAG search telemetry")
    agent_traces: List[AgentExecutionTrace] = Field(default_factory=list, description="Agent execution traces")
    performance: PerformanceTrace = Field(default_factory=PerformanceTrace, description="Latency breakdown")
    evaluation: EvaluationTrace = Field(default_factory=EvaluationTrace, description="Evaluation metrics")
