"""
Multi-Agent Orchestrator Module
Coordinates agent-to-agent communication, manages workflow execution,
and synthesizes multi-agent responses for Sri Lankan paddy farmers.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import copy
import hashlib
import uuid
from datetime import datetime
from typing import List, Optional, Callable, Any

from core.agent_messages import (
    AgentMessage, QueryIntent, AgentResponse,
    DiagnosticResult, FertilizerRecommendation, ReflectionResult,
    VisionAnalysisResult, ProcessingContext, RAGContextChunk,
    ConversationTurn, CaseMemory, MemoryMetadata, ConversationMemory
)
from core.observability import (
    RequestTrace, VisionTrace, RAGTrace,
    AgentExecutionTrace, PerformanceTrace, EvaluationTrace
)
from core.weather_service import WeatherService
from core.agents import DiagnosticAgent, FertilizerAgent, ReflectionAgent
from core.vision_processor import VisionProcessor
from config.model_provider import get_reasoning_model
from tools.tools import rag_search_tool, get_cached_vector_store


class PaddyAgentOrchestrator:
    """
    Main Orchestrator coordinating structured message exchange, Telemetry, Weather Context, & V3.0 Capabilities:
    - VisionProcessor (Vision Abstraction Layer)
    - DiagnosticAgent (Tool-Use Pattern)
    - FertilizerAgent (Planning & Calculation Pattern)
    - WeatherService (Environmental & Seasonal Rule Engine)
    - SynthesisAgent (Fallback Token Generator Reference)
    """

    def __init__(self):
        import time
        print("[ORCHESTRATOR V3.0] Initializing Core Services...")
        t_total_start = time.perf_counter()

        self.vision_processor = VisionProcessor()
        self.diagnostic_agent = DiagnosticAgent()
        self.fertilizer_agent = FertilizerAgent()
        self.weather_service = WeatherService()
        self.session_memory: Dict[str, ConversationMemory] = {}

        from core.agents.synthesis_agent import SynthesisAgent
        self.synthesis_agent = SynthesisAgent()

        # Pre-warm Vector Store (FAISS & MiniLM Embeddings)
        get_cached_vector_store()

        # Initialize V3.0 Architecture Capabilities & Tool Specs
        from core.plugins.providers.v3_plugin_bootstrap import init_v3_plugins
        init_v3_plugins()

        tot_dur_ms = (time.perf_counter() - t_total_start) * 1000.0
        print(f"[ORCHESTRATOR V3.0] Initialization Complete in {tot_dur_ms:.2f} ms ({tot_dur_ms/1000.0:.2f} s)", flush=True)

    def get_or_create_memory(self, session_id: str = "active_session") -> ConversationMemory:
        """Retrieves or initializes strongly typed ConversationMemory and CaseMemory."""
        if session_id not in self.session_memory:
            now_str = datetime.now().isoformat()
            self.session_memory[session_id] = ConversationMemory(
                metadata=MemoryMetadata(
                    session_id=session_id,
                    request_id=f"req_{uuid.uuid4().hex[:6]}",
                    created_at=now_str,
                    updated_at=now_str
                ),
                case_memory=CaseMemory(case_id=f"case_{session_id}"),
                turns=[]
            )
        return self.session_memory[session_id]

    def process_user_request(
        self,
        user_query: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
        step_callback: Optional[Callable[[int, str], None]] = None,
        session_id: str = "active_session"
    ) -> Any:
        import concurrent.futures
        import time
        from core.cache import response_cache
        from core.observability import TraceContext, telemetry_bus
        from core.context import RequestContext, ProcessingContext
        from core.planner import PlannerAgent, PlanningIntent
        from core.tools import AdaptiveToolResolver
        from core.executor import IntelligentExecutor
        from core.evidence import EvidenceGraph
        from core.synthesis import HybridResponseBuilder
        from core.reflection import RegulatoryReflection

        req_id = f"msg_{uuid.uuid4().hex[:8]}"
        trace_ctx = TraceContext(session_id=session_id, request_id=req_id)

        t_total_start = time.perf_counter()

        print(f"\n===========================================================================")
        print(f"[ORCHESTRATOR V3.0] Processing Query: '{user_query}' | Trace: {trace_ctx.trace_id} | Session: {session_id}")
        print(f"===========================================================================")

        conv_memory = self.get_or_create_memory(session_id)
        conv_memory.metadata.updated_at = datetime.now().isoformat()

        # 0. Check Response Cache
        image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
        cache_key = response_cache.generate_key(user_query, image_hash=image_hash)
        cached_res = response_cache.get(cache_key)
        if cached_res:
            print(f"[ORCHESTRATOR V3.0] CACHE HIT! Returning cached response in sub-10ms.")
            telemetry_bus.emit("RESPONSE_CACHE_HIT", {"query": user_query}, trace_id=trace_ctx.trace_id)
            if step_callback: step_callback(4, "ResponseCache: Served instant response (0 LLM Tokens)")
            
            cached_text = cached_res if isinstance(cached_res, str) else getattr(cached_res, "final_synthesis", str(cached_res))
            
            # Record cache-hit turns into ConversationMemory
            if cached_text and cached_text.strip():
                conv_memory.turns.append(
                    ConversationTurn(
                        role="user",
                        content=user_query,
                        has_image=image_bytes is not None,
                        vision_summary=None
                    )
                )
                conv_memory.turns.append(
                    ConversationTurn(
                        role="assistant",
                        content=cached_text
                    )
                )

            if isinstance(cached_res, str):
                cached_res = AgentResponse(
                    query=user_query,
                    intent=QueryIntent.GENERAL,
                    final_synthesis=cached_res
                )
            return cached_res, self.synthesis_agent

        # Step 0: Vision Processing Modality
        vision_result: Optional[VisionAnalysisResult] = None
        if image_bytes:
            if step_callback: step_callback(1, "Vision Layer: Extracting visual pathology features...")
            print("[ORCHESTRATOR V3.0] Ingesting attached image through Vision Processor...")
            vision_result = self.vision_processor.analyze_image(image_bytes, user_query=user_query)

        # Extract recent turns (sliding window of last 4 turns) from existing ConversationMemory
        recent_history = [
            {"role": turn.role, "content": turn.content}
            for turn in conv_memory.turns[-4:]
        ]

        # Step 1: Planner Agent (Lightweight JSON Compiler <300 tokens, <400ms)
        if step_callback: step_callback(1, "PlannerAgent V3.0: Compiling query into capability plan...")
        planner = PlannerAgent()
        planner_output = planner.plan(user_query, has_image=image_bytes is not None)
        telemetry_bus.emit("PLANNER_SUCCESS", planner_output.model_dump(), trace_id=trace_ctx.trace_id)

        # Step 2: Adaptive Tool Resolution & Intelligent DAG Execution
        if step_callback: step_callback(2, f"AdaptiveToolResolver: Resolving capabilities {planner_output.tasks}...")
        resolved_tools = AdaptiveToolResolver.resolve_capabilities(planner_output.tasks)

        # Assemble ProcessingContext
        processing_ctx = ProcessingContext(
            user_query=user_query,
            recent_history=recent_history,
            vision_analysis=vision_result
        )

        evidence_graph = EvidenceGraph(session_id=session_id, user_query=user_query)
        executor = IntelligentExecutor()
        
        if step_callback: step_callback(3, f"IntelligentExecutor: Executing parallel tool DAG...")
        evidence_graph = executor.execute_plan(resolved_tools, processing_ctx, evidence_graph)

        # Step 3: Hybrid Response Synthesis
        if step_callback: step_callback(4, "HybridResponseBuilder: Synthesizing grounded response...")
        final_text, ui_comps = HybridResponseBuilder.render_response(evidence_graph)

        # Step 4: Conditional Regulatory Reflection Audit (Pesticide Act No. 33)
        final_text, has_violation = RegulatoryReflection.audit_response(final_text)

        # Check for failed/empty synthesis before recording in memory
        is_failed_synthesis = (
            not final_text or not final_text.strip() or
            (bool(evidence_graph.errors_encountered) and "No specific evidence was collected" in final_text)
        )

        if not is_failed_synthesis:
            conv_memory.turns.append(
                ConversationTurn(
                    role="user",
                    content=user_query,
                    has_image=image_bytes is not None,
                    vision_summary=vision_result.raw_observations if vision_result else None
                )
            )
            conv_memory.turns.append(
                ConversationTurn(
                    role="assistant",
                    content=final_text
                )
            )

        # Update case memory tracking
        diag_art = evidence_graph.get_artifact("pathology_diagnosis")
        if diag_art and isinstance(diag_art, dict):
            dis = diag_art.get("suspected_disease")
            if dis and dis not in conv_memory.case_memory.previous_diagnoses:
                conv_memory.case_memory.previous_diagnoses.append(dis)
        if image_bytes:
            conv_memory.case_memory.uploaded_images_count += 1

        # Save to Response Cache (only for valid, successful responses)
        banned_failure_terms = (
            "No specific evidence was collected for this query",
            "An internal error occurred",
            "Pipeline execution failed",
            "LLM API error",
            "Service unavailable"
        )
        is_cacheable = (
            isinstance(final_text, str) and
            bool(final_text.strip()) and
            len(final_text.strip()) >= 10 and
            not any(term in final_text for term in banned_failure_terms) and
            not (bool(evidence_graph.errors_encountered) and not evidence_graph.artifacts)
        )
        if is_cacheable:
            response_cache.put(cache_key, final_text)

        tot_dur_ms = (time.perf_counter() - t_total_start) * 1000.0
        print(f"[ORCHESTRATOR V3.0] COMPLETE Pipeline Execution: {tot_dur_ms:.2f} ms")

        # Build structured AgentResponse object for 100% UI contract parity
        fert_art = evidence_graph.get_artifact("npk_formulation")
        weather_art = evidence_graph.get_artifact("weather_intelligence")

        response_obj = AgentResponse(
            query=user_query,
            intent=getattr(planner_output, "intent", QueryIntent.GENERAL),
            diagnostic_info=diag_art,
            fertilizer_info=fert_art,
            general_info=evidence_graph.get_artifact("knowledge_retrieval"),
            reflection_result=ReflectionResult(all_checks_passed=not has_violation),
            vision_info=vision_result,
            processing_context=processing_ctx,
            weather_info=weather_art,
            final_synthesis=final_text
        )

        return response_obj, self.synthesis_agent
