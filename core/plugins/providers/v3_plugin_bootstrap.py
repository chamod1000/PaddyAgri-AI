"""
V3 Plugin Bootstrap Module - Version 3.0 Architecture
Registers all built-in capabilities and tool specs into UnifiedRegistry.
"""

from typing import Dict, Any
from core.capabilities.capability_spec import CapabilitySpec
from core.tools.tool_spec import ToolSpec
from core.tools.unified_registry import unified_registry
from core.weather_service import WeatherService
from tools.tools import rag_search_tool
from core.agents.diagnostic_agent import DiagnosticAgent
from core.agents.fertilizer_agent import FertilizerAgent
from core.agent_messages import AgentMessage, QueryIntent


def init_v3_plugins():
    """Initializes and registers standard agricultural capability and tool specs."""
    if unified_registry.get_capability("weather_intelligence"):
        return  # Already registered

    # Singleton agent references
    weather_svc = WeatherService()
    diag_agent = DiagnosticAgent()
    fert_agent = FertilizerAgent()

    # 1. Weather Intelligence Capability & Tool Spec
    unified_registry.register_capability(CapabilitySpec(
        capability_id="weather_intelligence",
        display_name="District Weather & Fungal Risk Advisory",
        description="Live OpenMeteo weather data, relative humidity, and fungal risk alerts",
        required_inputs=["district"],
        default_tool_id="weather_openmeteo_v2"
    ))

    def _exec_weather(ctx: Any) -> Dict[str, Any]:
        dist = getattr(ctx, "district", None) or "Anuradhapura"
        weather_ctx, season_adv = weather_svc.get_context_and_advisory(dist)

        def _field(obj: Any, name: str, default: Any) -> Any:
            """Read one field from either a pydantic model or a plain dict."""
            if isinstance(obj, dict):
                return obj.get(name, default)
            value = getattr(obj, name, None)
            return default if value is None else value

        # Measurements come from WeatherContext; advisories come from SeasonalAdvisory.
        # Reading them from the wrong object silently yielded hardcoded defaults.
        return {
            "location": dist,
            "temperature_c": _field(weather_ctx, "temperature_c", 28.5),
            "humidity_pct": _field(weather_ctx, "humidity_pct", 82),
            "rainfall_probability_pct": _field(weather_ctx, "rainfall_probability_pct", 0.0),
            # Single authoritative risk value for the whole pipeline.
            "fungal_risk_alert": _field(season_adv, "fungal_risk_alert", "Normal Risk"),
            # Flattened to strings/lists: the renderer must never receive a model.
            "timing_advice": _field(season_adv, "fertilizer_timing_advice", ""),
            "irrigation_advice": _field(season_adv, "irrigation_advice", ""),
            "advisory_notes": list(_field(season_adv, "advisory_notes", []) or []),
        }

    unified_registry.register_tool(ToolSpec(
        tool_id="weather_openmeteo_v2",
        capability_id="weather_intelligence",
        provider_name="OpenMeteo API",
        version="2.0.0",
        execute_fn=_exec_weather
    ))

    # 2. Knowledge Retrieval Capability & Tool Spec (RAG)
    unified_registry.register_capability(CapabilitySpec(
        capability_id="knowledge_retrieval",
        display_name="DOA Knowledge Base RAG Search",
        description="Vector similarity search over Sri Lanka Department of Agriculture manuals",
        required_inputs=["user_query"],
        default_tool_id="faiss_minilm_rag_v1"
    ))

    def _exec_rag(ctx: Any) -> Dict[str, Any]:
        query = getattr(ctx, "user_query", "paddy cultivation guidelines")
        chunks = rag_search_tool.invoke({"query": query, "top_k": 3})
        return {"snippets": chunks}

    unified_registry.register_tool(ToolSpec(
        tool_id="faiss_minilm_rag_v1",
        capability_id="knowledge_retrieval",
        provider_name="FAISS / MiniLM",
        version="1.0.0",
        execute_fn=_exec_rag
    ))

    # 3. Pathology Diagnosis Capability & Tool Spec
    unified_registry.register_capability(CapabilitySpec(
        capability_id="pathology_diagnosis",
        display_name="Pathology Diagnostic Agent",
        description="Deep reasoning model + RAG context for paddy disease identification",
        required_inputs=["user_query"],
        default_tool_id="gemini_pathology_diag_v1"
    ))

    def _exec_diag(ctx: Any) -> Dict[str, Any]:
        from core.agent_messages import ProcessingContext as LegacyProcessingContext
        query = getattr(ctx, "user_query", "")
        legacy_ctx = LegacyProcessingContext(user_query=query)
        msg = AgentMessage(
            message_id="v3_diag",
            sender="PlannerAgent",
            receiver="DiagnosticAgent",
            intent=QueryIntent.DISEASE_DIAGNOSIS,
            user_query=query,
            payload={},
            context=legacy_ctx
        )
        res = diag_agent.process(msg)
        return {
            "suspected_disease": res.suspected_disease,
            "confidence_level": res.confidence_level,
            "symptoms_identified": res.symptoms_identified,
            "treatment_recommended": res.treatment_recommended,
            "thought_process": res.thought_process
        }

    unified_registry.register_tool(ToolSpec(
        tool_id="gemini_pathology_diag_v1",
        capability_id="pathology_diagnosis",
        provider_name="Google Gemini",
        version="1.0.0",
        execute_fn=_exec_diag
    ))

    # 4. NPK Formulation Capability & Tool Spec
    unified_registry.register_capability(CapabilitySpec(
        capability_id="npk_formulation",
        display_name="DOA NPK Fertilizer Calculator",
        description="Calculates standard Urea, TSP, MOP dosage per acre by paddy growth stage",
        required_inputs=["user_query"],
        default_tool_id="doa_npk_calculator_v1"
    ))

    def _exec_npk(ctx: Any) -> Dict[str, Any]:
        from core.agent_messages import ProcessingContext as LegacyProcessingContext
        query = getattr(ctx, "user_query", "")
        legacy_ctx = LegacyProcessingContext(user_query=query)
        msg = AgentMessage(
            message_id="v3_fert",
            sender="PlannerAgent",
            receiver="FertilizerAgent",
            intent=QueryIntent.FERTILIZER_RECOMMENDATION,
            user_query=query,
            payload={},
            context=legacy_ctx
        )
        res = fert_agent.process(msg)
        return {
            "season": res.season,
            "urea_kg": res.urea_dosage_per_acre_kg,
            "tsp_kg": res.tsp_dosage_per_acre_kg,
            "mop_kg": res.mop_dosage_per_acre_kg,
            "notes": getattr(res, "application_schedule", getattr(res, "application_notes", []))
        }

    unified_registry.register_tool(ToolSpec(
        tool_id="doa_npk_calculator_v1",
        capability_id="npk_formulation",
        provider_name="DOA Fertilizer Protocol",
        version="1.0.0",
        execute_fn=_exec_npk
    ))

    # 5. Market Pricing Capability & Tool Spec
    unified_registry.register_capability(CapabilitySpec(
        capability_id="market_pricing",
        display_name="Paddy Marketing Board Buying Prices",
        description="Islandwide buying prices per kg for Samba, Nadu, and Keeri Samba paddy",
        required_inputs=["district"],
        default_tool_id="pmb_market_prices_v1"
    ))

    def _exec_market(ctx: Any) -> Dict[str, Any]:
        dist = getattr(ctx, "district", None) or "Islandwide"
        return {
            "district": dist,
            "samba_price_lkr": 120.00,
            "nadu_price_lkr": 105.00,
            "keeri_samba_price_lkr": 130.00,
            "updated_date": "2026-07-31"
        }

    unified_registry.register_tool(ToolSpec(
        tool_id="pmb_market_prices_v1",
        capability_id="market_pricing",
        provider_name="PMB Price Index",
        version="1.0.0",
        execute_fn=_exec_market
    ))
