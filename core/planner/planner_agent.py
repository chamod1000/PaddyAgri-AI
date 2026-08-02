"""
PlannerAgent Module - Version 3.0 Architecture
Ultra-fast JSON Planner Compiler (<300 tokens, <400ms) for high-level capability task planning.
"""

import json
import time
from typing import Optional, List, Dict, Any
from core.agents.base_agent import BaseAgent
from core.planner.planner_schemas import PlannerOutputV3, PlanningIntent
from config.model_provider import get_router_model


class PlannerAgent(BaseAgent):
    """
    Lightweight JSON Planner Compiler.
    Does NOT generate answers. Compiles user query into PlannerOutputV3 JSON schema.
    """

    def __init__(self):
        super().__init__(name="PlannerAgent", model=None)

    def plan(self, user_query: str, has_image: bool = False, history: Optional[List[Dict[str, str]]] = None) -> PlannerOutputV3:
        t0 = time.perf_counter()
        q_lower = user_query.lower()

        # Fast Rule-Based Planning Compiler (< 1 ms latency)
        has_weather_kw = any(w in q_lower for w in ["weather", "forecast", "rain", "temperature", "humidity", "monsoon", "climate"])
        has_risk_kw = any(w in q_lower for w in ["risk", "fungal risk", "disease risk", "advisory"])
        has_diag_kw = any(w in q_lower for w in ["blast", "blight", "spot", "rot", "lesion", "yellowing", "symptom", "treatment", "cure", "diagnose"])
        has_fert_kw = any(w in q_lower for w in ["fertilizer", "fertiliser", "npk", "urea", "tsp", "mop", "potash", "dosage", "top dressing"])
        has_market_kw = any(w in q_lower for w in ["price", "prices", "market", "buying", "samba", "nadu", "cost per kg"])
        is_info_def = any(q_lower.startswith(p) for p in ["what is ", "what are ", "explain ", "tell me about ", "describe ", "define "]) and not has_image and not any(w in q_lower for w in ["my ", "diagnose", "symptom", "lesion", "yellowing", "dying"])

        tasks = []

        if has_weather_kw or (has_risk_kw and not has_diag_kw):
            intent = PlanningIntent.ENVIRONMENTAL_RISK
            tasks.append("weather_intelligence")
            tasks.append("knowledge_retrieval")

        elif has_fert_kw and has_diag_kw and not is_info_def:
            intent = PlanningIntent.CROP_DIAGNOSIS
            tasks.append("pathology_diagnosis")
            tasks.append("npk_formulation")
            tasks.append("knowledge_retrieval")

        elif (has_diag_kw or has_image) and not is_info_def:
            intent = PlanningIntent.CROP_DIAGNOSIS
            tasks.append("pathology_diagnosis")
            tasks.append("knowledge_retrieval")

        elif has_fert_kw:
            intent = PlanningIntent.FERTILIZER_ADVISORY
            tasks.append("npk_formulation")
            tasks.append("knowledge_retrieval")

        elif has_market_kw:
            intent = PlanningIntent.MARKET_INQUIRY
            tasks.append("market_pricing")

        else:
            intent = PlanningIntent.GENERAL_KNOWLEDGE
            tasks.append("knowledge_retrieval")

        dur_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[PLANNER] Planned in {dur_ms:.2f} ms | Intent: {intent.value} | Tasks: {tasks}")

        return PlannerOutputV3(
            intent=intent,
            confidence=0.98,
            missing_information=[],
            tasks=tasks
        )

    def process(self, message: Any) -> PlannerOutputV3:
        query = getattr(message, "user_query", str(message))
        ctx = getattr(message, "context", None)
        has_image = bool(getattr(ctx, "vision_analysis", None)) if ctx else False
        return self.plan(query, has_image=has_image)
