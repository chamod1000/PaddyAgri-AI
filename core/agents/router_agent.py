"""
RouterAgent Implementation
Agentic Pattern 1: Router Pattern using Ultra-Fast Keyword-Based Intent Classification.
"""

import re
from core.agent_messages import AgentMessage, QueryIntent
from core.agents.base_agent import BaseAgent
from config.model_provider import get_router_model


# ── Keyword dictionaries for instant intent classification ──
_DISEASE_KEYWORDS = {
    # English
    "disease", "blast", "blight", "sheath", "rot", "leaf spot", "brown spot",
    "bacterial", "fungal", "fungus", "virus", "wilt", "smut", "tungro",
    "yellowing", "lesion", "lesions", "symptom", "symptoms", "infection",
    "pest", "pests", "insect", "insects", "stem borer", "brown planthopper",
    "bph", "thrips", "mite", "caterpillar", "worm", "grub", "bug",
    "diagnosis", "diagnose", "pathology", "treatment", "fungicide",
    "insecticide", "pesticide", "spray", "control", "cure"
}

_FERTILIZER_KEYWORDS = {
    # English
    "fertilizer", "fertiliser", "npk", "urea", "tsp", "mop", "potash",
    "nitrogen", "phosphorus", "potassium", "dosage", "nutrient", "nutrients",
    "soil", "basal", "top dressing", "application schedule",
    "compost", "organic fertilizer", "manure"
}

_SEASON_KEYWORDS = {
    "yala", "maha", "season"
}


def _classify_intent(query: str) -> QueryIntent:
    """
    Zero-latency keyword-based intent classifier.
    Matches English keywords.
    Falls back to GENERAL if no strong signal detected.
    """
    q_lower = query.lower()

    has_disease = any(kw in q_lower for kw in _DISEASE_KEYWORDS)
    has_fertilizer = any(kw in q_lower for kw in _FERTILIZER_KEYWORDS)

    if has_disease and has_fertilizer:
        return QueryIntent.BOTH
    elif has_disease:
        return QueryIntent.DISEASE_DIAGNOSIS
    elif has_fertilizer:
        return QueryIntent.FERTILIZER_RECOMMENDATION
    else:
        return QueryIntent.GENERAL


class RouterAgent(BaseAgent):
    """
    Uses ultra-fast keyword-based classification for instant intent routing (0ms latency).
    Falls back to GENERAL intent if keyword matching is ambiguous.
    """

    def __init__(self):
        super().__init__(name="RouterAgent", model=None)

    def route_query(self, user_query: str) -> QueryIntent:
        """Instant keyword-based intent classification (0ms latency)."""
        intent = _classify_intent(user_query)
        print(f"[{self.name}] Instant keyword classification: {intent.value}")
        return intent

    def process(self, message: AgentMessage) -> QueryIntent:
        """Standard process interface for the base agent."""
        return self.route_query(message.user_query)
