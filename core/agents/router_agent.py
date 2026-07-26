"""
RouterAgent Implementation
Agentic Pattern 1: Router Pattern using fast Groq Llama 3.1 8B model.
Includes Sinhala & Singlish language auto-detection logic.
"""

from core.agent_messages import AgentMessage, QueryIntent
from core.agents.base_agent import BaseAgent
from config.model_provider import get_router_model, detect_language_and_script


class RouterAgent(BaseAgent):
    """
    Uses fast, low-latency LLM (Groq Llama 3.1 8B) to route incoming farmer queries.
    Also acts as language script detector helper.
    """

    def __init__(self):
        super().__init__(name="RouterAgent", model=get_router_model())

    def route_query(self, user_query: str) -> QueryIntent:
        """Dynamically classifies farmer query intent using AI LLM reasoning."""
        system_prompt = (
            "You are an intelligent agricultural intent routing agent for a Sri Lankan Paddy Farming System.\n"
            "Analyze the farmer's query and classify it into EXACTLY ONE of these categories:\n"
            "- DISEASE_DIAGNOSIS : If query specifically asks about plant leaf yellowing, spots, blights, pests, or disease symptoms.\n"
            "- FERTILIZER_RECOMMENDATION : If query specifically asks about NPK fertilizer, urea dosage, soil nutrients, or Yala/Maha fertilizer schedules.\n"
            "- BOTH : If query asks about both plant disease/pests AND fertilizer/soil treatment.\n"
            "- GENERAL : If query asks about general paddy cultivation steps (ගොයම් වගා කරන ආකාරය), land preparation, seed paddy standards, germination, quarantine acts, or general farming guidelines.\n\n"
            "Respond ONLY with the category name string (DISEASE_DIAGNOSIS, FERTILIZER_RECOMMENDATION, BOTH, or GENERAL)."
        )

        try:
            response = self.invoke_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            content = response.content.strip().upper()

            for intent in QueryIntent:
                if intent.value in content:
                    return intent
            return QueryIntent.GENERAL
        except Exception as e:
            self._log_error(e, "Routing failed, defaulting to BOTH")
            return QueryIntent.BOTH

    def process(self, message: AgentMessage) -> QueryIntent:
        """Standard process interface for the base agent."""
        return self.route_query(message.user_query)
