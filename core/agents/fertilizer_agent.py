"""
FertilizerAgent Implementation
Agentic Pattern 3: Planning & Calculation Pattern using NPK calculator + RAG + Groq Llama 3.3 70B / Google Gemini model.
"""

from typing import List

from core.agent_messages import AgentMessage, FertilizerRecommendation, RAGContextChunk
from core.agents.base_agent import BaseAgent
from config.model_provider import get_reasoning_model, detect_language_and_script
from tools.tools import rag_search_tool, fertilizer_calculator_tool


class FertilizerAgent(BaseAgent):
    """
    Specialized agent for soil nutrient requirements and fertilizer application scheduling.
    Uses deep reasoning model + Fertilizer Dosage Calculator tool + RAG search tool.
    Dynamically routes to Gemini if Sinhala or Singlish language is detected.
    """

    def __init__(self):
        # Default initialization with standard reasoning model
        super().__init__(name="FertilizerAgent", model=get_reasoning_model())

    def process(self, message: AgentMessage) -> FertilizerRecommendation:
        self._log_start(message)
        query = message.user_query

        # Step 0: Language & script auto-detection for routing to dedicated Gemini model
        is_sinhala_or_singlish = detect_language_and_script(query)
        self.model = get_reasoning_model(is_sinhala_or_singlish=is_sinhala_or_singlish)

        # Determine season and district from query context
        season = "Yala" if "yala" in query.lower() or "යල" in query else "Maha"
        district = "Polonnaruwa"  # Default major paddy district

        # Step 1: Tool-Use - Calculate base dosage
        calc_result = fertilizer_calculator_tool.invoke({
            "season": season,
            "district": district,
            "field_size_acres": 1.0
        })

        # Step 2: Tool-Use - RAG search for specific soil/fertilizer guidelines
        rag_results = rag_search_tool.invoke({"query": f"fertilizer recommendations {season} paddy", "top_k": 3})
        sources: List[RAGContextChunk] = []
        for chunk in rag_results:
            sources.append(RAGContextChunk(
                content=chunk["content"],
                filename=chunk["filename"],
                category=chunk["category"],
                page=chunk["page"],
                score=chunk["score"]
            ))

        result = FertilizerRecommendation(
            season=season,
            district_zone=district,
            urea_dosage_per_acre_kg=calc_result["urea_kg"],
            tsp_dosage_per_acre_kg=calc_result["tsp_kg"],
            mop_dosage_per_acre_kg=calc_result["mop_kg"],
            application_schedule=calc_result["schedule"],
            rag_sources=sources
        )
        self._log_success(message.message_id)
        return result
