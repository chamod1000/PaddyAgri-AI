"""
FertilizerAgent Implementation
Agentic Pattern 3: Planning & Calculation Pattern using NPK calculator + RAG + Groq Llama 3.3 70B / Google Gemini model.
"""

import json
from typing import List, Optional

from core.agent_messages import AgentMessage, FertilizerRecommendation, RAGContextChunk
from core.agents.base_agent import BaseAgent
from config.model_provider import get_reasoning_model
from tools.tools import rag_search_tool, fertilizer_calculator_tool


class FertilizerAgent(BaseAgent):
    """
    Specialized agent for soil nutrient requirements and fertilizer application scheduling.
    Uses deep reasoning model + Fertilizer Dosage Calculator tool + RAG search tool.
    Supports Explicit Chain-of-Thought (CoT) and Self-Correction Loop.
    """

    def __init__(self):
        # Default initialization with standard reasoning model
        super().__init__(name="FertilizerAgent", model=get_reasoning_model())

    def process(self, message: AgentMessage, feedback: Optional[str] = None) -> FertilizerRecommendation:
        self._log_start(message)
        query = message.user_query

        # Determine season and district from query context
        season = "Yala" if "yala" in query.lower() else "Maha"
        district = "Polonnaruwa"  # Default major paddy district

        # Step 1: Tool-Use - Calculate base dosage
        calc_result = fertilizer_calculator_tool.invoke({
            "season": season,
            "district": district,
            "field_size_acres": 1.0
        })

        # Step 2: Tool-Use - RAG search for specific soil/fertilizer guidelines
        shared_chunks = message.payload.get("shared_rag_chunks") if message.payload else None
        if shared_chunks is not None:
            print(f"[{self.name}] Using Shared RAG Context from Orchestrator Payload...")
            rag_results = shared_chunks
        else:
            print(f"[{self.name}] Executing RAG Vector Search...")
            rag_results = rag_search_tool.invoke({"query": query, "top_k": 3})
        sources: List[RAGContextChunk] = []
        context_str = ""
        for chunk in rag_results:
            sources.append(RAGContextChunk(
                content=chunk["content"],
                filename=chunk["filename"],
                category=chunk["category"],
                page=chunk["page"],
                score=chunk["score"]
            ))
            context_str += f"\n--- Source: {chunk['filename']} (Page {chunk['page']}) ---\n{chunk['content']}\n"

        # Step 3: High-Speed Structured Synthesis with Reasoning LLM
        system_prompt = (
            "You are a Sri Lankan Paddy Agronomist. Recommend NPK fertilizer using the calculator rates and RAG context.\n"
            "LIMITS: Urea ≤105kg/ac (Maha), ≤85kg/ac (Yala). TSP ≤35kg/ac. MOP ≤35kg/ac.\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "urea_dosage_per_acre_kg": 50.0,\n'
            '  "tsp_dosage_per_acre_kg": 25.0,\n'
            '  "mop_dosage_per_acre_kg": 25.0,\n'
            '  "season": "Yala",\n'
            '  "application_schedule": ["Basal: TSP 25kg + Urea 10kg", "Top 1 (2wk): Urea 20kg", "Top 2 (5wk): MOP 25kg + Urea 20kg"]\n'
            "}"
        )

        user_content = (
            f"Farmer Query: {query}\n"
            f"Base Calculator Rates for 1 Acre: Urea={calc_result['urea_kg']} kg, TSP={calc_result['tsp_kg']} kg, MOP={calc_result['mop_kg']} kg\n"
            f"Base Application Schedule: {calc_result['schedule']}\n\n"
            f"Retrieved Knowledge Base Context:\n{context_str}"
        )

        if feedback:
            user_content += (
                f"\n\n⚠️ REFLECTION CRITIQUE & SELF-CORRECTION FEEDBACK:\n{feedback}\n"
                f"Please review this critique, self-correct your previous analysis, and provide a rectified response."
            )

        try:
            response = self.invoke_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ])
            raw_text = response.content.strip()

            # Clean JSON formatting if wrapped in markdown blocks
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()

            parsed = json.loads(raw_text)
            result = FertilizerRecommendation(
                thought_process=parsed.get("thought_process", "Calculated via DOA NPK Guidelines"),
                season=season,
                district_zone=district,
                urea_dosage_per_acre_kg=float(parsed.get("urea_dosage_per_acre_kg", calc_result["urea_kg"])),
                tsp_dosage_per_acre_kg=float(parsed.get("tsp_dosage_per_acre_kg", calc_result["tsp_kg"])),
                mop_dosage_per_acre_kg=float(parsed.get("mop_dosage_per_acre_kg", calc_result["mop_kg"])),
                application_schedule=parsed.get("application_schedule", calc_result["schedule"]),
                rag_sources=sources
            )
            self._log_success(message.message_id)
            return result
        except Exception as e:
            self._log_error(e, "Fallback fertilizer synthesis")
            return FertilizerRecommendation(
                thought_process="Failed parsing response, fallback used.",
                season=season,
                district_zone=district,
                urea_dosage_per_acre_kg=calc_result["urea_kg"],
                tsp_dosage_per_acre_kg=calc_result["tsp_kg"],
                mop_dosage_per_acre_kg=calc_result["mop_kg"],
                application_schedule=calc_result["schedule"],
                rag_sources=sources
            )
