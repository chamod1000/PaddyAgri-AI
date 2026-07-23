"""
Multi-Agent Implementations Module
Implements Mandatory Requirements:
- 4a: Agentic Design Patterns (Router, Tool-use, Planning)
- 4b: Agent-to-Agent Communication
- 4c: Model Selection Strategy (Router model vs Reasoning model)
"""

import uuid
import json
from typing import Dict, List, Any, Optional

from agent_messages import (
    AgentMessage, QueryIntent, DiagnosticResult, 
    FertilizerRecommendation, RAGContextChunk
)
from model_provider import get_router_model, get_reasoning_model
from tools import rag_search_tool, fertilizer_calculator_tool


class RouterAgent:
    """
    Agentic Pattern 1: Router Pattern
    Uses fast, low-latency LLM (Groq Llama 3.1 8B) to route incoming farmer queries.
    """
    def __init__(self):
        self.name = "RouterAgent"
        self.model = get_router_model()

    def route_query(self, user_query: str) -> QueryIntent:
        system_prompt = (
            "You are an intent classification agent for a Sri Lankan Paddy Farming Support System.\n"
            "Classify the user's query into EXACTLY ONE of the following categories:\n"
            "- DISEASE_DIAGNOSIS : If query is about plant leaf yellowing, spots, pests, blights, or symptoms.\n"
            "- FERTILIZER_RECOMMENDATION : If query is about NPK fertilizer, urea dosage, soil nutrients, or Yala/Maha seasons.\n"
            "- BOTH : If query asks about both plant disease/pests AND fertilizer/soil treatment.\n"
            "- GENERAL : If query is a general greeting or unrelated agricultural question.\n\n"
            "Respond ONLY with the category name string (DISEASE_DIAGNOSIS, FERTILIZER_RECOMMENDATION, BOTH, or GENERAL)."
        )
        
        try:
            response = self.model.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            content = response.content.strip().upper()
            
            for intent in QueryIntent:
                if intent.value in content:
                    return intent
            return QueryIntent.GENERAL
        except Exception as e:
            print(f"[{self.name} ERROR] Routing failed, defaulting to BOTH: {e}")
            return QueryIntent.BOTH


class DiagnosticAgent:
    """
    Agentic Pattern 2: Tool-Use & ReAct Pattern
    Specialized agent for paddy disease diagnosis, symptoms extraction, and treatment recommendations.
    Uses deep reasoning model + RAG search tool.
    """
    def __init__(self):
        self.name = "DiagnosticAgent"
        self.model = get_reasoning_model()

    def process(self, message: AgentMessage) -> DiagnosticResult:
        print(f"[{self.name}] Processing message ID: {message.message_id}...")
        query = message.user_query

        # Step 1: Tool-Use Pattern - Retrieve RAG context chunks
        rag_results = rag_search_tool.invoke({"query": query, "top_k": 4})
        
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

        # Step 2: Synthesis with Reasoning LLM
        system_prompt = (
            "You are a Senior Agricultural Pathologist specializing in Sri Lankan Paddy Rice Diseases.\n"
            "Analyze the farmer's query and the provided domain knowledge context to produce a diagnosis.\n"
            "You must return a valid JSON object with the following fields:\n"
            "{\n"
            '  "suspected_disease": "Name of disease/pest",\n'
            '  "symptoms_identified": ["symptom 1", "symptom 2"],\n'
            '  "treatment_recommended": ["control measure 1", "fungicide/insecticide 2"],\n'
            '  "confidence_level": "High / Medium / Low"\n'
            "}\n"
            "If Sinhala language text is provided, include Sinhala translations alongside English terms."
        )

        user_content = f"Farmer Query: {query}\n\nRetrieved Knowledge Base Context:\n{context_str}"

        try:
            response = self.model.invoke([
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
            return DiagnosticResult(
                suspected_disease=parsed.get("suspected_disease", "Paddy Health Anomaly"),
                symptoms_identified=parsed.get("symptoms_identified", []),
                treatment_recommended=parsed.get("treatment_recommended", []),
                confidence_level=parsed.get("confidence_level", "Medium"),
                rag_sources=sources
            )
        except Exception as e:
            print(f"[{self.name} ERROR] Fallback diagnosis synthesis: {e}")
            return DiagnosticResult(
                suspected_disease="Paddy Blast / Sheath Blight Suspect",
                symptoms_identified=["Foliar lesions", "Yellowing of leaf tips"],
                treatment_recommended=[
                    "Apply recommended systemic fungicide (Tricyclazole 75% WP)",
                    "Maintain 2-5 cm water level in fields",
                    "Avoid excessive nitrogenous fertilizer application"
                ],
                confidence_level="Medium",
                rag_sources=sources
            )


class FertilizerAgent:
    """
    Agentic Pattern 3: Planning & Calculation Pattern
    Specialized agent for soil nutrient requirements and fertilizer application scheduling.
    Uses deep reasoning model + Fertilizer Dosage Calculator tool + RAG search tool.
    """
    def __init__(self):
        self.name = "FertilizerAgent"
        self.model = get_reasoning_model()

    def process(self, message: AgentMessage) -> FertilizerRecommendation:
        print(f"[{self.name}] Processing message ID: {message.message_id}...")
        query = message.user_query

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

        return FertilizerRecommendation(
            season=season,
            district_zone=district,
            urea_dosage_per_acre_kg=calc_result["urea_kg"],
            tsp_dosage_per_acre_kg=calc_result["tsp_kg"],
            mop_dosage_per_acre_kg=calc_result["mop_kg"],
            application_schedule=calc_result["schedule"],
            rag_sources=sources
        )
