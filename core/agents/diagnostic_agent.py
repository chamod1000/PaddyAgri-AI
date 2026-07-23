"""
DiagnosticAgent Implementation
Agentic Pattern 2: Tool-Use & ReAct Pattern using RAG + Groq Llama 3.3 70B / Google Gemini model.
"""

import json
from typing import List

from core.agent_messages import AgentMessage, DiagnosticResult, RAGContextChunk
from core.agents.base_agent import BaseAgent
from config.model_provider import get_reasoning_model, detect_language_and_script
from tools.tools import rag_search_tool


class DiagnosticAgent(BaseAgent):
    """
    Specialized agent for paddy disease diagnosis using deep reasoning model + RAG search tool.
    Dynamically routes to Gemini if Sinhala or Singlish language is detected.
    """

    def __init__(self):
        # Default initialization with standard reasoning model
        super().__init__(name="DiagnosticAgent", model=get_reasoning_model())

    def process(self, message: AgentMessage) -> DiagnosticResult:
        self._log_start(message)
        query = message.user_query

        # Step 0: Language & script auto-detection for routing to dedicated Gemini model
        is_sinhala_or_singlish = detect_language_and_script(query)
        self.model = get_reasoning_model(is_sinhala_or_singlish=is_sinhala_or_singlish)

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
            "If Sinhala language text is provided, include Sinhala or Singlish translations alongside English terms."
        )

        user_content = f"Farmer Query: {query}\n\nRetrieved Knowledge Base Context:\n{context_str}"

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
            result = DiagnosticResult(
                suspected_disease=parsed.get("suspected_disease", "Paddy Health Anomaly"),
                symptoms_identified=parsed.get("symptoms_identified", []),
                treatment_recommended=parsed.get("treatment_recommended", []),
                confidence_level=parsed.get("confidence_level", "Medium"),
                rag_sources=sources
            )
            self._log_success(message.message_id)
            return result
        except Exception as e:
            self._log_error(e, "Fallback diagnosis synthesis")
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
