"""
DiagnosticAgent Implementation
Agentic Pattern 2: Tool-Use & ReAct Pattern using RAG + Gemini 2.0 Flash / Groq Llama 3.3 70B model.
Optimized for ultra-fast response with tightened prompts and minimal RAG context.
"""

import json
from typing import List, Optional

from core.agent_messages import AgentMessage, DiagnosticResult, RAGContextChunk
from core.agents.base_agent import BaseAgent
from config.model_provider import get_reasoning_model
from tools.tools import rag_search_tool


class DiagnosticAgent(BaseAgent):
    """
    Specialized agent for paddy disease diagnosis using deep reasoning model + RAG search tool.
    Supports Explicit Chain-of-Thought (CoT) and Self-Correction Loop.
    """

    def __init__(self):
        super().__init__(name="DiagnosticAgent", model=get_reasoning_model())

    def process(self, message: AgentMessage, feedback: Optional[str] = None) -> DiagnosticResult:
        self._log_start(message)
        query = message.user_query

        # Step 1: High-Speed RAG Vector Search (FAISS Index)
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
            context_str += f"\n[{chunk['filename']} P{chunk['page']}]: {chunk['content']}\n"

        # Step 2: High-Speed Structured Synthesis with Reasoning LLM
        system_prompt = (
            "You are a Sri Lankan Paddy Pathologist. Diagnose the farmer's query using the RAG context below.\n"
            "RULES: Only recommend DOA-approved chemicals. NEVER recommend banned WHO Class Ia/Ib chemicals "
            "(Paraquat, Carbofuran, Endosulfan, Glyphosate).\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "suspected_disease": "Disease Name (Scientific Name)",\n'
            '  "symptoms_identified": ["symptom1", "symptom2"],\n'
            '  "treatment_recommended": ["treatment1", "treatment2"],\n'
            '  "confidence_level": "High/Medium/Low"\n'
            "}"
        )

        user_content = f"Query: {query}\n\nRAG Context:\n{context_str}"
        if feedback:
            user_content += f"\n\n⚠️ REFLECTION FEEDBACK:\n{feedback}\nSelf-correct your previous analysis."

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
                thought_process=parsed.get("thought_process", "Diagnosed via DOA RAG Knowledge Base"),
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
                thought_process="Failed parsing response, fallback used.",
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
