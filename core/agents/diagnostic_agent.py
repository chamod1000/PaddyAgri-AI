"""
DiagnosticAgent Implementation
Agentic Pattern 2: Tool-Use & ReAct Pattern using RAG + Groq Llama 3.3 70B / Google Gemini model.
"""

import json
from typing import List, Optional

from core.agent_messages import AgentMessage, DiagnosticResult, RAGContextChunk
from core.agents.base_agent import BaseAgent
from config.model_provider import get_reasoning_model, detect_language_and_script
from tools.tools import rag_search_tool


class DiagnosticAgent(BaseAgent):
    """
    Specialized agent for paddy disease diagnosis using deep reasoning model + RAG search tool.
    Dynamically routes to Gemini if Sinhala or Singlish language is detected.
    Supports Explicit Chain-of-Thought (CoT) and Self-Correction Loop.
    """

    def __init__(self):
        # Default initialization with standard reasoning model
        super().__init__(name="DiagnosticAgent", model=get_reasoning_model())

    def process(self, message: AgentMessage, feedback: Optional[str] = None) -> DiagnosticResult:
        self._log_start(message)
        query = message.user_query

        # Step 0: Language & script auto-detection for routing
        is_sinhala_or_singlish = detect_language_and_script(query)
        self.model = get_reasoning_model(is_sinhala_or_singlish=is_sinhala_or_singlish)

        # Step 1: High-Speed RAG Vector Search (FAISS Index)
        from tools.tools import rag_search_tool
        print(f"[{self.name}] Executing RAG Vector Search...")
        rag_results = rag_search_tool.invoke({"query": query, "top_k": 4})

        # Step 2: Fallback to Web Search ONLY if RAG yields insufficient chunks
        web_context = ""
        if not rag_results:
            try:
                from tools.tools import web_search_tool
                print(f"[{self.name}] RAG empty, executing fallback Web Search...")
                web_results = web_search_tool.invoke({"query": f"Sri Lanka paddy disease {query}", "max_results": 2})
                for w in web_results:
                    web_context += f"- [{w['title']}]({w['href']}): {w['body']}\n"
            except Exception as e:
                print(f"[{self.name}] Web search fallback skipped: {e}")

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

        # Step 2: Synthesis with Reasoning LLM & Explicit CoT prompting
        system_prompt = (
            "You are a Senior Agricultural Pathologist specializing in Sri Lankan Paddy Rice Diseases.\n"
            "Analyze the farmer's query and the provided domain knowledge context to produce a diagnosis.\n"
            "Respond in clear, professional Simple English.\n\n"
            "STRICT REGULATORY COMPLIANCE MANDATE:\n"
            "- Comply strictly with Sri Lanka Control of Pesticides Act No. 33 of 1980.\n"
            "- NEVER recommend banned WHO Class Ia/Ib toxic chemicals (e.g. Paraquat, Carbofuran, Endosulfan, Glyphosate).\n"
            "- Recommend ONLY DOA approved fungicides/insecticides (e.g., Tebuconazole, Azoxystrobin, Hexaconazole).\n\n"
            "Perform step-by-step reasoning under the 'thought_process' field, analyzing:\n"
            "A) Crop stage and symptoms.\n"
            "B) RAG Handbook document matching references.\n"
            "C) Department of Agriculture safety limits and regulatory compliance.\n\n"
            "Return a valid JSON object matching this structure:\n"
            "{\n"
            '  "thought_process": "Analysis of symptoms and RAG matches...",\n'
            '  "suspected_disease": "Paddy Blast Disease (Pyricularia oryzae)",\n'
            '  "symptoms_identified": ["Spindle-shaped brown lesions on leaves", "Leaf tip drying"],\n'
            '  "treatment_recommended": ["Apply recommended fungicide (e.g. Tebuconazole)", "Avoid excessive nitrogen application"],\n'
            '  "confidence_level": "High"\n'
            "}"
        )

        user_content = (
            f"Farmer Query: {query}\n\n"
            f"--- LIVE WEB SEARCH CONTEXT ---\n{web_context}\n\n"
            f"--- RAG KNOWLEDGE BASE CONTEXT ---\n{context_str}"
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
            result = DiagnosticResult(
                thought_process=parsed.get("thought_process", "CoT Completed"),
                suspected_disease=parsed.get("suspected_disease", "Paddy Health Anomaly"),
                symptoms_identified=parsed.get("symptoms_identified", []),
                treatment_recommended=parsed.get("treatment_recommended", []),
                confidence_level=parsed.get("confidence_level", "Medium"),
                rag_sources=sources
            )
            
            # Step 4: Auto-Learning (Second Brain) for Highly Confident Answers
            if result.confidence_level.upper() == "HIGH":
                from rag.rag_pipeline import auto_learn_text
                import uuid
                learned_content = (
                    f"Query: {query}\n"
                    f"Verified Disease: {result.suspected_disease}\n"
                    f"Symptoms: {', '.join(result.symptoms_identified)}\n"
                    f"Treatments: {', '.join(result.treatment_recommended)}"
                )
                try:
                    auto_learn_text(learned_content, f"learned_{uuid.uuid4().hex[:6]}")
                except Exception as e:
                    print(f"[{self.name}] Failed to auto-learn: {e}")

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
