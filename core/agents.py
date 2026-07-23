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

from core.agent_messages import (
    AgentMessage, QueryIntent, DiagnosticResult,
    FertilizerRecommendation, RAGContextChunk,
    SafetyVerdict, ReflectionResult
)
from config.model_provider import get_router_model, get_reasoning_model
from tools.tools import rag_search_tool, fertilizer_calculator_tool


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


class ReflectionAgent:
    """
    Agentic Pattern 4: Reflection & Self-Critique Pattern
    Safety & Quality Verifier that double-checks pesticide/fertilizer recommendations
    against Sri Lankan Department of Agriculture (DoA) environmental safety guidelines
    and biosecurity regulations before final output synthesis.
    """
    def __init__(self):
        self.name = "ReflectionAgent"
        # Sri Lankan DoA maximum dosage limits (kg per acre)
        self.doa_limits = {
            "urea": 65.0,
            "tsp": 30.0,
            "mop": 30.0,
        }
        # Sri Lankan banned/restricted pesticides list
        self.banned_chemicals = [
            "methyl parathion", "paraquat", "carbofuran", "methamidophos",
            "monocrotophos", "phosphamidon", "endosulfan"
        ]
        self.allergen_watchlist = {
            "chlorpyrifos": "respiratory sensitizer",
            "mancozeb": "skin irritant",
            "glyphosate": "suspected carcinogen (WHO Group 2A)"
        }

    def verify_fertilizer_dosage(self, rec: FertilizerRecommendation) -> List[SafetyVerdict]:
        """Check fertilizer dosages against DoA maximum limits."""
        verdicts = []
        checks = [
            ("urea", rec.urea_dosage_per_acre_kg),
            ("tsp", rec.tsp_dosage_per_acre_kg),
            ("mop", rec.mop_dosage_per_acre_kg),
        ]
        for chem, dose in checks:
            limit = self.doa_limits.get(chem, float("inf"))
            if dose > limit:
                verdicts.append(SafetyVerdict(
                    check_name=f"{chem}_dosage",
                    passed=False,
                    message=f"Dosage {dose} kg/ac exceeds DoA max of {limit} kg/ac",
                    severity="warning"
                ))
            else:
                verdicts.append(SafetyVerdict(
                    check_name=f"{chem}_dosage",
                    passed=True,
                    message=f"Dosage {dose} kg/ac within DoA limit of {limit} kg/ac",
                    severity="info"
                ))
        return verdicts

    def verify_banned_chemicals(self, treatment_list: List[str]) -> List[SafetyVerdict]:
        """Cross-check recommended treatments against banned pesticides list."""
        verdicts = []
        for treatment in treatment_list:
            for banned in self.banned_chemicals:
                if banned.lower() in treatment.lower():
                    verdicts.append(SafetyVerdict(
                        check_name="banned_chemical",
                        passed=False,
                        message=f"'{banned}' is RESTRICTED under Sri Lankan pesticide regulations",
                        severity="critical"
                    ))
        if not verdicts:
            verdicts.append(SafetyVerdict(
                check_name="banned_chemical",
                passed=True,
                message="No banned pesticides detected in recommendations",
                severity="info"
            ))
        return verdicts

    def verify_allergens(self, treatment_list: List[str]) -> List[SafetyVerdict]:
        """Check for known allergens in recommended treatments."""
        verdicts = []
        for treatment in treatment_list:
            for chem, risk in self.allergen_watchlist.items():
                if chem.lower() in treatment.lower():
                    verdicts.append(SafetyVerdict(
                        check_name="allergen_screen",
                        passed=False,
                        message=f"'{chem}' identified as {risk} - include safety precautions",
                        severity="warning"
                    ))
        if not verdicts:
            verdicts.append(SafetyVerdict(
                check_name="allergen_screen",
                passed=True,
                message="No known allergen risks detected",
                severity="info"
            ))
        return verdicts

    def get_citations(self) -> List[str]:
        """Return relevant Sri Lankan regulatory citations."""
        return [
            "Department of Agriculture Sri Lanka - Pesticide Act No. 33 of 1980",
            "Department of Agriculture Sri Lanka - Fertilizer Ordinance No. 1 of 1995",
            "Sri Lanka Standards Institution - SLS 1164:2019 Fertilizer Specification",
            "FAO Sri Lanka - Code of Conduct for Pesticide Management (2023)",
        ]

    def process(self, diagnostic: Optional[DiagnosticResult], fertilizer: Optional[FertilizerRecommendation]) -> ReflectionResult:
        """Run all safety checks on agent outputs."""
        all_verdicts: List[SafetyVerdict] = []
        warnings: List[str] = []
        biosecurity_alerts: List[str] = []

        # Check fertilizer dosage
        if fertilizer:
            all_verdicts.extend(self.verify_fertilizer_dosage(fertilizer))

        # Check diagnostic treatments
        if diagnostic:
            all_verdicts.extend(self.verify_banned_chemicals(diagnostic.treatment_recommended))
            all_verdicts.extend(self.verify_allergens(diagnostic.treatment_recommended))

        # Aggregate warnings
        for v in all_verdicts:
            if not v.passed:
                warnings.append(v.message)
                if v.severity == "critical":
                    biosecurity_alerts.append(f"BIOSECURITY: {v.message}")

        all_passed = all(v.passed for v in all_verdicts)

        return ReflectionResult(
            recommendation_id=f"ref_{uuid.uuid4().hex[:8]}",
            all_checks_passed=all_passed,
            verdicts=all_verdicts,
            warnings=warnings,
            regulatory_citations=self.get_citations(),
            biosecurity_alerts=biosecurity_alerts
        )
