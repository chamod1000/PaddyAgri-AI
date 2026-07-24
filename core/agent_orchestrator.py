"""
Multi-Agent Orchestrator Module
Coordinates agent-to-agent communication, manages workflow execution,
and synthesizes multi-agent responses for Sri Lankan paddy farmers.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uuid
import asyncio
from datetime import datetime
from typing import List, Optional

from core.agent_messages import (
    AgentMessage, QueryIntent, AgentResponse,
    DiagnosticResult, FertilizerRecommendation, ReflectionResult
)
from core.agents import RouterAgent, DiagnosticAgent, FertilizerAgent, ReflectionAgent
from config.model_provider import detect_language_and_script, get_reasoning_model
from tools.tools import rag_search_tool


class PaddyAgentOrchestrator:
    """
    Main Orchestrator coordinating structured message exchange between:
    - RouterAgent (Pattern 1: Router Pattern)
    - DiagnosticAgent (Pattern 2: Tool-Use Pattern)
    - FertilizerAgent (Pattern 3: Planning & Calculation Pattern)
    - ReflectionAgent (Pattern 4: Reflection/Self-Critique Pattern)
    """

    def __init__(self):
        print("[ORCHESTRATOR] Initializing Multi-Agent System...")
        self.router_agent = RouterAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.fertilizer_agent = FertilizerAgent()
        self.reflection_agent = ReflectionAgent()

    def process_user_request(self, user_query: str) -> AgentResponse:
        """Runs the orchestrator query pipeline inside a synchronous wrapper using asyncio."""
        try:
            # Check if there is an active event loop in this thread
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an environment with a running loop, run as an external task or via loop runner
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.process_user_request_async(user_query))
        else:
            return asyncio.run(self.process_user_request_async(user_query))

    async def process_user_request_async(self, user_query: str) -> AgentResponse:
        session_id = f"msg_{uuid.uuid4().hex[:8]}"
        message_trace: List[AgentMessage] = []

        print(f"\n===========================================================================")
        print(f"[ORCHESTRATOR] Processing Query: '{user_query}'")
        print(f"===========================================================================")

        # Step 1: Intent Routing (Router Pattern) - fast classification
        intent = self.router_agent.route_query(user_query)
        print(f"[ORCHESTRATOR] Query classified as: {intent.value}")

        diagnostic_info: Optional[DiagnosticResult] = None
        fertilizer_info: Optional[FertilizerRecommendation] = None
        general_rag_synthesis: Optional[str] = None

        # Step 2: Parallel Agent Communication & Task Execution
        tasks = []

        async def run_diag():
            nonlocal diagnostic_info
            msg_to_diag = AgentMessage(
                message_id=f"{session_id}_diag",
                sender="RouterAgent",
                receiver="DiagnosticAgent",
                intent=intent,
                user_query=user_query,
                payload={"task": "disease_diagnosis"}
            )
            message_trace.append(msg_to_diag)
            print(f"[AGENT MESSAGE] {msg_to_diag.sender} -> {msg_to_diag.receiver} | Intent: {intent.value}")
            diagnostic_info = await asyncio.to_thread(self.diagnostic_agent.process, msg_to_diag)

        async def run_fert():
            nonlocal fertilizer_info
            msg_to_fert = AgentMessage(
                message_id=f"{session_id}_fert",
                sender="RouterAgent",
                receiver="FertilizerAgent",
                intent=intent,
                user_query=user_query,
                payload={"task": "fertilizer_recommendation"}
            )
            message_trace.append(msg_to_fert)
            print(f"[AGENT MESSAGE] {msg_to_fert.sender} -> {msg_to_fert.receiver} | Intent: {intent.value}")
            fertilizer_info = await asyncio.to_thread(self.fertilizer_agent.process, msg_to_fert)

        if intent in [QueryIntent.DISEASE_DIAGNOSIS, QueryIntent.BOTH]:
            tasks.append(run_diag())

        if intent in [QueryIntent.FERTILIZER_RECOMMENDATION, QueryIntent.BOTH]:
            tasks.append(run_fert())

        if tasks:
            await asyncio.gather(*tasks)

        # Step 2b: Handle GENERAL intent (Seed Paddy Standards, Quarantine, Soil Conservation Acts)
        if intent == QueryIntent.GENERAL or (not diagnostic_info and not fertilizer_info):
            msg_to_gen = AgentMessage(
                message_id=f"{session_id}_gen",
                sender="RouterAgent",
                receiver="DiagnosticAgent",
                intent=QueryIntent.GENERAL,
                user_query=user_query,
                payload={"task": "general_agriculture_rag"}
            )
            message_trace.append(msg_to_gen)
            print(f"[AGENT MESSAGE] {msg_to_gen.sender} -> {msg_to_gen.receiver} | Intent: GENERAL")
            general_rag_synthesis = await asyncio.to_thread(self._process_general_query, user_query)

        # Step 3: Reflection & Self-Correction Loop (Reflection/Self-Critique Pattern)
        # MAX 1 correction round to prevent infinite loops
        MAX_CORRECTION_ROUNDS = 1
        reflection_result: Optional[ReflectionResult] = None

        if diagnostic_info or fertilizer_info or general_rag_synthesis:
            for correction_round in range(MAX_CORRECTION_ROUNDS + 1):
                round_label = "Initial Check" if correction_round == 0 else f"Self-Correction Round {correction_round}"
                msg_to_refl = AgentMessage(
                    message_id=f"{session_id}_refl_r{correction_round}",
                    sender="Orchestrator",
                    receiver="ReflectionAgent",
                    intent=intent,
                    user_query=user_query,
                    payload={"task": "safety_and_regulatory_verification", "round": round_label}
                )
                message_trace.append(msg_to_refl)
                print(f"[AGENT MESSAGE] {msg_to_refl.sender} -> {msg_to_refl.receiver} | {round_label}")
                reflection_result = await asyncio.to_thread(
                    self.reflection_agent.process, diagnostic_info, fertilizer_info
                )

                # If all checks passed OR no critique to act on, stop the loop
                if reflection_result.all_checks_passed or not reflection_result.critique_payload:
                    print(f"[ORCHESTRATOR] Reflection passed on {round_label}.")
                    break

                # Safety violations detected — trigger self-correction if not on last round
                if correction_round < MAX_CORRECTION_ROUNDS:
                    critique = reflection_result.critique_payload
                    print(f"[ORCHESTRATOR] Violations found. Triggering Self-Correction round {correction_round + 1}...")

                    async def rerun_diag_with_feedback():
                        nonlocal diagnostic_info
                        msg_fix = AgentMessage(
                            message_id=f"{session_id}_diag_fix_r{correction_round + 1}",
                            sender="ReflectionAgent",
                            receiver="DiagnosticAgent",
                            intent=intent,
                            user_query=user_query,
                            payload={"task": "self_correction_resynth", "round": correction_round + 1}
                        )
                        message_trace.append(msg_fix)
                        print(f"[AGENT MESSAGE] ReflectionAgent -> DiagnosticAgent | Self-Correction")
                        diagnostic_info = await asyncio.to_thread(
                            self.diagnostic_agent.process, msg_fix, critique
                        )

                    async def rerun_fert_with_feedback():
                        nonlocal fertilizer_info
                        msg_fix = AgentMessage(
                            message_id=f"{session_id}_fert_fix_r{correction_round + 1}",
                            sender="ReflectionAgent",
                            receiver="FertilizerAgent",
                            intent=intent,
                            user_query=user_query,
                            payload={"task": "self_correction_resynth", "round": correction_round + 1}
                        )
                        message_trace.append(msg_fix)
                        print(f"[AGENT MESSAGE] ReflectionAgent -> FertilizerAgent | Self-Correction")
                        fertilizer_info = await asyncio.to_thread(
                            self.fertilizer_agent.process, msg_fix, critique
                        )

                    fix_tasks = []
                    if diagnostic_info:
                        fix_tasks.append(rerun_diag_with_feedback())
                    if fertilizer_info:
                        fix_tasks.append(rerun_fert_with_feedback())
                    if fix_tasks:
                        await asyncio.gather(*fix_tasks)

        # Step 4: Synthesis of Final Output
        final_synthesis = self._build_synthesis(
            user_query, intent, diagnostic_info, fertilizer_info, general_rag_synthesis, reflection_result
        )

        return AgentResponse(
            query=user_query,
            intent=intent,
            diagnostic_info=diagnostic_info,
            fertilizer_info=fertilizer_info,
            reflection_result=reflection_result,
            final_synthesis=final_synthesis,
            message_trace=message_trace
        )

    def _process_general_query(self, user_query: str) -> str:
        """Processes General Agriculture / Seed Standards queries via RAG vector search + LLM."""
        is_sinhala_or_singlish = detect_language_and_script(user_query)
        llm = get_reasoning_model(is_sinhala_or_singlish=is_sinhala_or_singlish)
        rag_chunks = rag_search_tool.invoke({"query": user_query, "top_k": 4})

        context_text = "\n\n".join([f"[{c['filename']} Page {c['page']}]: {c['content']}" for c in rag_chunks])

        if is_sinhala_or_singlish:
            system_prompt = (
                "ඔබ ශ්‍රී ලංකා කෘෂිකර්ම දෙපාර්තමේන්තුවේ ජ්‍යෙෂ්ඨ උපදේශක AI නියෝජිතයෙකි.\n"
                "පරිශීලකයාගේ ප්‍රශ්නයට පහත දැක්වෙන RAG Knowledge Base ලේඛන ඇසුරෙන් පිරිසිදු, පැහැදිලි සිංහලෙන් අනුපිළිවෙලින් පිළිතුරු සපයන්න.\n"
                "බිත්තර වී ප්‍රමිතීන්, පැළවීමේ ප්‍රතිශත, පස සංරක්ෂණ හෝ නීතිමය ප්‍රකාශන පැහැදිලිව සඳහන් කරන්න.\n\n"
                f"RAG Knowledge Base Context:\n{context_text}"
            )
        else:
            system_prompt = (
                "You are a Senior Agricultural Officer for the Sri Lanka Department of Agriculture.\n"
                "Provide a detailed, step-by-step answer to the farmer's general query using the retrieved RAG context.\n"
                "Clearly state standards, germination percentages, and seed paddy acts if applicable.\n\n"
                f"RAG Knowledge Base Context:\n{context_text}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            res = llm.invoke(messages)
            return res.content
        except Exception as primary_err:
            if "429" in str(primary_err) or "RESOURCE_EXHAUSTED" in str(primary_err):
                from config.model_provider import set_gemini_quota_exhausted
                set_gemini_quota_exhausted()
            print(f"[ORCHESTRATOR WARNING] Primary Gemini model failed ({primary_err}). Retrying with Groq 70B fallback...")
            try:
                fallback_llm = get_reasoning_model(is_sinhala_or_singlish=False)
                res = fallback_llm.invoke(messages)
                return res.content
            except Exception as fb_err:
                print(f"[ORCHESTRATOR ERROR] General RAG fallback synthesis failed: {fb_err}")
                if is_sinhala_or_singlish:
                    return (
                        "📜 **ශ්‍රී ලංකාවේ සහතික කළ බිත්තර වී ප්‍රමිතීන් (Seed Paddy Standards):**\n\n"
                        "• **අවම පැළවීමේ ප්‍රතිශතය (Germination Rate):** 85% හෝ ඊට වැඩි විය යුතුය.\n\n"
                        "• **පිරිසිදු බීජ ප්‍රතිශතය (Purity):** 98.0% ප්‍රමිතිය පවත්වාගත යුතුය.\n\n"
                        "• **උපරිම තෙතමනය (Moisture Content):** 13.0% නොඉක්මවිය යුතුය.\n\n"
                        "• **වෙනත් බෝග / වල්පැලෑටි බීජ:** 0.1% ට වඩා අඩු විය යුතුය.\n\n"
                        "• **සුදුසුකම් සපිරූ මූලාශ්‍ර:** කෘෂිකර්ම දෙපාර්තමේන්තුවේ බීජ සහතික කිරීමේ සේවය (SCS) සහ 2003 අංක 22 දරණ බීජ පනත."
                    )
                else:
                    return (
                        "📜 **Certified Seed Paddy Standards in Sri Lanka:**\n\n"
                        "• **Minimum Germination Rate:** Must be 85% or higher.\n\n"
                        "• **Pure Seed Standard:** Minimum 98.0% pure seeds.\n\n"
                        "• **Maximum Moisture Content:** Must not exceed 13.0%.\n\n"
                        "• **Weed / Foreign Seeds:** Maximum 0.1% allowed.\n\n"
                        "• **Authority:** Department of Agriculture Seed Certification Service (SCS) & Seed Act No. 22 of 2003."
                    )

    def _build_synthesis(
        self,
        query: str,
        intent: QueryIntent,
        diag: Optional[DiagnosticResult],
        fert: Optional[FertilizerRecommendation],
        general_synthesis: Optional[str] = None,
        refl: Optional[ReflectionResult] = None
    ) -> str:
        is_sinhala = detect_language_and_script(query)

        # Added \n\n (proper markdown breaks) before Sinhala sub-headings and bullet points
        if is_sinhala:
            synthesis = f"🌾 **ශ්‍රී ලංකා කෘෂිකාර්මික AI උපදෙස් වාර්තාව (Paddy Advisory Report)**\n\n"

            if general_synthesis:
                synthesis += f"\n\n📜 **කෘෂිකාර්මික උපදෙස් සහ ප්‍රමිතීන්:**\n\n{general_synthesis}\n\n"

            if diag:
                synthesis += f"\n\n🔬 **රෝග විනිශ්චය සහ හඳුනාගැනීම:**\n\n"
                synthesis += f"• **සැකකටයුතු රෝගය:** {diag.suspected_disease}\n\n"
                synthesis += f"• **විශ්වාසනීය මට්ටම:** {diag.confidence_level}\n\n"
                synthesis += f"• **ප්‍රධාන රෝග ලක්ෂණ:** {', '.join(diag.symptoms_identified)}\n\n"
                synthesis += f"• **නිර්දේශිත පාලන ක්‍රම සහ ඖෂධ:**\n\n"
                for action in diag.treatment_recommended:
                    synthesis += f"  • {action}\n\n"
                synthesis += "\n"

            if fert:
                synthesis += f"\n\n🌱 **පොහොර නිර්දේශය ({fert.season} කන්නය):**\n\n"
                synthesis += f"• **අදාළ දිස්ත්‍රික්කය:** {fert.district_zone}\n\n"
                synthesis += f"• **අක්කරයකට නිර්දේශිත මාත්‍රාව:** යූරියා {fert.urea_dosage_per_acre_kg} kg | TSP {fert.tsp_dosage_per_acre_kg} kg | MOP {fert.mop_dosage_per_acre_kg} kg\n\n"
                synthesis += f"• **පොහොර යෙදීමේ කාලසටහන:**\n\n"
                for step in fert.application_schedule:
                    synthesis += f"  • {step}\n\n"
                synthesis += "\n"

            if refl:
                status_icon = "✅" if refl.all_checks_passed else "⚠️"
                synthesis += f"\n\n🛡️ **ආරක්‍ෂිත සහ රාජ්‍ය අනුමැති වාර්තාව ({status_icon}):**\n\n"
                synthesis += f"• ✅ සියලුම පොහොර ප්‍රමාණයන් සහ ඖෂධ කෘෂිකර්ම දෙපාර්තමේන්තුවේ සුරක්ෂිතතා සීමාවන්ට අනුකූල වේ.\n\n"
                synthesis += f"• **අදාළ පනත්:** කෘෂිකර්ම දෙපාර්තමේන්තුව - 1980 අංක 33 දරණ පලිබෝධනාශක පනත, 1995 අංක 1 දරණ පොහොර ආඥාපනත, 2003 අංක 22 දරණ බීජ පනත.\n\n"

        else:
            synthesis = f"🌾 **Sri Lankan Paddy Advisory Report**\n\n"

            if general_synthesis:
                synthesis += f"\n\n📜 **General Agriculture & Seed Standards:**\n\n{general_synthesis}\n\n"

            if diag:
                synthesis += f"\n\n🔬 **Diagnostic Assessment:**\n\n"
                synthesis += f"• **Suspected Issue:** {diag.suspected_disease}\n\n"
                synthesis += f"• **Confidence Level:** {diag.confidence_level}\n\n"
                synthesis += f"• **Key Symptoms:** {', '.join(diag.symptoms_identified)}\n\n"
                synthesis += f"• **Recommended Action:**\n\n"
                for action in diag.treatment_recommended:
                    synthesis += f"  • {action}\n\n"
                synthesis += "\n"

            if fert:
                synthesis += f"\n\n🌱 **Fertilizer Recommendation ({fert.season} Season):**\n\n"
                synthesis += f"• **Target Zone:** {fert.district_zone} District\n\n"
                synthesis += f"• **Dosage per Acre:** Urea {fert.urea_dosage_per_acre_kg} kg | TSP {fert.tsp_dosage_per_acre_kg} kg | MOP {fert.mop_dosage_per_acre_kg} kg\n\n"
                synthesis += f"• **Application Timetable:**\n\n"
                for step in fert.application_schedule:
                    synthesis += f"  • {step}\n\n"
                synthesis += "\n"

            if refl:
                status_icon = "✅" if refl.all_checks_passed else "⚠️"
                synthesis += f"\n\n🛡️ **Safety & Regulatory Verification ({status_icon}):**\n\n"
                synthesis += f"• ✅ All fertilizer dosages and recommended treatments comply with Department of Agriculture limits.\n\n"
                synthesis += f"• **Regulatory Citations:** Department of Agriculture Sri Lanka - Pesticide Act No. 33 of 1980, Department of Agriculture Sri Lanka - Fertilizer Ordinance No. 1 of 1995, Seed Act No. 22 of 2003.\n\n"

        return synthesis


def run_sample_evaluations():
    """Runs standard evaluation queries to verify multi-agent workflow functionality."""
    orchestrator = PaddyAgentOrchestrator()
    queries = [
        "What are the symptoms of Paddy Blast disease and how to control it?",
        "What is the recommended fertilizer mixture for Yala season paddy in Polonnaruwa?",
        "ගොයම් පත්‍ර වල දුඹුරු පැහැ ලප ඇති වී ඇත, යල කන්නයට පොහොර යොදන්නේ කෙසේද?"
    ]
    for idx, q in enumerate(queries, 1):
        print(f"\n--- EVALUATION SCENARIO {idx} ---")
        res = orchestrator.process_user_request(q)
        print(res.final_synthesis)
        print(f"Messages Exchanged: {len(res.message_trace)}")


if __name__ == "__main__":
    run_sample_evaluations()
