"""
Multi-Agent Orchestrator Module
Coordinates agent-to-agent communication, manages workflow execution,
and synthesizes multi-agent responses for Sri Lankan paddy farmers.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from agent_messages import (
    AgentMessage, QueryIntent, AgentResponse, 
    DiagnosticResult, FertilizerRecommendation
)
from agents import RouterAgent, DiagnosticAgent, FertilizerAgent


class PaddyAgentOrchestrator:
    """
    Main Orchestrator coordinating structured message exchange between:
    - RouterAgent (Fast Groq model)
    - DiagnosticAgent (Reasoning model + RAG)
    - FertilizerAgent (Reasoning model + Calculator tool)
    """

    def __init__(self):
        print("[ORCHESTRATOR] Initializing Multi-Agent System...")
        self.router_agent = RouterAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.fertilizer_agent = FertilizerAgent()

    def process_user_request(self, user_query: str) -> AgentResponse:
        """
        Executes multi-agent workflow:
        1. RouterAgent classifies intent.
        2. Sends structured AgentMessage to DiagnosticAgent and/or FertilizerAgent.
        3. Collects agent outputs and synthesizes complete response.
        """
        message_trace: List[AgentMessage] = []
        session_id = f"msg_{uuid.uuid4().hex[:8]}"

        print(f"\n{'='*75}\n[ORCHESTRATOR] Processing Query: '{user_query}'\n{'='*75}")

        # Step 1: Intent Routing (Router Pattern)
        intent = self.router_agent.route_query(user_query)
        print(f"[ORCHESTRATOR] Query classified as: {intent.value}")

        diagnostic_info: Optional[DiagnosticResult] = None
        fertilizer_info: Optional[FertilizerRecommendation] = None

        # Step 2: Agent Communication & Task Execution
        if intent in [QueryIntent.DISEASE_DIAGNOSIS, QueryIntent.BOTH]:
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

            diagnostic_info = self.diagnostic_agent.process(msg_to_diag)

        if intent in [QueryIntent.FERTILIZER_RECOMMENDATION, QueryIntent.BOTH]:
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

            fertilizer_info = self.fertilizer_agent.process(msg_to_fert)

        # Step 3: Synthesis of Final Output
        final_synthesis = self._build_synthesis(user_query, intent, diagnostic_info, fertilizer_info)

        return AgentResponse(
            query=user_query,
            intent=intent,
            diagnostic_info=diagnostic_info,
            fertilizer_info=fertilizer_info,
            final_synthesis=final_synthesis,
            message_trace=message_trace
        )

    def _build_synthesis(
        self,
        query: str,
        intent: QueryIntent,
        diag: Optional[DiagnosticResult],
        fert: Optional[FertilizerRecommendation]
    ) -> str:
        synthesis = f"🌾 **Sri Lankan Paddy Advisory Report**\n\n"

        if diag:
            synthesis += f"🔬 **Diagnostic Assessment:**\n"
            synthesis += f"- **Suspected Issue:** {diag.suspected_disease}\n"
            synthesis += f"- **Confidence Level:** {diag.confidence_level}\n"
            synthesis += f"- **Key Symptoms:** {', '.join(diag.symptoms_identified)}\n"
            synthesis += f"- **Recommended Action:**\n"
            for action in diag.treatment_recommended:
                synthesis += f"  • {action}\n"
            synthesis += "\n"

        if fert:
            synthesis += f"🌱 **Fertilizer Recommendation ({fert.season} Season):**\n"
            synthesis += f"- **Target Zone:** {fert.district_zone} District\n"
            synthesis += f"- **Dosage per Acre:** Urea {fert.urea_dosage_per_acre_kg} kg | TSP {fert.tsp_dosage_per_acre_kg} kg | MOP {fert.mop_dosage_per_acre_kg} kg\n"
            synthesis += f"- **Application Timetable:**\n"
            for step in fert.application_schedule:
                synthesis += f"  • {step}\n"
            synthesis += "\n"

        if not diag and not fert:
            synthesis += "General agricultural guidance requested. Please consult local Agrarian Services Center for specific soil testing."

        return synthesis


def run_sample_evaluations():
    """Runs 3 realistic test queries to evaluate multi-agent orchestration."""
    orchestrator = PaddyAgentOrchestrator()

    sample_queries = [
        "What are the symptoms of Paddy Blast disease and how to control it?",
        "What is the recommended fertilizer mixture for Yala season paddy in Polonnaruwa?",
        "ගොයම් පත්‍ර වල දුඹුරු පැහැ ලප ඇති වී ඇත, යල කන්නයට පොහොර යොදන්නේ කෙසේද?"
    ]

    for idx, q in enumerate(sample_queries, 1):
        print(f"\n--- EVALUATION SCENARIO {idx} ---")
        response = orchestrator.process_user_request(q)
        print("\n" + response.final_synthesis)
        print(f"Messages Exchanged: {len(response.message_trace)}")


if __name__ == "__main__":
    run_sample_evaluations()
