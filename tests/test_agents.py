"""
Unit & Negative Tests for Multi-Agent Orchestrator and Messages (tests/test_agents.py)
"""

import unittest
from core.agent_messages import (
    AgentMessage, QueryIntent, AgentResponse,
    DiagnosticResult, FertilizerRecommendation, ProcessingContext
)
from core.agent_orchestrator import PaddyAgentOrchestrator


class TestAgents(unittest.TestCase):

    def test_agent_messages_pydantic_validation(self):
        """Unit test for AgentResponse and ProcessingContext models."""
        context = ProcessingContext(user_query="Test query")
        self.assertEqual(context.user_query, "Test query")
        self.assertIsNone(context.vision_analysis)

        response = AgentResponse(
            query="Test query",
            intent=QueryIntent.DISEASE_DIAGNOSIS,
            final_synthesis="Test synthesis"
        )
        self.assertEqual(response.query, "Test query")
        self.assertEqual(response.intent, QueryIntent.DISEASE_DIAGNOSIS)
        self.assertEqual(response.final_synthesis, "Test synthesis")

    def test_orchestrator_initialization(self):
        """Unit test verifying orchestrator component instantiation."""
        orchestrator = PaddyAgentOrchestrator()
        self.assertIsNotNone(orchestrator.vision_processor)
        self.assertIsNotNone(orchestrator.diagnostic_agent)
        self.assertIsNotNone(orchestrator.fertilizer_agent)
        self.assertIsNotNone(orchestrator.synthesis_agent)
        self.assertIsNotNone(orchestrator.weather_service)

        # Regulatory reflection is a stateless classmethod facade, not an
        # instance attribute: core/agent_orchestrator.py imports it inline
        # (:103) and calls RegulatoryReflection.audit_response (:167).
        # Asserting orchestrator.reflection_agent tested a V2.0 shape.
        from core.reflection import RegulatoryReflection
        self.assertTrue(callable(RegulatoryReflection.audit_response))

        # CaseManager is likewise not held by the orchestrator; callers
        # instantiate it directly (see tests/test_integration.py).
        from core.case_manager import CaseManager
        self.assertIsNotNone(CaseManager())

    def test_memory_creation(self):
        """Unit test for session memory initialization."""
        orchestrator = PaddyAgentOrchestrator()
        mem = orchestrator.get_or_create_memory(session_id="test_session_123")
        self.assertEqual(mem.metadata.session_id, "test_session_123")
        self.assertEqual(len(mem.turns), 0)


if __name__ == "__main__":
    unittest.main()
