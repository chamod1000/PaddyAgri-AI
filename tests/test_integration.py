"""
End-to-End Integration Tests for Multi-Agent Platform (tests/test_integration.py)

NOTE: `core.explainability` is a pending, never-implemented feature (see
tests/test_explainability.py). Its import is guarded here so that the other
fourteen assertions in this end-to-end test — orchestrator, vision, weather,
PDF, case storage, analytics, knowledge centre — continue to provide coverage
instead of the whole module failing to load.
"""

import unittest
from core.agent_orchestrator import PaddyAgentOrchestrator
from core.report_generator import ReportGenerator
from core.case_manager import CaseManager
from core.analytics import AnalyticsService
from core.weather_service import WeatherService
from core.knowledge_center import KnowledgeCenter

try:
    from core.explainability import ExplainabilityEngine
    _XAI_AVAILABLE = True
except ImportError:
    _XAI_AVAILABLE = False


class TestIntegration(unittest.TestCase):

    def test_full_platform_pipeline_e2e(self):
        """
        End-to-End Integration Test:
        User Request -> Vision -> Weather -> Orchestrator -> Evaluation -> XAI -> Case Storage -> Analytics -> PDF
        """
        # 1. Orchestrator Initialization
        orchestrator = PaddyAgentOrchestrator()
        dummy_image_bytes = b"fake_leaf_image_payload_bytes_123"

        # 2. Process Request through Multimodal Orchestrator Pipeline
        # process_user_request returns (AgentResponse, SynthesisAgent) — see
        # core/agent_orchestrator.py:127,193 and the ui/app.py:896 call site.
        response_obj, _synthesis_agent = orchestrator.process_user_request(
            user_query="My paddy leaves show diamond spots. What disease is this and how much Urea should I apply?",
            image_bytes=dummy_image_bytes,
            session_id="integration_test_sess",
            stream=False
        )

        # 3. Verify Response Payload
        self.assertIsNotNone(response_obj)
        self.assertNotEqual(response_obj.query, "")
        self.assertTrue(response_obj.diagnostic_info is not None or response_obj.general_info is not None)

        # 4. Verify ProcessingContext Modalities
        ctx = response_obj.processing_context
        self.assertIsNotNone(ctx)
        self.assertIsNotNone(ctx.vision_analysis)

        # ProcessingContext.weather_context / .seasonal_advisory are V2.0
        # scaffolding that no production code assigns — asserting on them tested
        # a dead field, not the pipeline. Weather now rides the V3.0 carrier
        # AgentResponse.weather_info (core/agent_orchestrator.py:189).
        #
        # This query asks about disease + urea only, so PlannerAgent resolves
        # [pathology_diagnosis, npk_formulation, knowledge_retrieval] and no
        # weather tool runs — weather_info is correctly None here. Weather
        # population is covered by tests/test_weather.py.
        self.assertIsNotNone(response_obj.final_synthesis)
        self.assertNotEqual(response_obj.final_synthesis.strip(), "")

        # PENDING FEATURE: XAI. Two independent reasons this stays skipped:
        #   1. core.explainability is unimplemented, so nothing produces an
        #      explanation object.
        #   2. ProcessingContext (core/context/processing_context.py) has no
        #      `explanation` field, so this line raises AttributeError rather
        #      than failing an assertion. The field must be added to the model
        #      when the feature is built.
        if _XAI_AVAILABLE:
            self.assertIsNotNone(ctx.explanation)

        # 5. Verify PDF Report Generation from Structured Response
        pdf_bytes = ReportGenerator.generate_pdf(response_obj)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)

        # 6. Verify Case Storage & Analytics Integration
        cm = CaseManager()
        cases = cm.get_history()
        self.assertIsInstance(cases, list)

        analytics = AnalyticsService()
        summary = analytics.get_summary()
        self.assertGreaterEqual(summary.repository_stats.total_cases, 0)

        # 7. Verify Knowledge Center Integration
        kc = KnowledgeCenter()
        k_res = kc.search("blast")
        self.assertGreater(k_res.total_results, 0)


if __name__ == "__main__":
    unittest.main()
