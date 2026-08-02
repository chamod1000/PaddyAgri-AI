"""
Unit Tests for Explainable AI (XAI) Engine (tests/test_explainability.py)

STATUS: PENDING FEATURE — skipped, not deleted.

`core.explainability` has never existed in this repository. It is absent from
every commit, from the working tree, and from the backup tree; no production
module imports it. These tests are therefore a written specification for an
unbuilt feature, not a regression against deleted code.

They are retained because they define the intended contract:
    ExplainabilityEngine.generate_explanation(AgentResponse) -> DiagnosisExplanation
    with fields: disease_name, key_evidence, visual_evidence,
                 environmental_evidence, explanation_summary,
                 recommendation_rationale

`AgentResponse.explanation` (core/agent_messages.py) and the XAI section of
ReportGenerator (core/report_generator.py:178-185) are the consumer-side
scaffolding already in place for this feature. The XAI section is currently
dead code: nothing assigns `explanation`, so `if expl:` is always False.

Deleting these tests would erase the only surviving specification and leave
that scaffolding unexplained. To activate: implement core/explainability.py,
then remove the skip decorator below.
"""

import unittest

try:
    from core.agent_messages import AgentResponse, QueryIntent, DiagnosticResult
    from core.explainability import ExplainabilityEngine, DiagnosisExplanation
    from core.vision_processor import VisionAnalysisResult
    from core.weather_service import WeatherContext, SeasonalAdvisory
    _XAI_AVAILABLE = True
except ImportError:
    _XAI_AVAILABLE = False


@unittest.skipUnless(
    _XAI_AVAILABLE,
    "PENDING FEATURE: core.explainability is not implemented "
    "(never existed in any commit). Spec retained; implement the module to enable."
)
class TestExplainability(unittest.TestCase):

    def test_explainability_engine_evidence_mapping(self):
        """Unit test verifying evidence item mapping."""
        diag = DiagnosticResult(
            thought_process="...",
            suspected_disease="Paddy Blast",
            symptoms_identified=["leaf spots"],
            treatment_recommended=["Fungicide"],
            confidence_level="High"
        )
        vis = VisionAnalysisResult(
            has_image=True,
            visible_symptoms=["diamond-shaped lesions"],
            leaf_color="Yellowish-green",
            confidence_estimate="HIGH"
        )
        w_ctx = WeatherContext(humidity_pct=85.0)
        s_adv = SeasonalAdvisory(fungal_risk_alert="HIGH RISK")

        response = AgentResponse(
            query="Test query",
            intent=QueryIntent.DISEASE_DIAGNOSIS,
            diagnostic_info=diag,
            vision_info=vis,
            weather_info=w_ctx,
            seasonal_advisory=s_adv,
            final_synthesis=""
        )

        explanation = ExplainabilityEngine.generate_explanation(response)
        self.assertIsInstance(explanation, DiagnosisExplanation)
        self.assertEqual(explanation.disease_name, "Paddy Blast")
        self.assertGreater(len(explanation.key_evidence), 0)
        self.assertGreater(len(explanation.visual_evidence), 0)
        self.assertGreater(len(explanation.environmental_evidence), 0)
        self.assertIn("Paddy Blast", explanation.explanation_summary)


if __name__ == "__main__":
    unittest.main()
