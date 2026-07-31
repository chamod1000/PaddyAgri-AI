"""
Unit tests for ConversationExperienceEngine to verify production hardening improvements.
Tests intent classification, evidence assembly, CoT prompt integration, and fallback behavior.
"""

import unittest
from unittest.mock import Mock, patch
from core.synthesis.conversation_experience_engine import ConversationExperienceEngine, _Intent, _classify_intent


class TestConversationExperienceEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ConversationExperienceEngine

    def test_intent_classification_knowledge(self):
        intent = self.engine._classify_intent(
            query="What is paddy blast disease?",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, self.engine._Intent.KNOWLEDGE)

    def test_intent_classification_diagnosis(self):
        intent = self.engine._classify_intent(
            query="My rice leaves have brown spots, what should I do?",
            has_diagnosis=True,
            has_weather=False,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, self.engine._Intent.DIAGNOSIS)

    def test_intent_classification_weather(self):
        intent = self.engine._classify_intent(
            query="What is the weather forecast for Anuradhapura?",
            has_diagnosis=False,
            has_weather=True,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, self.engine._Intent.WEATHER)

    def test_intent_classification_fertilizer(self):
        intent = self.engine._classify_intent(
            query="How much urea should I apply this season?",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=True,
            has_vision=False
        )
        self.assertEqual(intent, self.engine._Intent.FERTILIZER)

    def test_intent_classification_image(self):
        intent = self.engine._classify_intent(
            query="Analyze this leaf photo",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=False,
            has_vision=True
        )
        self.assertEqual(intent, self.engine._Intent.IMAGE)

    def test_intent_classification_mixed(self):
        intent = self.engine._classify_intent(
            query="Diagnose my crop and recommend fertilizer",
            has_diagnosis=True,
            has_weather=False,
            has_fertilizer=True,
            has_vision=False
        )
        self.assertEqual(intent, self.engine._Intent.MIXED)

    def test_evidence_assembly_weather(self):
        evidence = self.engine._assemble_evidence(
            intent=self.engine._Intent.WEATHER,
            weather_info={"location": "Anuradhapura", "temperature_c": 31.5, "humidity_pct": 82, "fungal_risk_alert": "Moderate", "advisory_notes": ["Avoid spraying in rain"]},
            diagnostic_info=None,
            fertilizer_info=None,
            vision_info=None,
            general_info=None
        )
        self.assertIn("Anuradhapura", evidence)
        self.assertIn("31.5°C", evidence)
        self.assertIn("Moderate", evidence)

    def test_evidence_assembly_diagnosis(self):
        evidence = self.engine._assemble_evidence(
            intent=self.engine._Intent.DIAGNOSIS,
            weather_info=None,
            diagnostic_info={"suspected_disease": "Paddy Blast", "confidence_level": "High", "symptoms_identified": ["brown spots"], "treatment_recommended": ["Tricyclazole"]},
            fertilizer_info=None,
            vision_info=None,
            general_info=None
        )
        self.assertIn("Paddy Blast", evidence)
        self.assertIn("High", evidence)
        self.assertIn("brown spots", evidence)

    def test_evidence_assembly_fertilizer(self):
        evidence = self.engine._assemble_evidence(
            intent=self.engine._Intent.FERTILIZER,
            weather_info=None,
            diagnostic_info=None,
            fertilizer_info={"season": "Yala", "urea_dosage_per_acre_kg": 50.0, "tsp_dosage_per_acre_kg": 25.0, "mop_dosage_per_acre_kg": 25.0, "application_schedule": ["Tillering stage"]},
            vision_info=None,
            general_info=None
        )
        self.assertIn("Yala", evidence)
        self.assertIn("50.0 kg/ac", evidence)
        self.assertIn("Tillering stage", evidence)

    def test_evidence_assembly_general_info(self):
        evidence = self.engine._assemble_evidence(
            intent=self.engine._Intent.KNOWLEDGE,
            weather_info=None,
            diagnostic_info=None,
            fertilizer_info=None,
            vision_info=None,
            general_info={"snippets": [{"content": "Paddy Blast is a fungal disease", "filename": "ROP.pdf", "page": 42}]}
        )
        self.assertIn("DOA RESEARCH", evidence)
        self.assertIn("Paddy Blast is a fungal disease", evidence)

    def test_deterministic_fallback_uses_draft_content(self):
        draft = "This is a test draft response.\n\nIt has two paragraphs."
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.KNOWLEDGE,
            user_query="Test",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=None,
            general_info=None,
            draft_content=draft
        )
        self.assertIn("This is a test draft response.", result)
        self.assertIn("It has two paragraphs.", result)
        # Should not contain hardcoded Paddy Blast text
        self.assertNotIn("Paddy Blast", result)

    def test_deterministic_fallback_knowledge_uses_general_info(self):
        general = {"snippets": [{"content": "DOA recommends resistant varieties for Blast", "filename": "ROP.pdf", "page": 12}]}
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.KNOWLEDGE,
            user_query="What about Blast?",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=None,
            general_info=general,
            draft_content=None
        )
        self.assertIn("DOA recommends resistant varieties for Blast", result)
        # Should not contain hardcoded Paddy Blast paragraph
        self.assertNotIn("Paddy Blast (*Magnaporthe oryzae")

    def test_deterministic_fallback_knowledge_no_info(self):
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.KNOWLEDGE,
            user_query="Random topic",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=None,
            general_info=None,
            draft_content=None
        )
        self.assertIn("consulting the Department of Agriculture", result)

    def test_deterministic_fallback_diagnosis(self):
        diag = {"suspected_disease": "Brown Spot", "confidence_level": "Medium", "symptoms_identified": ["yellowing"], "treatment_recommended": ["copper oxychloride"]}
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.DIAGNOSIS,
            user_query="My leaves are yellowing",
            diagnostic_info=diag,
            fertilizer_info=None,
            weather_info=None,
            general_info=None,
            draft_content=None
        )
        self.assertIn("Brown Spot", result)
        self.assertIn("yellowing", result)
        self.assertIn("copper oxychloride", result)

    def test_deterministic_fallback_weather(self):
        weather = {"location": "Polonnaruwa", "humidity_pct": 85, "fungal_risk_alert": "High", "advisory_notes": ["Avoid field work"]}
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.WEATHER,
            user_query="Weather in Polonnaruwa",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=weather,
            general_info=None,
            draft_content=None
        )
        self.assertIn("Polonnaruwa", result)
        self.assertIn("85%", result)
        self.assertIn("High", result)

    def test_deterministic_fallback_fertilizer(self):
        fert = {"season": "Maha", "urea_dosage_per_acre_kg": 60.0, "application_schedule": ["Apply at tillering"]}
        result = self.engine._deterministic_fallback(
            intent=self.engine._Intent.FERTILIZER,
            user_query="Fertilizer rates",
            diagnostic_info=None,
            fertilizer_info=fert,
            weather_info=None,
            general_info=None,
            draft_content=None
        )
        self.assertIn("Maha", result)
        self.assertIn("60.0 kg", result)
        self.assertIn("tillering", result)

    def test_scrub_banned_opener_removes_ayubowan(self):
        text = "Ayubowan! Here is some advice about your paddy crop."
        result = self.engine._scrub_banned_opener(text)
        # The scrubber removes the banned opener and everything before the first sentence end
        self.assertNotIn("Ayubowan", result)
        # After removing the opener, the text may start with a newline or the next sentence
        self.assertFalse(result.startswith("Ayubowan"))

    def test_scrub_trailing_filler_removes_thank_you(self):
        text = "Here is the advice.\n\nThank you for asking."
        result = self.engine._scrub_trailing_filler(text)
        self.assertNotIn("Thank you", result)
        self.assertNotIn("for asking", result)

    def test_fix_paragraph_flow_collapses_blank_lines(self):
        text = "Line one.\n\n\n\nLine two."
        result = self.engine._fix_paragraph_flow(text)
        self.assertEqual(result.count("\n\n"), 1)

    def test_format_sources_with_known_publications(self):
        general = {"snippets": [
            {"content": "Test snippet", "filename": "ROP.pdf", "page": 10},
            {"content": "Another snippet", "filename": "Danapala.pdf", "page": 5}
        ]}
        result = self.engine._format_sources(general)
        self.assertIn("DOA Rice Operations & Pathology Guide", result)
        self.assertIn("Rice Pathology Research", result)
        self.assertIn("(Page 10)", result)
        self.assertIn("(Page 5)", result)

    def test_format_sources_with_unknown_file(self):
        general = {"snippets": [{"content": "Test", "filename": "unknown.pdf", "page": 1}]}
        result = self.engine._format_sources(general)
        self.assertIn("Department of Agriculture Sri Lanka", result)
        self.assertIn("(Page 1)", result)


if __name__ == "__main__":
    unittest.main()