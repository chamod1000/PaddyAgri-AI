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
        intent = _classify_intent(
            query="What is paddy blast disease?",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, _Intent.KNOWLEDGE)

    def test_intent_classification_diagnosis(self):
        intent = _classify_intent(
            query="My rice leaves have brown spots, what should I do?",
            has_diagnosis=True,
            has_weather=False,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, _Intent.DIAGNOSIS)

    def test_intent_classification_weather(self):
        intent = _classify_intent(
            query="What is the weather forecast for Anuradhapura?",
            has_diagnosis=False,
            has_weather=True,
            has_fertilizer=False,
            has_vision=False
        )
        self.assertEqual(intent, _Intent.WEATHER)

    def test_intent_classification_fertilizer(self):
        intent = _classify_intent(
            query="How much urea should I apply this season?",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=True,
            has_vision=False
        )
        self.assertEqual(intent, _Intent.FERTILIZER)

    def test_intent_classification_image(self):
        intent = _classify_intent(
            query="Analyze this leaf photo",
            has_diagnosis=False,
            has_weather=False,
            has_fertilizer=False,
            has_vision=True
        )
        self.assertEqual(intent, _Intent.IMAGE)

    def test_intent_classification_mixed(self):
        intent = _classify_intent(
            query="Diagnose my crop and recommend fertilizer",
            has_diagnosis=True,
            has_weather=False,
            has_fertilizer=True,
            has_vision=False
        )
        self.assertEqual(intent, _Intent.MIXED)

    def test_evidence_assembly_weather(self):
        evidence = self.engine._assemble_evidence(
            intent=_Intent.WEATHER,
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
            intent=_Intent.DIAGNOSIS,
            weather_info=None,
            diagnostic_info={"suspected_disease": "Paddy Blast", "confidence_level": "High", "symptoms_identified": ["brown spots"], "treatment_recommended": ["Tricyclazole"]},
            fertilizer_info=None,
            vision_info=None,
            general_info=None
        )
        self.assertIn("Paddy Blast", evidence)
        # The evidence block now carries a confidence *tier* plus the certainty
        # instruction the model must obey, instead of the bare label.
        self.assertIn("HIGH", evidence)
        self.assertIn("state this disease directly", evidence)
        self.assertIn("brown spots", evidence)

    def test_evidence_assembly_diagnosis_low_confidence_hedges(self):
        evidence = self.engine._assemble_evidence(
            intent=_Intent.DIAGNOSIS,
            weather_info=None,
            diagnostic_info={"suspected_disease": "Paddy Blast", "confidence_level": "Low",
                             "symptoms_identified": ["brown spots"], "treatment_recommended": ["Tricyclazole"]},
            fertilizer_info=None,
            vision_info=None,
            general_info=None
        )
        self.assertIn("LOW", evidence)
        self.assertIn("do NOT present this as the diagnosis", evidence)

    def test_evidence_assembly_diagnosis_missing_confidence_is_not_high(self):
        # A missing confidence value must not be read as certainty.
        evidence = self.engine._assemble_evidence(
            intent=_Intent.DIAGNOSIS,
            weather_info=None,
            diagnostic_info={"suspected_disease": "Paddy Blast",
                             "symptoms_identified": ["brown spots"], "treatment_recommended": ["Tricyclazole"]},
            fertilizer_info=None,
            vision_info=None,
            general_info=None
        )
        self.assertIn("LOW", evidence)
        self.assertNotIn("state this disease directly", evidence)

    def test_evidence_assembly_fertilizer(self):
        evidence = self.engine._assemble_evidence(
            intent=_Intent.FERTILIZER,
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
            intent=_Intent.KNOWLEDGE,
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
            intent=_Intent.KNOWLEDGE,
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
            intent=_Intent.KNOWLEDGE,
            user_query="What about Blast?",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=None,
            general_info=general,
            draft_content=None
        )
        self.assertIn("DOA recommends resistant varieties for Blast", result)
        # Should not contain hardcoded Paddy Blast paragraph
        self.assertNotIn("Paddy Blast (*Magnaporthe oryzae)", result)

    def test_deterministic_fallback_knowledge_no_info(self):
        result = self.engine._deterministic_fallback(
            intent=_Intent.KNOWLEDGE,
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
            intent=_Intent.DIAGNOSIS,
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
            intent=_Intent.WEATHER,
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
            intent=_Intent.FERTILIZER,
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

    def test_scrub_banned_opener_handles_none(self):
        with self.assertRaises(AssertionError):
            self.engine._scrub_banned_opener(None)

    def test_scrub_banned_opener_handles_empty_string(self):
        result = self.engine._scrub_banned_opener("")
        self.assertEqual(result, "")

    def test_scrub_trailing_filler_handles_none(self):
        with self.assertRaises(AssertionError):
            self.engine._scrub_trailing_filler(None)

    def test_scrub_trailing_filler_handles_empty_string(self):
        result = self.engine._scrub_trailing_filler("")
        self.assertEqual(result, "")

    def test_fix_paragraph_flow_handles_none(self):
        with self.assertRaises(AssertionError):
            self.engine._fix_paragraph_flow(None)

    def test_fix_paragraph_flow_handles_empty_string(self):
        result = self.engine._fix_paragraph_flow("")
        self.assertEqual(result, "")

    def test_extract_final_response_returns_none_on_malformed_xml(self):
        malformed_xml = "This is just plain text without XML tags"
        result = self.engine._extract_final_response(malformed_xml)
        self.assertIsNone(result)

    def test_compose_conversation_handles_malformed_llm_output(self):
        # This test verifies that when the LLM returns malformed output (no XML tags),
        # the system falls back to deterministic fallback and returns a valid string
        result = self.engine.compose_conversation(
            user_query="Test query",
            diagnostic_info=None,
            fertilizer_info=None,
            weather_info=None,
            general_info=None,
            reflection_result=None,
            final_synthesis="This is malformed output without XML tags"
        )
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertNotIn("None", result)

    def test_scrub_trailing_filler_removes_thank_you(self):
        text = "Here is the advice.\n\nThank you for asking."
        result = self.engine._scrub_trailing_filler(text)
        self.assertNotIn("Thank you", result)
        self.assertNotIn("for asking", result)

    def test_fix_paragraph_flow_collapses_blank_lines(self):
        text = "Line one.\n\n\n\nLine two."
        result = self.engine._fix_paragraph_flow(text)
        self.assertEqual(result.count("\n\n"), 1)

    def test_confidence_tier_mapping(self):
        from core.synthesis.conversation_experience_engine import _confidence_tier
        for raw, expected in (
            ("High", "high"), ("Very High", "high"), ("confirmed", "high"),
            ("Medium", "medium"), ("Moderate", "medium"), ("likely", "medium"),
            ("Low", "low"), ("unsure", "low"),
            ("85%", "high"), ("60%", "medium"), ("30%", "low"),
            (0.9, "high"), (0.6, "medium"), (0.2, "low"), (92, "high"),
        ):
            self.assertEqual(_confidence_tier(raw), expected, f"for {raw!r}")

    def test_confidence_tier_uncertain_is_not_certain(self):
        # "uncertain" contains "certain"; a substring scan promoted it to HIGH.
        from core.synthesis.conversation_experience_engine import _confidence_tier
        self.assertEqual(_confidence_tier("Uncertain"), "low")
        self.assertEqual(_confidence_tier("not conclusive"), "low")

    def test_confidence_tier_unknown_defaults_to_low(self):
        from core.synthesis.conversation_experience_engine import _confidence_tier
        for raw in (None, "", "banana"):
            self.assertEqual(_confidence_tier(raw), "low", f"for {raw!r}")

    def test_deterministic_fallback_diagnosis_high_states_directly(self):
        diag = {"suspected_disease": "Paddy Blast", "confidence_level": "High",
                "symptoms_identified": ["lesions"], "treatment_recommended": ["Tricyclazole"]}
        result = self.engine._deterministic_fallback(
            intent=_Intent.DIAGNOSIS, user_query="spots", diagnostic_info=diag,
            fertilizer_info=None, weather_info=None, general_info=None, draft_content=None)
        self.assertIn("Acting within the first week", result)
        self.assertNotIn("If this turns out to be the cause", result)

    def test_deterministic_fallback_diagnosis_low_hedges_and_gives_next_step(self):
        diag = {"suspected_disease": "Paddy Blast", "confidence_level": "Low",
                "symptoms_identified": ["lesions"], "treatment_recommended": ["Tricyclazole"]}
        result = self.engine._deterministic_fallback(
            intent=_Intent.DIAGNOSIS, user_query="spots", diagnostic_info=diag,
            fertilizer_info=None, weather_info=None, general_info=None, draft_content=None)
        self.assertIn("If this turns out to be the cause", result)
        self.assertIn("confirm the cause", result)
        self.assertNotIn("Acting within the first week", result)

    def test_scrub_trailing_filler_never_blanks_a_short_reply(self):
        # A one-line reply may legitimately end in a filler phrase.
        text = "Hi there! Feel free to ask any questions."
        self.assertTrue(self.engine._scrub_trailing_filler(text).strip())

    def test_format_sources_with_known_publications(self):
        general = {"snippets": [
            {"content": "Test snippet", "filename": "ROP.pdf", "page": 10},
            {"content": "Another snippet", "filename": "Danapala.pdf", "page": 5}
        ]}
        result = self.engine._format_sources(general)
        # Labels match _PUB_MAP, which names the real source documents:
        # Data/PDF/General_Cultivation_Guidelines/ROP_book.pdf and
        # "Dr. Danapala_Book_With Cover page Final.pdf". The previous
        # expectations ("DOA Rice Operations & Pathology Guide",
        # "Rice Pathology Research") were older label strings that conflated
        # the two publications.
        self.assertIn("Department of Agriculture Rice Operations Guide", result)
        self.assertIn("Dr. M. P. Dhanapala Rice Pathology", result)
        self.assertIn("(Page 10)", result)
        self.assertIn("(Page 5)", result)

    def test_format_sources_with_unknown_file(self):
        general = {"snippets": [{"content": "Test", "filename": "unknown.pdf", "page": 1}]}
        result = self.engine._format_sources(general)
        self.assertIn("Department of Agriculture Sri Lanka", result)
        self.assertIn("(Page 1)", result)


if __name__ == "__main__":
    unittest.main()