"""
Unit Test Suite for ReflectionAgent (Safety & Quality Verifier)
Tests Reflection & Self-Critique Pattern logic independently.
"""

import unittest
from agent_messages import DiagnosticResult, FertilizerRecommendation, SafetyVerdict, ReflectionResult
from agents import ReflectionAgent


class TestReflectionAgent(unittest.TestCase):
    def setUp(self):
        self.reflection_agent = ReflectionAgent()

    def test_valid_fertilizer_dosage(self):
        rec = FertilizerRecommendation(
            season="Yala",
            district_zone="Polonnaruwa",
            urea_dosage_per_acre_kg=50.0,
            tsp_dosage_per_acre_kg=25.0,
            mop_dosage_per_acre_kg=25.0,
            application_schedule=[]
        )
        verdicts = self.reflection_agent.verify_fertilizer_dosage(rec)
        self.assertTrue(all(v.passed for v in verdicts))

    def test_exceeded_fertilizer_dosage_warning(self):
        rec = FertilizerRecommendation(
            season="Yala",
            district_zone="Polonnaruwa",
            urea_dosage_per_acre_kg=85.0,  # Exceeds 65 kg max limit
            tsp_dosage_per_acre_kg=25.0,
            mop_dosage_per_acre_kg=25.0,
            application_schedule=[]
        )
        verdicts = self.reflection_agent.verify_fertilizer_dosage(rec)
        failed = [v for v in verdicts if not v.passed]
        self.assertEqual(len(failed), 1)
        self.assertIn("urea_dosage", failed[0].check_name)

    def test_banned_chemical_detection(self):
        treatments = ["Apply Paraquat solution", "Spray Tricyclazole"]
        verdicts = self.reflection_agent.verify_banned_chemicals(treatments)
        failed = [v for v in verdicts if not v.passed]
        self.assertEqual(len(failed), 1)
        self.assertIn("paraquat", failed[0].message.lower())

    def test_clean_treatments_pass(self):
        treatments = ["Apply Tricyclazole 75% WP", "Maintain proper drainage"]
        verdicts = self.reflection_agent.verify_banned_chemicals(treatments)
        self.assertTrue(all(v.passed for v in verdicts))


if __name__ == "__main__":
    unittest.main()
