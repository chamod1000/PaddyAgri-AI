"""
Unit & Deterministic Metrics Tests for Analytics Service (tests/test_analytics.py)
"""

import os
import tempfile
import unittest
from core.analytics import AnalyticsService, AnalyticsSummary
from core.case_manager import JSONCaseRepository, CaseRecord, DiagnosisSnapshot


class TestAnalytics(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analytics_empty_repository(self):
        """Negative test for AnalyticsService with empty repository."""
        temp_json = os.path.join(self.temp_dir.name, "empty_cases.json")
        repo = JSONCaseRepository(storage_path=temp_json)
        svc = AnalyticsService(repository=repo)

        summary = svc.get_summary()
        self.assertIsInstance(summary, AnalyticsSummary)
        self.assertEqual(summary.repository_stats.total_cases, 0)
        self.assertEqual(summary.repository_stats.total_diagnoses, 0)
        self.assertEqual(summary.disease_stats.most_common_disease, "None")

    def test_analytics_aggregation(self):
        """Unit test for AnalyticsService metric computations."""
        temp_json = os.path.join(self.temp_dir.name, "sample_cases.json")
        repo = JSONCaseRepository(storage_path=temp_json)
        
        c1 = CaseRecord(case_id="c1", session_id="s1", disease="Paddy Blast", confidence="High", snapshots=[DiagnosisSnapshot(diagnosis="Paddy Blast")])
        c2 = CaseRecord(case_id="c2", session_id="s2", disease="Paddy Blast", confidence="High", snapshots=[DiagnosisSnapshot(diagnosis="Paddy Blast")])
        c3 = CaseRecord(case_id="c3", session_id="s3", disease="Sheath Blight", confidence="Medium", snapshots=[DiagnosisSnapshot(diagnosis="Sheath Blight")])
        
        repo.save_case(c1)
        repo.save_case(c2)
        repo.save_case(c3)

        svc = AnalyticsService(repository=repo)
        summary = svc.get_summary()

        self.assertEqual(summary.repository_stats.total_cases, 3)
        self.assertEqual(summary.repository_stats.total_diagnoses, 3)
        self.assertEqual(summary.disease_stats.most_common_disease, "Paddy Blast")
        self.assertEqual(summary.disease_stats.disease_frequencies["Paddy Blast"], 2)


if __name__ == "__main__":
    unittest.main()
