"""
Unit & Repository Tests for Case Management System (tests/test_case_manager.py)
"""

import os
import tempfile
import unittest
from core.case_manager import (
    CaseManager, JSONCaseRepository, CaseRecord, DiagnosisSnapshot, FollowUpRecord
)
from core.agent_messages import AgentResponse, QueryIntent, DiagnosticResult


class TestCaseManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_json = os.path.join(self.temp_dir.name, "test_cases.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_json_case_repository(self):
        """Unit test for JSON repository save and retrieve."""
        repo = JSONCaseRepository(storage_path=self.temp_json)

        case = CaseRecord(
            case_id="case_test_1",
            session_id="sess_test_1",
            disease="Paddy Blast",
            confidence="High"
        )
        repo.save_case(case)

        retrieved = repo.get_case("case_test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.disease, "Paddy Blast")
        self.assertEqual(retrieved.confidence, "High")

    def test_case_manager_record_diagnosis(self):
        """Unit & integration test for CaseManager recording a diagnosis."""
        repo = JSONCaseRepository(storage_path=self.temp_json)
        cm = CaseManager(repository=repo)

        response = AgentResponse(
            query="Inspect leaf",
            intent=QueryIntent.DISEASE_DIAGNOSIS,
            diagnostic_info=DiagnosticResult(
                thought_process="Reasoning...",
                suspected_disease="Bacterial Leaf Blight",
                symptoms_identified=["yellowing"],
                treatment_recommended=["Copper hydroxide"],
                confidence_level="High"
            ),
            final_synthesis="Synthesis..."
        )

        record = cm.record_diagnosis(response)
        self.assertIsNotNone(record)
        self.assertEqual(record.disease, "Bacterial Leaf Blight")
        self.assertEqual(len(record.snapshots), 1)


if __name__ == "__main__":
    unittest.main()
