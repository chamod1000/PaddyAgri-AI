"""
Unit & Output Validation Tests for PDF Report Generator (tests/test_pdf.py)
"""

import unittest
from core.agent_messages import AgentResponse, QueryIntent, DiagnosticResult
from core.report_generator import ReportGenerator, CropHealthPDF


class TestPDF(unittest.TestCase):

    def test_pdf_report_generation(self):
        """Unit test for PDF binary buffer generation."""
        response = AgentResponse(
            query="What is Paddy Blast?",
            intent=QueryIntent.DISEASE_DIAGNOSIS,
            diagnostic_info=DiagnosticResult(
                thought_process="Reasoning...",
                suspected_disease="Paddy Blast",
                symptoms_identified=["diamond lesions"],
                treatment_recommended=["Tricyclazole"],
                confidence_level="High"
            ),
            final_synthesis="Synthesis text..."
        )

        pdf_bytes = ReportGenerator.generate_pdf(response)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_pdf_report_missing_fields_graceful(self):
        """Negative test verifying PDF generator handles missing fields gracefully."""
        empty_response = AgentResponse(
            query="General paddy query",
            intent=QueryIntent.GENERAL,
            final_synthesis=""
        )

        pdf_bytes = ReportGenerator.generate_pdf(empty_response)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)


if __name__ == "__main__":
    unittest.main()
