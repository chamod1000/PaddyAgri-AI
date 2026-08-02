"""
Unit & Negative Tests for Vision Processing Layer (tests/test_vision.py)
"""

import unittest
from core.vision_processor import VisionProcessor, VisionAnalysisResult


class TestVision(unittest.TestCase):

    def test_vision_processor_mock_extraction(self):
        """Unit test for vision feature extraction."""
        processor = VisionProcessor()
        dummy_bytes = b"fake_image_bytes_12345"
        result = processor.analyze_image(dummy_bytes)
        
        self.assertIsInstance(result, VisionAnalysisResult)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.visible_symptoms)

    def test_vision_processor_invalid_input(self):
        """Negative test for invalid / None image bytes handling."""
        processor = VisionProcessor()
        result = processor.analyze_image(None)
        self.assertIsInstance(result, VisionAnalysisResult)
        self.assertTrue(any("Empty image payload" in s for s in result.visible_symptoms))


if __name__ == "__main__":
    unittest.main()
