"""
Unit & Negative Tests for Knowledge Center (tests/test_knowledge_center.py)
"""

import unittest
from core.knowledge_center import KnowledgeCenter, KnowledgeSearchResult


class TestKnowledgeCenter(unittest.TestCase):

    def test_knowledge_center_search_hit(self):
        """Unit test verifying keyword search hit."""
        kc = KnowledgeCenter()
        result = kc.search(query="blast")
        self.assertIsInstance(result, KnowledgeSearchResult)
        self.assertGreater(result.total_results, 0)
        self.assertTrue(any("Blast" in a.title for a in result.matched_articles))

    def test_knowledge_center_category_filter(self):
        """Unit test verifying category filtering."""
        kc = KnowledgeCenter()
        result = kc.search(query="", category="Fertilizers")
        self.assertGreater(result.total_results, 0)
        self.assertTrue(all(a.category == "Fertilizers" for a in result.matched_articles))

    def test_knowledge_center_no_matches_negative(self):
        """Negative test verifying zero results for non-existent queries."""
        kc = KnowledgeCenter()
        result = kc.search(query="xyz_non_existent_term_12345")
        self.assertEqual(result.total_results, 0)
        self.assertEqual(len(result.matched_articles), 0)


if __name__ == "__main__":
    unittest.main()
