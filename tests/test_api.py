"""
Tests for REST API endpoints and data contracts.
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hinglish_absa import HinglishABSA

class TestAPIContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HinglishABSA()

    def test_single_inference_contract(self):
        """Verify API response contract for single text analysis."""
        text = "Phone ka camera bahut accha hai but battery backup bekar hai."
        res = self.engine.analyze(text)
        data = res.to_dict()

        self.assertIn("raw_text", data)
        self.assertIn("aspects", data)
        self.assertIn("aspect_count", data)
        self.assertIn("overall_polarity", data)
        self.assertGreaterEqual(len(data["aspects"]), 2)

        for asp in data["aspects"]:
            self.assertIn("aspect", asp)
            self.assertIn("category", asp)
            self.assertIn("sentiment", asp)
            self.assertIn("confidence", asp)
            self.assertIn("evidence", asp)

    def test_batch_inference_contract(self):
        """Verify API response contract for batch text analysis."""
        texts = [
            "Movie ki story mast thi but acting thodi boring thi.",
            "Laptop ka performance awesome hai, lekin heating issue bahut hai."
        ]
        results = self.engine.analyze_batch(texts)
        self.assertEqual(len(results), 2)
        dict_results = [r.to_dict() for r in results]
        self.assertEqual(len(dict_results), 2)
        self.assertTrue(dict_results[0]["aspect_count"] >= 2)
        self.assertTrue(dict_results[1]["aspect_count"] >= 2)

if __name__ == "__main__":
    unittest.main()
