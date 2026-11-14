"""
Tests for HinglishPreprocessor: slang, emojis, phonetic normalization, clause segmentation.
"""

import unittest
from hinglish_absa.preprocessor import HinglishPreprocessor

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = HinglishPreprocessor()

    def test_elongations(self):
        self.assertEqual(self.prep.remove_elongations("mastttt"), "mast")
        self.assertEqual(self.prep.remove_elongations("acchaaaa"), "accha")

    def test_phonetic_normalization(self):
        norm = self.prep.normalize_text("camera bhot acha hai")
        self.assertEqual(norm, "camera bahut accha hai")

    def test_emojis(self):
        emojis = self.prep.extract_emojis("Camera 🔥 but battery 🤮")
        self.assertEqual(len(emojis), 2)
        self.assertEqual(emojis[0]["sentiment"], "positive")
        self.assertEqual(emojis[1]["sentiment"], "negative")

    def test_clause_segmentation(self):
        clauses = self.prep.segment_clauses("Phone ka camera mast hai but battery bekar hai")
        self.assertEqual(len(clauses), 2)
        self.assertTrue(clauses[1]["is_contrastive"])

if __name__ == "__main__":
    unittest.main()
