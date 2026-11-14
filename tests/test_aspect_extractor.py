"""
Tests for HinglishAspectExtractor.
"""

import unittest
from hinglish_absa.aspect_extractor import HinglishAspectExtractor

class TestAspectExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = HinglishAspectExtractor()

    def test_single_aspect_extraction(self):
        aspects = self.extractor.extract_aspects("Camera quality acchi hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertTrue(any("camera" in t for t in terms))

    def test_multi_aspect_extraction(self):
        aspects = self.extractor.extract_aspects("Food hamesha fresh hai par service slow hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("food", terms)
        self.assertIn("service", terms)

    def test_compound_aspect_extraction(self):
        aspects = self.extractor.extract_aspects("Is phone ka battery backup aur heating issue check karo.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("battery backup", terms)
        self.assertIn("heating issue", terms)

    def test_parent_entity_suppression(self):
        aspects = self.extractor.extract_aspects("Phone ka camera bahut accha hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("camera", terms)
        self.assertNotIn("phone", terms)

if __name__ == "__main__":
    unittest.main()
