"""
Comprehensive test runner for Hinglish Aspect-Based Sentiment Analysis.
Uses standard Python unittest framework for zero-dependency execution.
"""

import unittest
from hinglish_absa import HinglishABSA, HinglishPreprocessor, HinglishAspectExtractor, HinglishSentimentClassifier

class TestUserExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.absa = HinglishABSA()

    def test_example_1_camera_vs_battery(self):
        """1. 'Phone ka camera bahut accha hai but battery backup bekar hai.'"""
        text = "Phone ka camera bahut accha hai but battery backup bekar hai."
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("camera", aspect_map)
        self.assertEqual(aspect_map["camera"], "positive")

        has_battery = any("battery" in k for k in aspect_map)
        self.assertTrue(has_battery, f"Expected battery in {aspect_map}")
        bat_key = [k for k in aspect_map if "battery" in k][0]
        self.assertEqual(aspect_map[bat_key], "negative")

    def test_example_2_story_vs_acting(self):
        """2. 'Movie ki story mast thi but acting thodi boring thi.'"""
        text = "Movie ki story mast thi but acting thodi boring thi."
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("story", aspect_map)
        self.assertEqual(aspect_map["story"], "positive")

        self.assertIn("acting", aspect_map)
        self.assertEqual(aspect_map["acting"], "negative")

    def test_example_3_performance_vs_heating(self):
        """3. 'Laptop ka performance awesome hai, lekin heating issue bahut hai.'"""
        text = "Laptop ka performance awesome hai, lekin heating issue bahut hai."
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("performance", aspect_map)
        self.assertEqual(aspect_map["performance"], "positive")

        has_heating = any("heat" in k for k in aspect_map)
        self.assertTrue(has_heating, f"Expected heating in {aspect_map}")
        heat_key = [k for k in aspect_map if "heat" in k][0]
        self.assertEqual(aspect_map[heat_key], "negative")

    def test_example_4_phone_slang_op_paisa_vasool(self):
        """4. 'Bhai phone ekdum OP hai, paisa vasool!'"""
        text = "Bhai phone ekdum OP hai, paisa vasool!"
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("phone", aspect_map)
        self.assertEqual(aspect_map["phone"], "positive")

    def test_hindi_devanagari_sentences(self):
        """Test pure Devanagari Hindi support."""
        text = "फोन का कैमरा बहुत अच्छा है लेकिन बैटरी बैकअप बेकार है।"
        res = self.absa.analyze(text)
        aspect_map = {a.aspect: a.polarity for a in res.aspects}

        self.assertIn("कैमरा", aspect_map)
        self.assertEqual(aspect_map["कैमरा"], "positive")
        self.assertIn("बैटरी बैकअप", aspect_map)
        self.assertEqual(aspect_map["बैटरी बैकअप"], "negative")

        # Movie test
        res_movie = self.absa.analyze("मूवी की कहानी मस्त थी पर एक्टिंग थोड़ी बोरिंग थी।")
        m_map = {a.aspect: a.polarity for a in res_movie.aspects}
        self.assertEqual(m_map.get("कहानी"), "positive")
        self.assertEqual(m_map.get("एक्टिंग"), "negative")


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


class TestAspectExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = HinglishAspectExtractor()

    def test_single_and_multi_aspects(self):
        aspects = self.extractor.extract_aspects("Food hamesha fresh hai par service slow hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("food", terms)
        self.assertIn("service", terms)

    def test_compound_aspects(self):
        aspects = self.extractor.extract_aspects("Battery backup aur heating issue dono hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("battery backup", terms)
        self.assertIn("heating issue", terms)

    def test_parent_suppression(self):
        aspects = self.extractor.extract_aspects("Phone ka camera bahut accha hai.")
        terms = [a.term.lower() for a in aspects]
        self.assertIn("camera", terms)
        self.assertNotIn("phone", terms)


class TestSentimentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = HinglishSentimentClassifier()

    def test_contrastive_clauses(self):
        sentence = "Phone ka camera bahut accha hai but battery backup bekar hai."
        res1 = self.clf.predict_sentiment(sentence, "camera", "Camera")
        res2 = self.clf.predict_sentiment(sentence, "battery backup", "Battery")
        self.assertEqual(res1.polarity, "positive")
        self.assertEqual(res2.polarity, "negative")

    def test_negation(self):
        sent_neg = "Camera bilkul accha nahi hai."
        res = self.clf.predict_sentiment(sent_neg, "camera", "Camera")
        self.assertEqual(res.polarity, "negative")

    def test_slang_polarity(self):
        sent = "Display ekdum jhakaas hai."
        res = self.clf.predict_sentiment(sent, "display", "Display")
        self.assertEqual(res.polarity, "positive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
