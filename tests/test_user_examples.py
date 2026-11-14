"""
Test suite verifying the user's specific Hinglish ABSA examples and edge cases.
"""

import unittest
from hinglish_absa import HinglishABSA

class TestUserExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.absa = HinglishABSA()

    def test_example_1_camera_vs_battery(self):
        """
        1. 'Phone ka camera bahut accha hai but battery backup bekar hai.'
        - Camera -> Positive
        - Battery backup -> Negative
        """
        text = "Phone ka camera bahut accha hai but battery backup bekar hai."
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("camera", aspect_map)
        self.assertEqual(aspect_map["camera"], "positive")

        has_battery = any("battery" in k for k in aspect_map)
        self.assertTrue(has_battery, f"Expected battery in {aspect_map}")
        battery_key = [k for k in aspect_map if "battery" in k][0]
        self.assertEqual(aspect_map[battery_key], "negative")

    def test_example_2_story_vs_acting(self):
        """
        2. 'Movie ki story mast thi but acting thodi boring thi.'
        - Story -> Positive
        - Acting -> Negative
        """
        text = "Movie ki story mast thi but acting thodi boring thi."
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("story", aspect_map)
        self.assertEqual(aspect_map["story"], "positive")

        self.assertIn("acting", aspect_map)
        self.assertEqual(aspect_map["acting"], "negative")

    def test_example_3_performance_vs_heating(self):
        """
        3. 'Laptop ka performance awesome hai, lekin heating issue bahut hai.'
        - Performance -> Positive
        - Heating / Heating issue -> Negative
        """
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
        """
        4. 'Bhai phone ekdum OP hai, paisa vasool!'
        - Phone quality -> Positive
        """
        text = "Bhai phone ekdum OP hai, paisa vasool!"
        res = self.absa.analyze(text)
        aspect_map = {a.aspect.lower(): a.polarity for a in res.aspects}

        self.assertIn("phone", aspect_map)
        self.assertEqual(aspect_map["phone"], "positive")

    def test_user_english_sentiment_categories(self):
        """Test English sentiment categories from user prompt."""
        # 1. Positive: "I love using this product, it's amazing!"
        res1 = self.absa.analyze("I love using this product, it's amazing!")
        self.assertTrue(res1.has_aspects)
        self.assertEqual(res1.aspects[0].polarity, "positive")

        # 2. Neutral: "The restaurant wasn't terrible, it wasn't great either, just okay."
        res2 = self.absa.analyze("The restaurant wasn't terrible, it wasn't great either, just okay.")
        self.assertTrue(res2.has_aspects)
        self.assertEqual(res2.aspects[0].polarity, "neutral")

        # 3. Negative: "I hate these updates, everything is slow and broken."
        res3 = self.absa.analyze("I hate these updates, everything is slow and broken.")
        self.assertTrue(res3.has_aspects)
        aspect_map3 = {a.aspect.lower(): a.polarity for a in res3.aspects}
        self.assertIn("updates", aspect_map3)
        self.assertEqual(aspect_map3["updates"], "negative")

    def test_user_hindi_devanagari_examples(self):
        """Test Hindi Devanagari examples from user prompt."""
        # 1. यह फिल्म बहुत अच्छी है। -> Positive
        res1 = self.absa.analyze("यह फिल्म बहुत अच्छी है।")
        self.assertTrue(res1.has_aspects)
        self.assertEqual(res1.aspects[0].polarity, "positive")

        # 2. मुझे यह खाना पसंद नहीं आया। -> Negative
        res2 = self.absa.analyze("मुझे यह खाना पसंद नहीं आया।")
        self.assertTrue(res2.has_aspects)
        self.assertEqual(res2.aspects[0].polarity, "negative")

        # 3. आज मौसम ठीक है। -> Neutral
        res3 = self.absa.analyze("आज मौसम ठीक है।")
        self.assertTrue(res3.has_aspects)
        self.assertEqual(res3.aspects[0].polarity, "neutral")

    def test_user_roman_hinglish_examples(self):
        """Test Roman Hinglish examples from user prompt."""
        # 1. Yeh movie bahut achhi hai. -> Positive
        res1 = self.absa.analyze("Yeh movie bahut achhi hai.")
        self.assertTrue(res1.has_aspects)
        self.assertEqual(res1.aspects[0].polarity, "positive")

        # 2. Mujhe yeh khana bilkul pasand nahi aaya. -> Negative
        res2 = self.absa.analyze("Mujhe yeh khana bilkul pasand nahi aaya.")
        self.assertTrue(res2.has_aspects)
        self.assertEqual(res2.aspects[0].polarity, "negative")

        # 3. Aaj mausam achha hai. -> Positive
        res3 = self.absa.analyze("Aaj mausam achha hai.")
        self.assertTrue(res3.has_aspects)
        self.assertEqual(res3.aspects[0].polarity, "positive")

        # 4. Maine naya mobile kharida. -> Neutral
        res4 = self.absa.analyze("Maine naya mobile kharida.")
        self.assertTrue(res4.has_aspects)
        self.assertEqual(res4.aspects[0].polarity, "neutral")

    def test_single_word_inputs(self):
        """Test single-word / short exclamation inputs."""
        res_neg = self.absa.analyze("Bakwas")
        self.assertEqual(res_neg.aspects[0].polarity, "negative")

        res_pos = self.absa.analyze("Mast")
        self.assertEqual(res_pos.aspects[0].polarity, "positive")

        res_hi_neg = self.absa.analyze("बकवास")
        self.assertEqual(res_hi_neg.aspects[0].polarity, "negative")

if __name__ == "__main__":
    unittest.main()
