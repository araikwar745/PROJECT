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

if __name__ == "__main__":
    unittest.main()
