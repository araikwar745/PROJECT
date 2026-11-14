"""
Tests for HinglishSentimentClassifier: negations, intensifiers, slang, contrastive clauses.
"""

import unittest
from hinglish_absa.sentiment_classifier import HinglishSentimentClassifier

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

        sent_bakwas = "Sound ekdum bakwas hai."
        res2 = self.clf.predict_sentiment(sent_bakwas, "sound", "Sound & Audio")
        self.assertEqual(res2.polarity, "negative")

if __name__ == "__main__":
    unittest.main()
