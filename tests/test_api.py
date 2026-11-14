"""
Tests for REST API endpoints using standard unittest and Starlette TestClient.
"""

import unittest
from starlette.testclient import TestClient
from hinglish_absa.api import app

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_api_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_api_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("endpoints", data)

    def test_api_analyze(self):
        payload = {"text": "Phone ka camera bahut accha hai but battery backup bekar hai."}
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("aspects", data)
        self.assertGreaterEqual(len(data["aspects"]), 2)

    def test_api_batch(self):
        payload = {"texts": [
            "Movie ki story mast thi but acting thodi boring thi.",
            "Laptop ka performance awesome hai, lekin heating issue bahut hai."
        ]}
        response = self.client.post("/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["results"]), 2)

if __name__ == "__main__":
    unittest.main()
