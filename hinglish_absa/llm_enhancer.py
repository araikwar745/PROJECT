"""
Optional LLM Integration Engine for Hinglish Aspect-Based Sentiment Analysis.
Enables plug-and-play support for Gemini API and OpenAI API when API keys are available,
while maintaining 100% offline functionality by default using the local hybrid ABSA pipeline.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

SYSTEM_PROMPT_ABSA = """You are an expert computational linguist specializing in Aspect-Based Sentiment Analysis (ABSA) for:
1. English
2. Hindi (Devanagari script)
3. Hinglish / Code-mixed Hindi-English (Roman script)

Your task:
Analyze the input text and extract all explicit or implicit aspects (entities/attributes/features being evaluated),
and classify the sentiment polarity for each aspect as one of: "positive", "negative", or "neutral".

For each aspect provide:
- aspect: the exact term or attribute
- category: standardized category (e.g., Camera, Battery, Performance, Display, Food Quality & Taste, Service & Staff, Ambience, Story & Plot, Acting & Cast, Price & Value, Weather & Environment, Software, OS & Updates, Delivery & Packaging, General / Product Quality)
- sentiment: "positive", "negative", or "neutral"
- confidence: float between 0.0 and 1.0
- explanation: brief justification citing specific words, slang, or negations in the text

If the text contains no specific noun or is a single word / general expression (e.g., 'Bakwas', 'Mast', 'बकवास'), use aspect 'Overall Experience' with category 'General / Product Quality'.

Respond ONLY with valid JSON in this format:
{
  "aspects": [
    {
      "aspect": "term",
      "category": "Category Name",
      "sentiment": "positive|negative|neutral",
      "confidence": 0.95,
      "explanation": "reasoning"
    }
  ],
  "overall_sentiment": "positive|negative|neutral|mixed",
  "document_summary": "brief summary"
}
"""

def _auto_load_dotenv():
    """Auto-load key-value pairs from .env in project root if present."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_file = os.path.join(base_dir, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_auto_load_dotenv()


class LLMEnhancer:
    """
    Manages optional cloud LLM API connectivity for ABSA inference.
    Supports Gemini API and OpenAI API.
    Zero external heavy SDK requirements (uses standard library urllib).
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
        _auto_load_dotenv()
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")

    def is_configured(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def analyze_with_gemini(self, text: str) -> Optional[Dict[str, Any]]:
        """Call Gemini API via REST."""
        if not self.is_configured():
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key.strip()}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT_ABSA},
                        {"text": f"Input text to analyze:\n\"{text}\""}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                raw_reply = res_json["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(raw_reply)
        except Exception as e:
            # Silently fallback to local pipeline if API network/quota fails
            return None

    def analyze_with_openai(self, text: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API via REST."""
        if not self.is_configured():
            return None

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_ABSA},
                {"role": "user", "content": f"Input text to analyze:\n\"{text}\""}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key.strip()}"
                }
            )
            with urllib.request.urlopen(req, timeout=12) as response:
                res_body = response.read().decode("utf-8")
                res_json = json.loads(res_body)
                raw_reply = res_json["choices"][0]["message"]["content"]
                return json.loads(raw_reply)
        except Exception:
            return None

    def analyze(self, text: str) -> Optional[Dict[str, Any]]:
        """Execute ABSA through configured LLM provider, or return None to use local."""
        if not self.is_configured():
            return None

        if self.provider == "openai":
            return self.analyze_with_openai(text)
        return self.analyze_with_gemini(text)
