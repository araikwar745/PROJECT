"""
Hinglish Aspect-Based Sentiment Analysis (ABSA)
A comprehensive NLP toolkit for identifying aspects and aspect-level sentiments
in Hindi-English code-mixed (Hinglish) text, with support for informal slang,
phonetic spelling variations, emojis, and multi-domain reviews.
"""

from .pipeline import HinglishABSA, ABSAResult
from .aspect_extractor import HinglishAspectExtractor, ExtractedAspect
from .sentiment_classifier import HinglishSentimentClassifier, AspectSentimentResult
from .preprocessor import HinglishPreprocessor
from .llm_enhancer import LLMEnhancer

__version__ = "1.0.0"
__all__ = [
    "HinglishABSA",
    "ABSAResult",
    "HinglishAspectExtractor",
    "ExtractedAspect",
    "HinglishSentimentClassifier",
    "AspectSentimentResult",
    "HinglishPreprocessor",
    "LLMEnhancer",
]
