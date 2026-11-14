"""
Unified Pipeline for Hinglish Aspect-Based Sentiment Analysis (ABSA).
Integrates preprocessing, aspect extraction, sentiment classification,
explanation generation, and optional LLM enhancement.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from .preprocessor import HinglishPreprocessor
from .aspect_extractor import HinglishAspectExtractor, ExtractedAspect
from .sentiment_classifier import HinglishSentimentClassifier, AspectSentimentResult
from .llm_enhancer import LLMEnhancer


class ABSAResult:
    """Encapsulates the end-to-end ABSA analysis result for a sentence or paragraph."""

    def __init__(
        self,
        raw_text: str,
        normalized_text: str,
        aspects: List[AspectSentimentResult],
        engine_source: str = "hybrid_local"
    ):
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.aspects = aspects
        self.engine_source = engine_source

    @property
    def has_aspects(self) -> bool:
        return len(self.aspects) > 0

    @property
    def aspect_breakdown(self) -> Dict[str, int]:
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in self.aspects:
            p = a.polarity.lower()
            if p in counts:
                counts[p] += 1
            else:
                counts["neutral"] += 1
        return counts

    @property
    def overall_polarity(self) -> str:
        """Derive overall sentiment from aspect distribution."""
        counts = self.aspect_breakdown
        pos = counts["positive"]
        neg = counts["negative"]
        neu = counts["neutral"]

        if pos > 0 and neg > 0:
            return "mixed"
        elif pos > 0 and neg == 0:
            return "positive"
        elif neg > 0 and pos == 0:
            return "negative"
        elif neu > 0:
            return "neutral"
        return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "overall_polarity": self.overall_polarity,
            "aspect_breakdown": self.aspect_breakdown,
            "aspect_count": len(self.aspects),
            "engine_source": self.engine_source,
            "aspects": [a.to_dict() for a in self.aspects]
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert detected aspects into a pandas DataFrame."""
        rows = []
        for a in self.aspects:
            rows.append({
                "Aspect": a.aspect,
                "Category": a.category,
                "Sentiment": a.polarity.capitalize(),
                "Confidence": f"{a.confidence * 100:.1f}%",
                "Sentiment Score": a.score,
                "Evidence": ", ".join(a.evidence) if a.evidence else "Model inference",
                "Context Scope": a.clause_context,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Produce a clean human-readable summary of detected aspects and sentiments."""
        if not self.aspects:
            return f"Sentence: \"{self.raw_text}\"\n  No explicit aspects detected."

        lines = [f"Input: \"{self.raw_text}\""]
        lines.append(f"Overall Tone: {self.overall_polarity.upper()} | Aspects Found: {len(self.aspects)}")
        for a in self.aspects:
            icon = "✅" if a.polarity == "positive" else ("❌" if a.polarity == "negative" else "⚪")
            lines.append(
                f"  {icon} {a.aspect.capitalize()} ({a.category}) ➔ {a.polarity.upper()} "
                f"[{a.confidence * 100:.1f}% conf] | Evidence: {', '.join(a.evidence) if a.evidence else 'N/A'}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"<ABSAResult: {len(self.aspects)} aspects ({self.overall_polarity}) in '{self.raw_text[:30]}...'>"


class HinglishABSA:
    """
    Main Hinglish Aspect-Based Sentiment Analysis pipeline.
    Combines rule-based extraction, Hinglish slang & phonetic lexicons,
    ABSA-MIX trained ML classifier, and optional plug-and-play LLM APIs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_provider: str = "gemini"
    ):
        self.preprocessor = HinglishPreprocessor()
        self.extractor = HinglishAspectExtractor(preprocessor=self.preprocessor)
        self.classifier = HinglishSentimentClassifier(
            model_path=model_path,
            preprocessor=self.preprocessor
        )
        self.llm_enhancer = LLMEnhancer(api_key=api_key, provider=api_provider)

    def set_api_key(self, api_key: str, provider: str = "gemini"):
        """Configure or update external API key at runtime."""
        self.llm_enhancer = LLMEnhancer(api_key=api_key, provider=provider)

    def is_api_configured(self) -> bool:
        """Check if an external LLM API key is connected."""
        return self.llm_enhancer.is_configured()

    def analyze(self, text: str, use_api: bool = False) -> ABSAResult:
        """
        Analyze an input text (word, sentence, or paragraph) end-to-end:
        1. If use_api is True and an API key is configured, invokes LLM enhancer.
        2. Otherwise (default), executes the fast local hybrid offline NLP pipeline:
           - Normalizes text & phonetics.
           - Extracts aspect candidates & categories.
           - Classifies polarity per aspect using context & slang modeling.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return ABSAResult("", "", [])

        clean_text = text.strip()

        # Try API enhancer if requested and configured
        if use_api and self.llm_enhancer.is_configured():
            llm_res = self.llm_enhancer.analyze(clean_text)
            if llm_res and "aspects" in llm_res and isinstance(llm_res["aspects"], list):
                aspect_results = []
                for item in llm_res["aspects"]:
                    asp_name = item.get("aspect", "Overall")
                    cat_name = item.get("category", "General / Product Quality")
                    pol_name = item.get("sentiment", "neutral").lower()
                    conf_val = float(item.get("confidence", 0.90))
                    expl_val = item.get("explanation", "LLM inference")
                    aspect_results.append(
                        AspectSentimentResult(
                            aspect=asp_name,
                            category=cat_name,
                            polarity=pol_name,
                            confidence=conf_val,
                            score=2.0 if pol_name == "positive" else (-2.0 if pol_name == "negative" else 0.0),
                            evidence=[expl_val] if isinstance(expl_val, str) else expl_val,
                            clause_context=clean_text,
                            probabilities={
                                "positive": 0.9 if pol_name == "positive" else 0.05,
                                "negative": 0.9 if pol_name == "negative" else 0.05,
                                "neutral": 0.9 if pol_name == "neutral" else 0.05
                            }
                        )
                    )
                return ABSAResult(
                    raw_text=clean_text,
                    normalized_text=self.preprocessor.normalize_text(clean_text),
                    aspects=aspect_results,
                    engine_source=f"llm_{self.llm_enhancer.provider}"
                )

        # Local Offline Engine
        norm_text = self.preprocessor.normalize_text(clean_text)
        extracted = self.extractor.extract_aspects(clean_text)

        aspect_results = []
        for asp in extracted:
            sent_res = self.classifier.predict_sentiment(
                sentence=clean_text,
                aspect_term=asp.term,
                category=asp.category
            )
            aspect_results.append(sent_res)

        return ABSAResult(
            raw_text=clean_text,
            normalized_text=norm_text,
            aspects=aspect_results,
            engine_source="hybrid_local"
        )

    def analyze_batch(self, texts: List[str], use_api: bool = False) -> List[ABSAResult]:
        """Analyze a list of sentences or reviews in batch."""
        return [self.analyze(t, use_api=use_api) for t in texts]
