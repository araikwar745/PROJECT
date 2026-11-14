"""
Unified Pipeline for Hinglish Aspect-Based Sentiment Analysis (ABSA).
Integrates preprocessing, aspect extraction, sentiment classification,
and explanation generation.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from .preprocessor import HinglishPreprocessor
from .aspect_extractor import HinglishAspectExtractor, ExtractedAspect
from .sentiment_classifier import HinglishSentimentClassifier, AspectSentimentResult

class ABSAResult:
    """Encapsulates the end-to-end ABSA analysis result for a sentence."""

    def __init__(
        self,
        raw_text: str,
        normalized_text: str,
        aspects: List[AspectSentimentResult]
    ):
        self.raw_text = raw_text
        self.normalized_text = normalized_text
        self.aspects = aspects

    @property
    def has_aspects(self) -> bool:
        return len(self.aspects) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "aspect_count": len(self.aspects),
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
        for a in self.aspects:
            icon = "✅" if a.polarity == "positive" else ("❌" if a.polarity == "negative" else "⚪")
            lines.append(
                f"  {icon} {a.aspect.capitalize()} ({a.category}) ➔ {a.polarity.upper()} "
                f"[{a.confidence * 100:.1f}% conf] | Evidence: {', '.join(a.evidence) if a.evidence else 'N/A'}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"<ABSAResult: {len(self.aspects)} aspects in '{self.raw_text[:30]}...'>"


class HinglishABSA:
    """
    Main Hinglish Aspect-Based Sentiment Analysis pipeline.
    """

    def __init__(
        self,
        model_path: Optional[str] = None
    ):
        self.preprocessor = HinglishPreprocessor()
        self.extractor = HinglishAspectExtractor(preprocessor=self.preprocessor)
        self.classifier = HinglishSentimentClassifier(
            model_path=model_path,
            preprocessor=self.preprocessor
        )

    def analyze(self, text: str) -> ABSAResult:
        """
        Analyze a Hinglish sentence end-to-end:
        1. Preprocess & normalize slang/phonetics.
        2. Extract domain aspects & noun chunks.
        3. Classify sentiment per aspect using hybrid context models.
        """
        if not text or not isinstance(text, str):
            return ABSAResult("", "", [])

        norm_text = self.preprocessor.normalize_text(text)
        extracted = self.extractor.extract_aspects(text)

        aspect_results = []
        for asp in extracted:
            sent_res = self.classifier.predict_sentiment(
                sentence=text,
                aspect_term=asp.term,
                category=asp.category
            )
            aspect_results.append(sent_res)

        return ABSAResult(
            raw_text=text,
            normalized_text=norm_text,
            aspects=aspect_results
        )

    def analyze_batch(self, texts: List[str]) -> List[ABSAResult]:
        """Analyze a list of sentences in batch."""
        return [self.analyze(t) for t in texts]
