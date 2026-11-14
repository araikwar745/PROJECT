"""
Aspect-Based Sentiment Classifier for Hinglish (Hindi-English Code-Mixed) text.
Combines clause-level syntactic scoping, Hinglish slang & emoji lexicons,
negation/intensifier modeling, and a machine learning classifier trained on ABSA-MIX.
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .lexicons import (
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON,
    NEUTRAL_LEXICON,
    NEGATION_WORDS,
    INTENSIFIERS,
    DIMINISHERS,
    CONTRASTIVE_CONJUNCTIONS,
    EMOJI_SENTIMENT,
)
from .preprocessor import HinglishPreprocessor

class AspectSentimentResult:
    """Detailed result of aspect sentiment classification."""

    def __init__(
        self,
        aspect: str,
        category: str,
        polarity: str,
        confidence: float,
        score: float,
        evidence: List[str],
        clause_context: str,
        probabilities: Dict[str, float]
    ):
        self.aspect = aspect
        self.category = category
        self.polarity = polarity
        self.confidence = confidence
        self.score = score
        self.evidence = evidence
        self.clause_context = clause_context
        self.probabilities = probabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect,
            "category": self.category,
            "sentiment": self.polarity,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 3),
            "evidence": self.evidence,
            "clause_context": self.clause_context,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()}
        }

    def __repr__(self):
        return f"<AspectSentiment '{self.aspect}': {self.polarity} (conf: {self.confidence:.2f})>"


class HinglishSentimentClassifier:
    """
    Hybrid Sentiment Classifier designed for Aspect-Based Sentiment Analysis
    in code-mixed Hindi-English text.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor: Optional[HinglishPreprocessor] = None
    ):
        self.preprocessor = preprocessor or HinglishPreprocessor()
        self.ml_model = None

        if model_path is None:
            curr_dir = os.path.dirname(__file__)
            default_path = os.path.join(curr_dir, "models", "sentiment_model.joblib")
            if os.path.exists(default_path):
                model_path = default_path

        if model_path and os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: Could not load ML model from {model_path}: {e}")
                self.ml_model = None

    def find_aspect_clause(self, sentence: str, aspect_term: str) -> str:
        """
        Identify the immediate grammatical clause in which the aspect resides,
        respecting sentence boundaries and contrastive conjunctions ('but', 'lekin', 'par', etc.).
        """
        # First isolate the specific sentence containing the aspect if multi-sentence / paragraph
        sentences = self.preprocessor.split_sentences(sentence)
        aspect_lower = aspect_term.lower().strip()
        target_sentence = sentence
        for s in sentences:
            if aspect_lower in s.lower():
                target_sentence = s
                break

        clauses = self.preprocessor.segment_clauses(target_sentence)

        for c in clauses:
            if aspect_lower in c["clause_text"].lower():
                return c["clause_text"]

        return self.preprocessor.normalize_text(target_sentence)

    def analyze_lexical_sentiment(
        self,
        clause_text: str,
        aspect_term: str
    ) -> Tuple[str, float, float, List[str]]:
        """
        Analyze sentiment using Hinglish slang, sentiment lexicons,
        negations, intensifiers, and diminishers.
        Returns (polarity, score, confidence, evidence_list).
        """
        norm_text = self.preprocessor.normalize_text(clause_text).lower()
        aspect_clean = self.preprocessor.normalize_text(aspect_term).lower()
        words = norm_text.split()

        evidence = []
        sentiment_score = 0.0

        # Check for explicit balanced neutral / indifferent expressions
        # e.g., "wasn't terrible, wasn't great either, just okay", "theek thaak"
        balanced_patterns = [
            r"\b(wasn'?t|not)\s+(terrible|bad|horrible|poor)\b.*\b(wasn'?t|not)\s+(great|good|awesome|amazing)\b",
            r"\b(wasn'?t|not)\s+(great|good|awesome|amazing)\b.*\b(wasn'?t|not)\s+(terrible|bad|horrible|poor)\b",
            r"\bneither\s+.+\s+nor\b",
            r"\b(just\s+okay|just\s+ok|just\s+fine|only\s+okay|only\s+ok)\b",
            r"\b(theek\s*thaak|thik\s*thak|thek\s*thak|theek-thaak|thik-thak)\b",
            r"\b(na\s+ac?chha|na\s+ac?cha)\b.*\b(na\s+bura|na\s+bekar)\b",
            r"\b(na\s+bura|na\s+bekar)\b.*\b(na\s+ac?chha|na\s+ac?cha)\b",
        ]
        for bp in balanced_patterns:
            if re.search(bp, norm_text, re.IGNORECASE):
                evidence.append("Balanced neutral tone ('neither good nor bad / just okay')")
                return "neutral", 0.0, 0.95, evidence

        # Check if the clause contains an inherent defect term like 'heating issue', 'problem', etc.
        has_active_negation = any(w in NEGATION_WORDS for w in words)
        
        # If the aspect itself contains defect words (English, Hinglish, Devanagari Hindi)
        is_defect_aspect = any(term in aspect_clean for term in [
            "issue", "problem", "defect", "flaw", "heating", "garam", "lag", "hang",
            "समस्या", "दिक्कत", "खराबी", "हीटिंग", "कमी"
        ])
        if is_defect_aspect:
            if has_active_negation:
                sentiment_score += 2.0
                evidence.append(f"Negated defect '{aspect_clean}' -> positive")
            else:
                sentiment_score -= 2.5
                evidence.append(f"Defect mention '{aspect_clean}' -> negative")

        # Scan for multi-word slang expressions (Roman and Devanagari)
        multi_word_slangs = {
            "paisa vasool": 3.0,
            "paisavasool": 3.0,
            "पैसा वसूल": 3.0,
            "dil jeet liya": 3.0,
            "दिल जीत लिया": 3.0,
            "chha gaya": 2.8,
            "छा गया": 2.8,
            "dimag kharab": -2.8,
            "दिमाग खराब": -2.8,
            "dimag ki dahi": -2.8,
            "vaat lag gayi": -3.0,
            "third class": -2.8,
            "थर्ड क्लास": -2.8,
            "loot macha rakhi": -3.0,
            "लूट मचा रखी": -3.0,
            "heating issue": -2.5,
            "हीटिंग की समस्या": -2.5,
            "हीटिंग समस्या": -2.5,
            "kuch khaas nahi": -1.8,
            "kuch khas nahi": -1.8,
            "kuch khas nhi": -1.8,
            "कुछ खास नहीं": -1.8,
        }

        has_matched_slang = False
        for slang, weight in multi_word_slangs.items():
            if slang in norm_text:
                sentiment_score += weight
                has_matched_slang = True
                if weight > 0:
                    evidence.append(f"Slang (+) '{slang}'")
                elif weight < 0:
                    evidence.append(f"Slang (-) '{slang}'")

        # Check emojis
        emojis = self.preprocessor.extract_emojis(clause_text)
        for emo in emojis:
            sentiment_score += emo["score"]
            evidence.append(f"Emoji {emo['emoji']} ({emo['sentiment']})")

        # Check individual tokens
        active_multiplier = 1.0
        negation_active = False

        has_positive_word = False
        has_negative_word = False
        has_neutral_word = False

        for i, word in enumerate(words):
            if word in NEGATION_WORDS:
                negation_active = True
                evidence.append(f"Negation '{word}'")
                continue

            if word in INTENSIFIERS:
                active_multiplier = INTENSIFIERS[word]
                evidence.append(f"Intensifier '{word}' ({active_multiplier}x)")
                continue

            if word in DIMINISHERS:
                active_multiplier = DIMINISHERS[word]
                evidence.append(f"Diminisher '{word}' ({active_multiplier}x)")
                continue

            # Skip aspect words
            if word in aspect_clean.split():
                continue

            # Check if followed by negation within 1-3 words (Hindi SOV syntax: 'accha nahi hai')
            post_negation = False
            for j in range(i + 1, min(len(words), i + 4)):
                if words[j] in NEGATION_WORDS:
                    post_negation = True
                    break

            is_negated = negation_active or post_negation

            w_score = 0.0
            if word in POSITIVE_LEXICON:
                has_positive_word = True
                w_score = POSITIVE_LEXICON[word] * active_multiplier
                if is_negated:
                    w_score = -w_score * 0.9
                    evidence.append(f"Negated positive '{word}' -> negative")
                    negation_active = False
                else:
                    evidence.append(f"Positive '{word}' (+{round(w_score, 1)})")
                active_multiplier = 1.0

            elif word in NEGATIVE_LEXICON:
                has_negative_word = True
                w_score = NEGATIVE_LEXICON[word] * active_multiplier
                if is_negated:
                    w_score = abs(w_score) * 0.5
                    evidence.append(f"Negated negative '{word}' -> neutral/positive")
                    negation_active = False
                else:
                    evidence.append(f"Negative '{word}' ({round(w_score, 1)})")
                active_multiplier = 1.0

            elif word in NEUTRAL_LEXICON:
                has_neutral_word = True
                evidence.append(f"Neutral '{word}'")

            sentiment_score += w_score

        # Factual / Non-opinion statement handling
        has_polar_cues = has_positive_word or has_negative_word or has_matched_slang or bool(emojis) or is_defect_aspect
        if not has_polar_cues:
            if has_neutral_word:
                return "neutral", 0.0, 0.90, evidence
            return "neutral", 0.0, 0.85, ["Factual statement with no sentiment polarity"]

        # Determine polarity & confidence
        if sentiment_score >= 0.8:
            polarity = "positive"
            conf = min(0.99, 0.75 + min(abs(sentiment_score) * 0.08, 0.23))
        elif sentiment_score <= -0.8:
            polarity = "negative"
            conf = min(0.99, 0.75 + min(abs(sentiment_score) * 0.08, 0.23))
        else:
            polarity = "neutral"
            conf = 0.75

        return polarity, sentiment_score, conf, evidence

    def predict_sentiment(
        self,
        sentence: str,
        aspect_term: str,
        category: str = "General"
    ) -> AspectSentimentResult:
        """Predict sentiment for an aspect term in sentence."""
        # Check if the containing sentence exhibits a balanced neutral evaluation
        sentences = self.preprocessor.split_sentences(sentence)
        aspect_lower = aspect_term.lower().strip()
        target_sentence = sentence
        for s in sentences:
            if aspect_lower in s.lower():
                target_sentence = s
                break

        norm_target_sent = self.preprocessor.normalize_text(target_sentence).lower()
        balanced_patterns = [
            r"\b(wasn'?t|not)\s+(terrible|bad|horrible|poor)\b.*\b(wasn'?t|not)\s+(great|good|awesome|amazing)\b",
            r"\b(wasn'?t|not)\s+(great|good|awesome|amazing)\b.*\b(wasn'?t|not)\s+(terrible|bad|horrible|poor)\b",
            r"\bneither\s+.+\s+nor\b",
            r"\b(just\s+okay|just\s+ok|just\s+fine|only\s+okay|only\s+ok)\b",
            r"\b(theek\s*thaak|thik\s*thak|thek\s*thak|theek-thaak|thik-thak)\b",
            r"\b(na\s+ac?chha|na\s+ac?cha)\b.*\b(na\s+bura|na\s+bekar)\b",
            r"\b(na\s+bura|na\s+bekar)\b.*\b(na\s+ac?chha|na\s+ac?cha)\b",
        ]
        if any(re.search(bp, norm_target_sent, re.IGNORECASE) for bp in balanced_patterns):
            return AspectSentimentResult(
                aspect=aspect_term,
                category=category,
                polarity="neutral",
                confidence=0.95,
                score=0.0,
                evidence=["Balanced neutral tone ('wasn't terrible, wasn't great, just okay')"],
                clause_context=target_sentence,
                probabilities={"positive": 0.05, "negative": 0.05, "neutral": 0.90}
            )

        clause = self.find_aspect_clause(sentence, aspect_term)

        lex_pol, lex_score, lex_conf, evidence = self.analyze_lexical_sentiment(clause, aspect_term)

        ml_probs = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        if self.ml_model:
            try:
                vec = self.ml_model["vectorizer"]
                clf = self.ml_model["classifier"]
                classes = self.ml_model["classes"]

                norm_sent = self.preprocessor.normalize_text(sentence).lower()
                norm_asp = self.preprocessor.normalize_text(aspect_term).lower()
                words = norm_sent.split()
                asp_words = norm_asp.split()

                start_idx = -1
                for i in range(len(words) - len(asp_words) + 1):
                    if words[i:i + len(asp_words)] == asp_words:
                        start_idx = i
                        break

                window_size = 6
                if start_idx != -1:
                    end_idx = start_idx + len(asp_words)
                    left = words[max(0, start_idx - window_size): start_idx]
                    right = words[end_idx: min(len(words), end_idx + window_size)]
                    w_text = " ".join(left + ["[TARGET]"] + asp_words + ["[TARGET]"] + right)
                else:
                    w_text = f"{norm_sent} [TARGET] {norm_asp}"

                combined_ctx = f"{norm_sent} [SEP] {norm_asp} [SEP] {w_text}"
                tfidf_feat = vec.transform([combined_ctx])

                pos_s = sum(POSITIVE_LEXICON.get(w, 0.0) for w in words)
                neg_s = sum(abs(NEGATIVE_LEXICON.get(w, 0.0)) for w in words)
                has_neg = sum(1.0 for w in words if w in NEGATION_WORDS)
                int_cnt = sum(1.0 for w in words if w in INTENSIFIERS)
                dim_cnt = sum(1.0 for w in words if w in DIMINISHERS)
                asp_l = float(len(asp_words))

                lex_dense = np.array([[pos_s, neg_s, pos_s - neg_s, has_neg, int_cnt, dim_cnt, asp_l]], dtype=np.float32)

                import scipy.sparse as sp
                feat_all = sp.hstack([tfidf_feat, lex_dense], format='csr')

                probs = clf.predict_proba(feat_all)[0]
                for cls_name, prob in zip(classes, probs):
                    ml_probs[cls_name] = float(prob)
            except Exception:
                pass

        # Hybrid decision fusion
        if lex_pol == "neutral":
            final_pol = "neutral"
            final_conf = lex_conf
            ml_probs = {"positive": 0.05, "negative": 0.05, "neutral": 0.90}
        elif abs(lex_score) >= 1.2:
            final_pol = lex_pol
            final_conf = max(lex_conf, ml_probs.get(lex_pol, 0.5))
        elif abs(lex_score) >= 0.5:
            ml_best = max(ml_probs, key=ml_probs.get)
            if ml_best == lex_pol:
                final_pol = lex_pol
                final_conf = min(0.98, (lex_conf + ml_probs[lex_pol]) / 2 + 0.1)
            else:
                final_pol = lex_pol if abs(lex_score) > 0.7 else ml_best
                final_conf = 0.70
        else:
            final_pol = max(ml_probs, key=ml_probs.get)
            final_conf = ml_probs[final_pol]

        return AspectSentimentResult(
            aspect=aspect_term,
            category=category,
            polarity=final_pol,
            confidence=final_conf,
            score=lex_score,
            evidence=evidence,
            clause_context=clause,
            probabilities=ml_probs
        )
