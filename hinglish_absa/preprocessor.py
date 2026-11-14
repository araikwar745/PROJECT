"""
Preprocessing and Normalization Pipeline for Hinglish Code-Mixed Text.
Handles informal expressions, phonetic spelling variations, vowel elongations,
Devanagari Hindi text, emojis, and clause segmentation for Aspect-Based Sentiment Analysis.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Any
from .lexicons import (
    EMOJI_SENTIMENT,
    CONTRASTIVE_CONJUNCTIONS,
    COORDINATING_CONJUNCTIONS,
    INTENSIFIERS,
    DIMINISHERS,
    NEGATION_WORDS,
)

# Common phonetic mapping table for Hinglish spelling variations
PHONETIC_NORMALIZATION_MAP = {
    # Positive / quality words
    r"\b(achha|acha|achhaa|achaa|achho|ache|achhe)\b": "accha",
    r"\b(achhi|achi|achhii)\b": "accha",
    r"\b(badhiya|badiya|bdhiya|badhia|badhya)\b": "badhiya",
    r"\b(mastt|masta|mst)\b": "mast",
    r"\b(jhakkas|jhakas|jhkas)\b": "jhakaas",
    r"\b(dhasu|dhaansu|dhansu)\b": "dhansu",
    r"\b(lallan top|lallantop)\b": "lallantop",
    r"\b(gajab|gazabb|gzb)\b": "gazab",
    r"\b(bawal|bawaal|bwl)\b": "bawaal",
    r"\b(kamal|kamaal)\b": "kamaal",
    r"\b(zabardast|jabardast|zbrdst)\b": "zabardast",
    r"\b(shaandar|shandar|shandar)\b": "shandar",
    r"\b(paisa\s*vasool|paisavasool|paisa-vasool|pv)\b": "paisa vasool",
    r"\b(dil\s*jeet\s*liya)\b": "dil jeet liya",

    # Negative words
    r"\b(bekaar|bkr|bekr)\b": "bekar",
    r"\b(bakwaas|bkwas|bakwas)\b": "bakwas",
    r"\b(gatiya|ghatiyaa|ghatiya)\b": "ghatiya",
    r"\b(kachara|kachra)\b": "kachra",
    r"\b(khraab|kharaab|kharb)\b": "kharab",
    r"\b(bogus|boguss)\b": "bogus",
    r"\b(khatara|khatra)\b": "khatara",
    r"\b(dimag\s*kharab|dimaag\s*kharab)\b": "dimag kharab",
    r"\b(third\s*class|3rd\s*class)\b": "third class",
    r"\b(heating\s*issue|heating\s*problem)\b": "heating issue",
    r"\b(battery\s*backup|battry\s*backup|battery\s*bkp)\b": "battery backup",

    # Negations & Modifiers
    r"\b(nhi|nahin|nae|ni|nhn|naa|na)\b": "nahi",
    r"\b(bhot|boht|bhout|bhaot|bhoooot|bhooot)\b": "bahut",
    r"\b(jyada|zaada|jada|zyadaa)\b": "zyada",
    r"\b(ekdam|ek\s*dam|ek\s*dum)\b": "ekdum",
    r"\b(thodi\s*si|thoda\s*sa)\b": "thoda",
    r"\b(thodi|thode|thoda)\b": "thoda",
    r"\b(lkn|per)\b": "lekin",
}

class HinglishPreprocessor:
    """Preprocessor for Hinglish and Hindi text."""

    def __init__(self):
        self.elongated_pattern = re.compile(r"([a-zA-Z])\1{2,}")
        self.normalization_rules = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in PHONETIC_NORMALIZATION_MAP.items()
        ]

    def remove_elongations(self, text: str) -> str:
        """
        Compress elongated characters:
        e.g., 'acchaaaa' -> 'accha', 'masttttt' -> 'mast', 'sooooo' -> 'so'
        Preserves natural double letters like 'good' or 'screen' (replaces 3+ occurrences with 1).
        """
        return self.elongated_pattern.sub(r"\1", text)

    def extract_emojis(self, text: str) -> List[Dict[str, Any]]:
        """Extract emoji symbols and their sentiment attributes from text."""
        detected = []
        for char in text:
            if char in EMOJI_SENTIMENT:
                detected.append({
                    "emoji": char,
                    "sentiment": EMOJI_SENTIMENT[char]["polarity"],
                    "score": EMOJI_SENTIMENT[char]["score"],
                    "name": EMOJI_SENTIMENT[char]["name"],
                })
        return detected

    def normalize_text(self, text: str) -> str:
        """
        Clean, normalize phonetics, and standardize Hinglish and Hindi text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Replace elongated characters (e.g. 'masttttt' -> 'mast')
        text = self.remove_elongations(text)

        # Apply phonetic mappings for Roman text
        for pattern, replacement in self.normalization_rules:
            text = pattern.sub(replacement, text)

        # Clean spaces before and after standard punctuation including Devanagari danda '।'
        text = re.sub(r"\s+([?.!,;:।])", r"\1", text)
        text = re.sub(r"([?.!,;:।])\s*", r"\1 ", text)

        # Clean multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def segment_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Segment a code-mixed or Hindi sentence into grammatical clauses
        based on contrastive conjunctions, punctuation, and Devanagari danda '।'.
        """
        normalized = self.normalize_text(text)

        # Pattern to split on contrastive conjunctions, commas, semicolons, full stops, and Devanagari danda
        contrast_words = "|".join([re.escape(w) for w in sorted(CONTRASTIVE_CONJUNCTIONS, key=len, reverse=True)])
        split_pattern = rf"(,\s*|\.\s+|\s*;\s*|।\s*|\s+(?:{contrast_words})\s+)"

        parts = re.split(split_pattern, normalized, flags=re.IGNORECASE)

        clauses = []
        current_conjunction = None

        for part in parts:
            part_str = part.strip()
            if not part_str:
                continue

            clean_part_lower = part_str.strip(",.;। ").lower()
            if clean_part_lower in CONTRASTIVE_CONJUNCTIONS or part_str in [",", ";", ".", "।"]:
                current_conjunction = clean_part_lower
                continue

            clauses.append({
                "clause_text": part_str,
                "leading_conjunction": current_conjunction,
                "is_contrastive": current_conjunction in CONTRASTIVE_CONJUNCTIONS,
                "emojis": self.extract_emojis(part_str),
            })
            current_conjunction = None

        if not clauses and normalized:
            clauses.append({
                "clause_text": normalized,
                "leading_conjunction": None,
                "is_contrastive": False,
                "emojis": self.extract_emojis(normalized),
            })

        return clauses
