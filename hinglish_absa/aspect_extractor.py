"""
Aspect Extractor for Hinglish and Hindi (Code-Mixed and Pure Hindi) Text.
Combines rule-based syntactic patterns, domain aspect ontologies,
and candidate extraction with category mapping.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .lexicons import (
    ASPECT_CATEGORIES,
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON,
    NEUTRAL_LEXICON,
    NEGATION_WORDS,
    INTENSIFIERS,
    DIMINISHERS,
    CONTRASTIVE_CONJUNCTIONS,
)
from .preprocessor import HinglishPreprocessor

# Hinglish and Hindi grammatical stop tokens and particles
HINGLISH_GRAMMAR_WORDS = {
    # Roman Hinglish particles
    "ka", "ki", "ke", "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "hoge",
    "bhi", "toh", "to", "ko", "se", "me", "mein", "par", "pe", "aur", "and",
    "lekin", "but", "magar", "bhai", "yaar", "sirf", "ye", "yeh", "wo", "woh",
    "ek", "kuch", "sab", "apna", "apni", "apne", "mera", "meri", "mere", "is",
    "ispe", "iska", "iski", "iske", "us", "uska", "uski", "uske", "kya", "kyu",
    "kyun", "kaha", "kab", "kaise", "hota", "hoti", "hote", "raha", "rahi", "rahe",
    "gaya", "gayi", "gaye", "karo", "karna", "kiya", "liye", "batao", "dekh", "dekho",
    "nahi", "nhi", "not", "so", "very", "too", "thoda", "thodi", "bahut", "bhot",
    "ekdum", "bilkul", "mast", "bekar", "awesome", "good", "bad", "great",

    # Devanagari Hindi particles
    "का", "की", "के", "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे",
    "भी", "तो", "को", "से", "में", "पर", "पे", "और", "तथा", "एवं", "लेकिन", "मगर",
    "यह", "वह", "इस", "उस", "एक", "कुछ", "सब", "अपना", "अपनी", "अपने", "मेरा", "मेरी",
    "क्या", "कहा", "कहाँ", "कब", "कैसे", "होता", "होती", "होते", "रहा", "रही", "रहे",
    "गया", "गई", "गए", "करना", "किया", "लिए", "नहीं", "ना", "मत", "बहुत", "एकदम", "बिल्कुल",
    "वाला", "वाली", "वाले", "हुए", "हुई", "हुआ"
}

PARENT_CONTAINER_ENTITIES = {
    "phone", "mobile", "smartphone", "laptop", "movie", "film", "cinema",
    "car", "bike", "hotel", "restaurant", "joint", "cafe", "app", "website",
    "फोन", "मोबाइल", "स्मार्टफोन", "लैपटॉप", "मूवी", "फिल्म", "सिनेमा",
    "कार", "होटल", "रेस्टोरेंट", "रेस्तरां", "ऐप"
}

class ExtractedAspect:
    """Represents an extracted aspect mention."""

    def __init__(
        self,
        term: str,
        category: str,
        start_char: int,
        end_char: int,
        confidence: float = 1.0,
        source: str = "pattern",
    ):
        self.term = term
        self.category = category
        self.start_char = start_char
        self.end_char = end_char
        self.confidence = confidence
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.term,
            "category": self.category,
            "span": [self.start_char, self.end_char],
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }

    def __repr__(self):
        return f"<Aspect '{self.term}' ({self.category}) [{self.start_char}:{self.end_char}]>"


class HinglishAspectExtractor:
    """Extracts aspect entities from code-mixed Hinglish and Hindi text."""

    def __init__(self, preprocessor: Optional[HinglishPreprocessor] = None):
        self.preprocessor = preprocessor or HinglishPreprocessor()

        # Build reverse lookup map: synonym/phrase -> Category
        self.term_to_category: Dict[str, str] = {}
        for category, terms in ASPECT_CATEGORIES.items():
            for t in terms:
                clean_term = t.lower().strip()
                if clean_term == "performance":
                    self.term_to_category[clean_term] = "Performance"
                elif clean_term not in self.term_to_category:
                    self.term_to_category[clean_term] = category

        # Multi-word aspect terms sorted longest first for greedy matching
        self.sorted_terms = sorted(self.term_to_category.keys(), key=lambda x: len(x), reverse=True)

        # Unicode-safe word boundary pattern for matching both English and Devanagari Hindi
        escaped_terms = [re.escape(t) for t in self.sorted_terms]
        # Lookbehind and lookahead for whitespace, punctuation (including '।'), or string edges
        self.aspect_regex = re.compile(
            r"(?:(?<=[\s,।;!?'\"()\[\]])|^)(" + "|".join(escaped_terms) + r")(?:(?=[\s,।;!?'\"()\[\]])|$)",
            re.IGNORECASE
        )

        # Syntactic patterns in Hinglish and Hindi
        # Pattern 1: [Target] [ka/ki/ke/का/की/के] [Aspect]
        self.genitive_pattern = re.compile(
            r"([^\s,।;!?]+)\s+(ka|ki|ke|का|की|के)\s+([^\s,।;!?]+(?:\s+[^\s,।;!?]+)?)",
            re.IGNORECASE
        )

        # Pattern 2: [Aspect] + issue / problem / quality / backup / समस्या / दिक्कत
        self.compound_modifier_pattern = re.compile(
            r"([^\s,।;!?]+)\s+(issue|issues|problem|problems|backup|quality|performance|life|speed|समस्या|दिक्कत|क्वालिटी|बैकअप)",
            re.IGNORECASE
        )

    def resolve_category(self, term: str, context: str = "") -> str:
        """Map an aspect term to its canonical category, with context disambiguation."""
        term_lower = term.lower().strip()
        context_lower = context.lower()

        # Disambiguate 'performance' / 'परफॉरमेंस'
        if term_lower in ["performance", "परफॉरमेंस"]:
            if any(w in context_lower for w in ["movie", "film", "actor", "acting", "hero", "heroine", "cast", "मूवी", "फिल्म", "अभिनय"]):
                return "Acting & Cast"
            return "Performance"

        if term_lower in self.term_to_category:
            return self.term_to_category[term_lower]

        # Check subword matching
        for key, cat in self.term_to_category.items():
            if key in term_lower or term_lower in key:
                return cat

        return "General / Feature"

    def is_valid_aspect_candidate(self, candidate: str) -> bool:
        """Check if candidate word is a meaningful aspect and not a stopword/sentiment word."""
        cand_lower = candidate.lower().strip()
        if len(cand_lower) < 2:
            return False

        if cand_lower in HINGLISH_GRAMMAR_WORDS:
            return False

        # Exclude pure sentiment words from being treated as aspects
        if cand_lower in POSITIVE_LEXICON or cand_lower in NEGATIVE_LEXICON or cand_lower in NEUTRAL_LEXICON:
            return False

        if cand_lower in INTENSIFIERS or cand_lower in DIMINISHERS or cand_lower in NEGATION_WORDS:
            return False

        if cand_lower in CONTRASTIVE_CONJUNCTIONS:
            return False

        return True

    def extract_aspects(self, text: str) -> List[ExtractedAspect]:
        """
        Extract all aspect terms from input text with character spans,
        categories, and confidence scores. Supports both Roman Hinglish and Devanagari Hindi.
        """
        if not text:
            return []

        extracted_spans: List[Tuple[int, int, str, str, float, str]] = []

        # 1. Direct ontology matching (highest priority for curated domain terms)
        for match in self.aspect_regex.finditer(text):
            matched_text = match.group(1) if match.lastindex else match.group(0)
            term_clean = matched_text.strip()
            cat = self.resolve_category(term_clean, context=text)
            start, end = match.start(1) if match.lastindex else match.start(), match.end(1) if match.lastindex else match.end()
            extracted_spans.append((start, end, term_clean, cat, 0.95, "ontology"))

        # 2. Check compound modifier patterns (e.g. 'heating issue', 'battery backup', 'हीटिंग समस्या')
        for match in self.compound_modifier_pattern.finditer(text):
            full_compound = match.group(0).strip()
            first_word = match.group(1).strip().lower()
            if self.is_valid_aspect_candidate(first_word):
                cat = self.resolve_category(full_compound, context=text)
                extracted_spans.append((match.start(), match.end(), full_compound, cat, 0.93, "compound_pattern"))

        # 3. Check for syntactic genitive patterns (e.g. 'phone ka camera', 'फोन का कैमरा', 'movie ki story')
        for match in self.genitive_pattern.finditer(text):
            raw_aspect = match.group(3).strip()
            # Split if multiple words and check first word
            aspect_words = raw_aspect.split()
            valid_cand = ""
            for w in aspect_words:
                if self.is_valid_aspect_candidate(w):
                    valid_cand += (" " + w if valid_cand else w)
                else:
                    break

            if valid_cand:
                cat = self.resolve_category(valid_cand, context=text)
                sub_start = match.start(3)
                sub_end = sub_start + len(valid_cand)
                extracted_spans.append((sub_start, sub_end, valid_cand, cat, 0.90, "genitive_pattern"))

        # Deduplicate overlapping spans:
        # Prefer longer compound term ('battery backup' over 'battery', 'बैटरी बैकअप' over 'बैटरी')
        extracted_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        filtered_spans = []
        for span in extracted_spans:
            s_start, s_end, s_term, s_cat, s_conf, s_src = span
            overlap = False
            for idx, existing in enumerate(filtered_spans):
                e_start, e_end, e_term, e_cat, e_conf, e_src = existing
                if not (s_end <= e_start or s_start >= e_end):
                    overlap = True
                    if (s_end - s_start) > (e_end - e_start):
                        filtered_spans[idx] = span
                    break
            if not overlap:
                filtered_spans.append(span)

        # Hierarchical entity check:
        # If specific sub-aspects exist (e.g. 'camera', 'battery backup', 'कैमरा', 'बैटरी बैकअप'),
        # filter out the parent container entity (e.g. 'phone' in 'Phone ka camera...', or 'फोन' in 'फोन का कैमरा...')
        sub_aspects = [
            s for s in filtered_spans
            if s[2].lower() not in PARENT_CONTAINER_ENTITIES
        ]

        final_spans = filtered_spans
        if sub_aspects:
            final_spans = []
            for s in filtered_spans:
                term_lower = s[2].lower()
                if term_lower in PARENT_CONTAINER_ENTITIES:
                    post_text = text[s[1]:s[1]+10].lower().strip()
                    if post_text.startswith(("ka ", "ki ", "ke ", "का ", "की ", "के ")):
                        continue
                final_spans.append(s)

        # Fallback for implicit reviews
        if not final_spans:
            phone_matches = re.finditer(
                r"(?:(?<=[\s,।;!?'\"()\[\]])|^)(phone|mobile|laptop|movie|film|hotel|restaurant|फोन|मोबाइल|लैपटॉप|मूवी|फिल्म|होटल)(?:(?=[\s,।;!?'\"()\[\]])|$)",
                text,
                re.IGNORECASE
            )
            for pm in phone_matches:
                t = (pm.group(1) if pm.lastindex else pm.group(0)).strip()
                final_spans.append((pm.start(), pm.end(), t, self.resolve_category(t, context=text), 0.85, "fallback_entity"))

        result = [
            ExtractedAspect(
                term=span[2],
                category=span[3],
                start_char=span[0],
                end_char=span[1],
                confidence=span[4],
                source=span[5],
            )
            for span in final_spans
        ]

        result.sort(key=lambda x: x.start_char)
        return result
