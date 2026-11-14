"""
Training and Evaluation Pipeline for Hinglish Aspect Sentiment Classifier.
Trains an Aspect-Based Sentiment Classification model on the ABSA-MIX dataset
(combining restaurant and laptop domains) and serializes the model weights.
"""

import os
import ast
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

from hinglish_absa.lexicons import (
    POSITIVE_LEXICON,
    NEGATIVE_LEXICON,
    NEUTRAL_LEXICON,
    NEGATION_WORDS,
    INTENSIFIERS,
    DIMINISHERS,
    CONTRASTIVE_CONJUNCTIONS,
)
from hinglish_absa.preprocessor import HinglishPreprocessor

preprocessor = HinglishPreprocessor()

def parse_absa_mix_csv(filepath: str) -> List[Dict[str, str]]:
    """Parse ABSA-MIX format CSV into list of {sentence, aspect, polarity}."""
    if not os.path.exists(filepath):
        return []

    df = pd.read_csv(filepath)
    col_asp = 'aspect_terms' if 'aspect_terms' in df.columns else 'aspect_term'
    col_pol = 'at_polarity'

    records = []
    for _, row in df.iterrows():
        raw_text = str(row.get('sentence', ''))
        raw_asps = row.get(col_asp, '[]')
        raw_pols = row.get(col_pol, '[]')

        try:
            if isinstance(raw_asps, str) and raw_asps.startswith('['):
                asps = ast.literal_eval(raw_asps)
            elif isinstance(raw_asps, list):
                asps = raw_asps
            else:
                asps = []

            if isinstance(raw_pols, str) and raw_pols.startswith('['):
                pols = ast.literal_eval(raw_pols)
            elif isinstance(raw_pols, list):
                pols = raw_pols
            else:
                pols = []
        except Exception:
            continue

        for a, p in zip(asps, pols):
            a_clean = str(a).strip().strip("\"'").lower()
            p_clean = str(p).strip().strip("\"'").lower()
            if a_clean and p_clean and p_clean in ['positive', 'negative', 'neutral']:
                records.append({
                    'sentence': raw_text,
                    'aspect': a_clean,
                    'polarity': p_clean
                })
    return records


class AspectContextExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts text representations for the aspect-in-context:
    1. Full sentence normalized
    2. Aspect term
    3. Window of tokens around the aspect (within +/- 6 words)
    """
    def __init__(self, window_size: int = 6):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is list of tuples: (sentence, aspect)
        contexts = []
        for sent, asp in X:
            norm_sent = preprocessor.normalize_text(sent).lower()
            norm_asp = preprocessor.normalize_text(asp).lower()

            words = norm_sent.split()
            asp_words = norm_asp.split()

            # Find position of aspect in sentence words
            start_idx = -1
            for i in range(len(words) - len(asp_words) + 1):
                if words[i:i + len(asp_words)] == asp_words:
                    start_idx = i
                    break

            if start_idx != -1:
                end_idx = start_idx + len(asp_words)
                left = words[max(0, start_idx - self.window_size): start_idx]
                right = words[end_idx: min(len(words), end_idx + self.window_size)]
                window_tokens = left + ["[TARGET]"] + asp_words + ["[TARGET]"] + right
                window_text = " ".join(window_tokens)
            else:
                window_text = norm_sent + " [TARGET] " + norm_asp

            # Combined representation: sentence + target marker + window
            combined = f"{norm_sent} [SEP] {norm_asp} [SEP] {window_text}"
            contexts.append(combined)
        return contexts


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Computes dense numeric features using Hinglish sentiment lexicons,
    negations, and intensifiers within the aspect's local scope.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for sent, asp in X:
            norm_sent = preprocessor.normalize_text(sent).lower()
            norm_asp = preprocessor.normalize_text(asp).lower()
            words = norm_sent.split()

            pos_score = 0.0
            neg_score = 0.0
            has_negation = 0.0
            intensifier_count = 0.0
            diminisher_count = 0.0

            # Scan words
            for i, w in enumerate(words):
                # Check sentiment
                if w in POSITIVE_LEXICON:
                    pos_score += POSITIVE_LEXICON[w]
                elif w in NEGATIVE_LEXICON:
                    neg_score += abs(NEGATIVE_LEXICON[w])

                if w in NEGATION_WORDS:
                    has_negation += 1.0
                if w in INTENSIFIERS:
                    intensifier_count += 1.0
                if w in DIMINISHERS:
                    diminisher_count += 1.0

            # Aspect length and position
            asp_len = len(norm_asp.split())

            features.append([
                pos_score,
                neg_score,
                pos_score - neg_score,
                has_negation,
                intensifier_count,
                diminisher_count,
                float(asp_len)
            ])
        return np.array(features, dtype=np.float32)


def build_augmented_training_data(absa_mix_train: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Augment ABSA-MIX training data with authentic Roman Hinglish sentences
    covering common slang, movie/tech/food domains, and user-provided patterns.
    """
    augmented = list(absa_mix_train)

    roman_hinglish_examples = [
        # Positive tech reviews with slang
        ("Phone ka camera bahut accha hai but battery backup bekar hai.", "camera", "positive"),
        ("Phone ka camera bahut accha hai but battery backup bekar hai.", "battery backup", "negative"),
        ("Movie ki story mast thi but acting thodi boring thi.", "story", "positive"),
        ("Movie ki story mast thi but acting thodi boring thi.", "acting", "negative"),
        ("Laptop ka performance awesome hai, lekin heating issue bahut hai.", "performance", "positive"),
        ("Laptop ka performance awesome hai, lekin heating issue bahut hai.", "heating issue", "negative"),
        ("Bhai phone ekdum OP hai, paisa vasool!", "phone", "positive"),
        ("Bhai phone ekdum OP hai, paisa vasool!", "paisa vasool", "positive"),

        ("Display ekdum jhakaas hai colors bahut crisp aate hai.", "display", "positive"),
        ("Sound quality ekdum bakwas hai, speaker phat jata hai.", "sound quality", "negative"),
        ("Sound quality ekdum bakwas hai, speaker phat jata hai.", "speaker", "negative"),
        ("Food ka taste lajawab tha par service boht slow thi.", "taste", "positive"),
        ("Food ka taste lajawab tha par service boht slow thi.", "service", "negative"),
        ("Is laptop ka trackpad thoda loose hai but keyboard best hai.", "trackpad", "negative"),
        ("Is laptop ka trackpad thoda loose hai but keyboard best hai.", "keyboard", "positive"),
        ("Processor kaafi powerful hai gaming smooth chalti hai.", "processor", "positive"),
        ("Processor kaafi powerful hai gaming smooth chalti hai.", "gaming", "positive"),
        ("Camera quality acchi nahi hai, pictures blur aati hai.", "camera quality", "negative"),
        ("Battery backup normal hai, ek din chal jata hai.", "battery backup", "neutral"),
        ("Design theek thaak hai par build quality mazboot hai.", "design", "neutral"),
        ("Design theek thaak hai par build quality mazboot hai.", "build quality", "positive"),
        ("Customer service ekdum ghatiya thi, koi sunta hi nahi.", "customer service", "negative"),
        ("Screen 120hz refresh rate ke sath super smooth lagti hai.", "screen", "positive"),
        ("Heating issue bilkul nahi hai full load par bhi cool rehta hai.", "heating issue", "positive"),
        ("Price thoda high hai but overall product solid hai.", "price", "negative"),
        ("Price thoda high hai but overall product solid hai.", "product", "positive"),
        ("Climax ekdum dhansu tha maza aa gaya.", "climax", "positive"),
        ("Songs kaafi bekar the movie me.", "songs", "negative"),
        ("Ambience bohot shaandar hai romantic dinner ke liye best place.", "ambience", "positive"),
        ("Packaging kachra thi sab leaked mila.", "packaging", "negative"),
        ("Delivery time par hui aur staff polite tha.", "delivery", "positive"),
        ("Delivery time par hui aur staff polite tha.", "staff", "positive"),
    ]

    for sent, asp, pol in roman_hinglish_examples:
        augmented.append({
            'sentence': sent,
            'aspect': asp,
            'polarity': pol
        })

    return augmented


def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_dir = os.path.join(base_dir, "ref_absa_mix")

    print("[1/5] Loading ABSA-MIX Datasets...")
    rest_train = parse_absa_mix_csv(os.path.join(ref_dir, "Complete_restaurant_train.csv"))
    lap_train = parse_absa_mix_csv(os.path.join(ref_dir, "complete_laptop_train.csv"))
    rest_test = parse_absa_mix_csv(os.path.join(ref_dir, "Complete_restaurant_test.csv"))
    lap_test = parse_absa_mix_csv(os.path.join(ref_dir, "complete_laptop_test.csv"))

    raw_train = rest_train + lap_train
    raw_test = rest_test + lap_test

    print(f"Loaded {len(raw_train)} ABSA-MIX training pairs.")
    print(f"Loaded {len(raw_test)} ABSA-MIX test pairs.")

    print("[2/5] Augmenting with authentic Roman Hinglish corpus...")
    train_data = build_augmented_training_data(raw_train)
    test_data = raw_test

    print(f"Total training instances: {len(train_data)}")
    print(f"Total testing instances:  {len(test_data)}")

    X_train_pairs = [(d['sentence'], d['aspect']) for d in train_data]
    y_train = [d['polarity'] for d in train_data]

    X_test_pairs = [(d['sentence'], d['aspect']) for d in test_data]
    y_test = [d['polarity'] for d in test_data]

    print("[3/5] Extracting Contextual N-Gram and Lexicon Features...")
    context_extractor = AspectContextExtractor(window_size=6)
    train_contexts = context_extractor.transform(X_train_pairs)
    test_contexts = context_extractor.transform(X_test_pairs)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(train_contexts)
    X_test_tfidf = vectorizer.transform(test_contexts)

    lex_extractor = LexiconFeatureExtractor()
    X_train_lex = lex_extractor.transform(X_train_pairs)
    X_test_lex = lex_extractor.transform(X_test_pairs)

    import scipy.sparse as sp
    X_train_all = sp.hstack([X_train_tfidf, X_train_lex], format='csr')
    X_test_all = sp.hstack([X_test_tfidf, X_test_lex], format='csr')

    print("[4/5] Training Calibrated Multi-Class Classifier...")
    clf = LogisticRegression(
        C=2.5,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    clf.fit(X_train_all, y_train)

    print("[5/5] Evaluating on Benchmark Test Sets...")
    y_pred = clf.predict(X_test_all)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print("BENCHMARK EVALUATION RESULTS ON ABSA-MIX TEST SET")
    print("=" * 50)
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save artifacts
    models_dir = os.path.join(base_dir, "hinglish_absa", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_package = {
        'vectorizer': vectorizer,
        'classifier': clf,
        'window_size': 6,
        'classes': clf.classes_.tolist(),
        'metrics': {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'test_size': len(y_test)
        }
    }

    model_save_path = os.path.join(models_dir, "sentiment_model.joblib")
    joblib.dump(model_package, model_save_path)
    print(f"Saved trained model and artifacts to: {model_save_path}")

if __name__ == "__main__":
    train_and_evaluate()
