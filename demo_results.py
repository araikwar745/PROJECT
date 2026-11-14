import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from hinglish_absa import HinglishABSA

absa = HinglishABSA()

examples = [
    # English
    ('English Positive', 'I love using this product, it is amazing!'),
    ('English Neutral', 'The restaurant was not terrible, it was not great either, just okay.'),
    ('English Negative', 'I hate these updates, everything is slow and broken.'),
    # Hindi Devanagari
    ('Hindi Positive', 'यह फिल्म बहुत अच्छी है।'),
    ('Hindi Negative', 'मुझे यह खाना पसंद नहीं आया।'),
    ('Hindi Neutral', 'आज मौसम ठीक है।'),
    # Roman Hinglish
    ('Hinglish Positive', 'Yeh movie bahut achhi hai.'),
    ('Hinglish Negative', 'Mujhe yeh khana bilkul pasand nahi aaya.'),
    ('Hinglish Positive', 'Aaj mausam achha hai.'),
    ('Hinglish Neutral', 'Maine naya mobile kharida.'),
    # Multi-Aspect
    ('Hinglish Multi-Aspect', 'Phone ka camera bahut accha hai but battery backup bekar hai.'),
    # Slang
    ('Hinglish Slang', 'Bhai phone ekdum OP hai, paisa vasool!'),
    # Single Words
    ('Single Word (Negative)', 'Bakwas'),
    ('Single Word (Positive)', 'Mast'),
    ('Single Word (Hindi)', 'बकवास')
]

print(f"{'Category':<24} | {'Input Text':<50} | {'Detected Aspects & Sentiments'}")
print("-" * 115)

for cat, text in examples:
    res = absa.analyze(text)
    aspect_strs = []
    for a in res.aspects:
        icon = '[+]' if a.polarity == 'positive' else ('[-]' if a.polarity == 'negative' else '[=]')
        aspect_strs.append(f"{icon} {a.aspect}: {a.polarity.upper()} ({a.confidence*100:.0f}%)")
    joined = ', '.join(aspect_strs) if aspect_strs else 'None'
    print(f"{cat:<24} | {text:<50} | {joined}")
