"""
Lexicons and Knowledge Bases for Hinglish Aspect-Based Sentiment Analysis.
Contains curated dictionaries for Hinglish slang, Hindi (Devanagari & Roman),
English sentiment words, negations, intensifiers, diminishers, and aspect taxonomies.
"""

# Positive sentiment words & slang (Roman Hinglish, Devanagari Hindi, and English)
POSITIVE_LEXICON = {
    # Hinglish & Hindi Slang (High intensity)
    "op": 3.0,
    "overpowered": 2.5,
    "paisa vasool": 3.0,
    "paisavasool": 3.0,
    "mast": 2.5,
    "masta": 2.2,
    "jhakaas": 3.0,
    "jhakas": 2.8,
    "dhansu": 2.8,
    "dhasu": 2.8,
    "lallantop": 3.0,
    "gazab": 2.8,
    "gajab": 2.5,
    "bawaal": 2.8,
    "bawal": 2.5,
    "kamaal": 2.8,
    "kamal": 2.5,
    "kadak": 2.5,
    "faadu": 2.8,
    "fadu": 2.5,
    "zabardast": 2.8,
    "shandar": 2.5,
    "shaandar": 2.8,
    "chha gaya": 2.8,
    "dil jeet liya": 3.0,
    "killer": 2.5,
    "fire": 2.5,
    "lit": 2.5,
    "bomb": 2.2,
    "topp": 2.5,
    "solid": 2.2,
    "bindaas": 2.2,
    "swad": 2.0,
    "lajawab": 2.8,
    "behtareen": 2.8,
    "tagda": 2.5,
    "chamatkar": 2.5,
    "badiya": 2.0,
    "badhiya": 2.2,
    "bdhiya": 2.0,
    "accha": 2.0,
    "achha": 2.0,
    "acha": 1.8,
    "achhi": 2.0,
    "achhe": 2.0,
    "sahi": 1.8,
    "pasand": 2.0,
    "pyara": 2.0,
    "pyaar": 2.0,
    "khush": 2.0,
    "maza": 2.2,
    "anand": 2.0,
    "tarif": 2.0,

    # English standard positive words
    "good": 1.8,
    "great": 2.2,
    "awesome": 2.8,
    "amazing": 2.8,
    "excellent": 2.8,
    "superb": 2.8,
    "fantastic": 2.8,
    "outstanding": 3.0,
    "perfect": 3.0,
    "love": 2.5,
    "loved": 2.5,
    "lovely": 2.2,
    "best": 2.8,
    "smooth": 2.0,
    "crisp": 2.0,
    "clear": 1.8,
    "fast": 1.8,
    "topnotch": 2.8,
    "top notch": 2.8,
    "reliable": 2.0,
    "durable": 2.0,
    "flawless": 2.8,
    "worth": 2.0,
    "worthy": 2.0,
    "stunning": 2.5,
    "fresh": 2.0,
    "clean": 1.8,
    "tasty": 2.2,
    "delicious": 2.5,
    "decent": 1.5,
    "nice": 1.8,
    "satisfying": 2.0,
    "satisfied": 2.0,
    "impressive": 2.2,
    "beautiful": 2.2,
    "brilliant": 2.5,
    "cool": 1.8,
    "super": 2.2,
    "wonderful": 2.5,

    # Devanagari Hindi positive words & slang
    "अच्छा": 2.0,
    "अच्छी": 2.0,
    "अच्छे": 2.0,
    "बढ़िया": 2.2,
    "शानदार": 2.8,
    "स्वादिष्ट": 2.5,
    "पसंद": 2.0,
    "ताज़ा": 2.0,
    "ताजा": 2.0,
    "बेहतरीन": 2.8,
    "लाजवाब": 2.8,
    "सुंदर": 2.0,
    "संतुष्ट": 2.0,
    "प्यार": 2.0,
    "मजा": 2.2,
    "मस्त": 2.5,
    "ज़बरदस्त": 2.8,
    "जबर्दस्त": 2.8,
    "कमाल": 2.8,
    "गज़ब": 2.8,
    "गजब": 2.5,
    "धमाकेदार": 2.8,
    "उम्दा": 2.5,
    "बवाल": 2.8,
    "पैसा वसूल": 3.0,
    "एकदम सही": 2.5,
}

# Negative sentiment words & slang
NEGATIVE_LEXICON = {
    # Hinglish & Hindi Slang (High intensity negative)
    "bekar": -2.5,
    "bekaar": -2.8,
    "bkr": -2.2,
    "bakwas": -2.8,
    "bakwaas": -3.0,
    "bkwas": -2.5,
    "ghatiya": -3.0,
    "gatiya": -2.8,
    "chindi": -2.2,
    "kachra": -2.8,
    "thanda": -1.8,
    "thandi": -1.8,
    "tatti": -3.0,
    "bogus": -2.5,
    "khatara": -2.8,
    "farzi": -2.5,
    "faltu": -2.5,
    "loot": -2.5,
    "loot macha rakhi": -3.0,
    "dhokha": -2.8,
    "kharab": -2.5,
    "khraab": -2.5,
    "raddi": -2.8,
    "fuddu": -2.8,
    "third class": -2.8,
    "third-class": -2.8,
    "dimag kharab": -2.8,
    "hang": -2.2,
    "lag": -2.0,
    "laggy": -2.2,
    "garam": -1.8,
    "heating": -2.0,
    "drain": -2.2,
    "draining": -2.2,
    "ganda": -2.5,
    "gandi": -2.5,
    "chutiya": -3.0,
    "niras": -2.0,
    "pareshan": -2.2,
    "sardard": -2.5,
    "sar dard": -2.5,
    "dhila": -1.8,
    "dheela": -1.8,
    "loss": -2.0,
    "ghata": -2.0,
    "fayda nahi": -2.2,
    "fayda nhi": -2.2,
    "boring": -2.0,
    "bore": -1.8,

    # English standard negative words
    "bad": -2.0,
    "terrible": -2.8,
    "horrible": -3.0,
    "worst": -3.0,
    "poor": -2.2,
    "waste": -2.5,
    "useless": -2.5,
    "slow": -1.8,
    "freeze": -2.2,
    "crashes": -2.5,
    "crash": -2.5,
    "buggy": -2.2,
    "bug": -1.8,
    "issue": -1.8,
    "issues": -1.8,
    "problem": -1.8,
    "problems": -1.8,
    "disappointing": -2.5,
    "disappointed": -2.5,
    "disappointment": -2.5,
    "heat": -1.8,
    "heavy": -1.5,
    "ugly": -2.0,
    "cheap": -1.8,
    "rough": -1.8,
    "noisy": -1.8,
    "defective": -2.8,
    "damaged": -2.5,
    "fail": -2.5,
    "failed": -2.5,
    "pathetic": -2.8,
    "disgusting": -3.0,
    "annoying": -2.2,
    "annoyed": -2.2,
    "rude": -2.5,
    "disinterested": -2.0,
    "awful": -2.8,
    "uncomfortable": -2.0,
    "confusing": -1.8,

    # Devanagari Hindi negative words & slang
    "खराब": -2.5,
    "बेकार": -2.8,
    "बकवास": -3.0,
    "घटिया": -3.0,
    "समस्या": -1.8,
    "दिक्कत": -1.8,
    "परेशान": -2.2,
    "धीमा": -1.8,
    "बुरा": -2.0,
    "बुरी": -2.0,
    "बुरे": -2.0,
    "कचरा": -2.8,
    "रद्दी": -2.8,
    "बोरिंग": -2.0,
    "फालतू": -2.5,
    "कमी": -1.8,
    "नुकसान": -2.0,
    "गंदा": -2.5,
    "गंदी": -2.5,
    "सिरदर्द": -2.5,
}

# Neutral / Average words
NEUTRAL_LEXICON = {
    "ok": 0.0,
    "okay": 0.0,
    "average": 0.0,
    "theek": 0.0,
    "thik": 0.0,
    "theek thaak": 0.0,
    "thik thak": 0.0,
    "normal": 0.0,
    "so so": 0.0,
    "soso": 0.0,
    "chalega": 0.0,
    "guzara": 0.0,
    "guzare layak": 0.0,
    "standard": 0.0,
    "medium": 0.0,
    "fine": 0.2,
    "moderate": 0.0,
    "fair": 0.1,
    "thek": 0.0,
    "theek-thaak": 0.0,
    "thik-thak": 0.0,
    "ठीक": 0.0,
    "सामान्य": 0.0,
    "औसत": 0.0,
    "ठीक-ठाक": 0.0,
    "ठीक ठाक": 0.0,
}

# Negation words (Roman, English, and Devanagari)
NEGATION_WORDS = {
    "nahi", "nhi", "nahin", "nae", "ni", "nhn", "na", "mat",
    "not", "never", "no", "dont", "don't", "didnt", "didn't",
    "wont", "won't", "cant", "can't", "couldnt", "couldn't",
    "without", "bina", "kabhi nahi", "kabhi nhi",
    "नहीं", "न", "ना", "बिना", "मत", "कदापि नहीं", "बिल्कुल नहीं"
}

# Intensifiers (multiply sentiment score > 1.0)
INTENSIFIERS = {
    "bahut": 1.6,
    "bhot": 1.6,
    "boht": 1.6,
    "bhout": 1.6,
    "bhaot": 1.6,
    "zyada": 1.4,
    "jyada": 1.4,
    "zaada": 1.4,
    "ekdum": 1.8,
    "ekdam": 1.8,
    "bilkul": 1.7,
    "kaafi": 1.4,
    "kafi": 1.4,
    "full": 1.5,
    "pura": 1.4,
    "poora": 1.4,
    "too": 1.5,
    "very": 1.6,
    "extremely": 1.8,
    "super": 1.6,
    "way": 1.3,
    "so": 1.4,
    "really": 1.5,
    "truly": 1.5,
    "totally": 1.6,
    "completely": 1.6,
    "ultra": 1.6,
    "pro": 1.4,
    "बहुत": 1.6,
    "एकदम": 1.8,
    "बिल्कुल": 1.7,
    "काफी": 1.4,
    "काफ़ी": 1.4,
    "अत्यधिक": 1.8,
    "ज्यादा": 1.4,
    "ज़्यादा": 1.4,
    "बेहद": 1.7,
}

# Diminishers (multiply sentiment score < 1.0)
DIMINISHERS = {
    "thoda": 0.6,
    "thodi": 0.6,
    "thode": 0.6,
    "thoda sa": 0.5,
    "thodi si": 0.5,
    "halka": 0.6,
    "halki": 0.6,
    "halke": 0.6,
    "slight": 0.6,
    "slightly": 0.6,
    "little": 0.6,
    "bit": 0.6,
    "a bit": 0.6,
    "somewhat": 0.7,
    "kind of": 0.7,
    "kinda": 0.7,
    "थोड़ा": 0.6,
    "थोड़ी": 0.6,
    "थोड़े": 0.6,
    "हल्का": 0.6,
    "हल्की": 0.6,
    "कम": 0.7,
    "जरा": 0.6,
    "ज़रा": 0.6,
}

# Contrastive conjunctions (indicate sentiment shift / clause boundary)
CONTRASTIVE_CONJUNCTIONS = {
    "but", "lekin", "lkn", "par", "magar", "halanki", "kintu", "parantu",
    "though", "although", "however", "while", "yet", "whereas", "nevertheless",
    "still", "per",
    "लेकिन", "पर", "मगर", "किंतु", "परंतु", "हालांकि", "हालाँकि", "जबकि"
}

# Coordinating conjunctions
COORDINATING_CONJUNCTIONS = {
    "and", "aur", "or", "plus", "sath me", "saath me", "also", "bhi",
    "तथा", "एवं", "और", "भी", "व"
}

# Common Emoji sentiment lexicon
EMOJI_SENTIMENT = {
    "🔥": {"polarity": "positive", "score": 2.5, "name": "fire"},
    "😍": {"polarity": "positive", "score": 2.8, "name": "heart_eyes"},
    "🥰": {"polarity": "positive", "score": 2.8, "name": "smiling_hearts"},
    "❤️": {"polarity": "positive", "score": 2.5, "name": "heart"},
    "💯": {"polarity": "positive", "score": 2.8, "name": "hundred"},
    "👍": {"polarity": "positive", "score": 2.0, "name": "thumbsup"},
    "👏": {"polarity": "positive", "score": 2.2, "name": "clapping"},
    "👌": {"polarity": "positive", "score": 2.2, "name": "ok_hand"},
    "🙌": {"polarity": "positive", "score": 2.2, "name": "raised_hands"},
    "🤩": {"polarity": "positive", "score": 2.8, "name": "star_struck"},
    "😎": {"polarity": "positive", "score": 2.0, "name": "sunglasses"},
    "✨": {"polarity": "positive", "score": 2.0, "name": "sparkles"},
    "🎉": {"polarity": "positive", "score": 2.2, "name": "party"},
    "🚀": {"polarity": "positive", "score": 2.5, "name": "rocket"},

    "💩": {"polarity": "negative", "score": -2.8, "name": "poop"},
    "🤮": {"polarity": "negative", "score": -3.0, "name": "vomit"},
    "🤢": {"polarity": "negative", "score": -2.5, "name": "nauseated"},
    "😡": {"polarity": "negative", "score": -2.8, "name": "angry"},
    "🤬": {"polarity": "negative", "score": -3.0, "name": "cursing"},
    "👎": {"polarity": "negative", "score": -2.2, "name": "thumbsdown"},
    "💔": {"polarity": "negative", "score": -2.5, "name": "broken_heart"},
    "😭": {"polarity": "negative", "score": -2.0, "name": "loud_cry"},
    "😢": {"polarity": "negative", "score": -1.8, "name": "sad_cry"},
    "😤": {"polarity": "negative", "score": -2.0, "name": "frustrated"},
    "🤦": {"polarity": "negative", "score": -2.0, "name": "facepalm"},
    "🗑️": {"polarity": "negative", "score": -2.5, "name": "wastebasket"},

    "😐": {"polarity": "neutral", "score": 0.0, "name": "neutral_face"},
    "🤔": {"polarity": "neutral", "score": 0.0, "name": "thinking"},
    "🤷": {"polarity": "neutral", "score": 0.0, "name": "shrug"},
    "😶": {"polarity": "neutral", "score": 0.0, "name": "silent"},
}

# Domain-specific aspect taxonomy and synonyms (English, Hinglish, and Devanagari Hindi)
ASPECT_CATEGORIES = {
    # Mobiles / Laptops / Tech Gadgets
    "Camera": [
        "camera", "cam", "selfie", "selfie camera", "rear camera", "front camera",
        "portrait mode", "night mode", "sensor", "lens", "zoom", "photo quality",
        "video quality", "photos", "picture quality", "pics", "macro",
        "कैमरा", "कैमरे", "फोटो", "पिक्चर क्वालिटी"
    ],
    "Battery": [
        "battery", "battery backup", "backup", "charging", "charger", "fast charging",
        "battery life", "battery drain", "battery time", "power", "cell",
        "बैटरी बैकअप", "बैटरी", "चार्जिंग", "चार्जर"
    ],
    "Performance": [
        "performance", "processor", "speed", "fast", "multitasking", "lag", "hanging",
        "smoothness", "ram", "cpu", "gpu", "chipset", "snapdragon", "mediatek", "apple silicon",
        "गेमिंग", "स्पीड", "परफॉरमेंस", "प्रोसेसर", "रैम"
    ],
    "Display": [
        "display", "screen", "screen quality", "amoled", "oled", "ips", "refresh rate",
        "120hz", "90hz", "brightness", "resolution", "touch", "touchscreen", "screen guard",
        "स्क्रीन क्वालिटी", "स्क्रीन", "डिस्प्ले"
    ],
    "Heating / Thermal": [
        "heating", "heating issue", "heat", "overheating", "temperature", "garam", "cooling",
        "fan", "thermal", "हीटिंग की समस्या", "हीटिंग समस्या", "हीटिंग", "गर्म"
    ],
    "Build & Design": [
        "build quality", "build", "design", "look", "looks", "body", "material", "finish",
        "weight", "heavy", "lightweight", "color", "colour", "frame", "hinge", "hinges",
        "बिल्ड क्वालिटी", "प्रीमियम लुक", "डिज़ाइन", "वजन"
    ],
    "Sound & Audio": [
        "sound", "audio", "speaker", "speakers", "volume", "sound quality", "mic",
        "microphone", "bass", "earphone", "headphone jack", "dolby",
        "साउंड क्वालिटी", "साउंड", "स्पीकर", "आवाज"
    ],
    "Storage & Memory": [
        "storage", "rom", "internal storage", "space", "memory card", "ram", "एसएसडी", "रैम", "मेमोरी"
    ],
    "Price & Value": [
        "price", "rate", "cost", "paisa", "value for money", "pricing", "budget", "paisa vasool",
        "पैसा वसूल", "मूल्य", "कीमत", "दाम", "पैसा"
    ],
    "Hardware & Connectivity": [
        "keyboard", "keypad", "trackpad", "touchpad", "mouse", "wifi", "bluetooth", "network",
        "5g", "sim", "ports", "usb", "charger cord", "cord", "कीबोर्ड", "माउस"
    ],

    # Movies / Entertainment
    "Story & Plot": [
        "story", "plot", "storyline", "script", "concept", "climax", "twist", "kahani",
        "स्टोरी", "कहानी", "कथा", "प्लॉट", "क्लाइमेक्स", "स्क्रिप्ट"
    ],
    "Acting & Cast": [
        "acting", "performance", "actors", "cast", "hero", "heroine", "villain", "lead",
        "acting skills", "character", "dialogue", "dialogues",
        "एक्टिंग", "अभिनय", "कलाकार", "हीरो", "अभिनेता", "डायलॉग"
    ],
    "Direction & Screenplay": [
        "direction", "director", "screenplay", "pacing", "cinematography", "visuals", "vfx",
        "निर्देशन", "निर्देशक", "सिनेमेटोग्राफी"
    ],
    "Music & Songs": [
        "music", "song", "songs", "soundtrack", "bgm", "background score", "singing",
        "म्यूजिक", "गीत", "संगीत", "गाने", "गाना"
    ],

    # Food & Restaurants
    "Food Quality & Taste": [
        "food", "taste", "flavor", "flavour", "dishes", "dish", "meal", "khana", "lunch",
        "dinner", "breakfast", "curry", "paneer", "biryani", "roti", "pizza", "burger",
        "coffee", "tea", "chai", "drinks", "beverage", "dessert", "snack", "sweets",
        "स्वादिष्ट", "खाना", "भोजन", "स्वाद", "कॉफ़ी", "चाय"
    ],
    "Service & Staff": [
        "service", "staff", "waiter", "manager", "hospitality", "behavior", "delivery",
        "customer service", "delivery boy", "speed of service", "service center", "support",
        "सर्विस", "स्टाफ", "वेटर", "सेवा", "डिलीवरी"
    ],
    "Ambience & Seating": [
        "ambience", "ambiance", "atmosphere", "vibe", "decor", "interior", "seating",
        "cleanliness", "hygiene", "place", "location", "space", "bar", "environment",
        "माहौल", "सजावट", "सिटिंग", "बैठने की जगह"
    ],

    # General Product / Overall
    "General / Product Quality": [
        "product", "phone", "mobile", "laptop", "movie", "film", "item", "quality",
        "overall", "experience", "purchase", "deal", "brand",
        "उत्पाद", "फोन", "मोबाइल", "लैपटॉप", "मूवी", "फिल्म"
    ]
}
