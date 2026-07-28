"""Multi-language query translation.

CCTV queries may arrive in Hindi or Gujarati (or any language). CLIP is trained
on English, so we translate the query to English before embedding it.

translate_query(text, source_lang='auto') -> (english_text, method)

Primary path: deep-translator (Google) for open-vocabulary translation.
Fallback: a small offline hi/gu -> en phrase dictionary, so a demo with no
internet still handles the common descriptive terms instead of hard-failing.
Translation is kept out of the search core (text_search) on purpose - the API
layer calls this, then passes the result as `translated_query`.
"""
from __future__ import annotations

# Common CCTV descriptive terms (colours, vehicles, people, carried items) in
# Hindi (Devanagari) and Gujarati -> English. Used only when online translation
# is unavailable. Token-level, so "<colour> <object>" phrases still work.
_OFFLINE = {
    # --- Hindi ---
    "सफेद": "white", "सफ़ेद": "white", "काला": "black", "काली": "black",
    "लाल": "red", "नीला": "blue", "नीली": "blue", "हरा": "green", "हरी": "green",
    "पीला": "yellow", "पीली": "yellow", "भूरा": "brown", "स्लेटी": "grey",
    "चांदी": "silver", "नारंगी": "orange",
    "ट्रक": "truck", "कार": "car", "गाड़ी": "vehicle", "वाहन": "vehicle",
    "बस": "bus", "बाइक": "motorcycle", "मोटरसाइकिल": "motorcycle",
    "मोटरसायकल": "motorcycle", "साइकिल": "bicycle", "स्कूटर": "scooter",
    "आदमी": "man", "व्यक्ति": "person", "औरत": "woman", "महिला": "woman",
    "लड़का": "boy", "लड़की": "girl", "बच्चा": "child", "भीड़": "crowd",
    "बैग": "bag", "बैकपैक": "backpack", "थैला": "bag", "छाता": "umbrella",
    "टोपी": "cap", "हेलमेट": "helmet", "चश्मा": "sunglasses", "मास्क": "mask",
    "पहन": "wearing", "पहने": "wearing", "वाला": "with", "वाली": "with",
    # --- Gujarati ---
    "સફેદ": "white", "કાળો": "black", "કાળી": "black", "લાલ": "red",
    "વાદળી": "blue", "લીલો": "green", "લીલી": "green", "પીળો": "yellow",
    "કથ્થઈ": "brown", "રાખોડી": "grey", "ચાંદી": "silver", "નારંગી": "orange",
    "ટ્રક": "truck", "કાર": "car", "ગાડી": "vehicle", "વાહન": "vehicle",
    "બસ": "bus", "બાઇક": "motorcycle", "મોટરસાઇકલ": "motorcycle",
    "સાયકલ": "bicycle", "સ્કૂટર": "scooter",
    "માણસ": "man", "વ્યક્તિ": "person", "સ્ત્રી": "woman", "મહિલા": "woman",
    "છોકરો": "boy", "છોકરી": "girl", "બાળક": "child", "ભીડ": "crowd",
    "બેગ": "bag", "થેલો": "bag", "છત્રી": "umbrella", "ટોપી": "cap",
    "હેલ્મેટ": "helmet", "ચશ્મા": "sunglasses", "માસ્ક": "mask",
    "પહેરેલ": "wearing", "વાળો": "with", "વાળી": "with",
}


def _looks_english(text: str) -> bool:
    """True if the text is essentially ASCII (already English / romanised)."""
    return all(ord(ch) < 128 for ch in text)


def _offline_translate(text: str) -> str:
    """Token-level dictionary translation; unknown tokens pass through unchanged."""
    out = []
    for token in text.replace(",", " ").split():
        key = token.strip(".,!?;:\"'()")
        out.append(_OFFLINE.get(key, token))
    return " ".join(out)


def translate_query(text: str, source_lang: str = "auto") -> tuple[str, str]:
    """Translate a query to English. Returns (english_text, method).

    method is one of: 'none' (already English), 'google', 'offline', 'empty'.
    """
    text = (text or "").strip()
    if not text:
        return text, "empty"
    # Explicit English, or pure-ASCII input, needs no translation.
    if source_lang == "en" or _looks_english(text):
        return text, "none"

    # Primary: online translation (open vocabulary). source='auto' is robust to
    # a mislabelled language tag.
    try:
        from deep_translator import GoogleTranslator
        english = GoogleTranslator(source="auto", target="en").translate(text)
        if english and english.strip():
            return english.strip(), "google"
    except Exception:
        pass  # no internet / service error -> fall back offline

    # Fallback: offline phrase dictionary.
    return _offline_translate(text), "offline"


if __name__ == "__main__":
    samples = [
        ("hi", "सफ़ेद ट्रक"),
        ("gu", "સફેદ ટ્રક"),
        ("hi", "लाल कार"),
        ("hi", "बैकपैक वाला आदमी"),
        ("en", "a white truck"),
    ]
    for lang, q in samples:
        eng, method = translate_query(q, lang)
        print(f"[{lang}] {q!r} -> {eng!r}  (via {method})")
