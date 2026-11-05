"""Language codes and lightweight script-based language identification."""
from __future__ import annotations

import re

# our short code -> IndicTrans2 (FLORES) code
IT2_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda", "ta": "tam_Taml"}
LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil"}
GTTS_CODES = {"hi": "hi", "kn": "kn", "ta": "ta", "en": "en"}

_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ta": (0x0B80, 0x0BFF),  # Tamil
}

# Native digits -> ASCII (Devanagari, Kannada, Tamil)
_DIGIT_MAP = {}
for _base in (0x0966, 0x0CE6, 0x0BE6):
    for _i in range(10):
        _DIGIT_MAP[chr(_base + _i)] = str(_i)
_DIGIT_TABLE = str.maketrans(_DIGIT_MAP)


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TABLE)


def detect_language(text: str, default: str = "en") -> str:
    """Pick the language whose script dominates the text (Latin => English)."""
    counts = {k: 0 for k in _SCRIPT_RANGES}
    latin = 0
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
                break
        else:
            if ch.isascii() and ch.isalpha():
                latin += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0 and latin == 0:
        return default
    return best if counts[best] >= latin else "en"


_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]
