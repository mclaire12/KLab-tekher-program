"""Simple, explainable text preprocessing for the language-detection baseline.

Design choices (important for English / French / Kinyarwanda):

* Accented letters are KEPT (é, è, ç, à ...). They are strong signals for French.
* Apostrophes inside words are KEPT and normalised to ``'``. Kinyarwanda writes
  many contractions such as ``nk'abandi`` or ``y'u Rwanda``; French has
  ``l'école``, ``qu'il``. Removing them would merge or split words wrongly.
* Broken quote characters from mis-decoded web text (``â€™``) are repaired.
* Digits, URLs, e-mails and punctuation are REMOVED: they carry no language
  information and are identical across the three languages.
* Stopwords are NOT removed by default. For *language identification* stopwords
  ("the", "le", "na") are among the most useful features. Optional stopword
  removal is provided for other tasks (e.g. the extractive summary).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

# Common "mojibake": UTF-8 quotes wrongly decoded as Windows-1252 (seen in web corpora).
_MOJIBAKE = {"â€™": "'", "â€˜": "'", "â€œ": '"', "â€": '"'}
_APOSTROPHES = re.compile(r"[‘’ʼ`´]")  # ‘ ’ ʼ ` ´ -> '
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
# Anything that is not a letter, an apostrophe or whitespace becomes a space.
_NON_LETTER = re.compile(r"[^\w'\s]|[\d_]")
_LONE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

# Small, hand-picked lists. Only used when explicitly requested.
STOPWORDS = {
    "english": {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
        "was", "were", "be", "been", "it", "this", "that", "with", "as", "by", "at",
        "from", "but", "not", "have", "has", "had", "they", "he", "she", "we", "you",
        "i", "his", "her", "their", "its", "which", "who", "will", "would", "can",
    },
    "french": {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "en", "dans",
        "sur", "pour", "par", "avec", "est", "sont", "été", "être", "il", "elle",
        "ils", "elles", "nous", "vous", "je", "ce", "cette", "ces", "qui", "que",
        "qu", "ne", "pas", "plus", "au", "aux", "son", "sa", "ses", "leur", "se",
    },
    # Kinyarwanda is agglutinative: grammatical words are often prefixes glued to
    # the next word. We only list very frequent free-standing function words and
    # keep the list deliberately short to avoid deleting meaningful content.
    "kinyarwanda": {
        "na", "ni", "ko", "mu", "ku", "kandi", "cyangwa", "ariko", "nka", "nk",
        "ya", "wa", "rya", "cya", "bya", "za", "kwa", "y", "w", "z", "ry", "by",
    },
}


@dataclass
class PreprocessingStats:
    original_characters: int
    cleaned_characters: int
    tokens: int
    unique_tokens: int
    removed_characters: int

    def as_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalise a raw string for TF-IDF. Returns '' for empty/None input."""
    if not text:
        return ""
    text = str(text)
    for broken, fixed in _MOJIBAKE.items():
        text = text.replace(broken, fixed)
    text = unicodedata.normalize("NFC", text)
    text = _APOSTROPHES.sub("'", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = _NON_LETTER.sub(" ", text)
    text = _LONE_APOSTROPHE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens (letters with internal apostrophes)."""
    return _TOKEN.findall(text)


def remove_stopwords(tokens: list[str], language: str | None = None) -> list[str]:
    """Remove stopwords for one language, or for all three if language is None."""
    if language is None:
        stop = set().union(*STOPWORDS.values())
    else:
        stop = STOPWORDS.get(language, set())
    return [t for t in tokens if t not in stop]


def preprocess(text: str) -> tuple[str, list[str], PreprocessingStats]:
    """Full pipeline: clean -> tokenize, plus statistics for the UI."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    stats = PreprocessingStats(
        original_characters=len(text or ""),
        cleaned_characters=len(cleaned),
        tokens=len(tokens),
        unique_tokens=len(set(tokens)),
        removed_characters=len(text or "") - len(cleaned),
    )
    return cleaned, tokens, stats


PREPROCESSING_DESCRIPTION = {
    "mojibake_repair": "â€™ -> '",
    "unicode_normalisation": "NFC",
    "lowercase": True,
    "apostrophes": "normalised to ' and kept inside words",
    "removed": ["URLs", "e-mails", "digits", "punctuation/symbols"],
    "whitespace": "collapsed to single spaces",
    "stopwords_removed_for_language_detection": False,
}
