"""Extractive summarisation (language-independent fallback / default).

Method (classic frequency-based extractive summary, Luhn 1958 style):
1. Split the document into sentences.
2. Count content-word frequencies (stopwords removed, frequencies normalised).
3. Score each sentence = mean normalised frequency of its words.
4. Keep the top-N sentences and return them in their ORIGINAL order.

No machine-learning model is trained here, and this is NOT the TF-IDF
language-detection model. Every summary sentence is copied verbatim from the
document, so the output can be traced back to the source.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from src.preprocessing import clean_text, remove_stopwords, tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"'”»)\]]*\s+|\n{2,}")


@dataclass
class SummaryResult:
    summary: str
    sentences: list[str]
    method: str
    model_name: str
    sentence_indices: list[int]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    parts = (s.strip().replace("\n", " ") for s in _SENTENCE_SPLIT.split(text))
    return [s for s in parts if len(s.split()) >= 3]


def extractive_summary(text: str, num_sentences: int = 5, language: str | None = None) -> SummaryResult:
    sentences = split_sentences(text)
    if not sentences:
        return SummaryResult("", [], "extractive", "Frequency-based sentence scoring", [])
    if len(sentences) <= num_sentences:
        return SummaryResult(" ".join(sentences), sentences, "extractive",
                             "Frequency-based sentence scoring", list(range(len(sentences))))

    sentence_tokens = [remove_stopwords(tokenize(clean_text(s)), language) for s in sentences]
    freq = Counter(t for toks in sentence_tokens for t in toks if len(t) > 2)
    if not freq:
        top = list(range(num_sentences))
    else:
        max_freq = max(freq.values())
        scores = [
            sum(freq[t] / max_freq for t in toks) / len(toks) if toks else 0.0
            for toks in sentence_tokens
        ]
        top = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences])
    chosen = [sentences[i] for i in top]
    return SummaryResult(" ".join(chosen), chosen, "extractive", "Frequency-based sentence scoring", top)
