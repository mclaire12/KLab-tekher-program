"""Retrieval-based question answering (extractive, explainable).

    Document -> overlapping chunks of sentences
             -> TF-IDF vectors (fitted on THIS document's chunks only)
             -> cosine similarity between the question and every chunk
             -> best chunk = source passage
             -> best-matching sentence inside it = answer

This is NOT a generative model: the answer is always a sentence copied from
the document, and the source passage is always returned with it. The TF-IDF
here is a separate, per-document retrieval index; it is unrelated to the
trained language-detection model.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.summarization import split_sentences

MIN_SCORE = 0.05  # below this, we report that no relevant passage was found


@dataclass
class Passage:
    text: str
    score: float
    chunk_index: int


@dataclass
class Answer:
    question: str
    answer: str | None
    score: float
    passages: list[Passage]
    method: str = "TF-IDF retrieval + cosine similarity (extractive)"


class DocumentRetriever:
    def __init__(self, text: str, sentences_per_chunk: int = 3, overlap: int = 1):
        sentences = split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]
        step = max(1, sentences_per_chunk - overlap)
        self.chunks = [
            " ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, max(1, len(sentences) - overlap), step)
        ] if sentences else []
        self.vectorizer = None
        if self.chunks:
            # word unigrams+bigrams with sublinear TF; char n-grams are not needed
            # here because the question and the document share the same language.
            self.vectorizer = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                              sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']*\b")
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, question: str, top_k: int = 3) -> list[Passage]:
        if self.vectorizer is None or not clean_text(question):
            return []
        sims = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        order = sims.argsort()[::-1][:top_k]
        return [Passage(self.chunks[i], float(sims[i]), int(i)) for i in order]

    def answer(self, question: str, top_k: int = 3) -> Answer:
        passages = self.retrieve(question, top_k)
        if not passages or passages[0].score < MIN_SCORE:
            return Answer(question, None, passages[0].score if passages else 0.0, passages)
        best = passages[0]
        sentences = split_sentences(best.text) or [best.text]
        sims = cosine_similarity(self.vectorizer.transform([question]),
                                 self.vectorizer.transform(sentences))[0]
        return Answer(question, sentences[int(sims.argmax())], best.score, passages)
