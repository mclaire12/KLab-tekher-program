"""Question answering over one document - exact, extractive answers.

    Question
       |
       |-- 1. Exact patterns (phone, e-mail, link, owner's name)   -> exact_answers.py
       |
       |-- 2. Retrieval: document -> overlapping chunks -> TF-IDF (fitted on THIS
       |      document only) -> cosine similarity -> top passages
       |      + pretrained extractive QA model picks the exact answer span
       |      (deepset/xlm-roberta-base-squad2)                    -> span_qa.py
       |
       '-- 3. Fallback when the QA model is unavailable: best-matching sentence

The answer is always text copied from the document (never generated), and the
source passage is always returned with it. The TF-IDF here is a per-document
retrieval index, unrelated to the trained language-detection model.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.summarization import split_sentences

from .exact_answers import exact_answer
from .span_qa import MODEL_NAME as SPAN_MODEL_NAME
from .span_qa import SpanQAUnavailableError, best_span

MIN_SCORE = 0.05  # below this, we report that no relevant passage was found
HEAD_CHARS = 1500  # start of the document (title, owner, contact details) is always searched


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
    method: str = ""
    source: str | None = None  # passage the answer was taken from


class DocumentRetriever:
    def __init__(self, text: str, sentences_per_chunk: int = 3, overlap: int = 1):
        self.text = text
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
        top_score = passages[0].score if passages else 0.0

        # 1. exact patterns (phone, e-mail, link, owner's name)
        exact = exact_answer(question, self.text)
        if exact is not None:
            return Answer(question, exact.answer, top_score, passages, exact.method, exact.source)

        # 2. pretrained extractive QA model on the relevant passages + document start
        # all top passages: a question can match a passage without sharing words with it
        contexts = [p.text for p in passages]
        head = self.text[:HEAD_CHARS].strip()
        if head and head not in contexts:
            contexts.append(head)
        try:
            spans = [span for c in contexts if (span := best_span(question, c)) is not None]
        except SpanQAUnavailableError:
            spans = None
        if spans is not None:
            best = max(spans, key=lambda sp: sp.score, default=None)
            if best is None or best.score < best.null_score:
                return Answer(question, None, top_score, passages, f"pretrained QA model {SPAN_MODEL_NAME}")
            return Answer(question, best.text, top_score, passages,
                          f"pretrained QA model {SPAN_MODEL_NAME}", best.context)

        # 3. fallback: best-matching sentence of the best passage
        if not passages or top_score < MIN_SCORE:
            return Answer(question, None, top_score, passages, "TF-IDF retrieval")
        sentences = split_sentences(passages[0].text) or [passages[0].text]
        sims = cosine_similarity(self.vectorizer.transform([question]),
                                 self.vectorizer.transform(sentences))[0]
        return Answer(question, sentences[int(sims.argmax())], top_score, passages,
                      "TF-IDF retrieval, best-matching sentence (QA model unavailable)", passages[0].text)
