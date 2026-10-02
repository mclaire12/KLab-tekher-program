"""PART 2 - FastText word embeddings (gensim) for language detection.

What FastText is (and is not)
-----------------------------
FastText (Bojanowski et al., 2017) learns one dense vector per word, like
Word2Vec, but represents every word as the SUM of vectors of its character
n-grams plus the word itself, e.g. with n = 3..6:

    "ishuri" -> <is, ish, shu, hur, uri, ri>, <ish, ..., <ishuri>

Consequences for this project:
  * rare and UNSEEN word forms still get a vector, built from their n-grams
    (useful for Kinyarwanda, where one stem appears with many prefixes:
    umwana / abana / kwiga / twiga ...);
  * words that share spelling patterns get similar vectors.

FastText is NOT contextual: a word always has the same vector, whatever
sentence it appears in (unlike BERT-style models).

The embeddings are trained here, unsupervised, on the training split only
(no labels are used, and validation/test text is never seen).
"""

from __future__ import annotations

import zlib

from src.config import RANDOM_SEED
from src.preprocessing import tokenize

# Configurable FastText settings shared by all configurations (see FASTTEXT_CONFIGS)
FASTTEXT_DEFAULTS = dict(
    vector_size=100,   # dimensions of each word vector
    window=5,          # context words on each side used to learn a word's vector
    min_count=2,       # words seen fewer times get no own vector (subwords still apply)
    epochs=20,         # passes over the training corpus
    sg=1,              # learning algorithm: 1 = skip-gram, 0 = CBOW
    min_n=3,           # shortest character n-gram
    max_n=6,           # longest character n-gram (0 = no subwords, i.e. plain word vectors)
    bucket=50_000,     # hash buckets for n-gram vectors (gensim default 2,000,000 -> ~800 MB)
    negative=5,        # negative sampling
    seed=RANDOM_SEED,
    workers=1,         # one thread: required for reproducible training
)


def stable_hash(text: str) -> int:
    """Deterministic hash for gensim's vector initialisation.

    gensim's default uses Python's built-in hash(), which changes between runs
    (hash randomisation). CRC32 makes training reproducible with a fixed seed.
    """
    return zlib.crc32(text.encode("utf-8"))


def train_fasttext(token_lists: list[list[str]], **settings):
    """Train gensim FastText on tokenised sentences; returns the KeyedVectors."""
    from gensim.models import FastText

    params = {**FASTTEXT_DEFAULTS, **settings}
    model = FastText(sentences=token_lists, hashfxn=stable_hash, **params)
    return model.wv


def tokenize_texts(texts) -> list[list[str]]:
    """Texts are already cleaned with src.preprocessing.clean_text (same as Part 1)."""
    return [tokenize(t) for t in texts]


# ------------------------------------------------- document vectors ----
# A classifier needs ONE fixed-size vector per text, but texts have different
# numbers of words. Word vectors are therefore pooled into a document vector:
#
#   "abana bagiye ku ishuri"
#     -> [v(abana), v(bagiye), v(ku), v(ishuri)]   4 x 100 numbers
#     -> mean over the 4 words                     1 x 100 numbers
#     -> L2-normalised (unit length, as TF-IDF rows are in Part 1)
#
# "mean_max" concatenates the mean and the element-wise maximum (1 x 200),
# a simple alternative that keeps some information about salient words.
POOLING_METHODS = ("mean", "mean_max")

# Configurations compared in Part 2 (each overrides FASTTEXT_DEFAULTS).
FASTTEXT_CONFIGS: dict[str, dict] = {
    # learning algorithm
    "fasttext_cbow_mean": dict(sg=0, pooling="mean"),
    "fasttext_skipgram_mean": dict(sg=1, pooling="mean"),
    # subword range matching Part 1's best TF-IDF character n-grams (2-5)
    "fasttext_skipgram_ng2-5_mean": dict(sg=1, min_n=2, max_n=5, pooling="mean"),
    # ablation: no subwords at all = plain word vectors (shows what subwords add)
    "fasttext_skipgram_nosubwords_mean": dict(sg=1, max_n=0, pooling="mean"),
    # alternative aggregation
    "fasttext_skipgram_meanmax": dict(sg=1, pooling="mean_max"),
}

_TRAINED_CACHE: dict = {}  # configs that differ only in pooling share one trained model


class FastTextDocumentVectorizer:
    """scikit-learn style representation: fit(texts) trains FastText,
    transform(texts) returns pooled, L2-normalised document vectors."""

    def __init__(self, pooling: str = "mean", **fasttext_settings):
        if pooling not in POOLING_METHODS:
            raise ValueError(f"pooling must be one of {POOLING_METHODS}")
        self.pooling = pooling
        self.settings = {**FASTTEXT_DEFAULTS, **fasttext_settings}
        self.wv = None

    # ------------------------------------------------------------ fit ----
    def fit(self, texts, y=None):
        texts = list(texts)
        key = (tuple(sorted(self.settings.items())), len(texts), hash(texts[0]), hash(texts[-1]))
        if key not in _TRAINED_CACHE:
            _TRAINED_CACHE[key] = train_fasttext(tokenize_texts(texts), **self.settings)
        self.wv = _TRAINED_CACHE[key]
        return self

    def fit_transform(self, texts, y=None):
        texts = list(texts)
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------ transform ----
    @property
    def dimension(self) -> int:
        size = self.wv.vector_size
        return size * 2 if self.pooling == "mean_max" else size

    def transform(self, texts):
        import numpy as np

        if self.wv is None:
            raise RuntimeError("FastTextDocumentVectorizer is not fitted")
        uses_subwords = self.settings["max_n"] >= self.settings["min_n"]
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, tokens in enumerate(tokenize_texts(texts)):
            if not uses_subwords:
                # plain word vectors: unseen words have NO vector and are skipped
                tokens = [t for t in tokens if t in self.wv.key_to_index]
            if not tokens:
                continue  # empty text (or only unseen words) -> zero vector
            vectors = self.wv[tokens]  # (n_words, vector_size); unseen words built from n-grams
            parts = [vectors.mean(axis=0)]
            if self.pooling == "mean_max":
                parts.append(vectors.max(axis=0))
            doc = np.concatenate(parts)
            norm = np.linalg.norm(doc)
            out[i] = doc / norm if norm > 0 else doc
        return out

    # ------------------------------------------------- save / describe ----
    def describe(self) -> dict:
        s = self.settings
        return {
            "algorithm": "skip-gram" if s["sg"] == 1 else "CBOW",
            "vector_size": s["vector_size"], "window": s["window"], "min_count": s["min_count"],
            "epochs": s["epochs"], "char_ngrams": [s["min_n"], s["max_n"]] if s["max_n"] else "none",
            "bucket": s["bucket"], "negative_sampling": s["negative"], "seed": s["seed"],
            "workers": s["workers"], "pooling": self.pooling, "document_vector_size": self.dimension,
            "l2_normalised": True,
            "vocabulary_size": len(self.wv.key_to_index) if self.wv is not None else None,
        }

    def save(self, model_dir) -> list[str]:
        """Only the KeyedVectors are needed for inference (word + n-gram vectors)."""
        from pathlib import Path

        path = Path(model_dir) / "fasttext.model"
        self.wv.save(str(path))
        return sorted(p.name for p in Path(model_dir).glob("fasttext.model*"))

    @classmethod
    def load(cls, model_dir, settings: dict) -> "FastTextDocumentVectorizer":
        from pathlib import Path

        from gensim.models.fasttext import FastTextKeyedVectors

        settings = dict(settings)
        vectorizer = cls.__new__(cls)
        vectorizer.pooling = settings.get("pooling", "mean")
        vectorizer.settings = settings
        vectorizer.wv = FastTextKeyedVectors.load(str(Path(model_dir) / "fasttext.model"))
        return vectorizer


def build_fasttext_vectorizer(config_name: str) -> FastTextDocumentVectorizer:
    if config_name not in FASTTEXT_CONFIGS:
        raise KeyError(f"Unknown FastText config '{config_name}'. Options: {list(FASTTEXT_CONFIGS)}")
    return FastTextDocumentVectorizer(**FASTTEXT_CONFIGS[config_name])
