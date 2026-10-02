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
