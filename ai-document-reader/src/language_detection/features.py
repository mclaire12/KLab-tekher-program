"""Text representations for language detection.

PART 1 uses only TF-IDF. Every representation is an object with scikit-learn's
``fit`` / ``transform`` interface, registered by name in ``FEATURE_CONFIGS``.
PART 2 can add word-embedding representations to this registry without
changing the training script or the app.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

# Each entry: name -> keyword arguments for TfidfVectorizer.
# Input text is already cleaned by src.preprocessing.clean_text, so the
# vectorizer's own lowercasing is disabled to keep one explicit pipeline.
FEATURE_CONFIGS: dict[str, dict] = {
    # Word unigrams: classic bag-of-words baseline.
    "tfidf_word_1gram": dict(
        analyzer="word", ngram_range=(1, 1), token_pattern=r"(?u)\b\w[\w']*\b",
        min_df=2, sublinear_tf=True, max_features=50_000,
    ),
    # Word unigrams + bigrams: adds short phrases ("de la", "of the", "mu rwanda").
    "tfidf_word_1-2gram": dict(
        analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w[\w']*\b",
        min_df=2, sublinear_tf=True, max_features=100_000,
    ),
    # Character n-grams inside word boundaries: captures spelling patterns
    # ("eau", "tion", "nyi", "rw"), robust to unseen words and short texts.
    "tfidf_char_wb_1-3gram": dict(
        analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True,
        max_features=50_000,
    ),
    "tfidf_char_wb_2-5gram": dict(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
        max_features=100_000,
    ),
}


def build_vectorizer(config_name: str) -> TfidfVectorizer:
    if config_name not in FEATURE_CONFIGS:
        raise KeyError(f"Unknown feature config '{config_name}'. Options: {list(FEATURE_CONFIGS)}")
    return TfidfVectorizer(lowercase=False, **FEATURE_CONFIGS[config_name])


def describe_vectorizer(vectorizer: TfidfVectorizer) -> dict:
    """JSON-serialisable description of a fitted vectorizer (for metadata.json)."""
    params = vectorizer.get_params()
    keys = ["analyzer", "ngram_range", "min_df", "max_df", "max_features",
            "sublinear_tf", "norm", "use_idf", "smooth_idf", "lowercase"]
    desc = {k: params[k] for k in keys}
    desc["ngram_range"] = list(desc["ngram_range"])
    desc["vocabulary_size"] = len(vectorizer.vocabulary_)
    return desc
