"""PART 2 - FastText word embeddings for language detection (English / French / Kinyarwanda).

Same raw text -> same preprocessing -> FastText word vectors -> document vectors
-> same classifiers -> same evaluation as Part 1.

Representation: gensim FastText trained (unsupervised) on the training split,
then each text = L2-normalised mean of its word vectors (see
src/language_detection/embeddings.py). Everything else (data, split,
classifiers, metrics, selection rule, calibration) is shared with Part 1
through training/common.py.

Outputs:
    models/language_detection/fasttext/{fasttext.model, classifier.pkl, metadata.json}
    results/fasttext_*  (results CSVs, classification report, charts)

Usage:
    python training/train_language_detection_embeddings.py
    python training/train_language_detection_embeddings.py --configs fasttext_skipgram_mean --classifiers LinearSVC
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gensim  # noqa: E402

from common import Family, run_experiment  # noqa: E402
from src.config import FASTTEXT_MODEL_DIR  # noqa: E402
from src.language_detection.embeddings import (  # noqa: E402
    FASTTEXT_CONFIGS, FASTTEXT_DEFAULTS, build_fasttext_vectorizer,
)


def save_fasttext(vectorizer, model_dir: Path) -> list[str]:
    for old in model_dir.glob("fasttext.model*"):
        old.unlink()  # remove files of a previous run
    return vectorizer.save(model_dir)


FASTTEXT = Family(
    key="fasttext",
    label="FastText",
    representation="FastText word embeddings (dense; mean word vector per document)",
    configs={name: {**FASTTEXT_DEFAULTS, **cfg} for name, cfg in FASTTEXT_CONFIGS.items()},
    build=build_fasttext_vectorizer,
    describe=lambda vectorizer: vectorizer.describe(),
    save=save_fasttext,
    model_dir=FASTTEXT_MODEL_DIR,
    classifier_file="classifier.pkl",
    model_version="part2-fasttext-v1",
    extra_metadata={
        "embedding_library": f"gensim {gensim.__version__} (FastText)",
        "embedding_training_data": "training split only (unsupervised; no labels, no validation/test text)",
        "contextual": False,
    },
)

if __name__ == "__main__":
    run_experiment(FASTTEXT)
