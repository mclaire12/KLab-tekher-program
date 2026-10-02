"""PART 1 - TF-IDF baseline for language detection (English / French / Kinyarwanda).

Representation: scikit-learn TfidfVectorizer (sparse, word or character n-grams).
Everything else (data, split, classifiers, metrics, selection, calibration) is
shared with Part 2 through training/common.py, so the two parts differ ONLY in
the text representation.

Outputs:
    models/language_detection/tfidf/{vectorizer.pkl, model.pkl, metadata.json}
    results/tfidf_*  (results CSVs, classification report, charts)

Usage:
    python training/train_language_detection_tfidf.py
    python training/train_language_detection_tfidf.py --configs tfidf_char_wb_2-5gram --classifiers LinearSVC
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import Family, run_experiment  # noqa: E402
from src.config import TFIDF_MODEL_DIR  # noqa: E402
from src.language_detection import FEATURE_CONFIGS, build_vectorizer, describe_vectorizer  # noqa: E402


def save_vectorizer(vectorizer, model_dir: Path) -> list[str]:
    joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
    return ["vectorizer.pkl"]


TFIDF = Family(
    key="tfidf",
    label="TF-IDF",
    representation="TF-IDF (sparse; no word embeddings)",
    configs=FEATURE_CONFIGS,
    build=build_vectorizer,
    describe=describe_vectorizer,
    save=save_vectorizer,
    model_dir=TFIDF_MODEL_DIR,
    classifier_file="model.pkl",
    model_version="part1-tfidf-baseline-v1",
)

if __name__ == "__main__":
    run_experiment(TFIDF)
