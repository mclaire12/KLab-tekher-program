"""Traditional classifiers compared in the baseline (all scikit-learn)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.config import RANDOM_SEED


def build_classifiers(seed: int = RANDOM_SEED) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=10.0, random_state=seed),
        "LinearSVC": LinearSVC(C=1.0, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed
        ),
        # Cosine distance is the natural metric for L2-normalised TF-IDF vectors.
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    }
