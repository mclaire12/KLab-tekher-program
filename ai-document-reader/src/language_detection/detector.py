"""Load the trained TF-IDF + classifier and predict a document's language."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.config import BEST_MODEL_PATH, METADATA_PATH, VECTORIZER_PATH
from src.preprocessing import clean_text

# Long documents are truncated before vectorising; a few thousand characters
# are more than enough to identify the language and keep prediction fast.
MAX_CHARS = 20_000


class ModelNotTrainedError(FileNotFoundError):
    pass


@dataclass
class LanguagePrediction:
    language: str
    # Probability of the predicted class, or None if the classifier does not
    # produce probabilities (e.g. LinearSVC). This is the model's confidence on
    # THIS input - it is NOT the model's accuracy.
    confidence: float | None
    probabilities: dict[str, float] | None
    # Raw decision-function margins (e.g. LinearSVC): larger = more strongly
    # that class. They are NOT probabilities and do not sum to 1.
    decision_scores: dict[str, float] | None
    model_name: str
    characters_used: int


class LanguageDetector:
    def __init__(self, vectorizer, model, metadata: dict):
        self.vectorizer = vectorizer
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(
        cls,
        vectorizer_path: Path = VECTORIZER_PATH,
        model_path: Path = BEST_MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> "LanguageDetector":
        missing = [p for p in (vectorizer_path, model_path, metadata_path) if not Path(p).exists()]
        if missing:
            raise ModelNotTrainedError(
                "Language detection model not found: "
                + ", ".join(str(p) for p in missing)
                + "\nTrain it first: python training/train_language_detection.py"
            )
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(joblib.load(vectorizer_path), joblib.load(model_path), metadata)

    @property
    def model_name(self) -> str:
        return self.metadata.get("selected_classifier", type(self.model).__name__)

    @property
    def supports_probabilities(self) -> bool:
        return hasattr(self.model, "predict_proba")

    def predict(self, text: str) -> LanguagePrediction:
        cleaned = clean_text(text)[:MAX_CHARS]
        if not cleaned:
            raise ValueError("Cannot detect language of an empty document.")
        X = self.vectorizer.transform([cleaned])
        label = str(self.model.predict(X)[0])

        probabilities = decision_scores = confidence = None
        classes = [str(c) for c in self.model.classes_]
        if self.supports_probabilities:
            proba = self.model.predict_proba(X)[0]
            probabilities = {c: float(p) for c, p in zip(classes, proba)}
            confidence = probabilities[label]
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)[0]
            decision_scores = {c: float(v) for c, v in zip(classes, scores)}
        return LanguagePrediction(label, confidence, probabilities, decision_scores,
                                  self.model_name, len(cleaned))

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(self.vectorizer.transform([clean_text(t) for t in texts]))
