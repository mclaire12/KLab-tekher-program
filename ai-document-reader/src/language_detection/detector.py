"""Load a trained language-detection model (representation + classifier) and predict.

Each model folder contains metadata.json, the fitted text representation and
the classifier, so Part 1 and Part 2 are loaded the same way:

    models/language_detection/tfidf/      Part 1 - TF-IDF baseline
    models/language_detection/fasttext/   Part 2 - FastText word embeddings
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.config import TFIDF_MODEL_DIR
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


def _load_representation(metadata: dict, model_dir: Path):
    kind = metadata.get("representation_type", "tfidf")
    if kind == "tfidf":
        return joblib.load(model_dir / "vectorizer.pkl")
    raise ValueError(f"Unknown representation type '{kind}' in {model_dir}")


class LanguageDetector:
    def __init__(self, vectorizer, model, metadata: dict):
        self.vectorizer = vectorizer  # any fitted representation with .transform(texts)
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path = TFIDF_MODEL_DIR) -> "LanguageDetector":
        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise ModelNotTrainedError(
                f"Language detection model not found in {model_dir}\n"
                f"Train it first: python training/train_language_detection_{model_dir.name}.py"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = [model_dir / f for f in metadata["representation_files"]]
        files.append(model_dir / metadata["classifier_file"])
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise ModelNotTrainedError("Model files missing: " + ", ".join(missing))
        vectorizer = _load_representation(metadata, model_dir)
        return cls(vectorizer, joblib.load(model_dir / metadata["classifier_file"]), metadata)

    @property
    def representation_label(self) -> str:
        return self.metadata.get("representation", self.metadata.get("representation_type", ""))

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
