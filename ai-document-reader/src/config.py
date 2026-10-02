"""Central paths and constants shared by training scripts and the app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "language_detection"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "language_detection"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results"

TEXT_COLUMN = "text"
LABEL_COLUMN = "language"
LANGUAGES = ["english", "french", "kinyarwanda"]

RANDOM_SEED = 42
