"""Central paths and constants shared by training scripts and the app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "language_detection"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "language_detection"
TFIDF_MODEL_DIR = MODEL_DIR / "tfidf"        # Part 1 - TF-IDF baseline
FASTTEXT_MODEL_DIR = MODEL_DIR / "fasttext"  # Part 2 - FastText word embeddings

RESULTS_DIR = PROJECT_ROOT / "results"

TEXT_COLUMN = "text"
LABEL_COLUMN = "language"
LANGUAGES = ["english", "french", "kinyarwanda"]

RANDOM_SEED = 42
