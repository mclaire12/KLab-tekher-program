"""Build data/language_detection/{train,test}.csv from REAL public corpora.

Source: Leipzig Corpora Collection (Universität Leipzig), licence CC BY 4.0
        https://wortschatz.uni-leipzig.de/en/download
    english     -> eng_news_2020_10K       (news sentences, 2020)
    french      -> fra_news_2020_10K       (news sentences, 2020)
    kinyarwanda -> kin_community_2017_10K  (web/community sentences, 2017)

Each corpus contains 10,000 real sentences. Nothing is generated or synthesised.
The script only downloads, labels, de-duplicates and splits the sentences.

Usage:
    python training/prepare_dataset.py
    python training/prepare_dataset.py --per-language 5000 --test-size 0.2

You can instead supply your own CSVs (columns: text,language) in
data/language_detection/ and skip this script entirely.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    DATA_DIR, LABEL_COLUMN, RANDOM_SEED, TEST_CSV, TEXT_COLUMN, TRAIN_CSV,
)
from src.preprocessing import clean_text  # noqa: E402

LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/{name}.tar.gz"
CORPORA = {
    "english": "eng_news_2020_10K",
    "french": "fra_news_2020_10K",
    "kinyarwanda": "kin_community_2017_10K",
}
RAW_DIR = DATA_DIR / "raw"
MIN_TOKENS = 3  # drop fragments like "Bigenze bite se kandi?" ... too short to be useful


def download(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / f"{name}.tar.gz"
    if archive.exists():
        print(f"  using cached {archive.name}")
        return archive
    url = LEIPZIG_URL.format(name=name)
    print(f"  downloading {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    archive.write_bytes(response.content)
    return archive


def read_sentences(archive: Path, name: str) -> list[str]:
    member = f"{name}/{name}-sentences.txt"
    with tarfile.open(archive, "r:gz") as tar:
        raw = tar.extractfile(member).read().decode("utf-8")
    # Format: "<id>\t<sentence>" per line
    return [line.split("\t", 1)[1] for line in raw.splitlines() if "\t" in line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=None,
                        help="Max sentences per language (default: all available, balanced to the smallest)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    frames = []
    for language, name in CORPORA.items():
        print(f"[{language}] {name}")
        sentences = read_sentences(download(name), name)
        df = pd.DataFrame({TEXT_COLUMN: sentences, LABEL_COLUMN: language})
        df[TEXT_COLUMN] = df[TEXT_COLUMN].str.strip()
        n_raw = len(df)
        cleaned = df[TEXT_COLUMN].map(clean_text)
        df = df[cleaned.str.split().str.len() >= MIN_TOKENS]
        df = df.loc[~cleaned[df.index].duplicated()]
        print(f"  {n_raw} sentences -> {len(df)} after removing short/duplicate sentences")
        frames.append(df)

    # Balance classes so accuracy is not inflated by a majority class.
    n = min(len(f) for f in frames)
    if args.per_language:
        n = min(n, args.per_language)
    data = pd.concat([f.sample(n=n, random_state=args.seed) for f in frames], ignore_index=True)

    # Remove sentences that appear in more than one language (ambiguous labels).
    key = data[TEXT_COLUMN].map(clean_text)
    ambiguous = key.duplicated(keep=False) & data.groupby(key)[LABEL_COLUMN].transform("nunique").gt(1)
    data = data[~ambiguous]

    train, test = train_test_split(
        data, test_size=args.test_size, stratify=data[LABEL_COLUMN], random_state=args.seed
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    test.to_csv(TEST_CSV, index=False, encoding="utf-8")

    print(f"\nSaved {len(train)} rows -> {TRAIN_CSV}")
    print(f"Saved {len(test)} rows -> {TEST_CSV}")
    print("\nClass distribution (train):")
    print(train[LABEL_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
