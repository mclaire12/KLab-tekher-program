"""Part 1 vs Part 2: TF-IDF baseline versus FastText embeddings.

Run AFTER both training scripts. Reads their saved results and models and writes:

    results/comparison.csv                same classifier, both representations, test set:
                                          accuracy / precision / recall / macro F1 + difference
    results/comparison_short_text.csv     same, on the first 1/2/3/5 words of each test sentence
    results/comparison_deployed.csv       the two deployed models (used by the app)
    results/comparison.png                bar chart (full sentences and 1-word inputs)
    results/comparison_short_text.png     line chart, macro F1 vs. number of words
    results/error_analysis.csv            texts where one representation is right and the other wrong
    results/error_analysis_summary.csv    counts of those cases per setting
    results/oov_analysis.csv              share of test words never seen in training, per language

difference = FastText score - TF-IDF score (positive = FastText better).

Usage:
    python training/compare_representations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    INK, INK_2, GRID, SERIES, SURFACE, compute_metrics, first_words, legend_above, load_splits,
    plot_lines, style_axes,
)
from src.config import FASTTEXT_MODEL_DIR, LABEL_COLUMN, RESULTS_DIR, TFIDF_MODEL_DIR  # noqa: E402
from src.language_detection import LanguageDetector  # noqa: E402
from src.preprocessing import tokenize  # noqa: E402

METRICS = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
           "recall_macro": "Recall (macro)", "f1_macro": "Macro F1"}
ERROR_SETTINGS = {"full sentence": None, "first 2 words": 2, "first 1 word": 1}
TFIDF_COLOR, FASTTEXT_COLOR = SERIES[0], SERIES[1]


def load_metadata(model_dir: Path) -> dict:
    path = model_dir / "metadata.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run both training scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))


def per_classifier(tfidf: pd.DataFrame, fasttext: pd.DataFrame, value_cols: list[str], keys: list[str]) -> pd.DataFrame:
    merged = tfidf.merge(fasttext, on=keys, suffixes=("_tfidf", "_fasttext"))
    rows = []
    for _, r in merged.iterrows():
        for col in value_cols:
            rows.append({**{k: r[k] for k in keys}, "metric": col,
                         "tfidf": r[f"{col}_tfidf"], "fasttext": r[f"{col}_fasttext"],
                         "difference": r[f"{col}_fasttext"] - r[f"{col}_tfidf"]})
    return pd.DataFrame(rows)


def plot_comparison(full: pd.DataFrame, one_word: pd.DataFrame, path: Path) -> None:
    """Two panels on one y-scale: macro F1 per classifier, TF-IDF vs FastText."""
    panels = [(full, "Full test sentences"), (one_word, "First word only")]
    lo = np.floor(min(full[["tfidf", "fasttext"]].min().min(), one_word[["tfidf", "fasttext"]].min().min()) * 20) / 20
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, sharey=True)
    for ax, (df, title) in zip(axes, panels):
        style_axes(ax)
        x = np.arange(len(df))
        for offset, col, color in ((-0.2, "tfidf", TFIDF_COLOR), (0.2, "fasttext", FASTTEXT_COLOR)):
            ax.bar(x + offset, df[col] - lo, bottom=lo, width=0.38, color=color)
            for xi, v in zip(x + offset, df[col]):
                ax.text(xi, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xticks(x, df["classifier"], color=INK)
        ax.set_ylim(lo, 1.012)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel(f"Test macro F1 (axis starts at {lo:.2f})", color=INK_2)
    fig.suptitle("Part 1 TF-IDF vs Part 2 FastText - same classifiers, same test set",
                 color=INK, fontsize=12, x=0.01, ha="left")
    legend_above(fig, [("TF-IDF (Part 1)", TFIDF_COLOR), ("FastText (Part 2)", FASTTEXT_COLOR)])
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def oov_rates(train_texts: pd.Series, test: pd.DataFrame, fasttext_vocab: set[str]) -> pd.DataFrame:
    train_vocab = {w for t in train_texts for w in tokenize(t)}
    rows = []
    for lang, group in test.groupby(LABEL_COLUMN):
        tokens = [w for t in group["clean"] for w in tokenize(t)]
        rows.append({
            "language": lang, "test_tokens": len(tokens),
            "unseen_in_training_text": np.mean([w not in train_vocab for w in tokens]),
            "no_own_fasttext_vector": np.mean([w not in fasttext_vocab for w in tokens]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    meta_t, meta_f = load_metadata(TFIDF_MODEL_DIR), load_metadata(FASTTEXT_MODEL_DIR)
    fp_t, fp_f = meta_t["dataset"].get("split_fingerprint"), meta_f["dataset"].get("split_fingerprint")
    if fp_t != fp_f:
        sys.exit(f"ERROR: the two models were not trained on the same split.\n  TF-IDF:   {fp_t}\n  FastText: {fp_f}")
    print(f"Same data split for both representations: {fp_t}")
    cfg_t, cfg_f = meta_t["feature_config"], meta_f["feature_config"]
    print(f"Compared representations: {cfg_t}  vs  {cfg_f}\n")

    # ------------------------------------------- per-classifier comparison ----
    res_t = pd.read_csv(RESULTS_DIR / "tfidf_results.csv")
    res_f = pd.read_csv(RESULTS_DIR / "fasttext_results.csv")
    cols = ["classifier", *METRICS]
    test_t = res_t[(res_t["split"] == "test") & (res_t["features"] == cfg_t)][cols]
    test_f = res_f[(res_f["split"] == "test") & (res_f["features"] == cfg_f)][cols]
    comparison = per_classifier(test_t, test_f, list(METRICS), ["classifier"])
    comparison.insert(1, "tfidf_features", cfg_t)
    comparison.insert(2, "fasttext_features", cfg_f)
    comparison.round(6).to_csv(RESULTS_DIR / "comparison.csv", index=False)

    short_t = pd.read_csv(RESULTS_DIR / "tfidf_short_text_robustness.csv")
    short_f = pd.read_csv(RESULTS_DIR / "fasttext_short_text_robustness.csv")
    keep = ["classifier", "words", "accuracy", "f1_macro"]
    short_cmp = per_classifier(short_t[short_t["features"] == cfg_t][keep],
                               short_f[short_f["features"] == cfg_f][keep],
                               ["accuracy", "f1_macro"], ["classifier", "words"])
    short_cmp.round(6).to_csv(RESULTS_DIR / "comparison_short_text.csv", index=False)

    f1_full = comparison[comparison["metric"] == "f1_macro"].reset_index(drop=True)
    f1_one = short_cmp[(short_cmp["metric"] == "f1_macro") & (short_cmp["words"] == 1)].reset_index(drop=True)
    plot_comparison(f1_full, f1_one, RESULTS_DIR / "comparison.png")

    # Same classifier (LinearSVC) for every line; word-level variants of both families as reference.
    clf = "LinearSVC"
    word_t, nosub_f = "tfidf_word_1gram", "fasttext_skipgram_nosubwords_mean"
    lines = {}
    for label, df, cfg in ((f"TF-IDF {cfg_t.removeprefix('tfidf_')}", short_t, cfg_t),
                           (f"FastText {cfg_f.removeprefix('fasttext_')}", short_f, cfg_f),
                           ("TF-IDF word 1gram", short_t, word_t),
                           ("FastText no subwords", short_f, nosub_f)):
        sub = df[(df["features"] == cfg) & (df["classifier"] == clf)]
        if len(sub):
            lines[label] = sub.set_index("words")["f1_macro"]
    plot_lines(lines, RESULTS_DIR / "comparison_short_text.png",
               f"Short inputs - TF-IDF vs FastText (all with {clf})", "Test macro F1")

    # ------------------------------------------------ deployed models ----
    data = load_splits()
    test = data.test
    det_t, det_f = LanguageDetector.load(TFIDF_MODEL_DIR), LanguageDetector.load(FASTTEXT_MODEL_DIR)
    deployed_rows, errors, summary = [], [], []
    for setting, n_words in ERROR_SETTINGS.items():
        texts = test["clean"] if n_words is None else first_words(test["clean"], n_words)
        y = test[LABEL_COLUMN].to_numpy()
        pred_t, pred_f = det_t.predict_batch(list(texts)), det_f.predict_batch(list(texts))
        for name, det, pred in (("TF-IDF", det_t, pred_t), ("FastText", det_f, pred_f)):
            deployed_rows.append({"setting": setting, "representation": name,
                                  "model": f"{det.metadata['feature_config']} + {det.model_name}",
                                  **{k: compute_metrics(y, pred)[k] for k in METRICS}})
        ok_t, ok_f = pred_t == y, pred_f == y
        summary.append({"setting": setting, "test_texts": len(y), "both_correct": int((ok_t & ok_f).sum()),
                        "tfidf_correct_fasttext_wrong": int((ok_t & ~ok_f).sum()),
                        "fasttext_correct_tfidf_wrong": int((~ok_t & ok_f).sum()),
                        "both_wrong": int((~ok_t & ~ok_f).sum())})
        for mask, case in ((ok_t & ~ok_f, "TF-IDF correct, FastText wrong"),
                           (~ok_t & ok_f, "FastText correct, TF-IDF wrong"),
                           (~ok_t & ~ok_f, "both wrong")):
            for text, true, pt, pf in zip(np.asarray(texts)[mask], y[mask], pred_t[mask], pred_f[mask]):
                errors.append({"setting": setting, "case": case, "text": text, "true_language": true,
                               "tfidf_prediction": pt, "fasttext_prediction": pf})

    deployed = pd.DataFrame(deployed_rows)
    deployed.round(6).to_csv(RESULTS_DIR / "comparison_deployed.csv", index=False)
    pd.DataFrame(errors).to_csv(RESULTS_DIR / "error_analysis.csv", index=False, encoding="utf-8")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / "error_analysis_summary.csv", index=False)

    oov = oov_rates(data.train["clean"], test, set(det_f.vectorizer.wv.key_to_index))
    oov.round(6).to_csv(RESULTS_DIR / "oov_analysis.csv", index=False)

    # ---------------------------------------------------------- report ----
    pd.set_option("display.width", 160)
    print("Per classifier, full test set (difference = FastText - TF-IDF):")
    print(comparison.pivot_table(index="classifier", columns="metric", values="difference")[list(METRICS)].round(4).to_string())
    print("\nMacro F1, full test set:")
    print(f1_full[["classifier", "tfidf", "fasttext", "difference"]].round(4).to_string(index=False))
    print("\nMacro F1 by number of words (difference = FastText - TF-IDF):")
    print(short_cmp[short_cmp["metric"] == "f1_macro"].pivot_table(index="classifier", columns="words",
                                                                     values="difference").round(4).to_string())
    print("\nDeployed models:")
    print(deployed.round(4).to_string(index=False))
    print("\nError analysis:")
    print(summary_df.to_string(index=False))
    print("\nWords never seen in training (share of test tokens):")
    print(oov.round(4).to_string(index=False))
    print(f"\nSaved comparison files to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
