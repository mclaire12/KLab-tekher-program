"""Train and compare TF-IDF + traditional classifiers for language detection.

Pipeline:
    load CSVs -> validate -> clean text -> split train/validation (stratified)
    -> for each TF-IDF configuration: fit vectorizer on the training split
       -> for each classifier: fit, evaluate on validation AND held-out test
    -> select best (feature config, classifier) by VALIDATION macro F1
       (ties broken by validation macro F1 on 3-word snippets, then training time)
    -> short-text robustness: evaluate on test sentences truncated to 1/2/3/5 words
    -> save vectorizer, best model, metadata, results, confusion matrices

The test set (test.csv) is never used for model selection; it only reports
how the selected model generalises.

Usage:
    python training/train_language_detection.py
    python training/train_language_detection.py --configs tfidf_char_wb_1-3gram --classifiers LogisticRegression
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    BEST_MODEL_PATH, LABEL_COLUMN, LANGUAGES, METADATA_PATH, MODEL_DIR,
    RANDOM_SEED, RESULTS_DIR, TEST_CSV, TEXT_COLUMN, TRAIN_CSV, VECTORIZER_PATH,
)
from src.language_detection import (  # noqa: E402
    FEATURE_CONFIGS, build_classifiers, build_vectorizer, describe_vectorizer,
)
from src.preprocessing import PREPROCESSING_DESCRIPTION, clean_text  # noqa: E402

MODEL_VERSION = "part1-tfidf-baseline-v1"
MIN_SAMPLES_PER_CLASS = 10
TIE_BREAK_WORDS = 3
SHORT_TEXT_WORDS = (1, 2, 3, 5)

# Chart styling (validated reference palette; see README "Evaluation")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4


class DatasetError(Exception):
    pass


# ----------------------------------------------------------------- data ----
def load_dataset(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise DatasetError(
            f"{name} dataset not found: {path}\n"
            "Create it with:  python training/prepare_dataset.py\n"
            "or place your own CSV (columns: text,language) there. "
            "See data/language_detection/README.md"
        )
    df = pd.read_csv(path, encoding="utf-8")
    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise DatasetError(f"{path} is missing column(s): {sorted(missing)}. Found: {list(df.columns)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    unknown = sorted(set(df[LABEL_COLUMN]) - set(LANGUAGES))
    if unknown:
        raise DatasetError(f"{path} contains unsupported labels {unknown}. Allowed: {LANGUAGES}")

    n_before = len(df)
    df["clean"] = df[TEXT_COLUMN].fillna("").map(clean_text)
    df = df[df["clean"].str.len() > 0]
    if len(df) < n_before:
        print(f"  {name}: dropped {n_before - len(df)} empty rows after cleaning")

    counts = df[LABEL_COLUMN].value_counts()
    for lang in LANGUAGES:
        if counts.get(lang, 0) < MIN_SAMPLES_PER_CLASS:
            raise DatasetError(
                f"{name}: language '{lang}' has {counts.get(lang, 0)} samples "
                f"(minimum {MIN_SAMPLES_PER_CLASS})."
            )
    print(f"  {name}: {len(df)} rows  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return df


# -------------------------------------------------------------- metrics ----
def compute_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LANGUAGES, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LANGUAGES, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "f1_weighted": f1_weighted,
    }


def first_words(texts: pd.Series, n: int) -> pd.Series:
    """Truncate cleaned texts to their first n words (real text, just shorter)."""
    return texts.str.split().str[:n].str.join(" ")


# --------------------------------------------------------------- charts ----
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK_2, length=0)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURFACE)
    _draw_cm(ax, cm, title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _draw_cm(ax, cm: np.ndarray, title: str, show_ylabel: bool = True) -> None:
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
    _style_axes(ax)
    n = len(LANGUAGES)
    ax.set_xticks(range(n), [l.capitalize() for l in LANGUAGES], color=INK)
    ax.set_yticks(range(n), [l.capitalize() for l in LANGUAGES], color=INK)
    ax.set_xlabel("Predicted language", color=INK_2)
    if show_ylabel:
        ax.set_ylabel("True language", color=INK_2)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    # 2px surface gap between cells (interior lines only)
    ax.set_xticks(np.arange(0.5, n - 1), minor=True)
    ax.set_yticks(np.arange(0.5, n - 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dark_cell = row_pct[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.1%}", ha="center", va="center",
                    fontsize=9, color="#ffffff" if dark_cell else INK)


def plot_all_confusion_matrices(cms: dict[str, np.ndarray], config: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(cms), figsize=(4.6 * len(cms), 4.6), facecolor=SURFACE)
    for k, (ax, (name, cm)) in enumerate(zip(np.atleast_1d(axes), cms.items())):
        _draw_cm(ax, cm, name, show_ylabel=(k == 0))
    fig.suptitle(f"Test-set confusion matrices - {config} (counts and row %)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _legend_above(fig, ax, labels_colors):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for l, c in labels_colors]
    legend = fig.legend(handles=handles, frameon=False, ncol=len(handles), loc="upper left",
                        bbox_to_anchor=(0.01, 0.93))
    for text in legend.get_texts():
        text.set_color(INK)


def plot_model_comparison(results: pd.DataFrame, path: Path, tie_col: str) -> None:
    """Two panels on one shared y-scale: validation macro F1 on full sentences
    and on short snippets. One bar per classifier, grouped by TF-IDF config."""
    val = results[results["split"] == "validation"]
    configs = list(dict.fromkeys(val["features"]))
    classifiers = list(dict.fromkeys(val["classifier"]))
    width = 0.8 / len(classifiers)
    lo = max(0.0, np.floor(min(val["f1_macro"].min(), val[tie_col].min()) * 20) / 20)
    panels = [("f1_macro", "Full sentences"), (tie_col, f"First {TIE_BREAK_WORDS} words only")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE, sharey=True)
    for ax, (col, subtitle) in zip(axes, panels):
        _style_axes(ax)
        for k, clf in enumerate(classifiers):
            sub = val[val["classifier"] == clf].set_index("features").reindex(configs)
            x = np.arange(len(configs)) + (k - (len(classifiers) - 1) / 2) * width
            ax.bar(x, sub[col] - lo, bottom=lo, width=width - 0.02, color=SERIES[k % len(SERIES)])
        ax.set_xticks(range(len(configs)), [c.replace("tfidf_", "").replace("_", " ") for c in configs],
                      color=INK)
        ax.set_ylim(lo, 1.0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(subtitle, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel(f"Validation macro F1 (axis starts at {lo:.2f})", color=INK_2)
    fig.suptitle("Classifier comparison by TF-IDF representation", color=INK, fontsize=12, x=0.01, ha="left")
    _legend_above(fig, axes[0], list(zip(classifiers, SERIES)))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_short_text_robustness(short: pd.DataFrame, classifier: str, path: Path) -> None:
    """Test macro F1 vs. number of words, one line per TF-IDF config (fixed classifier)."""
    sub = short[short["classifier"] == classifier]
    configs = list(dict.fromkeys(sub["features"]))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURFACE)
    _style_axes(ax)
    starts = []
    for k, config in enumerate(configs):
        line = sub[sub["features"] == config].sort_values("words")
        color = SERIES[k % len(SERIES)]
        ax.plot(line["words"], line["f1_macro"], color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=config.replace("tfidf_", ""))
        starts.append([line.iloc[0]["f1_macro"], line.iloc[0]["words"], config.replace("tfidf_", "")])
    # Direct labels left of the first point (where the lines are most spread),
    # nudged downward so they never overlap.
    gap = max((sub["f1_macro"].max() - sub["f1_macro"].min()) * 0.06, 1e-3)
    starts.sort(reverse=True)
    for i in range(1, len(starts)):
        starts[i][0] = min(starts[i][0], starts[i - 1][0] - gap)
    for y, x, label in starts:
        ax.annotate(label, (x, y), xytext=(-12, 0), textcoords="offset points", va="center",
                    ha="right", fontsize=9, color=INK)
    legend = ax.legend(frameon=False, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)
    words = sorted(sub["words"].unique())
    ax.set_xticks(words, [str(w) for w in words], color=INK)
    ax.set_xlim(min(words) - 1.4, max(words) + 0.3)
    ax.set_xlabel("Words of each test sentence given to the model", color=INK_2)
    ax.set_ylabel("Test macro F1", color=INK_2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Short-text robustness - {classifier}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(FEATURE_CONFIGS), choices=list(FEATURE_CONFIGS))
    parser.add_argument("--classifiers", nargs="+", default=list(build_classifiers()))
    parser.add_argument("--val-size", type=float, default=0.2, help="fraction of train.csv used for validation")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print("1) Loading and validating dataset")
    try:
        train_full = load_dataset(TRAIN_CSV, "train")
        test = load_dataset(TEST_CSV, "test")
    except DatasetError as exc:
        sys.exit(f"\nERROR: {exc}")

    overlap = set(train_full["clean"]) & set(test["clean"])
    if overlap:
        print(f"  WARNING: {len(overlap)} texts appear in both train and test (possible leakage)")

    print(f"2) Stratified split of train.csv -> train / validation ({1 - args.val_size:.0%}/{args.val_size:.0%}), seed={args.seed}")
    train, val = train_test_split(
        train_full, test_size=args.val_size, stratify=train_full[LABEL_COLUMN], random_state=args.seed
    )
    print(f"  train={len(train)}  validation={len(val)}  test={len(test)}")

    val_short = first_words(val["clean"], TIE_BREAK_WORDS)
    rows, short_rows, fitted = [], [], {}
    for config in args.configs:
        print(f"\n3) TF-IDF: {config}  {FEATURE_CONFIGS[config]}")
        vectorizer = build_vectorizer(config)
        X_train = vectorizer.fit_transform(train["clean"])  # fit on training split ONLY
        X_val, X_test = vectorizer.transform(val["clean"]), vectorizer.transform(test["clean"])
        print(f"   vocabulary size: {len(vectorizer.vocabulary_)}")

        all_classifiers = build_classifiers(args.seed)
        for name in args.classifiers:
            if name not in all_classifiers:
                sys.exit(f"Unknown classifier '{name}'. Options: {list(all_classifiers)}")
            model = all_classifiers[name]
            start = time.perf_counter()
            model.fit(X_train, train[LABEL_COLUMN])
            train_seconds = time.perf_counter() - start

            for split, X, y in (("validation", X_val, val[LABEL_COLUMN]), ("test", X_test, test[LABEL_COLUMN])):
                y_pred = model.predict(X)
                metrics = compute_metrics(y, y_pred)
                rows.append({"features": config, "classifier": name, "split": split, **metrics,
                             "train_seconds": train_seconds})
                if split == "validation":
                    short_pred = model.predict(vectorizer.transform(val_short))
                    rows[-1][f"f1_macro_{TIE_BREAK_WORDS}words"] = compute_metrics(val[LABEL_COLUMN], short_pred)["f1_macro"]
                else:
                    fitted[(config, name)] = (vectorizer, model, y_pred)
            for n in SHORT_TEXT_WORDS:
                pred = model.predict(vectorizer.transform(first_words(test["clean"], n)))
                m = compute_metrics(test[LABEL_COLUMN], pred)
                short_rows.append({"features": config, "classifier": name, "words": n,
                                   "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
            v = rows[-2]
            print(f"   {name:<20} val acc={v['accuracy']:.4f}  val macro-F1={v['f1_macro']:.4f}  "
                  f"{TIE_BREAK_WORDS}-word val F1={v[f'f1_macro_{TIE_BREAK_WORDS}words']:.4f}  ({train_seconds:.1f}s)")

    results = pd.DataFrame(rows)
    val_results = results[results["split"] == "validation"]
    tie_col = f"f1_macro_{TIE_BREAK_WORDS}words"
    best = val_results.sort_values(["f1_macro", tie_col, "train_seconds"],
                                   ascending=[False, False, True]).iloc[0]
    best_config, best_name = best["features"], best["classifier"]
    best_vectorizer, best_model, best_test_pred = fitted[(best_config, best_name)]
    test_metrics = results[(results["split"] == "test") & (results["features"] == best_config)
                           & (results["classifier"] == best_name)].iloc[0]

    print(f"\n4) Model selection (validation macro F1; ties -> {TIE_BREAK_WORDS}-word validation F1 -> training time)")
    print(f"   BEST: {best_name} + {best_config}  validation macro-F1={best['f1_macro']:.4f}")
    print(f"   Held-out test: accuracy={test_metrics['accuracy']:.4f}  macro-F1={test_metrics['f1_macro']:.4f}")

    # ---------------------------------------------------------- save ----
    print("\n5) Saving models and results")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, BEST_MODEL_PATH)

    clf_dir = MODEL_DIR / "classifiers"
    clf_dir.mkdir(exist_ok=True)
    for (config, name), (_, model, _) in fitted.items():
        if config == best_config:
            joblib.dump(model, clf_dir / f"{name}.pkl")

    report = classification_report(test[LABEL_COLUMN], best_test_pred, labels=LANGUAGES, digits=4)
    cm = confusion_matrix(test[LABEL_COLUMN], best_test_pred, labels=LANGUAGES)

    metadata = {
        "task": "language_detection",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "representation": "TF-IDF (no word embeddings)",
        "feature_config": best_config,
        "vectorizer_settings": describe_vectorizer(best_vectorizer),
        "selected_classifier": best_name,
        "classifier_params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                              for k, v in best_model.get_params().items()},
        "supports_probabilities": hasattr(best_model, "predict_proba"),
        "supported_languages": LANGUAGES,
        "selection_metric": f"validation macro F1 (tie-break: validation macro F1 on {TIE_BREAK_WORDS}-word snippets, then training time)",
        "validation_metrics": {k: float(best[k]) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]},
        "test_metrics": {k: float(test_metrics[k]) for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]},
        "dataset": {"train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
                    "train_csv": str(TRAIN_CSV.relative_to(TRAIN_CSV.parents[2])),
                    "test_csv": str(TEST_CSV.relative_to(TEST_CSV.parents[2]))},
        "preprocessing": PREPROCESSING_DESCRIPTION,
        "random_seed": args.seed,
        "scikit_learn_version": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results.round(6).to_csv(RESULTS_DIR / "language_detection_results.csv", index=False)
    short = pd.DataFrame(short_rows)
    short.round(6).to_csv(RESULTS_DIR / "short_text_robustness.csv", index=False)
    (val_results.groupby("features")["f1_macro"].agg(["max", "mean"])
     .rename(columns={"max": "best_val_f1_macro", "mean": "mean_val_f1_macro"})
     .reindex(args.configs).round(6).to_csv(RESULTS_DIR / "tfidf_config_comparison.csv"))
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Best model: {best_name} + {best_config}\nEvaluated on held-out test set ({len(test)} rows)\n\n"
        f"{report}\nConfusion matrix (rows=true, cols=predicted, order={LANGUAGES}):\n{cm}\n",
        encoding="utf-8",
    )
    plot_confusion_matrix(cm, f"{best_name} + {best_config} - held-out test set", RESULTS_DIR / "confusion_matrix.png")
    plot_all_confusion_matrices(
        {name: confusion_matrix(test[LABEL_COLUMN], pred, labels=LANGUAGES)
         for (config, name), (_, _, pred) in fitted.items() if config == best_config},
        best_config, RESULTS_DIR / "confusion_matrices_all_classifiers.png",
    )
    if len(args.configs) * len(args.classifiers) > 1:
        plot_model_comparison(results, RESULTS_DIR / "model_comparison.png", tie_col)
    plot_short_text_robustness(short, best_name, RESULTS_DIR / "short_text_robustness.png")

    print(f"   {VECTORIZER_PATH}\n   {BEST_MODEL_PATH}\n   {METADATA_PATH}\n   {RESULTS_DIR}")
    print("\n6) Results (macro-averaged; sorted by validation F1)")
    table = results.pivot_table(index=["features", "classifier"], columns="split",
                                values=["accuracy", "f1_macro"]).round(4)
    table.columns = [f"{split}_{metric}" for metric, split in table.columns]
    print(table.sort_values("validation_f1_macro", ascending=False).to_string())
    print("\n7) Short-text robustness - test macro F1 on the first N words of each sentence")
    print(short.pivot_table(index=["features", "classifier"], columns="words", values="f1_macro").round(4).to_string())
    print(f"\nClassification report - best model on held-out test set:\n{report}")
    print(f"Confusion matrix (rows=true {LANGUAGES}):\n{cm}")


if __name__ == "__main__":
    main()
