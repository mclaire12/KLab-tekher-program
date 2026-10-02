"""Educational 2-D view of the trained FastText word vectors (PCA).

    FastText vectors (100 dimensions) -> PCA -> 2 dimensions -> scatter plot

Only words that were seen at least MIN_COUNT times in the training text are
plotted, so every point is a word the model actually learned. Requested words
that are missing or too rare are listed (with their count) in
results/embedding_words.csv instead of being plotted from subwords alone.

PCA keeps only part of the variance (printed on the chart), and closeness in
the plot does NOT prove that two words mean the same thing.

Outputs:
    results/fasttext_embedding_pca.png
    results/embedding_words.csv          which words were plotted / excluded and why
    results/embedding_neighbours.csv     5 nearest words (cosine) for each plotted word

Usage:
    python training/visualize_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from common import INK, INK_2, GRID, SERIES, SURFACE, legend_above, style_axes  # noqa: E402
from src.config import FASTTEXT_MODEL_DIR, RANDOM_SEED, RESULTS_DIR  # noqa: E402

MIN_COUNT = 5
# Requested demo words (greetings) + concept groups: (english, french, kinyarwanda)
REQUESTED = {"english": ["hello", "school"], "french": ["bonjour", "école"], "kinyarwanda": ["muraho", "ishuri"]}
GROUPS = [
    ("school", "école", "ishuri"),
    ("government", "gouvernement", "leta"),
    ("president", "président", "perezida"),
    ("children", "enfants", "abana"),
    ("country", "pays", "igihugu"),
    ("people", "gens", "abantu"),
]
LANG_COLORS = {"english": SERIES[0], "french": SERIES[1], "kinyarwanda": SERIES[2]}


def _place_labels(fig, ax, labels: list[str], xy: np.ndarray) -> None:
    """Put each label at the first of four positions around its point that does not
    overlap an already placed label or any point (checked in screen coordinates)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    points = [to_px(p) for p in xy]
    point_boxes = [(px - 6, py - 6, px + 6, py + 6) for px, py in points]
    taken: list[tuple[float, float, float, float]] = []
    candidates = [((7, 5), "left", "bottom"), ((7, -5), "left", "top"),
                  ((-7, 5), "right", "bottom"), ((-7, -5), "right", "top")]

    def overlaps(a, b) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    for i, (label, (x, y)) in enumerate(zip(labels, xy)):
        for k, (offset, ha, va) in enumerate(candidates):
            ann = ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                              ha=ha, va=va, fontsize=9, color=INK)
            bb = ann.get_window_extent(renderer)
            box = (bb.x0, bb.y0, bb.x1, bb.y1)
            others = [b for j, b in enumerate(point_boxes) if j != i]
            if not any(overlaps(box, b) for b in taken + others) or k == len(candidates) - 1:
                taken.append(box)
                break
            ann.remove()


def main() -> None:
    from gensim.models.fasttext import FastTextKeyedVectors

    path = FASTTEXT_MODEL_DIR / "fasttext.model"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run training/train_language_detection_embeddings.py first.")
    wv = FastTextKeyedVectors.load(str(path))

    def count(word: str) -> int:
        return int(wv.get_vecattr(word, "count")) if word in wv.key_to_index else 0

    candidates = {w: lang for lang, words in REQUESTED.items() for w in words}
    for group in GROUPS:
        candidates.update(zip(group, ("english", "french", "kinyarwanda")))
    rows, plotted = [], []
    for word, lang in candidates.items():
        c = count(word)
        ok = c >= MIN_COUNT
        rows.append({"word": word, "language": lang, "training_count": c, "plotted": ok,
                     "reason": "learned vector" if ok else
                     ("never seen in training text" if c == 0 else f"seen only {c}x (< {MIN_COUNT})")})
        if ok:
            plotted.append((word, lang))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "embedding_words.csv", index=False, encoding="utf-8")

    words = [w for w, _ in plotted]
    vectors = np.stack([wv[w] for w in words])
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    xy = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
    style_axes(ax)
    for (word, lang), (x, y) in zip(plotted, xy):
        ax.scatter(x, y, s=70, color=LANG_COLORS[lang], edgecolor=SURFACE, linewidth=2, zorder=3)
    _place_labels(fig, ax, [w for w, _ in plotted], xy)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("PCA component 1", color=INK_2)
    ax.set_ylabel("PCA component 2", color=INK_2)
    fig.suptitle(f"FastText word vectors in 2-D (PCA keeps {explained:.0%} of the variance)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    legend_above(fig, [(lang.capitalize(), c) for lang, c in LANG_COLORS.items()], y=0.95)
    excluded = [r["word"] for r in rows if not r["plotted"]]
    if excluded:
        fig.text(0.01, 0.01, "Not plotted (missing or too rare in the training text): " + ", ".join(excluded),
                 fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(RESULTS_DIR / "fasttext_embedding_pca.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    neighbours = []
    for word, lang in plotted:
        for rank, (other, sim) in enumerate(wv.most_similar(word, topn=5), 1):
            neighbours.append({"word": word, "language": lang, "rank": rank, "neighbour": other,
                               "cosine_similarity": round(float(sim), 4)})
    pd.DataFrame(neighbours).to_csv(RESULTS_DIR / "embedding_neighbours.csv", index=False, encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nPCA explained variance (2 components): {explained:.1%}")
    nb = pd.DataFrame(neighbours)
    print(nb.groupby("word", sort=False)["neighbour"].apply(", ".join).to_string())
    print(f"\nSaved {RESULTS_DIR / 'fasttext_embedding_pca.png'}")


if __name__ == "__main__":
    main()
