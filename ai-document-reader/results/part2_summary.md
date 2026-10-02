# Part 2 Results Summary — TF-IDF vs FastText

All numbers come from `training/train_language_detection_tfidf.py`,
`training/train_language_detection_embeddings.py` and
`training/compare_representations.py`, run on 2026-10-02 with seed 42.
Both models were trained and tested on the **identical** split (fingerprint
`train 9119e708d3ce726b · validation 298416300f105cd3 · test 918d6f2a645d8230`).

## Headline table — deployed models, held-out test set (5,994 sentences)

| | TF-IDF (Part 1) | FastText (Part 2) | Difference |
|---|---:|---:|---:|
| Representation | char 2–5-grams, 99,533 sparse dims | skip-gram, subwords 2–5, mean of 100-dim word vectors | |
| Classifier (selected by validation) | LinearSVC (calibrated) | KNN (k = 5) | |
| Accuracy | 0.9997 | **1.0000** | +0.0003 |
| Precision (macro) | 0.9997 | **1.0000** | +0.0003 |
| Recall (macro) | 0.9997 | **1.0000** | +0.0003 |
| Macro F1 | 0.9997 | **1.0000** | +0.0003 |
| Errors on the test set | 2 | 0 | −2 |
| Macro F1, first 2 words only | 0.9581 | **0.9708** | +0.0127 |
| Macro F1, first word only | 0.9043 | **0.9299** | +0.0256 |

## Same classifier, different representation (test macro F1)

| Classifier | TF-IDF full | FastText full | Δ | TF-IDF 1 word | FastText 1 word | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9997 | 0.9998 | +0.0002 | 0.8901 | 0.9269 | +0.0368 |
| LinearSVC | 0.9997 | 1.0000 | +0.0003 | 0.9055 | 0.9308 | +0.0254 |
| Random Forest | 0.9995 | 0.9998 | +0.0003 | 0.8453 | 0.9171 | +0.0718 |
| KNN | 0.9960 | 1.0000 | +0.0040 | 0.9277 | 0.9299 | +0.0022 |

## What improved?

- **Short texts.** FastText is better for every classifier at 1, 2, 3 and 5
  words. The largest gains are on single words (up to +0.072 macro F1 for
  Random Forest; +0.026 for the deployed models).
- **Kinyarwanda gains most.** On single words, Kinyarwanda F1 rose from 0.922
  to 0.947, and 114 of the 227 single words that only FastText got right are
  Kinyarwanda (e.g. *leta*, *pasiteri*, *moto*). 15.1% of Kinyarwanda test
  words never appear in the training text (English 7.4%, French 8.5%);
  FastText still builds vectors for them from their character n-grams.
- **Every classifier benefits**, and the weakest ones on sparse TF-IDF
  (Random Forest, KNN) improve the most: dense 100-dimensional vectors suit them better
  than 99,533 sparse dimensions.
- **Both TF-IDF test errors were fixed**: a code-switched Kinyarwanda sentence
  (*"Ni ukwica intellectuellement generation yose"*) and a French sentence full of
  English names (*"Jennifer Aniston et Steve Carrell dans la série The Morning Show"*).

## What did not improve?

- **Full sentences: practically no difference.** TF-IDF was already at 0.9997;
  the gain is 2 sentences out of 5,994, too small to claim one representation is
  generally better on full sentences.
- **FastText is not better on every example.** On single words, TF-IDF was right
  and FastText wrong in 72 cases (FastText right / TF-IDF wrong: 227 cases).
  Many are names: *trump*, *donald* and *jacob* were labelled French, because
  embeddings learn from the contexts in the training text, and the French news
  corpus frequently mentions these people.
- **Cost.** Building the FastText representation took 78.8 s versus 2.7 s for
  TF-IDF, and the saved FastText vectors are 30.2 MB (TF-IDF vectorizer 3.4 MB).
- **Confidence is coarser.** The selected KNN's probability is the share of the 5
  nearest training sentences that agree (0, 20, 40, 60, 80 or 100%).

## Why might this have happened?

1. **Ceiling effect.** Full sentences already contain many clues for three very
   different languages, so both representations are almost perfect.
2. **Subwords help with short and unseen words.** The ablation without subwords
   (plain word vectors) drops to 0.8947 single-word F1 with LinearSVC, below
   FastText with subwords (0.9308) and even below TF-IDF character n-grams (0.9055).
   The subword information, not the "embedding" alone, explains most of the gain.
3. **Dense vectors carry learned similarity.** Words that occur in similar
   contexts, or share spelling, get similar vectors, so a short input is mapped
   near other words of its language even if it was never seen exactly.
4. **Embeddings also learn topic/domain.** Because English/French come from news and
   Kinyarwanda from web text, word vectors reflect who is mentioned where, not only
   the language, which explains errors on names.

FastText is **not** contextual: each word has one fixed vector regardless of the
sentence it is in.
