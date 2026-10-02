# Language Detection Dataset

The training script expects two CSV files in this folder:

```
data/language_detection/
    train.csv
    test.csv
```

## Format

UTF-8 CSV with a header and two columns:

| column     | description                                          |
|------------|------------------------------------------------------|
| `text`     | a sentence or paragraph                              |
| `language` | one of `english`, `french`, `kinyarwanda` (lowercase) |

```csv
text,language
"Hello, how are you today?",english
"Bonjour, comment allez-vous?",french
"Muraho, amakuru yawe?",kinyarwanda
```

## Option A — build it from real public corpora (used in this project)

```bash
python training/prepare_dataset.py
```

This downloads three corpora of real sentences from the
**Leipzig Corpora Collection** (Universität Leipzig, licence CC BY 4.0,
<https://wortschatz.uni-leipzig.de/en/download>):

| language    | corpus                   | domain               |
|-------------|--------------------------|----------------------|
| english     | `eng_news_2020_10K`      | news, 2020           |
| french      | `fra_news_2020_10K`      | news, 2020           |
| kinyarwanda | `kin_community_2017_10K` | web/community, 2017  |

The script removes sentences with fewer than 3 words and duplicates, balances
the three classes to the same size, and makes a stratified 80/20 split
(seed 42) into `train.csv` and `test.csv`. Raw archives are cached in
`raw/` (ignored by git).

Citation: D. Goldhahn, T. Eckart, U. Quasthoff (2012). *Building Large
Monolingual Dictionaries at the Leipzig Corpora Collection: From 100 to 200
Languages.* LREC 2012.

## Option B — use your own data

Place your own `train.csv` and `test.csv` (format above) in this folder.
If either file is missing or invalid, `training/train_language_detection.py`
stops with an explicit error message; it never invents data.
