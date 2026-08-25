# K Lab — AI Cohort Assignments

Python-for-AI coursework: fundamentals, NumPy/Pandas, and a small data-wrangling project.

## Setup

```
cd Assignment1
python -m venv ../venv          # if not already created
../venv/Scripts/Activate.ps1    # Windows PowerShell; macOS/Linux: source ../venv/bin/activate
pip install -r requirements.txt
jupyter notebook                # or open the folder in VS Code / Cursor
```

Copy `.env.example` to `.env` if you need local environment variables (none are required by the current notebooks).

## Repository structure

```
Assignment1/
├── data/
│   ├── raw/            # source data, untouched (titanic.csv)
│   └── processed/      # cleaned/feature-engineered output (titanic_clean.csv)
├── notebooks/
│   ├── assignment1.ipynb   # Day 1: Python fundamentals + NumPy/Pandas + one chart
│   └── assignment2.ipynb   # Assignment 2: data wrangling and exploratory analysis
├── reports/             # charts (.png) and written reports (.md)
├── src/                 # (reserved for shared helper code, currently empty)
└── requirements.txt
```

## Assignment 1 — Python for AI Fundamentals

`notebooks/assignment1.ipynb` covers Python basics applied to an AI context: variables and types, casting, f-strings, conditionals, loops, all four core collections, comprehensions, three utility functions (`normalise`, `summarise_scores`, `safe_divide`), a NumPy/Pandas warm-up, and one Matplotlib chart with a stated finding.

- **Output chart:** `reports/day01_chart.png`
- **Reflection:** `reports/day01_reflection.md`

## Assignment 2 — Data Wrangling and Exploratory Analysis

`notebooks/assignment2.ipynb` takes a real dataset through schema reporting, cleaning, feature engineering (`groupby` + merge, pivot), a vectorized NumPy computation, and two exploratory charts.

**Dataset**
- **Name:** Titanic passenger manifest (`data/raw/titanic.csv`, 891 rows)
- **Source:** https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv (seaborn's re-hosting of the Kaggle "Titanic: Machine Learning from Disaster" passenger list)
- **License:** `seaborn-data` repo is BSD-3-Clause; underlying passenger data is the public Titanic dataset used freely for education/competition
- **Why chosen:** meets the size/shape requirement and has real missingness plus a natural numeric/categorical mix (see the notebook's Task 1 markdown cell for the full rationale)

**Outputs**
- **Cleaned dataset:** `data/processed/titanic_clean.csv`
- **Charts:** `reports/a2_chart1.png` (survival rate by class and sex), `reports/a2_chart2.png` (age distribution by survival outcome)
- **Report + reflection:** `reports/weekend-a2-report.md`
