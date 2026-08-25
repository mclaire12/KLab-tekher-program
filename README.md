# KLab-tekher-program

Coursework repository for the K Lab AI cohort. Each class/assignment gets its own dated notebook plus supporting files, organized under folders like `Assignment1/`, with earlier loose exercises kept at the repo root for reference.

## How the repo is organized

```
KLab-tekher-program/
├── Assignment1/          # a self-contained coursework unit
│   ├── data/
│   │   ├── raw/              # source data, untouched
│   │   └── processed/        # cleaned/derived output
│   ├── notebooks/            # one .ipynb per exercise/assignment
│   ├── reports/              # charts (.png) and written reports (.md)
│   ├── src/                  # shared helper code, if any
│   ├── requirements.txt      # dependencies for this unit
│   └── README.md             # details specific to this unit
├── Claireproj1.ipynb      # early in-class warm-up notebook
├── project 2/              # early in-class exercises
└── venv/                   # local virtual environment (not committed)
```

Each top-level unit (like `Assignment1/`) is meant to be runnable on its own: it has its own `data/`, `notebooks/`, `reports/`, and `requirements.txt`, with a local `README.md` documenting what's inside. Notebooks read from `data/raw/`, write cleaned data to `data/processed/`, and save charts/reports to `reports/` — so outputs are reproducible by re-running the notebook rather than hand-edited afterward. As new assignments are added, they follow this same pattern (either as a new folder or a new notebook inside an existing one).

Loose files at the repo root (`Claireproj1.ipynb`, `project 2/`) predate this structure and hold early in-class exercises — they're kept for reference but aren't organized the same way.

## Setup

Each unit is set up independently, from inside its own folder:

```
cd Assignment1
python -m venv ../venv          # shared venv one level up, if not already created
../venv/Scripts/Activate.ps1    # Windows PowerShell; macOS/Linux: source ../venv/bin/activate
pip install -r requirements.txt
jupyter notebook                # or open the folder in VS Code / Cursor
```

`venv/` is gitignored, so it's created locally rather than committed. `.env.example` files (where present) show any local environment variables a unit expects — copy to `.env` and fill in values if needed.

## What's inside Assignments

See [Assignment1/README.md](Assignment1/README.md) for the notebooks, datasets, and reports specific to that unit (currently two notebooks: Python/NumPy/Pandas fundamentals, and a data-wrangling/exploratory-analysis pass on a real dataset).
