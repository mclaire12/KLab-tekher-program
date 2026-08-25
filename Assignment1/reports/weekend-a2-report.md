# Assignment 2 Report: Who Survived the Titanic?

## Dataset

The [Titanic passenger manifest](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv) (891 rows, distributed with `seaborn`, originally from the Kaggle "Titanic: Machine Learning from Disaster" competition, released for educational/non-commercial use). Full schema and license notes are in `notebooks/assignment2.ipynb` (Task 1).

## Question explored

Beyond the well-known "women and children first" headline, I wanted to know: **did class change the picture?** Specifically, did women and men across First, Second, and Third class all benefit from "women and children first" equally, or did class create a second, compounding disadvantage on top of sex?

## What I found

**Chart 1 — survival by class and sex** (`a2_chart1.png`): Women in First and Second class survived over 90% of the time (96.7% and 92.1%). Women in Third class survived only 50% of the time — still far better than any group of men, but roughly half the rate of women one or two classes up. Men fared badly everywhere (13.5%–36.9%), but even for men, First class nearly tripled the survival odds of Third class. So class did not just add a separate effect on top of sex — it interacted with it, and Third-class women paid a specific, visible penalty that Second- and First-class women mostly avoided.

**Chart 2 — survival by age** (`a2_chart2.png`): Children under 10 survived far more often than not, consistent with lifeboat priority for children. But the worst outcomes in the entire dataset belonged to passengers in their 20s, not the elderly — likely because that age band is dominated by Third-class and steerage men, the group Chart 1 shows had the lowest survival odds overall. Age and class/sex are entangled rather than independent effects.

**Supporting numbers** (Task 3 `groupby`/merge): mean fare by class was $13.68 (Third), $20.66 (Second), and $84.19 (First) — over a 6x gap between First and Third — while mean age rose from 24.8 (Third) to 38.2 (First), showing First-class passengers were both older and dramatically wealthier, which likely correlates with cabin location and proximity to lifeboats, a mechanism this dataset can't directly test.

## Limitation

After dropping the mostly-empty `deck` column and the redundant `alive` column (Task 2), 118 rows became exact duplicates of another row on every remaining field. This dataset has no passenger-ID or name column, so I can't confirm whether these are genuinely different people who happen to share a class/sex/age/fare/family-size profile (very plausible — e.g. several unrelated 40-year-old First-class men traveling alone), or true duplicated records. I chose not to drop them, since doing so risks deleting real, distinct passengers, but this means the survival-rate figures above could be mildly skewed if any of the 118 are actually re-counted people rather than coincidental look-alikes.

## Charts

- `reports/a2_chart1.png` — survival rate by class and sex
- `reports/a2_chart2.png` — age distribution split by survival outcome

## Reflection

**Which transform took the longest to get right, and why?**

The `groupby` + merge-back in Task 3 looked trivial on paper but took the longest in practice. My first attempt merged the per-class stats back on `pclass` (the integer column) while `class_stats` was grouped by `class` (the ordered category column) — pandas raised a merge key dtype error because I had one as `int64` and was trying to join it against a `category`. Once I re-grouped consistently on `class` end-to-end, it worked immediately. It was a good reminder to check `.dtype` on both sides of a join before assuming a name match is enough — advice this assignment's own "If You Get Stuck" section calls out directly.

**What would I do differently with another dataset this weekend?**

I'd check for a stable row identifier (or engineer one, like combining columns into a synthetic key) before doing any column-dropping, specifically so a duplicate-rows check is actually meaningful afterward rather than ambiguous the way it turned out here. I'd also compute the group-level numbers I plan to put in chart titles *before* writing the titles — I initially guessed at which class/sex pairing would show the "widest gap" from general Titanic knowledge, and the real computed numbers (Second class had the widest gap, not Third) contradicted my assumption.
