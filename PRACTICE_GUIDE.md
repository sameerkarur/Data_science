# Practice Guide — How to Become Pro with This Repo

This hub is designed for **depth**, not skim-reading. Every subtopic targets **50+ programming problems** with full solutions, plus **30 interview Q&A** items.

---

## Quick start (5 minutes)

```bash
cd Data_science
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open any `practice.ipynb` → run **Setup** → solve questions top to bottom.

---

## How datasets load in notebooks

All generated notebooks include a **Setup** cell that:

1. Finds the repo root by looking for `datasets/shared/`
2. Sets `DATA_DIR = REPO_ROOT / 'datasets/shared'`
3. Loads the topic's CSV (when applicable)

```python
from pathlib import Path
import pandas as pd

def find_repo_root(start=None):
    p = Path(start or '.').resolve()
    for candidate in [p, *p.parents]:
        if (candidate / 'datasets' / 'shared').exists():
            return candidate
    return Path('.')

REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / 'datasets/shared'

df = pd.read_csv(DATA_DIR / 'AusApparalSales4thQrt2020.csv')
```

**Important:** Start Jupyter from the **repo root** (`Data_science/`), or open notebooks via the file browser from that folder. If `DATA_DIR` shows "none", you're in the wrong working directory — `cd` to the repo root and restart Jupyter.

### Available shared datasets

| File | Use in |
|------|--------|
| `AusApparalSales4thQrt2020.csv` | NumPy, Pandas, Matplotlib, Seaborn, sales EDA |
| `marketing_data.csv` | Marketing campaigns, advanced stats |
| `HR_comma_sep.csv` | ML classification, turnover, wrangling |
| `rolling_stones_spotify.csv` | Clustering / song cohorts |
| `loan_data.csv` | Lending Club DL project |
| `Home_loan_data_sample.csv` | Home loan DL (50k sample; full file local/gitignored) |

Some subtopics also copy CSVs locally (e.g. NumPy folder) — both work; prefer `datasets/shared/` for consistency.

---

## Study workflow (per subtopic)

| Step | File | Time |
|------|------|------|
| 1 | `basics.md` | 15–30 min read |
| 2 | `practice.md` or `practice.ipynb` | 3–6 hours (50 Q) |
| 3 | `solutions.ipynb` | Compare after each attempt |
| 4 | `interview_qa.md` | 30 min aloud |
| 5 | Re-do Q41–Q50 next day | 45 min |

**Rule:** No solution peeking until you have *something* running in the cell.

---

## Question counts by course

| Course | Subtopics | Questions | Notes |
|--------|-----------|-----------|-------|
| **1 — Python** | 5 | **250+** | Variables, control flow, structures, OOP, file I/O |
| **2 — Data Science** | 11 | **670+** | 8 generated topics × 50 + NumPy(45) + Pandas(70) + Matplotlib(45) + Seaborn(60) |
| **3 — ML** | 5 | **250+** | EDA, clustering, classification, imbalance, evaluation |
| **4 — DL** | 4 | **200+** | NN basics, Keras, preprocessing, metrics |
| **5 — GenAI** | 3 | **150+** | Prompts, apps, optimization |

**Grand total: 1,500+** practice problems when including the full NumPy/Pandas/viz banks.

---

## Where the big banks live

These already had full content (220+ questions) and are mapped into the LMS structure:

- `02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy/numpy_practice.md` (+ `.ipynb`)
- `02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas/pandas_practice.md`
- `02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib/matplotlib_practice.md`
- `02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn/seaborn_practice.md`

Use the same dataset setup — load `AusApparalSales4thQrt2020.csv` from the subfolder copy or `datasets/shared/`.

---

## Comprehensive Master Learning Guide

For complete mathematical derivations, theoretical deep dives across all courses, formulas, 100+ interview Q&As, and architecture breakdowns:
👉 **See [LEARNING_GUIDE.md](LEARNING_GUIDE.md)**

---

## Dual Project Portfolio (Version 1 & Version 2)

All course and capstone projects have been implemented across two distinct technical architectures:
- **Version 1 (LMS Submissions):** Located inside each respective course folder (`01_.../projects`, `06_.../`, `07_.../`).
- **Version 2 (Next-Gen Production Implementations):** Located in **[`projects_version2/`](projects_version2/)**, featuring alternative algorithmic paradigms, advanced design patterns, and explainability layers.


---

## Pro progression (8 weeks)

| Week | Focus |
|------|-------|
| 1 | Course 1 all 250 Q |
| 2 | NumPy + Pandas banks |
| 3 | Matplotlib + Seaborn + stats topics |
| 4 | Course 2 projects + remaining DS topics |
| 5 | Course 3 ML all topics + projects |
| 6 | Course 4 DL + TensorFlow venv (`.venv_dl`) |
| 7 | Course 5 GenAI + projects |
| 8 | Mock interviews: all `interview_qa.md` files |

---

## Regenerate practice files

If you edit question banks:

```bash
python scripts/practice_generator/run.py
```

---

## Deep learning environment

TensorFlow requires Python ≤3.13:

```bash
python3.13 -m venv .venv_dl
source .venv_dl/bin/activate
pip install tensorflow jupyter pandas scikit-learn
```

Use `.venv_dl` kernels for Course 4 notebooks.

---

## Tips for interviews

1. Explain **out loud** while coding Q31–Q50.
2. For each ML/DL topic, tie answers to **your projects** (Employee Turnover, Lending Club, etc.).
3. Time yourself: **15 min per medium problem**, **25 min for hard ones**.
4. Keep a errors log — patterns you miss twice go on flashcards.

You have the volume here to reach pro-level fluency; consistency beats cramming.
