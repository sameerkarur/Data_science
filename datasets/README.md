# Shared Datasets — AIML Practice Hub

All datasets used across courses. Clone the repo and use these paths in your notebooks.

| File | Size | Used in |
|------|------|---------|
| `AusApparalSales4thQrt2020.csv` | ~320 KB | Course 2 — NumPy, Pandas, Matplotlib, Seaborn practice + Sales Analysis project |
| `marketing_data.csv` | ~220 KB | Course 2 — Marketing Campaigns project |
| `rolling_stones_spotify.csv` | ~317 KB | Course 3 — Creating Cohorts of Songs project |
| `HR_comma_sep.csv` | ~553 KB | Course 3 — Employee Turnover project |
| `loan_data.csv` | ~734 KB | Course 4 — Lending Club project |
| `Home_loan_data_sample.csv` | ~30 MB | Course 4 — Home Loan practice (50k rows; full 307k rows available locally) |
| `Home_loan_data.csv` | ~158 MB | Full dataset — use Git LFS locally (`brew install git-lfs`) |
| `Data_Dictionary.csv` | ~12 KB | Home Loan feature definitions |

## Load in Python

```python
import pandas as pd

# From repo root
df = pd.read_csv('datasets/shared/AusApparalSales4thQrt2020.csv')

# Or from a course subtopic folder (copies also in NumPy/Pandas folders)
df = pd.read_csv('../../datasets/shared/marketing_data.csv')
```

## Git LFS

`Home_loan_data.csv` is stored with **Git LFS** (over GitHub's 100 MB limit). After cloning:

```bash
git lfs install
git lfs pull
```
