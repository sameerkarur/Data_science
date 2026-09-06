"""Generate practice.md, practice.ipynb, solutions.ipynb, interview_qa.md for subtopics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Question:
    text: str
    solution: str
    section: str = "Core"
    starter: str = "# Your code here\n"


@dataclass
class InterviewQA:
    question: str
    answer: str
    section: str = "Conceptual"


@dataclass
class TopicSpec:
    rel_path: str
    title: str
    course: str
    dataset: str | None = None
    dataset_note: str = ""
    extra_imports: str = ""
    questions: list[Question] = field(default_factory=list)
    interview: list[InterviewQA] = field(default_factory=list)
    skip_if_exists: bool = False  # numpy/pandas already have full banks


SETUP_TEMPLATE = '''from pathlib import Path

def find_repo_root(start=None):
    p = Path(start or '.').resolve()
    for candidate in [p, *p.parents]:
        if (candidate / 'datasets' / 'shared').exists():
            return candidate
    return Path('.')

REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / 'datasets' / 'shared'
print(f"Repo root: {REPO_ROOT}")
print(f"Datasets : {DATA_DIR}")
print("Available CSVs:", sorted(p.name for p in DATA_DIR.glob('*.csv')) if DATA_DIR.exists() else "none")
'''

DATASET_LOAD = {
    "sales": '''import pandas as pd
import numpy as np

sales_path = DATA_DIR / 'AusApparalSales4thQrt2020.csv'
df = pd.read_csv(sales_path)
print(df.shape)
df.head()
''',
    "marketing": '''import pandas as pd

mkt_path = DATA_DIR / 'marketing_data.csv'
df = pd.read_csv(mkt_path)
print(df.shape)
df.head()
''',
    "hr": '''import pandas as pd

hr_path = DATA_DIR / 'HR_comma_sep.csv'
df = pd.read_csv(hr_path)
print(df.shape)
df.head()
''',
    "spotify": '''import pandas as pd

spotify_path = DATA_DIR / 'rolling_stones_spotify.csv'
df = pd.read_csv(spotify_path)
print(df.shape)
df.head()
''',
    "loan": '''import pandas as pd

loan_path = DATA_DIR / 'loan_data.csv'
df = pd.read_csv(loan_path)
print(df.shape)
df.head()
''',
    "home_loan": '''import pandas as pd

home_path = DATA_DIR / 'Home_loan_data_sample.csv'
df = pd.read_csv(home_path)
print(df.shape)
df.head()
''',
    "none": "# Pure Python topic — no dataset required for most exercises\npass\n",
}


HOW_TO_PRACTICE = """## How to practice (read this first)

1. **Start Jupyter from repo root** so dataset paths resolve:
   ```bash
   cd Data_science
   jupyter notebook
   ```
2. Open this notebook (`practice.ipynb`) in the subtopic folder.
3. Run the **Setup** cell — it finds `datasets/shared/` automatically.
4. Read `basics.md` for syntax, then solve **one question at a time** in the code cell below it.
5. Do **not** peek at `solutions.ipynb` until you have a working attempt (even if wrong).
6. After all questions, review `interview_qa.md` aloud as mock interview answers.
7. **Pro tip:** Re-solve Q41–Q50 from memory the next day without notes.

**Target:** 50 questions × 5 courses ≈ **1,250+ problems** in this repo (+ NumPy/Pandas/Matplotlib/Seaborn banks).
"""


def _md_escape(text: str) -> str:
    return text.replace("`", "\\`")


def write_practice_md(topic: TopicSpec, out_dir: Path) -> None:
    lines = [
        f"# {topic.title} — Practice Questions (50+)",
        "",
        HOW_TO_PRACTICE.replace("## How to practice (read this first)", "## How to practice"),
        "",
    ]
    if topic.dataset:
        lines.extend([
            f"### Dataset for this topic: `{topic.dataset}`",
            topic.dataset_note or "",
            "",
            "```python",
            DATASET_LOAD.get(topic.dataset, DATASET_LOAD["none"]).strip(),
            "```",
            "",
        ])

    current_section = None
    for i, q in enumerate(topic.questions, 1):
        if q.section != current_section:
            current_section = q.section
            lines.extend(["", f"## {current_section}", ""])
        lines.append(f"**Q{i}.** {q.text}")
        lines.append("")

    lines.extend([
        "---",
        f"**Total: {len(topic.questions)} questions**",
        "",
        "Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.",
    ])
    (out_dir / "practice.md").write_text("\n".join(lines), encoding="utf-8")


def _notebook_cell(cell_type: str, source: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True) if source else [],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def write_practice_ipynb(topic: TopicSpec, out_dir: Path) -> None:
    cells = [
        _notebook_cell("markdown", f"# {topic.title} — Practice\n\n{HOW_TO_PRACTICE}"),
        _notebook_cell("markdown", "## Setup & dataset loading"),
        _notebook_cell(
            "code",
            SETUP_TEMPLATE + "\n" + topic.extra_imports + "\n"
            + DATASET_LOAD.get(topic.dataset or "none", DATASET_LOAD["none"]),
        ),
    ]

    current_section = None
    for i, q in enumerate(topic.questions, 1):
        if q.section != current_section:
            current_section = q.section
            cells.append(_notebook_cell("markdown", f"## {current_section}"))
        cells.append(_notebook_cell("markdown", f"**Q{i}.** {q.text}"))
        cells.append(_notebook_cell("code", q.starter))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (out_dir / "practice.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


def write_solutions_ipynb(topic: TopicSpec, out_dir: Path) -> None:
    cells = [
        _notebook_cell("markdown", f"# {topic.title} — Solutions\n\nRun setup first, then each solution cell."),
        _notebook_cell("markdown", "## Setup"),
        _notebook_cell(
            "code",
            SETUP_TEMPLATE + "\n" + topic.extra_imports + "\n"
            + DATASET_LOAD.get(topic.dataset or "none", DATASET_LOAD["none"]),
        ),
    ]

    current_section = None
    for i, q in enumerate(topic.questions, 1):
        if q.section != current_section:
            current_section = q.section
            cells.append(_notebook_cell("markdown", f"## {current_section}"))
        cells.append(_notebook_cell("markdown", f"**Q{i}.** {q.text}"))
        cells.append(_notebook_cell("code", q.solution))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    (out_dir / "solutions.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


def write_interview_qa(topic: TopicSpec, out_dir: Path) -> None:
    lines = [
        f"# Interview Q&A — {topic.title}",
        "",
        f"> **{len(topic.interview)} questions** — read aloud, then explain without looking.",
        "",
    ]
    current = None
    for i, item in enumerate(topic.interview, 1):
        if item.section != current:
            current = item.section
            lines.extend(["", f"## {current}", ""])
        lines.extend([
            f"### Q{i}. {item.question}",
            "",
            item.answer,
            "",
        ])
    (out_dir / "interview_qa.md").write_text("\n".join(lines), encoding="utf-8")


def generate_topic(topic: TopicSpec) -> None:
    out_dir = REPO_ROOT / topic.rel_path
    if topic.skip_if_exists:
        return
    if not topic.questions:
        raise ValueError(f"No questions for {topic.rel_path}")
    if len(topic.questions) < 50:
        raise ValueError(f"{topic.rel_path} has only {len(topic.questions)} questions (need 50+)")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_practice_md(topic, out_dir)
    write_practice_ipynb(topic, out_dir)
    write_solutions_ipynb(topic, out_dir)
    write_interview_qa(topic, out_dir)
    print(f"OK  {topic.rel_path} ({len(topic.questions)} Q, {len(topic.interview)} interview)")


def generate_all(topics: Iterable[TopicSpec]) -> None:
    for topic in topics:
        generate_topic(topic)
