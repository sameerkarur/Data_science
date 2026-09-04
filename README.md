# AIML Practice Hub — From Python Basics to GenAI

**The complete open learning guide for IIT Kanpur AIML (and any AI/ML career path).**

Structured exactly like the program: **5 courses → subtopic folders → hands-on practice → 2 portfolio projects per course → interview Q&A.**

> Built by [Sameer Karur](https://github.com/sameerkarur) while completing the IITK Professional Certificate in Generative AI & Machine Learning.

---

## Open in browser

**[index.html](index.html)** — click any course/topic to jump to notebooks (works on GitHub Pages or locally).

---

## What's inside

| Course | Topics | Practice | Projects |
|--------|--------|----------|----------|
| **1 — Python Foundations** | 5 subtopics | basics + practice notebooks + interview Q&A | Expense Tracker, Task Manager |
| **2 — Applied Data Science** | 11 subtopics | **220+ questions** (NumPy/Pandas/Matplotlib/Seaborn) + stats/probability | Marketing Campaigns, Sales Analysis |
| **3 — Machine Learning** | 5 subtopics | Clustering, SMOTE, CV, evaluation | Song Cohorts, Employee Turnover |
| **4 — Deep Learning** | 4 subtopics | Keras/TensorFlow, AUC, sensitivity | Lending Club, Home Loan |
| **5 — Generative AI** | 3 subtopics | Prompt engineering, ChatGPT apps | Storytelling, PM Consultant |

Each subtopic folder contains:

```
subtopic/
├── basics.md           ← Syntax cheat sheet
├── practice.md         ← Question list
├── practice.ipynb      ← Type & run your code here
├── solutions.ipynb     ← Check answers
└── interview_qa.md     ← Interview prep Q&A
```

---

## Quick start

```bash
git clone https://github.com/sameerkarur/Data_science.git
cd Data_science
pip install -r requirements.txt

# Large Home Loan dataset (158 MB) uses Git LFS
git lfs install
git lfs pull

jupyter notebook
```

1. Open **[index.html](index.html)** or pick a course below  
2. Open `practice.ipynb` in any subtopic  
3. Read `basics.md` when stuck  
4. Review `interview_qa.md` before interviews  

---

## Folder structure

```
Data_science/
├── index.html                          ← Web navigation hub
├── README.md
├── requirements.txt
├── datasets/shared/                    ← ALL datasets (included in repo)
│   ├── AusApparalSales4thQrt2020.csv
│   ├── marketing_data.csv
│   ├── HR_comma_sep.csv
│   ├── rolling_stones_spotify.csv
│   ├── loan_data.csv
│   └── Home_loan_data.csv              ← Git LFS
│
├── Course_01_Python_Foundations/
│   ├── 01_variables_datatypes/ … 05_file_io_exceptions/
│   └── projects/ Personal_Expense_Tracker, Task_Manager
│
├── Course_02_Applied_Data_Science/
│   ├── 03_numpy/          ← 45 questions (full Python_Practice content)
│   ├── 08_pandas/           ← 70 questions
│   ├── 10_data_visualization/matplotlib/  ← 45 questions
│   ├── 10_data_visualization/seaborn/     ← 60 questions
│   └── projects/ Marketing_Campaigns, Sales_Analysis
│
├── Course_03_Machine_Learning/
│   └── projects/ Creating_Cohorts_of_Songs, Employee_Turnover_Analytics
│
├── Course_04_Deep_Learning/
│   └── projects/ Lending_Club_Loan_Analysis, Home_Loan_Data_Analysis
│
└── Course_05_Generative_AI/
    └── projects/ ChatGPT_Based_Storytelling, Virtual_PM_Consultant
```

---

## Datasets (all included)

See **[datasets/README.md](datasets/README.md)** for full list and load examples.

No external downloads needed — clone and practice.

---

## Interview prep path

| Goal | Focus on |
|------|----------|
| **Data Analyst** | Course 2 — Pandas, SQL-like ops, visualization, statistics |
| **ML Engineer** | Course 2 + 3 — feature engineering, SMOTE, model metrics |
| **Deep Learning** | Course 4 — Keras, imbalanced data, ROC/AUC, sensitivity |
| **AI Engineer / GenAI** | Course 5 — prompt engineering, LLM applications |

Every subtopic has **`interview_qa.md`**. Course 2 assessment-style questions are covered in statistics/probability topics.

---

## Related repo

Extended NumPy/Pandas-only practice: **[Python_Practice](https://github.com/sameerkarur/Python_Practice)** (same 220+ question style).

---

## Connect

- GitHub: [sameerkarur](https://github.com/sameerkarur)
- LinkedIn: [Sameer Karur](https://www.linkedin.com/in/sameer-karur-a5648224b/)

If this hub helps your AI/ML journey, **star the repo** — it helps others discover it.

**Happy learning.**
