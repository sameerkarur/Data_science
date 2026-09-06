# Intro to Data Science — Practice Questions (50+)

## How to practice

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


### Dataset for this topic: `sales`
Primary: AusApparalSales4thQrt2020.csv

```python
import pandas as pd
import numpy as np

sales_path = DATA_DIR / 'AusApparalSales4thQrt2020.csv'
df = pd.read_csv(sales_path)
print(df.shape)
df.head()
```


## Data Science mindset

**Q1.** Define data science in one sentence and list 5 steps of a DS project.

**Q2.** Load AusApparal sales CSV from DATA_DIR. Print shape and columns.

**Q3.** Explain population vs sample using HR dataset example.

**Q4.** Identify structured vs unstructured data examples in AIML.

**Q5.** List 3 types of analytics: descriptive, diagnostic, predictive, prescriptive — give one example each.

**Q6.** What is a feature vs label in ML?

**Q7.** Load marketing_data.csv; print dtypes and missing counts.

**Q8.** Compute basic KPIs: total sales, avg units from apparel dataset.

**Q9.** Create a simple data dictionary dict for sales columns.

**Q10.** Explain CRISP-DM phases briefly.

**Q11.** Compare batch vs real-time analytics.

**Q12.** What is ETL vs ELT?

**Q13.** Name 3 data quality dimensions (accuracy, completeness, timeliness).

**Q14.** Detect duplicates in first 1000 rows of HR data.

**Q15.** Sample 5 random rows with seed=42 for reproducibility.

**Q16.** Group sales by State; print top 3 states by revenue.

**Q17.** Plot histogram of Sales (matplotlib) — 20 bins.

**Q18.** Explain bias-variance tradeoff intuitively.

**Q19.** What is train/validation/test split purpose?

**Q20.** Define overfitting and one prevention method.

**Q21.** Compare supervised vs unsupervised learning.

**Q22.** What is a confounding variable?

**Q23.** Explain correlation does not imply causation with example.

**Q24.** List Python DS stack: NumPy, Pandas, Matplotlib, Seaborn, Sklearn.

**Q25.** Create Jupyter markdown checklist for project workflow.

**Q26.** Save cleaned sample to CSV in project folder (head 100 rows).

**Q27.** Compute sales per unit; handle division safely.

**Q28.** Identify outliers in Sales using IQR rule.

**Q29.** Cross-tab State vs Group counts.

**Q30.** Explain p-value in plain English.

**Q31.** What is A/B testing?

**Q32.** Define KPI for marketing campaign project.

**Q33.** Document assumptions when analyzing sample home loan data.

**Q34.** Compare CSV vs Parquet for analytics.

**Q35.** Use .describe(include='all') on marketing data.

**Q36.** Parse Date column to datetime in sales data.

**Q37.** Extract month from Date; group mean Sales by month.

**Q38.** Explain reproducibility with random_state.

**Q39.** Create function load_dataset(name) using DATA_DIR.

**Q40.** What is data leakage?

**Q41.** Name ethical concerns in HR attrition modeling.

**Q42.** Summarize loan_data target column distribution.

**Q43.** Compare mean vs median for skewed Sales.

**Q44.** Explain dimensionality in tabular data.

**Q45.** List 3 visualization types for numeric vs categorical.

**Q46.** Write pseudo-code for hypothesis test workflow.

**Q47.** Explain central limit theorem one line.

**Q48.** What is a cohort?

**Q49.** Define retention vs churn.

**Q50.** Build mini report dict with title, rows, findings keys.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.