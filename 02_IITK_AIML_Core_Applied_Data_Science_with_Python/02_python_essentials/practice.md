# Python Essentials for DS — Practice Questions (50+)

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


```python
import pandas as pd
import numpy as np

sales_path = DATA_DIR / 'AusApparalSales4thQrt2020.csv'
df = pd.read_csv(sales_path)
print(df.shape)
df.head()
```


## Python for DS

**Q1.** Import numpy, pandas, matplotlib; print versions.

**Q2.** Create ndarray 1..12 reshape (3,4).

**Q3.** Vectorized: multiply sales array by 1.1.

**Q4.** Apply function to column with .apply(lambda).

**Q5.** List comprehension filter Sales > 30000.

**Q6.** Dict comprehension state → total sales.

**Q7.** Use pathlib to list all CSV in DATA_DIR.

**Q8.** Read CSV with usecols to load only Sales, State.

**Q9.** Timing: %%timeit sum Python list vs numpy.

**Q10.** Use f-string format dataframe shape in message.

**Q11.** Try/except reading missing file gracefully.

**Q12.** Assert no negative Sales values.

**Q13.** Use typing hint def top_n(series, n: int) -> pd.Series.

**Q14.** Lambda in sort: sort states by name length.

**Q15.** Map values: Group M→Men, W→Women if present.

**Q16.** Filter dataframe with boolean mask.

**Q17.** Chain methods: groupby mean sort descending head.

**Q18.** Use .pipe for custom function in chain.

**Q19.** Set pandas display options max 5 columns.

**Q20.** Reset display options.

**Q21.** Use random seed in numpy choice sample.

**Q22.** Bin Sales into quartiles with qcut.

**Q23.** Cut Sales into custom bins.

**Q24.** Merge two small summary dataframes.

**Q25.** Pivot table mean Sales by State and Group.

**Q26.** Melt wide to long (create wide first).

**Q27.** Use assign to add column inline.

**Q28.** Explode list column demo.

**Q29.** String accessor: lower state names.

**Q30.** Extract regex pattern from column if string.

**Q31.** Replace values with .replace dict.

**Q32.** Drop duplicates subset State, Date.

**Q33.** Fillna median for numeric column demo.

**Q34.** Clip Sales to percentile 1-99.

**Q35.** Use agg multiple functions.

**Q36.** Named aggregation in groupby.

**Q37.** Rolling mean window 7 on daily aggregated sales.

**Q38.** Shift for lag feature demo.

**Q39.** Cumulative sum sales over time.

**Q40.** Save figure to project folder.

**Q41.** Use os.environ.get for optional config.

**Q42.** Write reusable load_sales() function.

**Q43.** Profile memory usage .memory_usage(deep=True).

**Q44.** Convert column to category dtype.

**Q45.** Use pd.to_numeric errors='coerce'.

**Q46.** Build pytest-style assert for mean > 0.

**Q47.** Docstring module-level practice function.

**Q48.** Use __name__ == '__main__' guard pattern.

**Q49.** Create requirements check list programmatically.

**Q50.** Use pd.read_csv with parse_dates and infer_datetime_format.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.