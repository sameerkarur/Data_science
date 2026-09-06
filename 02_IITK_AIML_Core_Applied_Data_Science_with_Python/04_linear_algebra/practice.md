# Linear Algebra for DS — Practice Questions (50+)

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


## Section 1: Concepts

**Q1.** Define Linear Algebra and give AIML example.

**Q2.** Load AusApparalSales4thQrt2020.csv and describe `Sales`.

**Q3.** Compute mean, median, std of `Sales`.

**Q4.** Explain when to use Linear Algebra test vs alternative.

**Q5.** Simulate 1000 draws from normal; plot histogram.

**Q6.** Calculate 95% CI for mean of `Sales`.

**Q7.** State null and alternative hypothesis for Linear Algebra.

**Q8.** What alpha=0.05 means in Linear Algebra.

**Q9.** Type I vs Type II error in Linear Algebra.

**Q10.** Check skewness of `Sales`.


## Section 2: Computation

**Q11.** Define Linear Algebra and give AIML example.

**Q12.** Load AusApparalSales4thQrt2020.csv and describe `Sales`.

**Q13.** Compute mean, median, std of `Sales`.

**Q14.** Explain when to use Linear Algebra test vs alternative.

**Q15.** Simulate 1000 draws from normal; plot histogram.

**Q16.** Calculate 95% CI for mean of `Sales`.

**Q17.** State null and alternative hypothesis for Linear Algebra.

**Q18.** What alpha=0.05 means in Linear Algebra.

**Q19.** Type I vs Type II error in Linear Algebra.

**Q20.** Check skewness of `Sales`.


## Section 3: Interpretation

**Q21.** Define Linear Algebra and give AIML example.

**Q22.** Load AusApparalSales4thQrt2020.csv and describe `Sales`.

**Q23.** Compute mean, median, std of `Sales`.

**Q24.** Explain when to use Linear Algebra test vs alternative.

**Q25.** Simulate 1000 draws from normal; plot histogram.

**Q26.** Calculate 95% CI for mean of `Sales`.

**Q27.** State null and alternative hypothesis for Linear Algebra.

**Q28.** What alpha=0.05 means in Linear Algebra.

**Q29.** Type I vs Type II error in Linear Algebra.

**Q30.** Check skewness of `Sales`.


## Section 4: Applied

**Q31.** Define Linear Algebra and give AIML example.

**Q32.** Load AusApparalSales4thQrt2020.csv and describe `Sales`.

**Q33.** Compute mean, median, std of `Sales`.

**Q34.** Explain when to use Linear Algebra test vs alternative.

**Q35.** Simulate 1000 draws from normal; plot histogram.

**Q36.** Calculate 95% CI for mean of `Sales`.

**Q37.** State null and alternative hypothesis for Linear Algebra.

**Q38.** What alpha=0.05 means in Linear Algebra.

**Q39.** Type I vs Type II error in Linear Algebra.

**Q40.** Check skewness of `Sales`.


## Section 5: Challenge

**Q41.** Define Linear Algebra and give AIML example.

**Q42.** Load AusApparalSales4thQrt2020.csv and describe `Sales`.

**Q43.** Compute mean, median, std of `Sales`.

**Q44.** Explain when to use Linear Algebra test vs alternative.

**Q45.** Simulate 1000 draws from normal; plot histogram.

**Q46.** Calculate 95% CI for mean of `Sales`.

**Q47.** State null and alternative hypothesis for Linear Algebra.

**Q48.** What alpha=0.05 means in Linear Algebra.

**Q49.** Type I vs Type II error in Linear Algebra.

**Q50.** Check skewness of `Sales`.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.