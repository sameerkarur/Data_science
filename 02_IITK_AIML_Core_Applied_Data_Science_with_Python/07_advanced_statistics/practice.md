# Advanced Statistics — Practice Questions (50+)

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


### Dataset for this topic: `marketing`


```python
import pandas as pd

mkt_path = DATA_DIR / 'marketing_data.csv'
df = pd.read_csv(mkt_path)
print(df.shape)
df.head()
```


## Section 1: Concepts

**Q1.** Define Advanced Stats and give AIML example.

**Q2.** Load marketing_data.csv and describe `Recency`.

**Q3.** Compute mean, median, std of `Recency`.

**Q4.** Explain when to use Advanced Stats test vs alternative.

**Q5.** Simulate 1000 draws from normal; plot histogram.

**Q6.** Calculate 95% CI for mean of `Recency`.

**Q7.** State null and alternative hypothesis for Advanced Stats.

**Q8.** What alpha=0.05 means in Advanced Stats.

**Q9.** Type I vs Type II error in Advanced Stats.

**Q10.** Check skewness of `Recency`.


## Section 2: Computation

**Q11.** Define Advanced Stats and give AIML example.

**Q12.** Load marketing_data.csv and describe `Recency`.

**Q13.** Compute mean, median, std of `Recency`.

**Q14.** Explain when to use Advanced Stats test vs alternative.

**Q15.** Simulate 1000 draws from normal; plot histogram.

**Q16.** Calculate 95% CI for mean of `Recency`.

**Q17.** State null and alternative hypothesis for Advanced Stats.

**Q18.** What alpha=0.05 means in Advanced Stats.

**Q19.** Type I vs Type II error in Advanced Stats.

**Q20.** Check skewness of `Recency`.


## Section 3: Interpretation

**Q21.** Define Advanced Stats and give AIML example.

**Q22.** Load marketing_data.csv and describe `Recency`.

**Q23.** Compute mean, median, std of `Recency`.

**Q24.** Explain when to use Advanced Stats test vs alternative.

**Q25.** Simulate 1000 draws from normal; plot histogram.

**Q26.** Calculate 95% CI for mean of `Recency`.

**Q27.** State null and alternative hypothesis for Advanced Stats.

**Q28.** What alpha=0.05 means in Advanced Stats.

**Q29.** Type I vs Type II error in Advanced Stats.

**Q30.** Check skewness of `Recency`.


## Section 4: Applied

**Q31.** Define Advanced Stats and give AIML example.

**Q32.** Load marketing_data.csv and describe `Recency`.

**Q33.** Compute mean, median, std of `Recency`.

**Q34.** Explain when to use Advanced Stats test vs alternative.

**Q35.** Simulate 1000 draws from normal; plot histogram.

**Q36.** Calculate 95% CI for mean of `Recency`.

**Q37.** State null and alternative hypothesis for Advanced Stats.

**Q38.** What alpha=0.05 means in Advanced Stats.

**Q39.** Type I vs Type II error in Advanced Stats.

**Q40.** Check skewness of `Recency`.


## Section 5: Challenge

**Q41.** Define Advanced Stats and give AIML example.

**Q42.** Load marketing_data.csv and describe `Recency`.

**Q43.** Compute mean, median, std of `Recency`.

**Q44.** Explain when to use Advanced Stats test vs alternative.

**Q45.** Simulate 1000 draws from normal; plot histogram.

**Q46.** Calculate 95% CI for mean of `Recency`.

**Q47.** State null and alternative hypothesis for Advanced Stats.

**Q48.** What alpha=0.05 means in Advanced Stats.

**Q49.** Type I vs Type II error in Advanced Stats.

**Q50.** Check skewness of `Recency`.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.