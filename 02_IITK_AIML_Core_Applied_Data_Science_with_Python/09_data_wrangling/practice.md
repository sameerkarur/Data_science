# Data Wrangling — Practice Questions (50+)

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


### Dataset for this topic: `hr`


```python
import pandas as pd

hr_path = DATA_DIR / 'HR_comma_sep.csv'
df = pd.read_csv(hr_path)
print(df.shape)
df.head()
```


## Section 1: Concepts

**Q1.** Define Wrangling and give AIML example.

**Q2.** Load HR_comma_sep.csv and describe `satisfaction_level`.

**Q3.** Compute mean, median, std of `satisfaction_level`.

**Q4.** Explain when to use Wrangling test vs alternative.

**Q5.** Simulate 1000 draws from normal; plot histogram.

**Q6.** Calculate 95% CI for mean of `satisfaction_level`.

**Q7.** State null and alternative hypothesis for Wrangling.

**Q8.** What alpha=0.05 means in Wrangling.

**Q9.** Type I vs Type II error in Wrangling.

**Q10.** Check skewness of `satisfaction_level`.


## Section 2: Computation

**Q11.** Define Wrangling and give AIML example.

**Q12.** Load HR_comma_sep.csv and describe `satisfaction_level`.

**Q13.** Compute mean, median, std of `satisfaction_level`.

**Q14.** Explain when to use Wrangling test vs alternative.

**Q15.** Simulate 1000 draws from normal; plot histogram.

**Q16.** Calculate 95% CI for mean of `satisfaction_level`.

**Q17.** State null and alternative hypothesis for Wrangling.

**Q18.** What alpha=0.05 means in Wrangling.

**Q19.** Type I vs Type II error in Wrangling.

**Q20.** Check skewness of `satisfaction_level`.


## Section 3: Interpretation

**Q21.** Define Wrangling and give AIML example.

**Q22.** Load HR_comma_sep.csv and describe `satisfaction_level`.

**Q23.** Compute mean, median, std of `satisfaction_level`.

**Q24.** Explain when to use Wrangling test vs alternative.

**Q25.** Simulate 1000 draws from normal; plot histogram.

**Q26.** Calculate 95% CI for mean of `satisfaction_level`.

**Q27.** State null and alternative hypothesis for Wrangling.

**Q28.** What alpha=0.05 means in Wrangling.

**Q29.** Type I vs Type II error in Wrangling.

**Q30.** Check skewness of `satisfaction_level`.


## Section 4: Applied

**Q31.** Define Wrangling and give AIML example.

**Q32.** Load HR_comma_sep.csv and describe `satisfaction_level`.

**Q33.** Compute mean, median, std of `satisfaction_level`.

**Q34.** Explain when to use Wrangling test vs alternative.

**Q35.** Simulate 1000 draws from normal; plot histogram.

**Q36.** Calculate 95% CI for mean of `satisfaction_level`.

**Q37.** State null and alternative hypothesis for Wrangling.

**Q38.** What alpha=0.05 means in Wrangling.

**Q39.** Type I vs Type II error in Wrangling.

**Q40.** Check skewness of `satisfaction_level`.


## Section 5: Challenge

**Q41.** Define Wrangling and give AIML example.

**Q42.** Load HR_comma_sep.csv and describe `satisfaction_level`.

**Q43.** Compute mean, median, std of `satisfaction_level`.

**Q44.** Explain when to use Wrangling test vs alternative.

**Q45.** Simulate 1000 draws from normal; plot histogram.

**Q46.** Calculate 95% CI for mean of `satisfaction_level`.

**Q47.** State null and alternative hypothesis for Wrangling.

**Q48.** What alpha=0.05 means in Wrangling.

**Q49.** Type I vs Type II error in Wrangling.

**Q50.** Check skewness of `satisfaction_level`.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.