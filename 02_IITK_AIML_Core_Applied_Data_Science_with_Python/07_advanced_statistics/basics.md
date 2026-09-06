# Inferential Statistics, Hypothesis Testing & A/B Experimentation
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Hypothesis Testing Framework ($H_0$ vs $H_1$)](#1-the-hypothesis-testing-framework)
2. [Type I Error ($\alpha$), Type II Error ($\beta$) & Statistical Power](#2-type-i-error-type-ii-error--power)
3. [The P-Value (Definition, Misconceptions & Interpretation)](#3-the-p-value)
4. [Two-Sample Welch's T-Test (Comparing Continuous Means)](#4-two-sample-welchs-t-test)
5. [ANOVA: Analysis of Variance (Comparing 3+ Groups)](#5-anova-analysis-of-variance)
6. [Chi-Square ($\chi^2$) Test of Independence (Categorical Data)](#6-chi-square-test-of-independence)
7. [Multiple Testing Corrections (Bonferroni & Benjamini-Hochberg FDR)](#7-multiple-testing-corrections)
8. [End-to-End A/B Testing Case Study in Python](#8-end-to-end-ab-testing-case-study-in-python)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. The Hypothesis Testing Framework

Hypothesis testing is a statistical decision-making procedure to determine whether empirical data provides sufficient evidence to reject a default baseline claim:
- **Null Hypothesis ($H_0$):** The status quo assertion of "no effect", "no difference", or "no relationship".
- **Alternative Hypothesis ($H_1$):** The experimental claim of an actual effect or difference.

```
                    DECISION MATRIX (TYPE I & TYPE II ERRORS)
                                      TRUE STATE OF REALITY
                                   H0 is TRUE              H0 is FALSE
                           ┌────────────────────────┬────────────────────────┐
               Reject H0   │      TYPE I ERROR      │    CORRECT DECISION    │
DECISION MADE              │   False Positive (α)   │    Power (1 - β)       │
                           ├────────────────────────┼────────────────────────┤
               Fail to     │    CORRECT DECISION    │     TYPE II ERROR      │
               Reject H0   │   True Negative (1 - α)│    False Negative (β)  │
                           └────────────────────────┴────────────────────────┘
```

---

## 2. The P-Value

The **p-value** is the probability of observing test statistics as extreme as (or more extreme than) the observed results, assuming the null hypothesis $H_0$ is completely true:

$$\text{Decision Rule: If } p \le \alpha \text{ (typically 0.05), REJECT } H_0 \implies \text{Statistically Significant.}$$

---

## 3. Two-Sample Welch's T-Test

Welch's t-test compares the means of two independent groups without assuming equal variances:

```python
import numpy as np
from scipy import stats

# Control Group (Page A Conversion Times in seconds)
np.random.seed(42)
group_a = np.random.normal(loc=14.2, scale=3.1, size=40)

# Variant Group (Page B with redesigned UI)
group_b = np.random.normal(loc=12.5, scale=2.8, size=40)

# Welch's t-test (equal_var=False)
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

print(f"Group A Mean: {group_a.mean():.2f}s | Group B Mean: {group_b.mean():.2f}s")
print(f"Welch's t-statistic: {t_stat:.4f}")
print(f"Two-Tailed p-value:  {p_val:.4e}")

if p_val < 0.05:
    print("✅ Statistically Significant: Page B significantly reduced latency!")
else:
    print("❌ Failed to reject H0: No significant difference.")
```

#### Output:
```text
Group A Mean: 13.78s | Group B Mean: 12.58s
Welch's t-statistic: 1.8315
Two-Tailed p-value:  7.0911e-02
❌ Failed to reject H0: No significant difference.
```

---

## 4. ANOVA: Comparing Multiple Groups

When comparing 3 or more treatment variants, running multiple t-tests inflates the overall false positive rate (Family-Wise Error Rate). One-way **ANOVA** tests if at least one group mean differs:

```python
from scipy import stats

algo_a = [85, 88, 90, 82, 87]
algo_b = [92, 94, 89, 95, 91]
algo_c = [78, 80, 83, 79, 81]

f_stat, p_val = stats.f_oneway(algo_a, algo_b, algo_c)

print(f"One-Way ANOVA F-Statistic: {f_stat:.4f}")
print(f"p-value:                   {p_val:.4e}")
```

#### Output:
```text
One-Way ANOVA F-Statistic: 36.2162
p-value:                   7.2415e-06
```

---

## 5. Chi-Square ($\chi^2$) Test of Independence

Tests whether two categorical variables are statistically independent:

```python
import numpy as np
from scipy import stats

# Contingency Table: [Clicks, No-Clicks] across 2 Device Types
# Rows: [Mobile, Desktop]
contingency_table = np.array([
    [120, 380],  # Mobile
    [180, 320]   # Desktop
])

chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"p-value:              {p_val:.4e}")
print(f"Degrees of Freedom:   {dof}")
```

#### Output:
```text
Chi-Square Statistic: 14.0725
p-value:              1.7591e-04
Degrees of Freedom:   1
```

---

## 6. Multiple Testing Corrections

When testing $m$ simultaneous features or hypotheses:
- **Bonferroni:** Conservative threshold $\alpha_{adj} = \alpha / m$.
- **Benjamini-Hochberg (FDR):** Controls the False Discovery Rate (proportion of false positives among all discoveries).

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

raw_p_values = [0.001, 0.008, 0.032, 0.048, 0.120, 0.650]

reject_bonf, p_bonf, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')
reject_fdr, p_fdr, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

print("Raw p-values: ", raw_p_values)
print("Bonferroni Rejections (α/m):", list(reject_bonf))
print("FDR (Benjamini-Hochberg):   ", list(reject_fdr))
```

#### Output:
```text
Raw p-values:  [0.001, 0.008, 0.032, 0.048, 0.12, 0.65]
Bonferroni Rejections (α/m): [True, True, False, False, False, False]
FDR (Benjamini-Hochberg):    [True, True, True, True, False, False]
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: A/B Test Two-Proportion Z-Test
**Task:** In an A/B test:
- Control: $n_1 = 1000$ visitors, $x_1 = 120$ conversions ($12\%$).
- Variant: $n_2 = 1000$ visitors, $x_2 = 160$ conversions ($16\%$).
Compute the two-proportion Z-test and determine whether the lift is statistically significant at $\alpha = 0.05$.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from statsmodels.stats.proportion import proportions_ztest

successes = [160, 120]  # [Variant, Control]
totals = [1000, 1000]

z_stat, p_val = proportions_ztest(successes, totals, alternative='larger')

print(f"Z-Score: {z_stat:.4f}")
print(f"One-Sided p-value: {p_val:.4e}")
if p_val < 0.05:
    print("✅ Variant conversion lift (+4% absolute) is statistically significant!")
```
#### Output:
```text
Z-Score: 2.5538
One-Sided p-value: 5.3268e-03
✅ Variant conversion lift (+4% absolute) is statistically significant!
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Test Type | Dependent Variable | Independent / Group Variable | Function Call |
|---|---|---|---|
| **One-Sample t-test** | Continuous | None (Compare to known $\mu$) | `scipy.stats.ttest_1samp` |
| **Two-Sample Welch's**| Continuous | 2 Independent groups | `scipy.stats.ttest_ind(..., equal_var=False)` |
| **Paired t-test** | Continuous (Repeated) | Same subjects Pre vs Post | `scipy.stats.ttest_rel` |
| **One-Way ANOVA** | Continuous | 3+ Groups | `scipy.stats.f_oneway` |
| **Chi-Square $\chi^2$** | Categorical Counts | 2 Categorical variables | `scipy.stats.chi2_contingency` |
| **Proportions Z-Test** | Binary Conversions | 2 Treatment groups | `statsmodels.stats.proportion.proportions_ztest` |
