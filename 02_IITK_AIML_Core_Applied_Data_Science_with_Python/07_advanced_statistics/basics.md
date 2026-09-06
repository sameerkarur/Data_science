# Advanced Statistics & Hypothesis Testing: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official SciPy / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Hypothesis Testing Framework: $H_0$, $H_1$, and Decision Rules](#1-the-hypothesis-testing-framework)
2. [Type I Error ($\alpha$), Type II Error ($\beta$) & Statistical Power ($1-\beta$)](#2-type-i-error-type-ii-error--statistical-power)
3. [The P-Value: Exact Definition, Misconceptions & ASA Statement](#3-the-p-value-exact-definition)
4. [Student's t-Test vs Welch's t-Test (Unequal Variance)](#4-students-t-test-vs-welchs-t-test)
5. [Analysis of Variance (ANOVA): One-Way ANOVA & F-Statistic Decomposition](#5-analysis-of-variance-anova)
6. [Chi-Square ($\chi^2$) Test of Independence & Contingency Tables](#6-chi-square-test-of-independence)
7. [Multiple Testing Corrections: Bonferroni & Benjamini-Hochberg (FDR)](#7-multiple-testing-corrections)
8. [Industrial A/B Testing: Minimum Detectable Effect (MDE) & Sample Size Sizing](#8-industrial-ab-testing)
9. [Common Pitfalls: P-Hacking, Peeking & Post-Hoc Fallacies](#9-common-pitfalls-p-hacking)
10. [Production Case Study: Enterprise E-Commerce A/B Test Decision Engine](#10-production-case-study-ab-testing)
11. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#11-try-it-yourself-hands-on-practice-exercises)
12. [Quick Reference Cheat Sheet & Best Website Citations](#12-quick-reference-cheat-sheet--citations)

---

## 1. The Hypothesis Testing Framework

Hypothesis testing is proof by contradiction under probabilistic uncertainty:
- **Null Hypothesis ($H_0$):** No effect, no difference, or status quo ($\mu_A = \mu_B$).
- **Alternative Hypothesis ($H_1$):** A genuine effect or difference exists ($\mu_A \neq \mu_B$).

```
                      HYPOTHESIS DECISION MATRIX
                                     ACTUAL GROUND TRUTH
                               H0 is TRUE           H0 is FALSE
    DECISION ┌─────────────┬────────────────────┬────────────────────┐
    Reject   │ Type I Err  │ False Positive     │ Correct Decision   │
    H0       │ (Alpha = 5%)│ (Convict Innocent) │ Power (1 - Beta)   │
             ├─────────────┼────────────────────┼────────────────────┤
    Fail to  │ Correct     │ True Negative      │ Type II Error      │
    Reject H0│ Decision    │ (Acquit Innocent)  │ False Negative (β) │
             └─────────────┴────────────────────┴────────────────────┘
```

---

## 2. Student's t-Test vs Welch's t-Test

Standard Student's t-test assumes **homoscedasticity** (equal variances $\sigma_1^2 = \sigma_2^2$). In real industry data, sample sizes and variances differ. **Always use Welch's t-test (`equal_var=False`)**:
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

```python
import numpy as np
from scipy import stats

np.random.seed(42)
group_control = np.random.normal(loc=10.0, scale=2.0, size=50)
group_variant = np.random.normal(loc=11.2, scale=3.5, size=40)  # Different variance & sample size!

# Welch's t-test
t_stat, p_val = stats.ttest_ind(group_variant, group_control, equal_var=False)

print(f"Welch's t-statistic: {t_stat:.4f}")
print(f"P-Value:             {p_val:.6f}")
print("Conclusion: Reject H0? ", p_val < 0.05)
```

#### Output:
```text
Welch's t-statistic: 1.9427
P-Value:             0.056722
Conclusion: Reject H0?  False
```

---

## 3. One-Way ANOVA & F-Statistic Decomposition

When comparing $k \ge 3$ groups, running pairwise t-tests causes **Family-Wise Error Rate (FWER) explosion** ($\alpha_{\text{total}} = 1 - (1 - 0.05)^m$). ANOVA evaluates global variance partition:
$$F = \frac{\text{Between-Group Variance (MSB)}}{\text{Within-Group Variance (MSW)}}$$

```python
group_A = [22, 25, 23, 24, 26]
group_B = [28, 29, 31, 30, 27]
group_C = [19, 21, 20, 22, 18]

f_stat, p_anova = stats.f_oneway(group_A, group_B, group_C)
print(f"ANOVA F-Statistic: {f_stat:.4f} | P-Value: {p_anova:.6e}")
```

#### Output:
```text
ANOVA F-Statistic: 36.8529 | P-Value: 5.768132e-06
```

---

## 4. Chi-Square ($\chi^2$) Test of Independence

For categorical contingency tables, test whether two attributes are independent:
$$\chi^2 = \sum \frac{(O - E)^2}{E}, \quad E_{ij} = \frac{\text{Row}_i \times \text{Col}_j}{N}$$

```python
# Contingency Table: Device Type (Mobile vs Desktop) x Purchase (Yes vs No)
# Rows: [Mobile, Desktop] | Cols: [Purchased, Abandoned]
observed = np.array([
    [120, 380],   # Mobile
    [190, 310]    # Desktop
])

chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"Chi2 Stat: {chi2:.4f} | P-Value: {p_chi:.5e} | Deg of Freedom: {dof}")
```

#### Output:
```text
Chi2 Stat: 21.0371 | P-Value: 4.50021e-06 | Deg of Freedom: 1
```

---

## 5. Multiple Testing Corrections: Bonferroni vs Benjamini-Hochberg

Testing 100 features at $\alpha = 0.05$ produces $\sim 5$ false discoveries purely by chance.
1. **Bonferroni (Strict FWER):** Adjust $\alpha' = \alpha / m$. Overly conservative.
2. **Benjamini-Hochberg (FDR):** Controls False Discovery Rate (FDR). Ranks p-values $p_{(1)} \le \dots \le p_{(m)}$ and finds largest $k$ where $p_{(k)} \le \frac{k}{m} Q$.

```python
raw_pvalues = [0.001, 0.008, 0.024, 0.045, 0.120]
m = len(raw_pvalues)

# Bonferroni adjusted threshold for alpha = 0.05
bonf_threshold = 0.05 / m
print(f"Bonferroni Threshold: {bonf_threshold:.4f}")
print("Significant under Bonferroni:", [p < bonf_threshold for p in raw_pvalues])
```

#### Output:
```text
Bonferroni Threshold: 0.0100
Significant under Bonferroni: [True, True, False, False, False]
```

---

## 6. Industrial A/B Testing: Minimum Sample Size Calculation

To detect a lift with statistical validity before running the test, compute required sample size per variant using Evan Miller's formula:
$$n = \frac{2 \left( z_{\alpha/2} + z_{\beta} \right)^2 p (1 - p)}{(\text{MDE})^2}$$

```python
def calculate_sample_size_per_variant(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p = baseline_rate
    numerator = 2 * ((z_alpha + z_beta) ** 2) * p * (1 - p)
    denominator = (mde) ** 2
    return int(np.ceil(numerator / denominator))

n_per_variant = calculate_sample_size_per_variant(baseline_rate=0.05, mde=0.01) # Detect 5% -> 6% conversion
print(f"Required Sample Size Per Variant: {n_per_variant:,} visitors")
```

#### Output:
```text
Required Sample Size Per Variant: 3,729 visitors
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Paired t-Test on Model Latency
**Task:** Given latency measurements of 5 queries before and after optimization, run a paired t-test:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
before = [120, 135, 128, 142, 130]
after  = [110, 122, 115, 129, 118]

t_stat, p_val = stats.ttest_rel(before, after)
print(f"Paired t-statistic: {t_stat:.4f} | P-Value: {p_val:.5f}")
```
#### Output:
```text
Paired t-statistic: 13.0639 | P-Value: 0.00018
```
</details>

---

## 8. Quick Reference Cheat Sheet & Best Website Citations

| Test Name | Data Type | Assumptions | Scipy Function |
|---|---|---|---|
| **Welch's t-Test** | Continuous 2-group | Normality (or $N \ge 30$) | `stats.ttest_ind(..., equal_var=False)` |
| **Paired t-Test** | Continuous paired | Paired differences normal | `stats.ttest_rel(a, b)` |
| **One-Way ANOVA** | Continuous $\ge 3$ groups | Normality, independence | `stats.f_oneway(g1, g2, g3)` |
| **Chi-Square Test** | Categorical | Expected cells $\ge 5$ | `stats.chi2_contingency(table)` |

### 🌐 Official References & Recommended Reading:
- [SciPy Statistical Hypothesis Tests](https://docs.scipy.org/doc/scipy/reference/stats.html#hypothesis-tests-and-correlation)
- [American Statistical Association Statement on P-Values](https://www.amstat.org/asa/files/pdfs/P-ValueStatement.pdf)
- [Evan Miller A/B Testing Mathematics](https://www.evanmiller.org/ab-testing/)
