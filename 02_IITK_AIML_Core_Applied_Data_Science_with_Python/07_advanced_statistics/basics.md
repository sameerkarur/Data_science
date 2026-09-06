# Chapter 7: Inferential Statistics & Hypothesis Testing
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Inferential statistics allows engineers to make rigorous decisions under uncertainty by quantifying whether an observed difference between groups is statistically genuine or merely an artifact of random sampling noise.

```
                    HYPOTHESIS TESTING DECISION TREE
       Define Null Hypothesis (H₀) & Alternative (H₁)
                           │
       Compute Test Statistic (t, z, F, or χ²)
                           │
                 Obtain p-value from CDF
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
          p < α (0.05)            p >= α (0.05)
       Reject Null (H₀)      Fail to Reject Null (H₀)
     Statistically Significant   Insufficient Evidence
```

---

## 2. Deep Theoretical Foundations

### 1. Decision Theory: Type I vs Type II Errors
In any statistical test:
- **Type I Error ($\alpha$):** Rejecting $H_0$ when $H_0$ is true (False Positive rate). Typically set to $\alpha = 0.05$.
- **Type II Error ($\beta$):** Failing to reject $H_0$ when $H_1$ is true (False Negative rate).
- **Statistical Power ($1 - \beta$):** The probability of correctly detecting a genuine effect. In industrial A/B testing, target power is typically $80\% - 90\%$.

### 2. Welch's t-test (Unequal Variances)
Standard Student's t-test assumes homoscedasticity (equal variance $\sigma_1^2 = \sigma_2^2$). In real-world data, Welch's t-test does not assume equal variances:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
With degrees of freedom approximated via the Welch-Satterthwaite equation:
$$\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

### 3. ANOVA (Analysis of Variance) & F-Ratio
Used to test equality of means across $K \ge 3$ groups simultaneously without inflating the family-wise error rate:
$$F = \frac{\text{Mean Square Between (MSB)}}{\text{Mean Square Within (MSW)}} = \frac{\frac{1}{K - 1}\sum_{i=1}^K n_i (\bar{X}_i - \bar{X})^2}{\frac{1}{N - K}\sum_{i=1}^K \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2}$$

### 4. Multiple Testing Corrections
Running $M$ independent hypothesis tests at significance level $\alpha = 0.05$ inflates the family-wise false positive probability to $1 - (1 - \alpha)^M$ (at $M = 20$, probability of $\ge 1$ false discovery exceeds $64\%$).
- **Bonferroni Correction:** Controls Family-Wise Error Rate (FWER) conservatively:
  $$\alpha_{\text{adjusted}} = \frac{\alpha}{M}$$
- **Benjamini-Hochberg (FDR):** Controls the False Discovery Rate (fraction of declared discoveries that are false) with higher statistical power:
  $$p_{(i)} \le \frac{i}{M} Q$$

---

## 3. Production Implementation: Automated A/B Testing Engine

```python
import numpy as np
from scipy import stats

def evaluate_ab_experiment(control_conversions: np.ndarray, 
                           treatment_conversions: np.ndarray, 
                           alpha: float = 0.05) -> dict[str, float | bool]:
    """Evaluates continuous KPI uplifts using Welch's t-test with effect size."""
    n_ctrl, n_treat = len(control_conversions), len(treatment_conversions)
    mean_ctrl, mean_treat = np.mean(control_conversions), np.mean(treatment_conversions)
    
    # Welch's two-sample t-test (robust to unequal variances)
    t_stat, p_val = stats.ttest_ind(treatment_conversions, control_conversions, equal_var=False)
    
    # Cohen's d effect size
    s_pooled = np.sqrt(((n_ctrl - 1)*np.var(control_conversions, ddof=1) + 
                        (n_treat - 1)*np.var(treatment_conversions, ddof=1)) / (n_ctrl + n_treat - 2))
    cohens_d = (mean_treat - mean_ctrl) / s_pooled
    
    relative_lift = (mean_treat - mean_ctrl) / mean_ctrl if mean_ctrl != 0 else 0.0
    
    return {
        "control_mean": float(mean_ctrl),
        "treatment_mean": float(mean_treat),
        "relative_lift_pct": float(relative_lift * 100),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "statistically_significant": bool(p_val < alpha)
    }
```
