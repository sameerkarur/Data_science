# Inferential Statistics & Hypothesis Testing
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Type I vs Type II Errors
- **Type I Error ($lpha$):** Rejecting null hypothesis $H_0$ when it was actually true (False Positive).
- **Type II Error ($eta$):** Failing to reject $H_0$ when it was false (False Negative).
- **Statistical Power ($1 - eta$):** Probability of correctly detecting a genuine effect.

### 2. ANOVA (Analysis of Variance)
Compares means across $\ge 3$ groups by assessing variance between groups relative to variance within groups:
$$F = rac{	ext{Mean Square Between (MSB)}}{	ext{Mean Square Within (MSW)}}$$
