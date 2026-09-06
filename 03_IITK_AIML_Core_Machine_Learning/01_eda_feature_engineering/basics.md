# Exploratory Data Analysis & Advanced Feature Engineering: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Scikit-Learn / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The EDA Philosophy & John Tukey's Statistical Mindset](#1-the-eda-philosophy)
2. [Univariate, Bivariate & Multivariate Analysis Framework](#2-univariate-bivariate--multivariate-analysis)
3. [Mathematical Feature Transformations (Log, Box-Cox, Yeo-Johnson)](#3-mathematical-feature-transformations)
4. [Feature Interaction & Polynomial Features Architecture](#4-feature-interaction--polynomial-features)
5. [Information-Theoretic Feature Selection: Mutual Information vs ANOVA F-Value](#5-information-theoretic-feature-selection)
6. [Dimensionality Reduction: PCA vs t-SNE vs UMAP](#6-dimensionality-reduction-pca-vs-tsne-vs-umap)
7. [Automated Feature Engineering with Featuretools & Deep Feature Synthesis](#7-automated-feature-engineering)
8. [Common Pitfalls: Target Leakage & Lookahead Bias](#8-common-pitfalls-target-leakage)
9. [Production Case Study: Feature Engineering Pipeline for E-Commerce Customer Lifetime Value (LTV)](#9-production-case-study-ltv-pipeline)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. The EDA Philosophy & John Tukey's Mindset

EDA is an iterative discovery cycle:
```
                      THE RECURSIVE EDA LIFECYCLE
    ┌──────────────────────┐
    │ 1. Raw Distribution │ ──► Check Skewness, Kurtosis, Missingness
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 2. Relationships     │ ──► Pearson/Spearman Correlation, Mutual Information
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 3. Hypothesis & Test │ ──► Welch's t-test, Chi-Square Independence
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ 4. Feature Synthesis │ ──► Transformations, Domain Ratios, Aggregations
    └──────────────────────┘
```

---

## 2. Mathematical Transformations: Log, Box-Cox & Yeo-Johnson

Linear models and distance-based estimators assume feature normality and homoscedasticity:
1. **Natural Logarithm:** $y = \ln(x + 1)$ (Requires $x \ge 0$).
2. **Box-Cox Transformation:** Requires strictly positive values $x > 0$:
$$y^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(x) & \text{if } \lambda = 0 \end{cases}$$
3. **Yeo-Johnson Transformation:** Handles zero and negative values ($x \in \mathbb{R}$), estimating optimal $\lambda$ via Maximum Likelihood.

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats

np.random.seed(42)
raw_skewed = np.random.exponential(scale=3.0, size=1000) - 2.0  # Contains negatives!

# Yeo-Johnson handles negative values seamlessly
pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(raw_skewed.reshape(-1, 1)).flatten()

print(f"Original Skewness:    {stats.skew(raw_skewed):.3f} (Severe Right-Skew)")
print(f"Transformed Skewness: {stats.skew(transformed):.3f} (Near 0 = Normal Gaussian!)")
print(f"Optimal Lambda (λ):   {pt.lambdas_[0]:.3f}")
```

#### Output:
```text
Original Skewness:    1.954 (Severe Right-Skew)
Transformed Skewness: 0.082 (Near 0 = Normal Gaussian!)
Optimal Lambda (λ):   0.142
```

---

## 3. Information-Theoretic Feature Selection: Mutual Information

Unlike Pearson correlation which only detects **linear** associations, **Mutual Information (MI)** measures both linear and non-linear dependencies:
$$I(X; Y) = \iint p(x, y) \ln \frac{p(x, y)}{p(x) p(y)} dx dy$$

```python
from sklearn.feature_selection import mutual_info_regression

# Synthetic non-linear data: y = x^2 (Pearson correlation is ~0, but MI is huge!)
x_vals = np.linspace(-3, 3, 500)
y_vals = x_vals ** 2 + np.random.normal(0, 0.2, 500)

pearson_corr = np.corrcoef(x_vals, y_vals)[0, 1]
mi_score = mutual_info_regression(x_vals.reshape(-1, 1), y_vals)[0]

print(f"Pearson Correlation (Linear):     {pearson_corr:.4f} (Fails to see relationship!)")
print(f"Mutual Information (Non-Linear):  {mi_score:.4f} (Strongly detects non-linear link!)")
```

#### Output:
```text
Pearson Correlation (Linear):     0.0241 (Fails to see relationship!)
Mutual Information (Non-Linear):  0.8654 (Strongly detects non-linear link!)
```

---

## 4. Production Case Study: E-Commerce Customer LTV Pipeline

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RFMFeatureExtractor(BaseEstimator, TransformerMixin):
    """Computes Recency, Frequency, Monetary (RFM) aggregations per customer."""
    def __init__(self, reference_date: str = '2026-09-01'):
        self.ref_date = pd.to_datetime(reference_date)

    def fit(self, X, y=None):
        return self

    def transform(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        df = transactions_df.copy()
        df['tx_date'] = pd.to_datetime(df['tx_date'])

        # GroupBy customer
        rfm = df.groupby('customer_id').agg(
            recency_days=('tx_date', lambda x: (self.ref_date - x.max()).days),
            tx_frequency=('tx_id', 'count'),
            monetary_total=('amount', 'sum'),
            avg_basket_value=('amount', 'mean')
        ).reset_index()

        # Ratio features
        rfm['monetary_per_frequency'] = rfm['monetary_total'] / (rfm['tx_frequency'] + 1e-5)
        return rfm

raw_tx = pd.DataFrame({
    'customer_id': [101, 101, 102, 103, 101],
    'tx_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
    'tx_date': ['2026-08-15', '2026-08-28', '2026-07-10', '2026-08-30', '2026-08-31'],
    'amount': [120.0, 45.0, 310.0, 25.0, 85.0]
})

rfm_engine = RFMFeatureExtractor()
engineered_df = rfm_engine.transform(raw_tx)
print("Engineered Customer RFM Matrix:\n", engineered_df)
```

#### Output:
```text
Engineered Customer RFM Matrix:
    customer_id  recency_days  tx_frequency  monetary_total  avg_basket_value  monetary_per_frequency
0          101             1             3           250.0         83.333333               83.333056
1          102            53             1           310.0        310.000000              309.996900
2          103             2             1            25.0         25.000000               24.999750
```

---

## 5. Quick Reference Cheat Sheet & Best Website Citations

| Technique | Goal | Scikit-Learn Class | Non-Linear? |
|---|---|---|---|
| **Power Transformer** | Normality & homoscedasticity | `PowerTransformer(method='yeo-johnson')` | Yes |
| **Mutual Information** | Non-linear feature importance | `mutual_info_classif` / `regression` | Yes |
| **SelectKBest** | Top $K$ feature filter | `SelectKBest(score_func=...)` | Both |

### 🌐 Official References & Recommended Reading:
- [Scikit-Learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [John Tukey — Exploratory Data Analysis (Addison-Wesley)](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
- [W3Schools Machine Learning Feature Selection](https://www.w3schools.com/python/python_ml_scale.asp)
