# Exploratory Data Analysis & Feature Selection Pipelines: The Definitive Textbook
**Comprehensive Academic & Industry Engineering Handbook (John Tukey / Scikit-Learn Grade)**

---

## 📑 Table of Contents
1. [The Philosophy & Foundations of Exploratory Data Analysis](#1-the-philosophy--foundations-of-eda)
   - [John Tukey's Exploratory Paradigm vs Confirmatory Statistics](#11-john-tukeys-exploratory-paradigm)
   - [Univariate Analysis: Skewness, Kurtosis & Heavy Tails](#12-univariate-analysis)
   - [Bivariate & Multivariate Relationships: Scatter Matrices & Covariance](#13-bivariate--multivariate-relationships)
2. [Mathematical Transformations for Non-Gaussian Features](#2-mathematical-transformations-for-non-gaussian-features)
   - [Logarithmic Transformation ($\log(1+x)$)](#21-logarithmic-transformation)
   - [Box-Cox Power Transformation & Maximum Likelihood Estimation](#22-box-cox-power-transformation)
   - [Yeo-Johnson Transformation for Real-Valued Data ($x \in \mathbb{R}$)](#23-yeo-johnson-transformation)
   - [Quantile Uniform & Gaussian Mapping](#24-quantile-uniform--gaussian-mapping)
3. [Feature Engineering & Interaction Architectures](#3-feature-engineering--interaction-architectures)
   - [Polynomial & Cross-Product Interactions](#31-polynomial--cross-product-interactions)
   - [Temporal Cyclical Encoding ($\sin / \cos$)](#32-temporal-cyclical-encoding)
   - [Domain-Specific Aggregations & Split-Apply-Combine](#33-domain-specific-aggregations)
4. [Statistical & Information-Theoretic Feature Selection](#4-statistical--information-theoretic-feature-selection)
   - [Filter Methods: Pearson vs Spearman Rank vs Kendall Tau](#41-filter-methods)
   - [Mutual Information (Kullback-Leibler Divergence Formulation)](#42-mutual-information)
   - [ANOVA F-Statistic vs Chi-Square ($\chi^2$) Contingency Tests](#43-anova-f-statistic-vs-chi-square)
   - [Recursive Feature Elimination (RFE) & Wrapper Search](#44-recursive-feature-elimination)
   - [Embedded Selection: L1 Lasso Sparsity & Tree Importance](#45-embedded-selection)
5. [Dimensionality Reduction: Manifold & Spectral Learning](#5-dimensionality-reduction-manifold--spectral-learning)
   - [Principal Component Analysis (PCA): Spectral Eigendecomposition](#51-principal-component-analysis)
   - [t-Distributed Stochastic Neighbor Embedding (t-SNE) Dynamics](#52-t-distributed-stochastic-neighbor-embedding)
   - [Uniform Manifold Approximation and Projection (UMAP)](#53-uniform-manifold-approximation-and-projection)
6. [Data Leakage Prevention in Production Pipelines](#6-data-leakage-prevention-in-production-pipelines)
   - [Target Leakage vs Temporal Lookahead Contamination](#61-target-leakage-vs-temporal-contamination)
   - [Scikit-Learn `ColumnTransformer` & Pipeline Architecture](#62-scikit-learn-columntransformer)
7. [Production Case Study: E-Commerce Customer Lifetime Value (LTV) Pipeline](#7-production-case-study-ltv-pipeline)
8. [Common Pitfalls & EDA Anti-Patterns](#8-common-pitfalls--eda-anti-patterns)
9. [Try It Yourself! (Hands-On Practice Exercises with Full Solutions)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Staff-Level Technical Interview Questions & Model Answers](#10-staff-level-technical-interview-questions)

---

## 1. The Philosophy & Foundations of EDA

### 1.1 John Tukey's Exploratory Paradigm
In 1977, statistician John Tukey established Exploratory Data Analysis (EDA) as the scientific discipline of discovering structural signals, unexpected anomalies, and latent dynamics before formulating rigid parametric models:
- **Confirmatory Data Analysis (CDA):** Tests whether an a priori hypothesis is statistically significant ($p < 0.05$).
- **Exploratory Data Analysis (EDA):** Generates hypotheses by inspecting distributions, correlations, clustering tendencies, and boundary discontinuities.

```
                      THE RECURSIVE EDA LIFECYCLE
    ┌──────────────────────┐
    │ 1. Raw Distribution  │ ──► Compute Skewness, Kurtosis, Missingness
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

### 1.2 Univariate Analysis: Skewness, Kurtosis & Heavy Tails
- **Fisher-Pearson Standardized Skewness ($g_1$):** Measures asymmetric distribution tilt:
  $$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{N}\sum (x_i - \bar{x})^3}{\left(\frac{1}{N}\sum (x_i - \bar{x})^2\right)^{3/2}}$$
  - $g_1 = 0$: Symmetric distribution (Gaussian-like).
  - $g_1 > 1.0$: Severe positive (right) skew (e.g. income, transaction amounts, latency).
- **Excess Kurtosis ($g_2$):** Measures tail weight and outlier propensity relative to Gaussian:
  $$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{N}\sum (x_i - \bar{x})^4}{\left(\frac{1}{N}\sum (x_i - \bar{x})^2\right)^2} - 3$$
  - $g_2 > 0$ (Leptokurtic): Heavy tails with extreme outlier risk (financial crashes, network bursts).

---

## 2. Mathematical Transformations for Non-Gaussian Features

### 2.1 The Box-Cox Transformation ($y > 0$)
Estimates parameter $\lambda$ via Maximum Likelihood to map strictly positive variables onto a Gaussian distribution:
$$y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(y) & \text{if } \lambda = 0 \end{cases}$$

### 2.2 The Yeo-Johnson Transformation ($y \in \mathbb{R}$)
Overcomes the positivity constraint of Box-Cox, handling zero and negative values smoothly:
$$\psi(\lambda, y) = \begin{cases} \frac{(y + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \ge 0 \\ \ln(y + 1) & \text{if } \lambda = 0, y \ge 0 \\ -\frac{(-y + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, y < 0 \\ -\ln(-y + 1) & \text{if } \lambda = 2, y < 0 \end{cases}$$

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats

np.random.seed(42)
raw_skewed = np.random.exponential(scale=3.0, size=1000) - 2.0  # Contains negative values!

pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(raw_skewed.reshape(-1, 1)).flatten()

print(f"Original Skewness:    {stats.skew(raw_skewed):.3f}")
print(f"Transformed Skewness: {stats.skew(transformed):.3f}")
print(f"Estimated Lambda (λ): {pt.lambdas_[0]:.3f}")
```

#### Output:
```text
Original Skewness:    1.954
Transformed Skewness: 0.082
Estimated Lambda (λ): 0.142
```

---

## 3. Information-Theoretic Feature Selection: Mutual Information

Unlike Pearson correlation which only detects linear associations, **Mutual Information (MI)** measures general non-linear dependencies:
$$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy = H(X) - H(X | Y)$$

```python
from sklearn.feature_selection import mutual_info_regression

# Synthetic non-linear data: y = x^2
x_vals = np.linspace(-3, 3, 500)
y_vals = x_vals ** 2 + np.random.normal(0, 0.2, 500)

pearson_corr = np.corrcoef(x_vals, y_vals)[0, 1]
mi_score = mutual_info_regression(x_vals.reshape(-1, 1), y_vals)[0]

print(f"Pearson Correlation (Linear):    {pearson_corr:.4f} (Misses non-linear relationship!)")
print(f"Mutual Information (Non-Linear): {mi_score:.4f} (Captures strong relationship!)")
```

#### Output:
```text
Pearson Correlation (Linear):    0.0241 (Misses non-linear relationship!)
Mutual Information (Non-Linear): 0.8654 (Captures strong relationship!)
```

---

## 4. Production Scikit-Learn Pipeline Implementation

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class CyclicalTimeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cycle=24.0):
        self.cycle = cycle

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        sin_feat = np.sin(2 * np.pi * X / self.cycle)
        cos_feat = np.cos(2 * np.pi * X / self.cycle)
        return np.column_stack([sin_feat, cos_feat])

# Leak-free column pipeline
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, ["age", "income", "credit_score"]),
        ("cat", cat_pipe, ["education", "home_ownership"]),
        ("hour_cycle", CyclicalTimeEncoder(cycle=24.0), ["transaction_hour"])
    ]
)
```

---

## 5. Staff-Level Technical Interview Questions & Model Answers

### Q1: What is Target Leakage, and how do you systematically detect and prevent it in automated feature engineering?
**Model Answer:**
Target leakage occurs when an input feature contains information about the ground-truth target label that would not be available at the exact moment a prediction is made in production. 

**Common Manifestations:**
1. **Temporal Lookahead:** Using future transaction totals to predict current-day churn.
2. **Proxy IDs:** Using an internal `account_closure_ticket_id` feature that only gets created when an account is marked for closure.
3. **Imputation Leakage:** Fitting an imputer or scaler on the entire dataset before performing train/test split.

**Systematic Prevention:**
1. **Strict Temporal Splitting:** Split data chronologically rather than randomly for time-dependent phenomena.
2. **Point-in-Time Joins:** Restrict feature lookups to transactions strictly strictly preceding the event timestamp ($t_{\text{feature}} < t_{\text{event}}$).
3. **Pipeline Encapsulation:** Enforce that all transformers (`fit()`) execute exclusively within cross-validation training folds using Scikit-Learn `Pipeline`.

---

## 6. Academic Citations
1. **Tukey, J. W. (1977).** *Exploratory Data Analysis*. Addison-Wesley.
2. **Box, G. E., & Cox, D. R. (1964).** An analysis of transformations. *JRSS Series B*.
3. **Yeo, I. K., & Johnson, R. A. (2000).** A new family of power transformations. *Biometrika*.
