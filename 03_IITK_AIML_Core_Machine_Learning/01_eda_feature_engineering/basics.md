# Chapter 1: Exploratory Data Analysis & Feature Engineering
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

Feature engineering is the process of transforming raw observational data into mathematical representations that expose the underlying geometry of the problem to machine learning algorithms. High-performing tabular systems rely far more on intelligent feature synthesis (interactions, domain encodings, aggregations) than on raw model complexity.

```
                  FEATURE ENGINEERING PIPELINE
    Raw Features ──► 1. Missingness & Non-Linear Power Transforms (Yeo-Johnson)
                            │
                     2. Categorical Target Encoding with Empirical Bayes Smoothing
                            │
                     3. Interaction Terms & Polynomial Features (x₁ · x₂)
                            │
                     4. Permutation Feature Importance & Mutual Information Selection
                            │
                     Output Matrix X ──► ML Estimator
```

---

## 2. Deep Theoretical Foundations

### 1. Target Encoding with Empirical Bayes Smoothing
Target encoding replaces a categorical level $k$ with the expected target value. To prevent catastrophic target leakage and overfitting on rare categories, Bayesian m-estimate smoothing shrinks category estimates toward the global target prior:
$$\hat{S}_k = \lambda(n_k) \cdot \bar{y}_k + (1 - \lambda(n_k)) \cdot \bar{y}_{\text{global}}$$
Where the smoothing weight $\lambda(n_k)$ is a logistic function of sample size $n_k$:
$$\lambda(n_k) = \frac{1}{1 + e^{-(n_k - m) / s}}$$
Here, $m$ represents the min-sample inflection threshold and $s$ controls smoothing curvature.

### 2. Mutual Information & Non-Linear Dependency
Linear correlation (Pearson $r$) fails to capture non-linear relationships (e.g. $y = x^2$ yields $r \approx 0$). Mutual Information (MI), rooted in Shannon Information Theory, measures the reduction in entropy of target $Y$ given feature $X$:
$$I(X; Y) = \iint p(x, y) \log \left(\frac{p(x, y)}{p(x)p(y)}\right) dx \, dy = H(Y) - H(Y \mid X)$$
$I(X; Y) = 0 \iff X$ and $Y$ are strictly statistically independent.

### 3. Yeo-Johnson Power Transformation
Standard Box-Cox transformations require strictly positive values ($x > 0$). The Yeo-Johnson transformation extends variance-stabilizing power transformations to real values (including zero and negative values):
$$\psi(\lambda, x) = \begin{cases} 
\frac{(x + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \\
\ln(x + 1) & \text{if } \lambda = 0, x \ge 0 \\
-\frac{(-x + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{if } \lambda \neq 2, x < 0 \\
-\ln(-x + 1) & \text{if } \lambda = 2, x < 0 
\end{cases}$$
The parameter $\lambda$ is estimated via Maximum Likelihood Estimation (MLE) to maximize normality.

---

## 3. Production Implementation: Leak-Free Target Encoder with Out-of-Fold Cross-Fitting

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Leakage-free out-of-fold target encoder with Empirical Bayes m-smoothing."""
    def __init__(self, m_smooth: float = 10.0, n_splits: int = 5, random_state: int = 42):
        self.m_smooth = m_smooth
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = 0.0
        self.mapping_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OutOfFoldTargetEncoder":
        self.global_mean_ = float(y.mean())
        self.mapping_ = {}
        for col in X.columns:
            stats = y.groupby(X[col]).agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + self.m_smooth * self.global_mean_) / (stats['count'] + self.m_smooth)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.global_mean_ = float(y.mean())
        self.fit(X, y)
        X_out = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for col in X.columns:
            oof_col = pd.Series(index=X.index, dtype='float64')
            for trn_idx, val_idx in kf.split(X):
                X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
                stats = y_trn.groupby(X_trn[col]).agg(['count', 'mean'])
                smoothed = (stats['count'] * stats['mean'] + self.m_smooth * self.global_mean_) / (stats['count'] + self.m_smooth)
                oof_col.iloc[val_idx] = X.iloc[val_idx][col].map(smoothed).fillna(self.global_mean_)
            X_out[col] = oof_col
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in X.columns:
            X_out[col] = X_out[col].map(self.mapping_[col]).fillna(self.global_mean_)
        return X_out
```

---

## 4. Complexity & Selection Matrix

| Feature Transformation | Time Complexity | Out-of-Fold Required? | Handles High Cardinality? |
|---|---|---|---|
| One-Hot Encoding | $O(N \cdot K)$ | No | ❌ Explodes feature dimensions |
| Target Encoding (Naive) | $O(N)$ | ❌ High Leakage! | ✅ Retains 1D column |
| Out-of-Fold Target Encoding | $O(K_{\text{folds}} \cdot N)$ | ✅ Mandatory in Production | ✅ Optimal for Tabular ML |
| Mutual Information Selection | $O(N \log N)$ | No | ✅ Captures Non-linear curves |
