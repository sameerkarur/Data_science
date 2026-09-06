# Chapter 9: Data Wrangling, Imputation & Outlier Engineering
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Raw industrial data is noisy, incomplete, and filled with anomalies. Robust data wrangling requires diagnosing missingness mechanisms before applying imputation, and utilizing non-parametric outlier boundaries.

```
                 DATA CLEANING & IMPUTATION LIFECYCLE
       Raw Data ──► Detect Missingness Patterns:
                      ├── MCAR (Missing Completely at Random)
                      ├── MAR  (Missing at Random)
                      └── MNAR (Missing Not at Random)
                           │
       Screen Outliers via IQR Tukey Fences / Isolation Forest
                           │
       Apply Domain Scalers (StandardScaler / RobustScaler)
```

---

## 2. Deep Theoretical Foundations

### 1. Missingness Taxonomy (Little & Rubin)
- **MCAR (Missing Completely at Random):** Missingness is entirely independent of observed and unobserved data: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$. Deletion does not introduce bias.
- **MAR (Missing at Random):** Missingness depends systematically on observed features, but not on the missing value itself: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$. Resolved via conditional imputation (MICE / IterativeImputer).
- **MNAR (Missing Not at Random):** Missingness depends directly on the unobserved value (e.g. high-income respondents refusing to state income). Requires explicit missingness indicator flags.

### 2. Multivariate Imputation by Chained Equations (MICE)
Rather than simple mean or median imputation (which destroys feature variance and correlations), MICE models each variable with missing values as a function of all other variables in an iterative round-robin Gibbs sampler:
$$\hat{Y}_j^{(t)} \sim P(Y_j \mid Y_{-j}^{(t)}, \theta_j)$$
Iterating across 10–20 cycles converges to the true joint conditional distribution.

### 3. Isolation Forest for Multi-Dimensional Outlier Detection
Isolation Forest isolates anomalies by randomly selecting a feature and a random split value:
- Since anomalies occupy sparse, isolated regions of the feature space, they require significantly fewer recursive splits to be isolated.
- The anomaly score $s(x, n)$ is derived from the average tree path length $E(h(x))$:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}, \quad \text{where } c(n) = 2\ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
  An anomaly score $s \to 1$ indicates a definite outlier.

---

## 3. Production Implementation: Leakage-Safe Imputer & Outlier Filter

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest

def clean_tabular_dataset(df: pd.DataFrame, contamination: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
    """Applies MICE imputation and Isolation Forest outlier filtering."""
    df_clean = df.copy()
    
    # 1. MICE Iterative Multivariate Imputation
    imputer = IterativeImputer(max_iter=10, random_state=42)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # 2. Multi-Dimensional Anomaly Isolation
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = iso.fit_predict(df_clean[numeric_cols])
    is_inlier = pd.Series(outlier_labels == 1, index=df_clean.index)
    
    return df_clean[is_inlier], is_inlier
```
