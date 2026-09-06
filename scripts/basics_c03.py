"""
Textbook-Scale Architectural & Conceptual Guides for Course 3:
IITK AIML Core: Machine Learning
Modules:
  01_eda_feature_engineering
  02_clustering
  03_classification
  04_imbalanced_data
  05_model_evaluation
"""

C03_BASICS = {}

# =====================================================================
# 1. Exploratory Data Analysis & Advanced Feature Engineering
# =====================================================================
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering"] = r'''# Chapter 1: Exploratory Data Analysis & Feature Engineering
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
'''

# =====================================================================
# 2. Unsupervised Clustering & Density-Based Partitioning
# =====================================================================
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/02_clustering"] = r'''# Chapter 2: Unsupervised Clustering & Density Manifolds
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

Clustering discovers natural groupings in unlabeled data. Different algorithms make fundamentally different geometric assumptions: K-Means assumes convex, spherical isotropic clusters; DBSCAN discovers non-linear density contours while isolating noise points; Gaussian Mixture Models (GMM) formulate soft probabilistic boundaries.

```
                 K-MEANS++ INITIALIZATION & VORONOI TESSELLATION
       Choose Initial Centroid c₁ Uniformly at Random
                           │
       Compute Squared Distance D(x)² from Nearest Centroid
                           │
       Sample Next Centroid cᵢ with Probability P(x) = D(x)² / Σ D(x')²
                           │
       Iterate: E-Step (Voronoi Assignment) ➔ M-Step (Mean Relocation)
                           │
       Convergence: Centroid Drift Δc < Tolerance ε
```

---

## 2. Deep Theoretical Foundations

### 1. K-Means Objective Function (Inertia)
K-Means solves the non-convex optimization problem of minimizing within-cluster sum of squares (WCSS):
$$\mathcal{J} = \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|_2^2$$
Because finding the global minimum is NP-hard, standard K-Means uses Lloyd's heuristic (Expectation-Maximization). **K-Means++ initialization** guarantees an expected approximation ratio of $O(\log K)$ relative to the optimal clustering.

### 2. Silhouette Coefficient Analysis
Evaluates cluster cohesion versus separation for each sample $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: Mean intra-cluster distance between sample $i$ and all other points in its own cluster.
- $b(i)$: Mean nearest-cluster distance between sample $i$ and points in the closest neighboring cluster.
- Score ranges from $-1$ (incorrect cluster assignment) to $+1$ (dense, highly separated clusters).

### 3. Density-Based Spatial Clustering (DBSCAN & HDBSCAN)
DBSCAN defines clusters as continuous regions of high point density separated by low-density regions:
- **$\epsilon$-Neighborhood:** $N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \epsilon\}$.
- **Core Point:** $|N_\epsilon(p)| \ge \text{MinPts}$.
- **Density-Reachable:** A point $p$ is density-reachable from core point $q$ if there is a chain of core points connecting them.
HDBSCAN (Hierarchical DBSCAN) constructs a minimum spanning tree over the mutual reachability distance graph, extracting clusters across variable density thresholds without requiring a global $\epsilon$.

---

## 3. Production Implementation: Optimal Cluster Discovery Pipeline

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 10)) -> dict[str, int | dict]:
    """Sweeps cluster counts and identifies optimal K via Silhouette & Davies-Bouldin metrics."""
    inertias, silhouettes, db_scores = {}, {}, {}
    best_k = 2
    best_sil = -1.0

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X)
        
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        
        inertias[k] = float(km.inertia_)
        silhouettes[k] = sil
        db_scores[k] = db
        
        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "optimal_k": best_k,
        "best_silhouette": best_sil,
        "silhouette_profile": silhouettes,
        "davies_bouldin_profile": db_scores,
        "inertia_profile": inertias
    }
```
'''

# =====================================================================
# 3. Supervised Classification & Ensemble Architectures
# =====================================================================
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/03_classification"] = r'''# Chapter 3: Classification Algorithms & Ensemble Theory
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

Classification maps continuous and categorical feature spaces into discrete class distributions. Ensemble algorithms combine multiple weak base learners to construct strong predictive systems with provably reduced generalization error.

```
           BAGGING (RANDOM FOREST) vs BOOSTING (GRADIENT BOOSTING)
    ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
    │ BAGGING: Parallel Independent   │   │ BOOSTING: Sequential Residuals  │
    │ ┌──────┐ ┌──────┐ ┌──────┐      │   │ Tree 1 ──► Residual e₁          │
    │ │Tree 1│ │Tree 2│ │Tree 3│      │   │              ▼                  │
    │ └──────┘ └──────┘ └──────┘      │   │            Tree 2 ──► Resid e₂  │
    │   Average / Majority Vote       │   │                         ▼       │
    │   Reduces Model VARIANCE!       │   │                       Tree 3    │
    └─────────────────────────────────┘   │   Reduces Model BIAS!           │
                                          └─────────────────────────────────┘
```

---

## 2. Deep Theoretical Foundations

### 1. Logistic Regression & Binary Cross-Entropy
Logistic regression models the log-odds (logit) of the positive class as a linear combination of inputs:
$$\ln\left(\frac{P(Y=1 \mid x)}{1 - P(Y=1 \mid x)}\right) = w^T x + b \iff P(Y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
Optimized by minimizing negative log-likelihood (Binary Cross-Entropy):
$$\mathcal{L}(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \right]$$

### 2. Decision Tree Splitting Criteria
- **Gini Impurity:**
  $$I_G(p) = 1 - \sum_{k=1}^C p_k^2$$
- **Shannon Entropy:**
  $$H(p) = -\sum_{k=1}^C p_k \log_2(p_k)$$
Information Gain measures the reduction in impurity achieved by splitting node $D$ on feature $A$:
$$\text{Gain}(D, A) = I(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} I(D_v)$$

### 3. Gradient Boosting & Second-Order Taylor Expansion (XGBoost)
Unlike gradient descent in parameter space, gradient boosting performs gradient descent in **function space**. At step $t$, the objective is:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
Where:
- First-order gradient: $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$
- Second-order Hessian: $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$
This 2nd-order Taylor expansion allows analytic computation of optimal leaf weights $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$.

---

## 3. Production Implementation: Advanced Gradient Boosting Pipeline

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier

def train_production_boosting(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    """Trains histogram-based gradient boosting with early stopping and monotonic constraints."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
    print(f"Validation Log-Loss: {log_loss(y_val, val_probs):.4f}")
    return clf
```

---

## 4. Algorithmic Complexity Comparison

| Algorithm | Training Time | Inference Latency | Non-Linearity Handling | Interpretability |
|---|---|---|---|---|
| **Logistic Regression** | $O(N \cdot D)$ (L-BFGS) | $O(D)$ (Vector dot) | Linear (Requires manual terms) | High (Coefficients) |
| **Random Forest** | $O(M \cdot D \cdot N \log N)$ | $O(M \cdot \text{depth})$ | High | Medium (MDI, Permutation) |
| **Gradient Boosting (XGB)** | $O(M \cdot K \cdot N)$ (Hist) | $O(M \cdot \text{depth})$ | SOTA for Tabular | High via SHAP |
'''

# =====================================================================
# 4. Class Imbalance Mitigation & Cost-Sensitive Learning
# =====================================================================
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data"] = r'''# Chapter 4: Class Imbalance & Cost-Sensitive Learning
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

When the target class distribution is severely skewed (e.g. 1 positive case per 1,000 negative cases in fraud or rare disease detection), standard loss functions minimize total error by predicting the majority class exclusively. Remedying imbalance requires synthetic feature space interpolation, cost-sensitive re-weighting, and precision-recall threshold optimization.

```
                 SMOTE (SYNTHETIC MINORITY OVERSAMPLING)
       Select Minority Instance xᵢ
                    │
       Find k Nearest Minority Neighbors {x₁ₙ, x₂ₙ, ...}
                    │
       Pick Random Neighbor xᵣ
                    │
       Synthesize New Point: x_new = xᵢ + λ · (xᵣ - xᵢ),  where λ ∈ [0, 1]
```

---

## 2. Deep Theoretical Foundations

### 1. SMOTE & ADASYN Mechanics
- **SMOTE (Synthetic Minority Over-sampling Technique):** Creates synthetic minority samples along the line segments joining $k$-nearest minority neighbors:
  $$x_{\text{new}} = x_i + \lambda (x_{zi} - x_i), \quad \lambda \sim \mathcal{U}(0, 1)$$
- **ADASYN (Adaptive Synthetic):** Uses a density distribution $r_i$ to generate more synthetic examples for minority instances that are harder to learn (those surrounded by majority class points).

### 2. Cost-Sensitive Loss Re-Weighting
Instead of altering the training data distribution via resampling, cost-sensitive learning modifies the empirical risk objective:
$$\mathcal{L}_{\text{cost}} = - \frac{1}{N} \sum_{i=1}^N \left[ w_1 \cdot y_i \log \hat{y}_i + w_0 \cdot (1 - y_i) \log(1 - \hat{y}_i) \right]$$
Where weights are typically inverse-frequency balanced:
$$w_1 = \frac{N}{2 \cdot N_{\text{minority}}}, \quad w_0 = \frac{N}{2 \cdot N_{\text{majority}}}$$

### 3. Threshold Moving via Youden's J Statistic
The default threshold $p \ge 0.5$ assumes symmetric error costs. In imbalanced problems, optimal decision boundaries are discovered by maximizing Youden's J statistic or maximizing the F1 score over the Precision-Recall curve:
$$J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}$$

---

## 3. Production Implementation: Imbalanced Pipeline with Dynamic Thresholding

```python
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

def optimize_classification_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    """Finds decision threshold that strictly maximizes the F1-score on imbalanced validation data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    # Exclude division by zero
    f1_scores = np.where((precisions + recalls) > 0, 
                         2 * (precisions * recalls) / (precisions + recalls), 0.0)
    best_idx = np.argmax(f1_scores)
    # Thresholds has length len(precisions) - 1
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    best_f1 = float(f1_scores[best_idx])
    
    print(f"Optimal Threshold: {best_threshold:.4f} | Peak F1: {best_f1:.4f} (Default 0.5 F1: {f1_score(y_true, y_probs >= 0.5):.4f})")
    return best_threshold, best_f1
```
'''

# =====================================================================
# 5. Model Evaluation, Validation Rigor & Interpretability
# =====================================================================
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/05_model_evaluation"] = r'''# Chapter 5: Model Evaluation, Validation Rigor & Explainability
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

A machine learning model is only as reliable as its validation framework. In production systems, evaluation must quantify probabilistic calibration, out-of-fold generalizability, temporal stability, and feature attribution.

```
                 STRATIFIED & TIME-SERIES CROSS VALIDATION
    Standard Stratified K-Fold:      Purged Time-Series Split:
    ┌───┬───┬───┬───┬───┐            ┌─────────┬─────────┬───────┐
    │Tst│Trn│Trn│Trn│Trn│            │  Train  │ Embargo │ Test  │
    └───┴───┴───┴───┴───┘            └─────────┴─────────┴───────┘
    (Preserves Class Ratio)          (Prevents Temporal Lookahead Leakage!)
```

---

## 2. Deep Theoretical Foundations

### 1. Brier Score & Probability Calibration
Accuracy and ROC-AUC assess ranking; they do not assess whether model probabilities represent true event frequencies.
The **Brier Score** strictly evaluates probability calibration:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
- **Platt Scaling:** Fits a logistic regression model on uncalibrated model scores: $P(y=1 \mid f) = \frac{1}{1 + \exp(A f + B)}$.
- **Isotonic Regression:** Fits a non-parametric piecewise constant isotonic step function.

### 2. SHAP (SHapley Additive exPlanations)
Derived from cooperative game theory, Shapley values provide the unique fair allocation of credit among features that satisfies **Efficiency**, **Symmetry**, **Dummy**, and **Additivity**:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
SHAP decomposes any prediction $\hat{y}$ into the sum of feature contributions plus baseline expectation:
$$\hat{y} = \mathbb{E}[f(X)] + \sum_{i=1}^M \phi_i$$

---

## 3. Production Implementation: Probability Calibration & Diagnostics

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier

def calibrate_estimator(base_model: RandomForestClassifier, 
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> CalibratedClassifierCV:
    """Calibrates model output probabilities using Isotonic Regression with Brier verification."""
    # Pre-fit model
    base_model.fit(X_train, y_train)
    raw_probs = base_model.predict_proba(X_val)[:, 1]
    raw_brier = brier_score_loss(y_val, raw_probs)
    
    # Apply Isotonic Calibration
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)
    cal_probs = calibrated_model.predict_proba(X_val)[:, 1]
    cal_brier = brier_score_loss(y_val, cal_probs)
    
    print(f"Brier Score Improvement: {raw_brier:.4f} ➔ {cal_brier:.4f} (Lower is better)")
    return calibrated_model
```
'''

print(f"Loaded {len(C03_BASICS)} textbook chapters for Course 3.")
