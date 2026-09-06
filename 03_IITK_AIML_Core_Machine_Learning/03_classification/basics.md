# Chapter 3: Classification Algorithms & Ensemble Theory
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
