# Classification Algorithms & Ensemble Theory
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Logistic Regression Log-Odds Formulation
$$\ln\left(rac{p}{1-p}ight) = w_0 + w_1 x_1 + \dots + w_d x_d \iff p = \sigma(w^T x) = rac{1}{1 + e^{-w^T x}}$$
Optimized via Cross-Entropy (Log Loss) minimization using gradient descent or L-BFGS.

### 2. Gradient Boosting (XGBoost) Second-Order Expansion
Gradient boosting optimizes arbitrary loss functions by fitting trees to negative gradients:
$$\mathcal{L}^{(t)} pprox \sum_{i=1}^n \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + rac{1}{2} h_i f_t^2(x_i) ight] + \Omega(f_t)$$
*(where $g_i$ is 1st derivative gradient, $h_i$ is 2nd derivative Hessian)*
