"""
Comprehensive, high-depth Basics & Architecture Guides for Course 3:
Core Machine Learning (5 modules)
"""

C03_BASICS = {}

# 1. EDA & Feature Engineering
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering"] = """# Exploratory Data Analysis & Advanced Feature Engineering
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Feature engineering transforms raw tabular observations into predictive numerical signals while avoiding target leakage.

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

## 🧭 Deep Theoretical Foundations

### 1. Target Encoding with Empirical Bayes Smoothing
Target encoding replaces a categorical level with the expected target value, but risks catastrophic overfitting on rare categories. Smoothing blends the category conditional mean with the global prior:
$$\hat{S}_i = \lambda(n_i) \cdot \bar{y}_i + (1 - \lambda(n_i)) \cdot \bar{y}_{\text{global}}, \quad \text{where } \lambda(n_i) = \frac{1}{1 + e^{-(n_i - k) / f}}$$

### 2. Permutation Feature Importance vs Gini Impurity
Default Random Forest feature importances (Mean Decrease in Impurity - MDI) are severely biased toward continuous features with high cardinality. Permutation Feature Importance measures true drop in validation performance when feature values are randomly shuffled out-of-fold.
"""

# 2. Clustering
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/02_clustering"] = """# Unsupervised Clustering & Density-Based Partitioning
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. K-Means Objective (Inertia)
$$\mathcal{J} = \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|^2$$
K-Means minimizes within-cluster sum of squares, assuming spherical clusters with isotropic variance.

### 2. Density-Based Clustering (DBSCAN & HDBSCAN)
Unlike K-Means, DBSCAN separates dense regions from sparse noise:
- Core Point: At least `min_samples` within radius $\epsilon$.
- Border Point: Within $\epsilon$ of a core point.
- Noise Point: Neither; filtered out as an anomaly.
"""

# 3. Classification
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/03_classification"] = """# Classification Algorithms & Ensemble Theory
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
$$\ln\left(\frac{p}{1-p}\right) = w_0 + w_1 x_1 + \dots + w_d x_d \iff p = \sigma(w^T x) = \frac{1}{1 + e^{-w^T x}}$$
Optimized via Cross-Entropy (Log Loss) minimization using gradient descent or L-BFGS.

### 2. Gradient Boosting (XGBoost) Second-Order Expansion
Gradient boosting optimizes arbitrary loss functions by fitting trees to negative gradients:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
*(where $g_i$ is 1st derivative gradient, $h_i$ is 2nd derivative Hessian)*
"""

# 4. Imbalanced Data
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data"] = """# Class Imbalance Mitigation & Cost-Sensitive Learning
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Cost-Sensitive Matrix & Loss Re-Weighting
Standard cross-entropy penalizes all errors symmetrically. Cost-sensitive cross-entropy scales the minority loss:
$$\mathcal{L}_{\text{cost}} = - \left( w_1 \cdot y \log \hat{y} + w_0 \cdot (1-y) \log(1 - \hat{y}) \right), \quad \text{where } w_1 = \frac{N_{\text{total}}}{2 \cdot N_{\text{minority}}}$$

### 2. Precision-Recall AUC vs ROC-AUC
On highly skewed classes (e.g. 99.5% negatives), ROC-AUC paints an overly optimistic portrait because True Negative Count dominates the False Positive Rate denominator. PR-AUC evaluates true precision among predicted minority alerts.
"""

# 5. Model Evaluation
C03_BASICS["03_IITK_AIML_Core_Machine_Learning/05_model_evaluation"] = """# Model Evaluation, Validation Rigor & Probability Calibration
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 STRATIFIED & TIME-SERIES CROSS VALIDATION
    Standard Stratified K-Fold:      Purged Time-Series Split:
    ┌───┬───┬───┬───┬───┐            ┌─────────┬─────────┬───────┐
    │Tst│Trn│Trn│Trn│Trn│            │  Train  │ Embargo │ Test  │
    └───┴───┴───┴───┴───┘            └─────────┴─────────┴───────┘
    (Preserves Class Ratio)          (Prevents Temporal Lookahead Leakage!)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Brier Score & Probability Calibration
Accuracy only checks binary thresholding ($p > 0.5$). The Brier Score measures true posterior calibration:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (f_i - y_i)^2$$
Calibrated probabilities ensure that when a model outputs 0.90 confidence, exactly 90 out of 100 predictions are positive. Calibrate via **Platt Scaling** (logistic sigmoid) or **Isotonic Regression**.
"""

print(f"Loaded {len(C03_BASICS)} comprehensive guides for Course 3.")
