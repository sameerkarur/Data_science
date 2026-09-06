# Exploratory Data Analysis & Advanced Feature Engineering
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
$$\hat{S}_i = \lambda(n_i) \cdot ar{y}_i + (1 - \lambda(n_i)) \cdot ar{y}_{	ext{global}}, \quad 	ext{where } \lambda(n_i) = rac{1}{1 + e^{-(n_i - k) / f}}$$

### 2. Permutation Feature Importance vs Gini Impurity
Default Random Forest feature importances (Mean Decrease in Impurity - MDI) are severely biased toward continuous features with high cardinality. Permutation Feature Importance measures true drop in validation performance when feature values are randomly shuffled out-of-fold.
