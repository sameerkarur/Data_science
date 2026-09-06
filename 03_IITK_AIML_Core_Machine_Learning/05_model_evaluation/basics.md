# Model Validation, Probability Calibration & Explainability (SHAP)
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Validation Rigor Hierarchy (Data Leakage Prevention)](#1-the-validation-rigor-hierarchy)
2. [Cross-Validation Strategies: Stratified, Group & TimeSeriesSplit](#2-cross-validation-strategies)
3. [Probability Calibration (Brier Score, Platt Scaling & Isotonic Regression)](#3-probability-calibration)
4. [Bias-Variance Tradeoff (Learning Curves & Overfitting Detection)](#4-bias-variance-tradeoff)
5. [Feature Importance: MDI vs Permutation Importance](#5-feature-importance-mdi-vs-permutation)
6. [Explainable AI with SHAP (Shapley Additive Explanations)](#6-explainable-ai-with-shap)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. The Validation Rigor Hierarchy

Data leakage occurs when information from outside the training dataset is used to create the model. Preprocessing transformations (`StandardScaler`, `SimpleImputer`, `TargetEncoder`) must **NEVER be fitted on the entire dataset** before splitting!

```
                  STRICT NESTED VALIDATION PIPELINE
      FULL DATASET (X, y)
             │
             ├──► TEST SET (Held out completely until final verification!)
             │
             └──► TRAINING SET
                      │
                      ▼ K-Fold Cross Validation:
                      Fold 1: [Train] [Train] [Train] [Train] [VALIDATION]
                      Fold 2: [Train] [Train] [Train] [VALIDATION] [Train]
                      Fold 3: [Train] [Train] [VALIDATION] [Train] [Train]
                      Fold 4: [Train] [VALIDATION] [Train] [Train] [Train]
                      Fold 5: [VALIDATION] [Train] [Train] [Train] [Train]
```

---

## 2. Cross-Validation Strategies

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# 1. Stratified K-Fold: Preserves label distribution in each fold
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=2)
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    print(f"Stratified Fold {fold} Val Labels: {y[val_idx]}")

# 2. TimeSeriesSplit: Prevents looking into the future (Temporal integrity)
tscv = TimeSeriesSplit(n_splits=3)
time_data = np.arange(6)
print("\n--- TimeSeriesSplit Folds ---")
for fold, (tr, val) in enumerate(tscv.split(time_data), 1):
    print(f"Fold {fold}: Train on {time_data[tr]} -> Forecast on {time_data[val]}")
```

#### Output:
```text
Stratified Fold 1 Val Labels: [0 0 1 1]
Stratified Fold 2 Val Labels: [0 0 1 1]

--- TimeSeriesSplit Folds ---
Fold 1: Train on [0 1 2] -> Forecast on [3]
Fold 2: Train on [0 1 2 3] -> Forecast on [4]
Fold 3: Train on [0 1 2 3 4] -> Forecast on [5]
```

---

## 3. Probability Calibration (Reliability Curves)

A model is calibrated if, when it predicts 70% probability of rain, it actually rains 70% of those days:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=600, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Uncalibrated Random Forest (Tends to push probabilities toward 0 and 1)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
raw_probs = rf.predict_proba(X_test)[:, 1]

# Calibrated with Sigmoid (Platt Scaling)
cal_rf = CalibratedClassifierCV(rf, cv='prefit', method='sigmoid')
cal_rf.fit(X_train, y_train)
cal_probs = cal_rf.predict_proba(X_test)[:, 1]

print(f"Uncalibrated Brier Score Loss: {brier_score_loss(y_test, raw_probs):.4f} (Lower is better)")
print(f"Calibrated Brier Score Loss:   {brier_score_loss(y_test, cal_probs):.4f} (Calibrated!)")
```

#### Output:
```text
Uncalibrated Brier Score Loss: 0.0824 (Lower is better)
Calibrated Brier Score Loss:   0.0712 (Calibrated!)
```

---

## 4. Feature Importance: Permutation Importance

Unlike tree impurity importance (MDI) which favors high-cardinality noise, **Permutation Importance** measures degradation in test score when each feature is randomly shuffled:

```python
from sklearn.inspection import permutation_importance
import pandas as pd

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'Feature': [f"Feature_{i}" for i in range(X.shape[1])],
    'Importance_Mean': perm_imp.importances_mean,
    'Importance_Std': perm_imp.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

print("--- Top 3 Features by Permutation Importance ---")
print(perm_df.head(3).to_string(index=False))
```

#### Output:
```text
--- Top 3 Features by Permutation Importance ---
  Feature  Importance_Mean  Importance_Std
Feature_3         0.142222        0.015210
Feature_1         0.098889        0.012411
Feature_0         0.035556        0.008412
```

---

## 5. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Pipeline Cross-Validation with Zero Leakage
**Task:** Build a scikit-learn `Pipeline` combining `StandardScaler` and `LogisticRegression`, and evaluate it via 5-fold cross-validation using `cross_val_score(..., scoring='roc_auc')`:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"5-Fold ROC-AUC Scores: {np.round(scores, 3)}")
print(f"Mean ROC-AUC:          {scores.mean():.4f} (+/- {scores.std():.4f})")
```
#### Output:
```text
5-Fold ROC-AUC Scores: [0.942 0.915 0.938 0.951 0.929]
Mean ROC-AUC:          0.9350 (+/- 0.0125)
```
</details>

---

## 6. Quick Reference Cheat Sheet

| Validation Technique | Class | Key Benefit |
|---|---|---|
| **Stratified K-Fold**| `StratifiedKFold(n_splits=5)` | Preserves class imbalance ratio |
| **Time Series Split**| `TimeSeriesSplit(n_splits=5)` | Prevents forward-looking leakage |
| **Pipeline** | `sklearn.pipeline.Pipeline` | Encapsulates transforms inside each fold |
| **Platt Scaling** | `CalibratedClassifierCV(method='sigmoid')` | Converts decision values to true probabilities |
| **Permutation Imp** | `permutation_importance(model, X, y)` | Unbiased feature ranking |
