# Class Imbalance Mitigation, Cost-Sensitive Learning & Resampling
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Problem with Class Imbalance (The Accuracy Paradox)](#1-the-problem-with-class-imbalance)
2. [Algorithmic Approaches vs Resampling Approaches](#2-algorithmic-vs-resampling)
3. [Oversampling: SMOTE & ADASYN (Synthetic Minority Generation)](#3-oversampling-smote--adasyn)
4. [Undersampling: Random Under-Sampler & Tomek Links](#4-undersampling-techniques)
5. [Cost-Sensitive Learning (`class_weight='balanced'`)](#5-cost-sensitive-learning)
6. [Optimal Decision Threshold Moving (Youden's J Statistic)](#6-optimal-decision-threshold-moving)
7. [Focal Loss for Extreme Imbalance](#7-focal-loss-for-extreme-imbalance)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. The Accuracy Paradox

In domains such as credit card fraud detection (99.8% legitimate, 0.2% fraud) or disease screening:
A naive model predicting `Legitimate` 100% of the time achieves **99.8% Accuracy**, yet catches **0% of fraud** (Recall = 0.0), resulting in catastrophic real-world failure.

```
                      THE SMOTE SYNTHETIC INTERPOLATION
      Feature 2
         ▲
         │                                   Majority Samples (O)
         │           O    O        O
         │              O    O   O
         │
         │                Minority Sample A (X)
         │                  \
         │                   \  Synthetic Sample X_new = A + λ*(B - A)
         │                    * ◄── Generated point on line segment!
         │                     \
         │                      \
         │                       Minority Sample B (X)
         └────────────────────────────────────────────────────────► Feature 1
```

---

## 2. SMOTE & Cost-Sensitive Classification in Python

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Generate imbalanced dataset (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.95, 0.05],
                           n_informative=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Raw Training Counts: Class 0 = {(y_train==0).sum()}, Class 1 = {(y_train==1).sum()}")

# Baseline Model (Naive)
base_clf = RandomForestClassifier(random_state=42)
base_clf.fit(X_train, y_train)

# Solution 1: Cost-Sensitive Learning (class_weight='balanced')
cost_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
cost_clf.fit(X_train, y_train)

# Solution 2: SMOTE Resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
smote_clf = RandomForestClassifier(random_state=42)
smote_clf.fit(X_resampled, y_resampled)

print("\n--- Model Comparisons on Unbalanced Test Set ---")
print("1. Baseline Recall (Class 1):       {:.1%}".format(
    classification_report(y_test, base_clf.predict(X_test), output_dict=True)['1']['recall']))
print("2. Cost-Sensitive Recall (Class 1): {:.1%}".format(
    classification_report(y_test, cost_clf.predict(X_test), output_dict=True)['1']['recall']))
print("3. SMOTE Model Recall (Class 1):    {:.1%}".format(
    classification_report(y_test, smote_clf.predict(X_test), output_dict=True)['1']['recall']))
```

#### Output:
```text
Raw Training Counts: Class 0 = 665, Class 1 = 35

--- Model Comparisons on Unbalanced Test Set ---
1. Baseline Recall (Class 1):       46.7%
2. Cost-Sensitive Recall (Class 1): 80.0%
3. SMOTE Model Recall (Class 1):    80.0%
```

---

## 3. Threshold Moving: Youden's J Statistic

Rather than blindly using the default threshold $\tau = 0.5$, threshold moving sweeps across ROC curve operating points to find the threshold maximizing Sensitivity + Specificity:

$$J = \text{TPR} - \text{FPR} = \text{Recall} + \text{Specificity} - 1$$

```python
from sklearn.metrics import roc_curve
import numpy as np

probs = smote_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Find threshold that maximizes Youden's J
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Optimal Operating Threshold: {best_threshold:.3f}")
print(f"Optimized TPR (Recall):      {tpr[best_idx]:.1%}")
print(f"Associated FPR:              {fpr[best_idx]:.1%}")
```

#### Output:
```text
Optimal Operating Threshold: 0.280
Optimized TPR (Recall):      86.7%
Associated FPR:              6.3%
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Precision-Recall Curve Optimal Operating Point
**Task:** In high-imbalance problems, the Precision-Recall curve is preferred over ROC. Use `precision_recall_curve` to find the threshold that achieves at least $80\%$ Precision while maximizing Recall:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Filter where precision >= 0.80
valid_indices = np.where(precisions[:-1] >= 0.80)[0]
if len(valid_indices) > 0:
    best_i = valid_indices[np.argmax(recalls[valid_indices])]
    print(f"Target Threshold: {thresholds[best_i]:.3f}")
    print(f"Precision:        {precisions[best_i]:.1%}")
    print(f"Recall:           {recalls[best_i]:.1%}")
```
#### Output:
```text
Target Threshold: 0.420
Precision:        83.3%
Recall:           66.7%
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Approach | Implementation | Key Advantage |
|---|---|---|
| **SMOTE** | `imblearn.over_sampling.SMOTE()` | Synthesizes new minority examples |
| **ADASYN** | `imblearn.over_sampling.ADASYN()` | Focuses synthesis on borderline hard examples |
| **Cost-Sensitive** | `class_weight='balanced'` | Penalizes minority misclassification directly |
| **Focal Loss** | $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ | Down-weights easy majority examples |
| **Threshold Tuning**| Youden's J statistic | Zero-cost post-processing probability shift |
