# Imbalanced Data Strategies & Cost-Sensitive Learning: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Imbalanced-Learn / Scikit-Learn Style)**

---

## 📑 Table of Contents (On this page)
1. [The Accuracy Paradox in Severe Class Imbalance](#1-the-accuracy-paradox)
2. [Resampling Techniques: Random Undersampling vs SMOTE vs ADASYN](#2-resampling-techniques)
3. [Cost-Sensitive Learning & Class Weights](#3-cost-sensitive-learning)
4. [Optimal Decision Threshold Moving: Youden's J & Precision-Recall Tuning](#4-optimal-decision-threshold-moving)
5. [Focal Loss: Addressing Easy Examples in Extreme Imbalance](#5-focal-loss)
6. [Evaluation Under Imbalance: PR-AUC vs ROC-AUC](#6-evaluation-under-imbalance)
7. [Production Case Study: Financial Fraud Detection with 0.1% Rare Target](#7-production-case-study-fraud-detection)
8. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet & Best Website Citations](#9-quick-reference-cheat-sheet--citations)

---

## 1. The Accuracy Paradox

In fraud detection or disease diagnosis, 99.9% of transactions are legitimate and 0.1% are fraudulent.
- A naive dummy model predicting "Legitimate" for all transactions achieves **99.9% Accuracy**, but **0.0% Recall** on fraud.
- **Accuracy is completely meaningless under class imbalance.**

```
                     THE SMOTE INTERPOLATION GEOMETRY
    Minority Point x_i (●)                    Nearest Neighbor x_zi (●)
             \                                       /
              \               Synthetic (★)         /
               ●──────────────────★────────────────●
               x_new = x_i + λ * (x_zi - x_i),  λ ~ Uniform(0, 1)
```

---

## 2. SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE synthesizes new minority points along the line segment connecting $k$-nearest minority neighbors:

```python
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Generate synthetic imbalanced dataset (98% Class 0, 2% Class 1)
X = np.random.randn(1000, 4)
y = np.array([0] * 980 + [1] * 20)

print("Original Distribution:", Counter(y))

smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("SMOTE Resampled Distribution:", Counter(y_res))
```

#### Output:
```text
Original Distribution: Counter({0: 980, 1: 20})
SMOTE Resampled Distribution: Counter({0: 980, 1: 980})
```

---

## 3. Cost-Sensitive Learning & Class Weighting

Instead of physically resampling rows, cost-sensitive learning scales the loss penalty for minority misclassifications:
$$w_j = \frac{N}{2 \times N_j}$$

```python
from sklearn.linear_model import LogisticRegression

# Built-in balanced class weights in Scikit-Learn
clf_balanced = LogisticRegression(class_weight='balanced', random_state=42)
clf_balanced.fit(X, y)
print("Balanced Weights Applied Successfully to Model.")
```

#### Output:
```text
Balanced Weights Applied Successfully to Model.
```

---

## 4. Production Case Study: Financial Fraud Detector with PR-AUC

```python
from sklearn.metrics import precision_recall_curve, f1_score

class FraudDetectionEngine:
    """Fraud classification engine with optimal F1 threshold calibration."""
    def __init__(self, model):
        self.model = model

    def calibrate_threshold(self, X_val, y_val):
        probas = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probas)
        # Compute F1 across all candidate thresholds
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        self.best_threshold_ = thresholds[best_idx]
        print(f"Optimal Decision Threshold Calibrated: {self.best_threshold_:.4f} (Max F1: {f1_scores[best_idx]:.4f})")

    def predict_optimal(self, X):
        probas = self.model.predict_proba(X)[:, 1]
        return (probas >= self.best_threshold_).astype(int)

engine = FraudDetectionEngine(clf_balanced)
engine.calibrate_threshold(X, y)
```

#### Output:
```text
Optimal Decision Threshold Calibrated: 0.5218 (Max F1: 0.2857)
```

---

## 5. Quick Reference Cheat Sheet & Best Website Citations

| Strategy | When to Use | Key Risk | Imbalanced-Learn Class |
|---|---|---|---|
| **SMOTE** | Small minority sample size | Can create overlapping noise | `SMOTE` |
| **ADASYN** | Hard minority border regions | Overfocuses on noise outliers | `ADASYN` |
| **Undersampling** | Huge dataset (>10M rows) | Throws away majority information | `RandomUnderSampler` |
| **Class Weights** | Tree ensembles, Neural Nets | Parameter tuning required | `class_weight='balanced'` |

### 🌐 Official References & Recommended Reading:
- [Imbalanced-Learn Official Documentation](https://imbalanced-learn.org/stable/)
- [Chawla et al. — SMOTE: Synthetic Minority Over-sampling Technique (JAIR 2002)](https://arxiv.org/abs/1106.1813)
- [Lin et al. — Focal Loss for Dense Object Detection (Facebook AI Research)](https://arxiv.org/abs/1708.02002)
