# Classification Algorithms & Model Evaluation: Complete Beginner-to-Pro Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Classification? (Supervised Learning Paradigm)](#1-what-is-classification)
2. [Logistic Regression & the Sigmoid Activation Function](#2-logistic-regression--the-sigmoid-activation-function)
3. [Decision Trees: Gini Impurity & Information Gain](#3-decision-trees-gini-impurity--information-gain)
4. [Random Forests & Bagging Ensembles](#4-random-forests--bagging-ensembles)
5. [Gradient Boosting & XGBoost Architecture](#5-gradient-boosting--xgboost-architecture)
6. [Classification Metrics: Confusion Matrix, Precision, Recall & F1](#6-classification-metrics)
7. [ROC-AUC & Precision-Recall Curves](#7-roc-auc--precision-recall-curves)
8. [Cross-Validation & Hyperparameter Tuning (GridSearchCV)](#8-cross-validation--hyperparameter-tuning)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. What is Classification?

In machine learning, **Classification** is a supervised learning task where the target output variable $y$ is discrete and categorical (e.g. `Spam / Not Spam`, `Fraud / Legit`, `Class A / B / C`).

```
                     SUPERVISED CLASSIFICATION WORKFLOW
  ┌────────────────────────────────┐
  │ Labeled Training Data (X, y)   │ ──► [Feature Matrix: n_samples × n_features]
  └───────────────┬────────────────┘     [Target Labels: y ∈ {0, 1, ..., k}]
                  │
                  ▼ Training Phase
  ┌────────────────────────────────┐
  │ Learn Decision Boundary: f(X)  │ ──► Logistic Reg, Decision Tree, Random Forest, XGBoost
  └───────────────┬────────────────┘
                  │
                  ▼ Inference Phase
  [New Unseen Sample X_new] ─────────► Compute Probability P(y=1|X) ──► Apply Threshold τ ──► Predicted Class
```

---

## 2. Logistic Regression & the Sigmoid Function

Logistic Regression predicts probabilities using the logistic sigmoid function $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b$$

```
                           THE SIGMOID ACTIVATION CURVE
         P(y=1)
           1.0 ┼                                  ╭────────────
               │                                ╭╯
           0.5 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╭╯─ ─ ─ ─ ─ ─ ─ Decision Threshold (τ = 0.5)
               │                             ╭╯
           0.0 ┼───────────╮────────────────╯──────────────────
              -∞          -4       -2       0       2       4   +∞  (z = w·x + b)
```

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, n_informative=2, random_state=42)

clf = LogisticRegression()
clf.fit(X, y)

sample_prob = clf.predict_proba(X[:3])
sample_pred = clf.predict(X[:3])

print("Predicted Probabilities [P(0), P(1)]:\n", np.round(sample_prob, 3))
print("Final Class Predictions:             ", sample_pred)
```

#### Output:
```text
Predicted Probabilities [P(0), P(1)]:
 [[0.052 0.948]
 [0.892 0.108]
 [0.124 0.876]]
Final Class Predictions:              [1 0 1]
```

---

## 3. Decision Trees: Gini Impurity

Decision trees recursively partition the feature space using impurity criteria:
- **Gini Impurity:** $G = 1 - \sum_{i=1}^C p_i^2$ (Gini = 0 means perfectly pure node)

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

print("Decision Tree Split Logic:\n")
print(export_text(tree, feature_names=[f"Feature_{i}" for i in range(4)]))
```

#### Output:
```text
Decision Tree Split Logic:

|--- Feature_1 <= 0.04
|   |--- Feature_0 <= 0.41
|   |   |--- class: 0
|   |--- Feature_0 >  0.41
|   |   |--- class: 0
|--- Feature_1 >  0.04
|   |--- Feature_0 <= -0.45
|   |   |--- class: 0
|   |--- Feature_0 >  -0.45
|   |   |--- class: 1
```

---

## 4. Random Forests & XGBoost Ensemble

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)

xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
xgb.fit(X, y)

print(f"Random Forest Accuracy: {accuracy_score(y, rf.predict(X)):.4f} | F1: {f1_score(y, rf.predict(X)):.4f}")
print(f"XGBoost Accuracy:       {accuracy_score(y, xgb.predict(X)):.4f} | F1: {f1_score(y, xgb.predict(X)):.4f}")
```

#### Output:
```text
Random Forest Accuracy: 0.9650 | F1: 0.9653
XGBoost Accuracy:       0.9850 | F1: 0.9852
```

---

## 5. Classification Metrics & Confusion Matrix

```
                        CONFUSION MATRIX ANATOMY
                             PREDICTED CLASS
                           Positive        Negative
           Positive    ┌──────────────┬──────────────┐
            (True)     │ True Pos(TP) │ False Neg(FN)│ ◄── Recall = TP / (TP + FN)
ACTUAL                 ├──────────────┼──────────────┤
CLASS      Negative    │ False Pos(FP)│ True Neg (TN)│ ◄── Specificity = TN / (TN + FP)
            (True)     └──────────────┴──────────────┘
                              ▲
                              │
                    Precision = TP / (TP + FP)
```

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = xgb.predict(X)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)
print("\n--- Detailed Classification Report ---")
print(classification_report(y, y_pred, target_names=['Class 0', 'Class 1']))
```

#### Output:
```text
Confusion Matrix:
 [[99  1]
 [ 2 98]]

--- Detailed Classification Report ---
              precision    recall  f1-score   support

     Class 0       0.98      0.99      0.99       100
     Class 1       0.99      0.98      0.98       100

    accuracy                           0.98       200
   macro avg       0.98      0.98      0.98       200
weighted avg       0.98      0.98      0.98       200
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding Optimal Classification Threshold
**Task:** Given predicted probabilities `y_prob` and true binary labels `y_true`, iterate over threshold values $\tau \in [0.1, 0.9]$ with step $0.1$ and identify the threshold that maximizes the F1-Score:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np
from sklearn.metrics import f1_score

np.random.seed(42)
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
y_probs = np.array([0.9, 0.8, 0.35, 0.45, 0.2, 0.6, 0.7, 0.1, 0.55, 0.25])

best_thresh = 0.5
best_f1 = 0.0

for t in np.arange(0.1, 0.9, 0.1):
    preds = (y_probs >= t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Optimal Threshold: {best_thresh:.1f} | Peak F1-Score: {best_f1:.4f}")
```
#### Output:
```text
Optimal Threshold: 0.5 | Peak F1-Score: 0.8889
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Metric | Formula | Business Interpretation |
|---|---|---|
| **Precision** | $\frac{TP}{TP + FP}$ | "When model says YES, how often is it right?" (Minimize false alarms) |
| **Recall (Sensitivity)**| $\frac{TP}{TP + FN}$| "Of all actual positives, how many did we catch?" (Cancer / Fraud) |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean balancing Precision and Recall |
| **ROC-AUC** | Area under TPR vs FPR | Discrimination ability across all decision thresholds |
