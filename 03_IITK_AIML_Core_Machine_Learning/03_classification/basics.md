# Supervised Classification & Ensemble Methods: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Scikit-Learn / XGBoost / W3Schools Style)**

---

## 📑 Table of Contents (On this page)
1. [Supervised Classification Taxonomy: Binary, Multiclass & Multi-label](#1-supervised-classification-taxonomy)
2. [Logistic Regression: Logit Link, Sigmoid & Binary Cross-Entropy Loss](#2-logistic-regression)
3. [Decision Trees: Shannon Entropy, Gini Impurity & CART Pruning](#3-decision-trees)
4. [Random Forests: Bagging & Out-of-Bag (OOB) Generalization](#4-random-forests)
5. [Gradient Boosting & XGBoost: Second-Order Taylor Expansion & Regularization](#5-gradient-boosting--xgboost)
6. [Comprehensive Evaluation Metrics: Confusion Matrix, ROC-AUC & PR-AUC](#6-comprehensive-evaluation-metrics)
7. [Common Pitfalls: Evaluating Imbalanced Classifiers with Accuracy](#7-common-pitfalls)
8. [Production Case Study: Enterprise Loan Default Risk Engine](#8-production-case-study-loan-default)
9. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet & Best Website Citations](#10-quick-reference-cheat-sheet--citations)

---

## 1. Logistic Regression & Binary Cross-Entropy

Logistic regression models the probability $p = P(y=1 \mid \mathbf{x})$ using the **Sigmoid function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b$$

The objective is to minimize **Binary Cross-Entropy (Log Loss)** via Gradient Descent:
$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(z_i) + (1 - y_i) \ln (1 - \sigma(z_i)) \right]$$

```
                         THE SIGMOID ACTIVATION
                       1.0 ┌───────────────────******
                           │             ******
                           │          ***
                       0.5 ┼─────────* (Decision Boundary at z=0)
                           │      ***
                           │******
                       0.0 └─────────────────────────
                          -6  -4  -2   0   2   4   6  (z)
```

---

## 2. Decision Trees: Gini Impurity vs Shannon Entropy

At each candidate split, CART selects feature $j$ and threshold $t$ that maximizes Impurity Reduction:
$$\text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2, \quad \text{Entropy}(D) = -\sum_{k=1}^K p_k \log_2 p_k$$

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
tree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
tree.fit(data.data, data.target)

print(f"Trained Tree Depth: {tree.get_depth()} | Leaf Nodes: {tree.get_n_leaves()}")
print(f"Top Split Feature: {data.feature_names[tree.tree_.feature[0]]}")
```

#### Output:
```text
Trained Tree Depth: 3 | Leaf Nodes: 8
Top Split Feature: worst perimeter
```

---

## 3. Gradient Boosting & XGBoost: The Mathematics

While Random Forests train trees independently in parallel (**Bagging**), Gradient Boosting trains trees **sequentially** on the negative gradients (pseudo-residuals) of the loss function:
$$\tilde{y}_i = -\left[ \frac{\partial \mathcal{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

XGBoost incorporates 2nd-order Taylor expansion and $L_1/L_2$ leaf regularization:
$$\text{Obj}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t(x_i)^2 \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
where $g_i$ is the gradient and $h_i$ is the Hessian.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

preds_proba = xgb_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, preds_proba)
print(f"XGBoost Test ROC-AUC Score: {auc_score:.4f}")
```

#### Output:
```text
XGBoost Test ROC-AUC Score: 0.9934
```

---

## 4. Evaluation Metrics: The Complete Confusion Matrix

```
                      CONFUSION MATRIX GEOMETRY
                                  ACTUAL CLASS
                             Positive (1)     Negative (0)
        PREDICTED  Positive  [ True Pos (TP)  | False Pos (FP) ] -> Precision = TP / (TP+FP)
        CLASS      Negative  [ False Neg (FN) | True Neg (TN)  ]
                                  │
                                  ▼
                        Recall / Sensitivity = TP / (TP+FN)
```

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Production Case Study: Loan Default Risk Engine

```python
from sklearn.metrics import classification_report

class LoanDefaultClassifier:
    """Production credit underwriting scoring engine."""
    def __init__(self):
        self.clf = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.08, eval_metric='logloss')

    def fit_and_report(self, X_tr, y_tr, X_te, y_te):
        self.clf.fit(X_tr, y_tr)
        preds = self.clf.predict(X_te)
        return classification_report(y_te, preds, target_names=["Good Credit", "Default"])

engine = LoanDefaultClassifier()
report = engine.fit_and_report(X_train, y_train, X_test, y_test)
print("Credit Underwriting Classification Report:\n", report)
```

#### Output:
```text
Credit Underwriting Classification Report:
               precision    recall  f1-score   support

 Good Credit       0.95      0.93      0.94        43
     Default       0.96      0.97      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```

---

## 6. Quick Reference Cheat Sheet & Best Website Citations

| Model | Linear? | Interpretability | Outlier Sensitivity | Typical Hyperparameters |
|---|---|---|---|---|
| **Logistic Regression** | Yes | High (odds ratios) | High | `C`, `penalty='l1'/'l2'` |
| **Random Forest** | No | Medium | Low | `n_estimators`, `max_depth` |
| **XGBoost** | No | Medium-Low | Low | `learning_rate`, `subsample` |

### 🌐 Official References & Recommended Reading:
- [Scikit-Learn Supervised Models Guide](https://scikit-learn.org/stable/supervised_learning.html)
- [XGBoost Official Documentation](https://xgboost.readthedocs.io/en/stable/)
- [W3Schools Logistic Regression & Decision Trees](https://www.w3schools.com/python/python_ml_logistic_regression.asp)
