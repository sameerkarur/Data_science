# Chapter 4: Class Imbalance & Cost-Sensitive Learning
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
