# Class Imbalance Mitigation & Cost-Sensitive Learning
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
$$\mathcal{L}_{	ext{cost}} = - \left( w_1 \cdot y \log \hat{y} + w_0 \cdot (1-y) \log(1 - \hat{y}) ight), \quad 	ext{where } w_1 = rac{N_{	ext{total}}}{2 \cdot N_{	ext{minority}}}$$

### 2. Precision-Recall AUC vs ROC-AUC
On highly skewed classes (e.g. 99.5% negatives), ROC-AUC paints an overly optimistic portrait because True Negative Count dominates the False Positive Rate denominator. PR-AUC evaluates true precision among predicted minority alerts.
