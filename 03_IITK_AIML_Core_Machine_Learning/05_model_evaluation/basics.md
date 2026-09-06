# Model Evaluation, Validation Rigor & Probability Calibration
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
$$	ext{BS} = rac{1}{N} \sum_{i=1}^N (f_i - y_i)^2$$
Calibrated probabilities ensure that when a model outputs 0.90 confidence, exactly 90 out of 100 predictions are positive. Calibrate via **Platt Scaling** (logistic sigmoid) or **Isotonic Regression**.
