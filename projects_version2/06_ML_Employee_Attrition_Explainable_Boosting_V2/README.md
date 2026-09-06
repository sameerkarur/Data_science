# Project 06 (V2): Explainable Gradient Boosting & Threshold Tuning
**Next-Generation Machine Learning Architecture**  
*Curriculum: Core — Machine Learning*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Model Family** | Random Forest (Bagging) | Gradient Boosted Decision Trees (Boosting with shrinkage $\eta=0.08$) |
| **Imbalance Strategy** | SMOTE artificial point synthesis | Cost-Sensitive Loss Weighting (`pos_weight = N_{neg} / N_{pos}`) |
| **Threshold Decision** | Fixed 0.50 threshold (sub-optimal for skewed classes) | Dynamic Precision-Recall F1-optimal threshold tuning |
| **Explainability (XAI)**| Basic impurity importances | Tree-based attribution ranking providing actionable employee retention insights |
| **Tenure Dynamics** | Static tenure histogram | Empirical Hazard Rate curve identifying peak flight-risk windows |

---

## 🚀 How to Run

```bash
python projects_version2/06_ML_Employee_Attrition_Explainable_Boosting_V2/employee_attrition_xgboost_shap_v2.py
```
