# Capstone 2 (V2): Hierarchical Stacking Ensemble & Safety Stock Engine
**Next-Generation Time-Series Forecasting Architecture**  
*Curriculum: Capstone Project 2*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Model Architecture** | Single-model XGBoost Regressor | Two-Tier Stacking Ensemble (Gradient Boosting + Random Forest + Ridge Meta-Learner) |
| **Seasonality Modeling**| Simple integer day/month columns | Fourier Harmonic Oscillators ($\sin, \cos$ components) capturing annual and weekly cycles |
| **Validation Framework**| Random train/test split | Strict Forward Chronological Split preventing future data leakage |
| **Output Type** | Point forecast only | Probabilistic Uncertainty Bands ($P_{10}, P_{50}, P_{90}$) |
| **Supply Chain Impact** | Abstract error reporting | Automated Safety Stock Buffer ($SS = Z \times \sigma_L$) and Reorder Point calculation |

---

## 🚀 How to Run

```bash
python projects_version2/14_Capstone2_Hierarchical_Probabilistic_Demand_Forecasting_V2/sales_demand_forecasting_v2.py
```
