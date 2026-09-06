# Project 08 (V2): Attentive Feature Masking & Calibrated Default Risk
**Next-Generation Deep Learning Architecture**  
*Curriculum: Core — Deep Learning with Keras & TensorFlow*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Feature Processing** | Uniform dense feeding to all neurons | Attentive Feature Masking Layer (dynamically weighting salient features) |
| **Model Transparency** | Black-box latent hidden weights | Direct inspection of per-feature attention coefficient masks |
| **Overfitting Control** | Standard Dropout layer only | Feature-level Sparsity and Batch Normalization |
| **Underwriting Suitability**| Raw uncalibrated Sigmoid output | Calibrated probability alignment evaluated via Brier Score |

---

## 🚀 How to Run

```bash
python projects_version2/08_DL_Home_Loan_Risk_TabNet_V2/home_loan_tabnet_v2.py
```
