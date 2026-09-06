# Project 07 (V2): Tabular ResNet & Focal Loss for Credit Risk
**Next-Generation Deep Learning Architecture**  
*Curriculum: Core — Deep Learning with Keras & TensorFlow*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Neural Topology** | Standard sequential feedforward Multi-Layer Perceptron (MLP) | Deep Tabular ResNet with residual skip connections ($x + \mathcal{F}(x)$) |
| **Gradient Flow** | Susceptible to saturation and vanishing gradients in deep layers | Preserved identity shortcuts and Batch Normalization |
| **Loss Function** | Standard Binary Cross-Entropy (overwhelmed by non-default class) | Focal Loss $\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ focusing on hard minority cases |
| **Uncertainty Quantification**| Deterministic point estimate | Monte Carlo Dropout at inference quantifying model epistemic uncertainty |
| **Calibration** | Uncalibrated raw probabilities | Evaluated using Brier score calibration metric |

---

## 🚀 How to Run

```bash
python projects_version2/07_DL_Lending_Club_Tabular_ResNet_V2/lending_club_tabular_resnet_v2.py
```
