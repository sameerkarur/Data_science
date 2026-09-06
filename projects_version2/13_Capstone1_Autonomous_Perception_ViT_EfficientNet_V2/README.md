# Capstone 1 (V2): EfficientNet Vision & Spatio-Temporal Hazard Model
**Next-Generation Computer Vision & Safety Architecture**  
*Curriculum: Capstone Project 1*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Vision Backbone** | MobileNetV2 | Modern EfficientNet-B0 with compound depth/width/resolution scaling |
| **Model Interpretability**| Raw output probabilities | Grad-CAM (Gradient-Weighted Class Activation Mapping) visual heatmaps |
| **Data Augmentation** | Minimal horizontal flip | Multi-stage augmentations: Random Rotation, Zoom & Contrast Jitter |
| **Safety Analytics** | Historical EDA charts on Tesla crashes | Real-Time Spatio-Temporal Hazard Index predicting dynamic collision risk |
| **Deployment Role** | Offline vehicle crop classifier | Dual perception-safety module ready for embedded ADAS integration |

---

## 🚀 How to Run

```bash
python projects_version2/13_Capstone1_Autonomous_Perception_ViT_EfficientNet_V2/autonomous_perception_v2.py
```
