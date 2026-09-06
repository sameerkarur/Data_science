# Capstone 3 (V2): ResNet50 Vision & Bayesian Personalized Ranking (BPR)
**Next-Generation Deep Vision & Tourism Recommender Architecture**  
*Curriculum: Capstone Project 3*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Vision Architecture** | MobileNetV2 with standard global pooling | Deep ResNet50 with Global Attention Pooling emphasizing architectural details |
| **Recommendation Paradigm**| Explicit SVD matrix factorization on 1-5 ratings | Bayesian Personalized Ranking (BPR) for Implicit Feedback (visits/bookmarks) |
| **Loss Optimization** | Root Mean Squared Error (pointwise) | Pairwise Ranking Optimization directly maximizing relative preference margins |
| **Cold-Start Strategy** | Global mean rating imputation | Hybrid Fallback blending visual aesthetic embeddings with user itinerary history |
| **Personalization** | Static recommendation list | Dynamic personalized rankings that filter out previously visited monuments |

---

## 🚀 How to Run

```bash
python projects_version2/15_Capstone3_Dual_Vision_BPR_Recommender_V2/heritage_vision_bpr_recommender_v2.py
```
