# Project 05 (V2): Density-Based Clustering & PCA Manifold Engine
**Next-Generation Machine Learning Architecture**  
*Curriculum: Core — Machine Learning*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Clustering Geometry** | Spherical, isotropic clusters (K-Means) | Arbitrary non-convex density shapes via DBSCAN & Hierarchical Ward linkage |
| **Outlier Handling** | Forces all points into a cluster | Natural outlier isolation (`noise_label = -1`) |
| **Dimensionality** | Manual subset of 4 raw features | Orthogonal PCA dimensionality reduction explaining >90% variance |
| **Validation Rigor** | Visual inertia elbow inspection | Quantitative Silhouette Score & Davies-Bouldin separation metrics |
| **Downstream Utility** | Generic cluster assignment table | Automated acoustic persona & mood playlist generation engine |

---

## 🚀 How to Run

```bash
python projects_version2/05_ML_Spotify_Cohorts_HDBSCAN_PCA_V2/spotify_hdbscan_pca_v2.py
```
