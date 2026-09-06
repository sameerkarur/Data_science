# Unsupervised Clustering & Density-Based Partitioning
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 K-MEANS++ INITIALIZATION & VORONOI TESSELLATION
       Choose Initial Centroid c₁ Uniformly at Random
                           │
       Compute Squared Distance D(x)² from Nearest Centroid
                           │
       Sample Next Centroid cᵢ with Probability P(x) = D(x)² / Σ D(x')²
                           │
       Iterate: E-Step (Voronoi Assignment) ➔ M-Step (Mean Relocation)
                           │
       Convergence: Centroid Drift Δc < Tolerance ε
```

---

## 🧭 Deep Theoretical Foundations

### 1. K-Means Objective (Inertia)
$$\mathcal{J} = \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|^2$$
K-Means minimizes within-cluster sum of squares, assuming spherical clusters with isotropic variance.

### 2. Density-Based Clustering (DBSCAN & HDBSCAN)
Unlike K-Means, DBSCAN separates dense regions from sparse noise:
- Core Point: At least `min_samples` within radius $\epsilon$.
- Border Point: Within $\epsilon$ of a core point.
- Noise Point: Neither; filtered out as an anomaly.
