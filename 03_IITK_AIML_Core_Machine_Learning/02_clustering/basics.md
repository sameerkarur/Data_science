# Chapter 2: Unsupervised Clustering & Density Manifolds
**Comprehensive Textbook Guide — Advanced Machine Learning**

---

## 1. Executive Overview & Mental Models

Clustering discovers natural groupings in unlabeled data. Different algorithms make fundamentally different geometric assumptions: K-Means assumes convex, spherical isotropic clusters; DBSCAN discovers non-linear density contours while isolating noise points; Gaussian Mixture Models (GMM) formulate soft probabilistic boundaries.

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

## 2. Deep Theoretical Foundations

### 1. K-Means Objective Function (Inertia)
K-Means solves the non-convex optimization problem of minimizing within-cluster sum of squares (WCSS):
$$\mathcal{J} = \sum_{k=1}^K \sum_{x \in S_k} \| x - \mu_k \|_2^2$$
Because finding the global minimum is NP-hard, standard K-Means uses Lloyd's heuristic (Expectation-Maximization). **K-Means++ initialization** guarantees an expected approximation ratio of $O(\log K)$ relative to the optimal clustering.

### 2. Silhouette Coefficient Analysis
Evaluates cluster cohesion versus separation for each sample $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: Mean intra-cluster distance between sample $i$ and all other points in its own cluster.
- $b(i)$: Mean nearest-cluster distance between sample $i$ and points in the closest neighboring cluster.
- Score ranges from $-1$ (incorrect cluster assignment) to $+1$ (dense, highly separated clusters).

### 3. Density-Based Spatial Clustering (DBSCAN & HDBSCAN)
DBSCAN defines clusters as continuous regions of high point density separated by low-density regions:
- **$\epsilon$-Neighborhood:** $N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \epsilon\}$.
- **Core Point:** $|N_\epsilon(p)| \ge \text{MinPts}$.
- **Density-Reachable:** A point $p$ is density-reachable from core point $q$ if there is a chain of core points connecting them.
HDBSCAN (Hierarchical DBSCAN) constructs a minimum spanning tree over the mutual reachability distance graph, extracting clusters across variable density thresholds without requiring a global $\epsilon$.

---

## 3. Production Implementation: Optimal Cluster Discovery Pipeline

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_optimal_k(X: np.ndarray, k_range: range = range(2, 10)) -> dict[str, int | dict]:
    """Sweeps cluster counts and identifies optimal K via Silhouette & Davies-Bouldin metrics."""
    inertias, silhouettes, db_scores = {}, {}, {}
    best_k = 2
    best_sil = -1.0

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(X)
        
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        
        inertias[k] = float(km.inertia_)
        silhouettes[k] = sil
        db_scores[k] = db
        
        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "optimal_k": best_k,
        "best_silhouette": best_sil,
        "silhouette_profile": silhouettes,
        "davies_bouldin_profile": db_scores,
        "inertia_profile": inertias
    }
```
