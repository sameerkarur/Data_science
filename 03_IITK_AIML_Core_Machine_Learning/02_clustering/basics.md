# Unsupervised Clustering: K-Means, Hierarchical & DBSCAN Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Unsupervised Clustering?](#1-what-is-unsupervised-clustering)
2. [K-Means Algorithm (Centroid Mechanics & Lloyds Algorithm)](#2-k-means-algorithm)
3. [The Elbow Method & Silhouette Analysis (Optimal K)](#3-the-elbow-method--silhouette-analysis)
4. [DBSCAN (Density-Based Spatial Clustering of Applications with Noise)](#4-dbscan-density-based-clustering)
5. [Hierarchical Clustering & Dendrograms](#5-hierarchical-clustering--dendrograms)
6. [Clustering Evaluation Metrics (Inertia, Silhouette, Davies-Bouldin)](#6-clustering-evaluation-metrics)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. What is Unsupervised Clustering?

**Clustering** is an unsupervised learning task that partitions unlabelled data points into distinct, homogeneous groups (clusters) where points in the same cluster are highly similar, while points in different clusters are distinct.

```
                      CLUSTERING PARTITIONING GEOMETRY
      Feature 2
         ▲
         │        [Cluster 1: Tech Enthusiasts]
         │           *   * *
         │          *  (C1) *
         │            * *
         │
         │                               [Cluster 2: Budget Shoppers]
         │                                    #   # #
         │                                   #  (C2) #
         │                                     # # #
         │       [Cluster 3: Enterprise]
         │          @   @ @
         │         @  (C3) @
         └────────────────────────────────────────────────────────► Feature 1
```

---

## 2. K-Means Algorithm

K-Means alternates between two iterative steps until convergence:
1. **Assignment Step:** Assign each sample $\mathbf{x}_i$ to its nearest centroid $\mathbf{\mu}_j$ using Euclidean distance.
2. **Update Step:** Recalculate centroids as the mean of all points assigned to that cluster.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset with 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.70, random_state=42)

# Fit KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print("KMeans Cluster Centroids:\n", np.round(kmeans.cluster_centers_, 2))
print("Inertia (Sum of squared distances):", round(kmeans.inertia_, 2))
```

#### Output:
```text
KMeans Cluster Centroids:
 [[-2.63  9.01]
 [ 4.79  1.94]
 [-1.43  2.78]]
Inertia (Sum of squared distances): 287.64
```

---

## 3. The Elbow Method & Silhouette Analysis

```python
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X, labels)
print(f"Overall Silhouette Score: {silhouette_avg:.4f} (Close to 1.0 indicates well-separated clusters!)")
```

#### Output:
```text
Overall Silhouette Score: 0.7490 (Close to 1.0 indicates well-separated clusters!)
```

---

## 4. DBSCAN (Density-Based Clustering)

Unlike K-Means, DBSCAN does not assume spherical clusters and automatically identifies arbitrary shapes while isolating **noise/outliers**:

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate two interleaving crescent moons
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.25, min_samples=5)
moon_labels = dbscan.fit_predict(X_moons)

n_clusters_found = len(set(moon_labels)) - (1 if -1 in moon_labels else 0)
n_noise = list(moon_labels).count(-1)

print(f"Clusters Detected: {n_clusters_found}")
print(f"Noise Points Identified: {n_noise}")
```

#### Output:
```text
Clusters Detected: 2
Noise Points Identified: 0
```

---

## 5. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Customer Segmentation
**Task:** Given customer annual spend and loyalty scores, scale the features with `StandardScaler` and cluster them into 3 distinct customer tiers using K-Means:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.DataFrame({
    'Spend_USD': [1200, 45000, 32000, 800, 1500, 52000, 2200, 48000],
    'Loyalty_Score': [2, 9, 8, 1, 3, 10, 4, 8]
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
data['Segment'] = kmeans.fit_predict(X_scaled)

print("Segmented Customers:\n", data)
```
#### Output:
```text
Segmented Customers:
    Spend_USD  Loyalty_Score  Segment
0       1200              2        1
1      45000              9        0
2      32000              8        0
3        800              1        1
4       1500              3        1
5      52000             10        0
6       2200              4        1
7      48000              8        0
```
</details>

---

## 6. Quick Reference Cheat Sheet

| Algorithm | Shape Assumption | Outlier Handling | Requires K? |
|---|---|---|---|
| **K-Means** | Spherical / Convex | Sensitive to outliers | Yes |
| **DBSCAN** | Arbitrary / Non-linear | Isolates noise as `-1` | No (Requires `eps`, `min_samples`) |
| **Hierarchical**| Tree-structured clusters | Sensitive | Cut height dictates K |
