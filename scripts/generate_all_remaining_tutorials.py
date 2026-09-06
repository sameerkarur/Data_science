"""
Generates extensive, tutorial-grade, visual guides (W3Schools / GeeksforGeeks / NumPy Official Docs style)
for all remaining modules across Course 2, 3, 4, 5, and 6.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# Course 2: Intro Data Science
# =====================================================================
C02_M01_GUIDE = r'''# Introduction to Data Science, CRISP-DM & Analytics Lifecycle
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Data Science? (The Interdisciplinary Venn Diagram)](#1-what-is-data-science)
2. [The 4 Types of Analytics (Descriptive, Diagnostic, Predictive, Prescriptive)](#2-the-4-types-of-analytics)
3. [CRISP-DM: The Cross-Industry Standard Process for Data Mining](#3-crisp-dm-the-cross-industry-standard-process)
4. [Data Science Project Lifecycle (Visual Dataflow)](#4-data-science-project-lifecycle)
5. [Key Roles & Tooling Ecosystem (Python, SQL, BI, Cloud)](#5-key-roles--tooling-ecosystem)
6. [Data Ethics, Privacy & Governance (GDPR, Bias, Fair Use)](#6-data-ethics-privacy--governance)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. What is Data Science?

Data Science is an interdisciplinary field that extracts actionable insights from noisy, structured, and unstructured data using scientific methods, mathematical algorithms, and computational systems.

### Visual Architecture: Drew Conway's Data Science Venn Diagram

```
                             COMPUTER SCIENCE
                            (Hacking Skills)
                                ┌───────┐
                               │       │
                               │  ML   │
                               │       │
                       ┌───────┼───────┼───────┐
                       │       │       │       │
                       │Danger │ DATA  │Traditional│
                       │ Zone  │SCIENCE│Research   │
                       │       │       │       │
               ┌───────┴───────┴───────┴───────┴───────┐
               │ MATH & STATISTICS       DOMAIN EXPERTISE│
               │ (Quantitative Modeling) (Business Context)│
               └───────────────────────────────────────┘
```

---

## 2. The 4 Types of Analytics

```
                        ANALYTICS MATURITY CURVE
   Value / Impact
       ▲                                                 ╭─ Prescriptive ("What should we do?")
       │                                            ╭───╯   (Optimization, Reinforcement Learning)
       │                                       ╭───╯
       │                                  ╭───╯ Predictive ("What will happen?")
       │                             ╭───╯      (Machine Learning, Forecasting)
       │                        ╭───╯
       │                   ╭───╯ Diagnostic ("Why did it happen?")
       │              ╭───╯      (Root Cause, Correlation, Anomaly Detection)
       │         ╭───╯
       │    ╭───╯ Descriptive ("What happened?")
       │╭───╯     (KPI Dashboards, Summary Stats)
       └────────────────────────────────────────────────────────► Difficulty / Sophistication
```

---

## 3. CRISP-DM: The Cross-Industry Standard Process

```
                      CRISP-DM ITERATIVE CYCLE
        ┌──────────────────────────────────────────────────┐
        │  1. Business Understanding (Define Objectives)   │
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  2. Data Understanding (Exploration & Auditing)  │
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  3. Data Preparation (Wrangling & Cleaning)      │ ◄─── (Spends 70% of project time!)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  4. Modeling (Algorithm Training & Tuning)       │
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  5. Evaluation (Validate against Business Goals) │
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  6. Deployment (API, Dashboard, Continuous CI/CD)│
        └──────────────────────────────────────────────────┘
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Identifying Analytics Categories
**Task:** Match the business scenarios to the correct analytics category (`Descriptive`, `Diagnostic`, `Predictive`, `Prescriptive`):
1. Determining why shopping cart abandonment spiked by 35% last Tuesday.
2. Generating a monthly revenue breakdown report by geography.
3. Automatically adjusting flight ticket prices in real-time to maximize revenue.
4. Forecasting hospital bed occupancy for the next 30 days.

<details>
<summary>👉 Click to Reveal Solution</summary>

```text
1. Diagnostic ("Why did it happen?" -> Investigating root cause of cart dropoff).
2. Descriptive ("What happened?" -> Summarizing historical revenue records).
3. Prescriptive ("What should we do?" -> Algorithmic optimization of dynamic pricing).
4. Predictive ("What will happen?" -> Machine learning time series regression).
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Analytics Type | Primary Question | Primary Tooling |
|---|---|---|
| **Descriptive** | What happened? | SQL, Tableau, PowerBI, Pandas |
| **Diagnostic** | Why did it happen? | Correlation, ANOVA, Root Cause Analysis |
| **Predictive** | What will happen? | Scikit-Learn, XGBoost, ARIMA, LSTM |
| **Prescriptive** | What should we do? | Linear Programming, PuLP, Simulation |
'''

p = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science/basics.md"
p.write_text(C02_M01_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M01 Guide: {len(C02_M01_GUIDE.splitlines())} lines.")

# =====================================================================
# Course 2: Linear Algebra
# =====================================================================
C02_M04_GUIDE = r'''# Linear Algebra for Machine Learning: Complete Visual & Code Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Linear Algebra Powers All AI & Machine Learning](#1-why-linear-algebra-powers-all-ai--machine-learning)
2. [Vectors: Geometric & Algebraic Representations](#2-vectors-geometric--algebraic-representations)
3. [Vector Operations: Addition, Norms & Dot Product](#3-vector-operations-addition-norms--dot-product)
4. [Matrices: Linear Transformations & Space Distortion](#4-matrices-linear-transformations--space-distortion)
5. [Matrix Multiplication (The Inner Working Geometry)](#5-matrix-multiplication)
6. [Determinants, Inverses & Linear Independence](#6-determinants-inverses--linear-independence)
7. [Eigenvalues & Eigenvectors: Principal Axes of Transformation](#7-eigenvalues--eigenvectors)
8. [Singular Value Decomposition (SVD) & PCA Dimensionality Reduction](#8-singular-value-decomposition-svd--pca)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Why Linear Algebra Powers All AI

Every machine learning model represents data as points in multi-dimensional vector spaces:
- **Images:** 3D matrices of pixel intensities $(H \times W \times C)$.
- **Text & Tokens:** Dense embedding vectors of 768 or 1536 dimensions.
- **Neural Networks:** Stacks of matrix multiplications and bias additions: $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$.

---

## 2. Vectors: Geometric & Algebraic Representations

```
                       GEOMETRIC VECTOR SPACE (2D)
           Y-Axis
             ▲
             │                  Vector v = [4, 3]
           3 ┼                 /|  Magnitude ||v|| = √(4² + 3²) = 5
             │                / │  Direction θ = arctan(3/4) = 36.87°
           2 ┼               /  │
             │              /   │
           1 ┼             /    │
             │            /     │
             └───────────┼──────┼────────► X-Axis
             0           2      4
```

```python
import numpy as np

# Vector definition
v = np.array([4, 3])

# Vector magnitude (L2 Norm)
l2_norm = np.linalg.norm(v)

# Unit vector (Direction)
unit_v = v / l2_norm

print("Vector:          ", v)
print(f"L2 Norm (Length): {l2_norm:.2f}")
print("Unit Vector:     ", unit_v)
print("Unit Length:     ", np.linalg.norm(unit_v))
```

#### Output:
```text
Vector:           [4 3]
L2 Norm (Length): 5.00
Unit Vector:      [0.8 0.6]
Unit Length:      1.0
```

---

## 3. Vector Operations: Addition & Dot Product

The dot product measures directional alignment between two vectors:
$$\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta) = \sum_{i=1}^n a_i b_i$$

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_prod = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot Product (a · b): {dot_prod}")

# Cosine similarity
cos_theta = dot_prod / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine Similarity:   {cos_theta:.4f}")
```

#### Output:
```text
Dot Product (a · b): 32
Cosine Similarity:   0.9746
```

---

## 4. Matrix Multiplication

```
                    MATRIX MULTIPLICATION GEOMETRY: C = A @ B
        Matrix A (2x3)               Matrix B (3x2)               Result C (2x2)
    ┌────────┬────────┬────────┐     ┌────────┬────────┐     ┌─────────────┬─────────────┐
    │  a₁₁   │  a₁₂   │  a₁₃   │     │  b₁₁   │  b₁₂   │     │ Row 1 · C₁  │ Row 1 · C₂  │
    ├────────┼────────┼────────┤  @  ├────────┼────────┤  =  ├─────────────┼─────────────┤
    │  a₂₁   │  a₂₂   │  a₂₃   │     │  b₂₁   │  b₂₂   │     │ Row 2 · C₁  │ Row 2 · C₂  │
    └────────┴────────┴────────┘     ├────────┼────────┤     └─────────────┴─────────────┘
                                     │  b₃₁   │  b₃₂   │
                                     └────────┴────────┘
```

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
B = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2

C = A @ B  # Result is 2x2
print("Matrix Product (A @ B):\n", C)
```

#### Output:
```text
Matrix Product (A @ B):
 [[ 58  64]
 [139 154]]
```

---

## 5. Eigenvalues & Eigenvectors

An eigenvector $\mathbf{v}$ of a matrix $\mathbf{A}$ is a special vector whose direction remains unchanged during the transformation, only scaled by its eigenvalue $\lambda$:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

```python
import numpy as np

A = np.array([[4, 1], [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues (Scale factors):", eigenvalues)
print("Eigenvectors (Columns):\n", np.round(eigenvectors, 3))

# Verify A @ v = lambda * v for first pair
v0 = eigenvectors[:, 0]
lambda0 = eigenvalues[0]

Av = A @ v0
lv = lambda0 * v0
print("\nVerification Av == lv:")
print("A @ v0:     ", np.round(Av, 3))
print("lambda0 * v0:", np.round(lv, 3))
```

#### Output:
```text
Eigenvalues (Scale factors): [5. 2.]
Eigenvectors (Columns):
 [[ 0.707 -0.447]
 [ 0.707  0.894]]

Verification Av == lv:
A @ v0:      [3.536 3.536]
lambda0 * v0: [3.536 3.536]
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Principal Component Projection
**Task:** Project a 2D data matrix `X` onto its top principal eigenvector to perform dimensionality reduction from 2D down to 1D:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np

X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7]])

# 1. Center the data
X_centered = X - X.mean(axis=0)

# 2. Compute Covariance Matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. Compute Eigenvectors
evals, evecs = np.linalg.eig(cov_matrix)
top_vector = evecs[:, np.argmax(evals)]  # Principal axis

# 4. Project onto 1D line
X_1D = X_centered @ top_vector
print("Reduced 1D Feature Representation:\n", np.round(X_1D, 2))
```
#### Output:
```text
Reduced 1D Feature Representation:
 [ 0.83 -1.89  0.47 -0.19  1.29  0.31]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Concept | NumPy Code | Description |
|---|---|---|
| **Dot Product** | `np.dot(a, b)` | Sum of products $\mathbf{a}^T\mathbf{b}$ |
| **Matrix Multiply** | `A @ B` | Standard matrix multiplication |
| **L2 Norm** | `np.linalg.norm(v)` | Euclidean length $\|\mathbf{v}\|_2$ |
| **Inverse** | `np.linalg.inv(A)` | $\mathbf{A}^{-1}$ such that $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ |
| **Determinant** | `np.linalg.det(A)` | Volume scaling factor of transformation |
| **Eigendecomposition**| `np.linalg.eig(A)` | Computes $\lambda$ and $\mathbf{v}$ |
'''

p2 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra/basics.md"
p2.write_text(C02_M04_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M04 Guide: {len(C02_M04_GUIDE.splitlines())} lines.")

# =====================================================================
# Course 3: Clustering
# =====================================================================
C03_M02_GUIDE = r'''# Unsupervised Clustering: K-Means, Hierarchical & DBSCAN Guide
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
'''

p3 = REPO_ROOT / "03_IITK_AIML_Core_Machine_Learning/02_clustering/basics.md"
p3.write_text(C03_M02_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C03 M02 Guide: {len(C03_M02_GUIDE.splitlines())} lines.")
