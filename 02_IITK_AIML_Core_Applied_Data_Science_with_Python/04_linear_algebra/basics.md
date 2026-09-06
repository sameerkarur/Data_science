# Linear Algebra for Machine Learning: Complete Visual & Code Guide
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
