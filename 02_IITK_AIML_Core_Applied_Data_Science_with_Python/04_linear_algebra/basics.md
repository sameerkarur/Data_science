# Linear Algebra for Machine Learning & Vector Spaces
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Linear Algebra provides the language for transforming high-dimensional data manifolds.

```
                      SINGULAR VALUE DECOMPOSITION (SVD)
    ┌───────────────┐     ┌───────────────┐   ┌─────────┐   ┌───────────────┐
    │               │     │               │   │ Σ (Diag)│   │               │
    │   Data (A)    │  =  │   Left (U)    │ · │ Singular│ · │   Right (Vᵀ)  │
    │   (m × n)     │     │   (m × m)     │   │ Values  │   │   (n × n)     │
    │               │     │  Eigenvectors │   │ (m × n) │   │  Eigenvectors │
    └───────────────┘     └───────────────┘   └─────────┘   └───────────────┘
```

---

## 🧭 Deep Theoretical Foundations

### 1. Eigenvalues & Eigenvectors
For a square matrix $A$, a non-zero vector $v$ is an eigenvector with eigenvalue $\lambda$ if:
$$A v = \lambda v \iff (A - \lambda I)v = 0$$
Eigenvectors define the principal axes of variance in feature space, underpinning Principal Component Analysis (PCA).

### 2. Singular Value Decomposition (SVD)
Any real matrix $A \in \mathbb{R}^{m 	imes n}$ decomposes as $A = U \Sigma V^T$, where:
- $U$ contains orthogonal eigenvectors of $AA^T$.
- $V$ contains orthogonal eigenvectors of $A^T A$.
- $\Sigma$ contains ordered non-negative singular values $\sigma_i = \sqrt{\lambda_i}$.

---

## 💻 Production Implementation: PCA from First Principles

```python
import numpy as np

def compute_pca(X: np.ndarray, n_components: int = 2):
    """Calculates PCA via Covariance Eigendecomposition."""
    # 1. Mean center data
    X_centered = X - np.mean(X, axis=0)
    # 2. Covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 4. Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, sort_idx[:n_components]]
    # 5. Project onto principal manifold
    return np.dot(X_centered, top_components)
```
