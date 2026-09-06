# Chapter 4: Linear Algebra for Machine Learning & Vector Spaces
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Linear algebra provides the mathematical language for transforming high-dimensional data spaces. In machine learning:
- A dataset is a collection of vectors in an $n$-dimensional Euclidean vector space $\mathbb{R}^n$.
- Neural network layers and projections are linear transformations represented as matrix multiplications.
- Principal Component Analysis (PCA) and dimensionality reduction are orthogonal projections onto eigenspaces.

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

## 2. Architectural Flowchart: Principal Component Analysis (PCA) Projection

```
                  PCA EIGENSYSTEM DIMENSIONALITY REDUCTION
                  
       Raw Data Matrix X (m samples, n features)
                           │
                           ▼
       Step 1: Mean Center Data
       X_c = X - μ_X  (Center of mass relocated to origin)
                           │
                           ▼
       Step 2: Empirical Covariance Matrix
       Σ = (1 / (m - 1)) · X_cᵀ X_c  (Size: n × n)
                           │
                           ▼
       Step 3: Spectral Eigendecomposition
       Σ vᵢ = λᵢ vᵢ  (Compute eigenvalues λ and eigenvectors v)
                           │
                           ▼
       Step 4: Rank-Sort Components Descending
       λ₁ ≥ λ₂ ≥ ... ≥ λ_n
                           │
                           ▼
       Step 5: Select Top-k Eigenvectors (Projection Matrix W_k)
                           │
                           ▼
       Step 6: Project onto Subspace Manifold
       Z = X_c · W_k  (Dimension reduced from n ➔ k with maximal variance!)
```

---

## 3. Deep Theoretical Foundations

### 1. Vector Spaces, Linear Independence & Rank
A set of vectors $\{v_1, v_2, \dots, v_k\}$ in $\mathbb{R}^n$ is **linearly independent** if:
$$c_1 v_1 + c_2 v_2 + \dots + c_k v_k = 0 \iff c_1 = c_2 = \dots = c_k = 0$$
The **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the maximal number of linearly independent column (or row) vectors. If $\text{rank}(A) < \min(m, n)$, the matrix is rank-deficient, indicating collinearity among features.

### 2. Spectral Theorem & Singular Value Decomposition (SVD)
Any real matrix $A \in \mathbb{R}^{m \times n}$ factorizes into:
$$A = U \Sigma V^T$$
Where:
- $U \in \mathbb{R}^{m \times m}$ is an orthonormal matrix containing the eigenvectors of $A A^T$.
- $V \in \mathbb{R}^{n \times n}$ is an orthonormal matrix containing the eigenvectors of $A^T A$.
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing non-negative singular values $\sigma_i = \sqrt{\lambda_i}$.

### 3. Eckart-Young-Mirsky Theorem
The optimal rank-$k$ approximation $A_k$ of matrix $A$ in terms of Frobenius norm is obtained by truncating the SVD at the top $k$ singular values:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T, \quad \min_{\text{rank}(B)=k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{j=k+1}^{\min(m,n)} \sigma_j^2}$$
This theorem is the mathematical backbone of Latent Semantic Analysis (LSA), image compression, and collaborative filtering recommendation systems.

---

## 4. Production Implementation: Full PCA from First Principles

```python
import numpy as np

class PrincipalComponentAnalysis:
    """Rigorous PCA via Covariance Matrix Eigendecomposition."""
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        m, n = X.shape
        # 1. Mean centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Covariance matrix computation
        cov_matrix = np.dot(X_centered.T, X_centered) / (m - 1)
        
        # 3. Hermitian Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        sorted_evals = eigenvalues[idx]
        sorted_evecs = eigenvectors[:, idx]
        
        # 5. Extract top k components
        self.components_ = sorted_evecs[:, :self.n_components]
        total_variance = np.sum(sorted_evals)
        self.explained_variance_ratio_ = sorted_evals[:self.n_components] / total_variance
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
```
