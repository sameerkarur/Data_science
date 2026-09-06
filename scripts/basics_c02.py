"""
Textbook-Scale Architectural & Conceptual Guides for Course 2 (Part 1):
IITK AIML Core: Applied Data Science with Python
Modules:
  01_intro_data_science
  02_python_essentials
  03_numpy
  04_linear_algebra
  05_statistics_fundamentals
  06_probability_distributions
"""

C02_BASICS_P1 = {}

# =====================================================================
# 1. Intro to Data Science & The Scientific Lifecycle
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science"] = r'''# Chapter 1: Data Science Foundations & CRISP-DM Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Data science is not merely the ad-hoc fitting of machine learning models to tabular matrices; it is a systematic, hypothesis-driven engineering discipline. The **Cross-Industry Standard Process for Data Mining (CRISP-DM)** provides the industry-standard cyclic blueprint that connects real-world business ROI to mathematical optimization.

```
                      CRISP-DM ITERATIVE ENGINEERING CYCLE
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. Business Understanding ◄────────────┐                        │
    │    │ (KPIs, Success Metrics, ROI)      │                        │
    │    ▼                                   │ Feedback Loop          │
    │ 2. Data Understanding                  │                        │
    │    │ (EDA, Distributions, Quality)     │                        │
    │    ▼                                   │                        │
    │ 3. Data Preparation                    │                        │
    │    │ (Cleaning, Scaling, Encoding)     │                        │
    │    ▼                                   │                        │
    │ 4. Modeling ───────────────────────────┘                        │
    │    │ (Algorithm Selection, Tuning)                              │
    │    ▼                                                            │
    │ 5. Evaluation ──[Meets Business Criteria?]──► 6. Deployment    │
    │         └── No ──► Re-assess Business & Data Pipeline          │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Flowchart: Data Leakage Elimination Pipeline

Data leakage—the inadvertent presence of validation/test or target-derived information in training features—is the single greatest cause of model failures in production.

```
                 STRICT LEAKAGE-FREE PARTITIONING PIPELINE
                 
       Raw Unprocessed Data Pool (Features X, Labels y)
                             │
                             ▼
       Step 1: Out-of-Time or Stratified Train-Test Split (80 / 20)
           ┌─────────────────┴─────────────────┐
           ▼                                   ▼
       Train Set (X_train, y_train)        Test Set (X_test, y_test)
           │                                   │ (SEALED VAULT: UNTOUCHED!)
           ▼                                   │
       Step 2: Fit Transformers                │
       • Compute train mean & std              │
       • Compute train medians for impute      │
       • Fit Target Encoders with Out-of-Fold  │
           │                                   │
           ▼                                   ▼
       Step 3: Transform Train             Step 4: Transform Test
       X_train_clean = scaler.transform()  X_test_clean = scaler.transform()
           │                               (Applies Train Stats ONLY!)
           ▼                                   │
       Step 5: Train Estimator                 ▼
       model.fit(X_train_clean, y_train) ──► Step 6: Evaluate Model on Test
```

---

## 3. Deep Theoretical Foundations

### 1. The Mathematical Decomposition of Generalization Error
For any supervised regression estimator $\hat{f}(x)$ trained on dataset $\mathcal{D}$, the expected mean squared error on an unseen sample $(x, y)$ decomposes into three mathematically distinct components:
$$\mathbb{E}_{\mathcal{D}, \epsilon}\left[(y - \hat{f}(x))^2\right] = \text{Bias}\left[\hat{f}(x)\right]^2 + \text{Var}\left[\hat{f}(x)\right] + \sigma^2_{\text{irreducible}}$$

Where:
- **Bias:** The error introduced by approximating an inherently complex real-world phenomenon with a simplified model:
  $$\text{Bias}\left[\hat{f}(x)\right] = \mathbb{E}_{\mathcal{D}}\left[\hat{f}(x)\right] - f(x)$$
- **Variance:** The variability of the model prediction across different training set samplings:
  $$\text{Var}\left[\hat{f}(x)\right] = \mathbb{E}_{\mathcal{D}}\left[\left(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}\left[\hat{f}(x)\right]\right)^2\right]$$
- **Irreducible Error ($\sigma^2$):** The intrinsic stochastic noise in the data generating process that cannot be eliminated by any estimator.

```
       Error
         ▲
         │       Total Expected Error = Bias² + Variance + Noise
         │         \             /
         │          \           /  ◄── Variance (Overfitting)
         │           \         /
         │   Bias² ───\_______/ 
         │ (Underfit)
         └──────────────────────────────────────► Model Complexity
```

---

## 4. Production Implementation: End-to-End Leakage-Proof Scikit-Learn Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report

# 1. Synthesizing Sample Heterogeneous Data
np.random.seed(42)
n_samples = 1000
raw_df = pd.DataFrame({
    'age': np.random.choice([25, 35, 45, np.nan, 60], size=n_samples),
    'income': np.random.exponential(scale=50000, size=n_samples),
    'education': np.random.choice(['HighSchool', 'Bachelors', 'Masters', 'PhD'], size=n_samples),
    'converted': np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples)
})

X = raw_df[['age', 'income', 'education']]
y = raw_df['converted']

# 2. Strict Partitioning BEFORE Any Computation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Defining Feature Subspaces
num_cols = ['age', 'income']
cat_cols = ['education']

# 4. Building Subspace Transformers
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipe, num_cols),
    ('cat', categorical_pipe, cat_cols)
])

# 5. Master Production Estimator Pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(class_weight='balanced'))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 5. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Fitting Preprocessors on Full Datasets
Scaling or imputing prior to cross-validation introduces optimistic bias. A model may appear to achieve 95% accuracy in notebook experiments, only to degrade severely in production because the test distribution statistics leaked into training scalers.

### Pitfall 2: Optimizing the Wrong Evaluation Metric
In fraud detection or disease screening where positive cases represent $< 1\%$ of data, an estimator predicting all zeros achieves $> 99\%$ accuracy while failing completely. In imbalanced problems, accuracy must be discarded in favor of **Precision-Recall AUC (PR-AUC)** and **Expected Financial Utility Matrices**.
'''

# =====================================================================
# 2. Python Essentials for Data Science & Vectorization
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials"] = r'''# Chapter 2: High-Performance Python & Vectorization Essentials
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Pure Python code executes via the CPython virtual machine bytecode interpreter. While expressive, it suffers from heavy pointer dereferencing, dynamic type checking, and the **Global Interpreter Lock (GIL)**. Vectorized scientific computing replaces interpreted scalar loops with compiled C/Fortran SIMD operations over contiguous memory blocks.

```
       PURE PYTHON ITERATION LOOP (SLOW)           VECTORIZED C-CONTIGUOUS EXECUTION (FAST)
    ┌───────────────────────────────────┐        ┌─────────────────────────────────────────┐
    │ For each element:                 │        │ Single Instruction Multiple Data (SIMD) │
    │ 1. Fetch PyObject pointer         │        │ ┌───────────────┬───────────────┐       │
    │ 2. Unpack integer data            │        │ | Chunk [0..3]  | Chunk [4..7]  |       │
    │ 3. Perform dynamic type dispatch  │        │ └───────┬───────┴───────┬───────┘       │
    │ 4. Pack result into new PyObject  │        │         ▼               ▼               │
    │ Execution Speed: ~1.0x (Baseline) │        │ Hardware AVX-512 CPU Vector Registers   │
    └───────────────────────────────────┘        │ Execution Speed: ~50x–300x Acceleration │
                                                 └─────────────────────────────────────────┘
```

---

## 2. Architectural Flowchart: Hardware Cache Locality & Memory Bounding

```
                  CPU CACHE HIERARCHY & MEMORY THROUGHPUT
                  
    CPU Core ──► L1 Cache (32KB, ~1 ns latency, 64-byte Cache Lines)
                    │
                    ▼
                 L2 Cache (512KB - 1MB, ~3-5 ns latency)
                    │
                    ▼
                 L3 Cache (Shared 16-64MB, ~10-15 ns latency)
                    │
                    ▼
                 Main RAM (DDR4/DDR5, ~60-100 ns latency)
                 
    PYTHON LIST: Non-contiguous pointers scattered across heap.
                 Causes frequent L1/L2 CACHE MISSES (Pointer Chasing).
                 
    NUMPY NDARRAY: Packed contiguous raw C array.
                   Fills entire 64-byte cache line per read!
```

---

## 3. Deep Theoretical Foundations

### 1. The Global Interpreter Lock (GIL) Mechanics
In CPython, memory management is non-thread-safe due to the reference counting mechanism (`ob_refcnt`). The GIL is a mutual exclusion lock that prevents multiple native OS threads from executing Python bytecodes concurrently. However, vectorized numerical libraries (NumPy, SciPy, PyTorch) explicitly release the GIL (`Py_BEGIN_ALLOW_THREADS`) before entering C routines, enabling true multicore CPU parallelism.

### 2. SIMD (Single Instruction Multiple Data)
Modern CPUs contain specialized 256-bit (AVX2) and 512-bit (AVX-512) vector registers. Instead of performing 4 separate scalar float64 multiplications across 4 clock cycles, an AVX instruction loads four 64-bit floats into a single vector register and computes all four products in a single hardware cycle.

---

## 4. Production Implementation: Profiling & Accelerating Kernels with Numba

```python
import time
import numpy as np
import numba

def python_monte_carlo_pi(nsamples: int) -> float:
    """Calculates Pi using pure interpreted Python."""
    import random
    acc = 0
    for _ in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) <= 1.0:
            acc += 1
    return 4.0 * acc / nsamples

@numba.njit(parallel=True, fastmath=True)
def numba_monte_carlo_pi(nsamples: int) -> float:
    """JIT-compiled to native machine code with multi-threaded SIMD."""
    acc = 0
    for i in numba.prange(nsamples):
        x = np.random.random()
        y = np.random.random()
        if (x*x + y*y) <= 1.0:
            acc += 1
    return 4.0 * acc / nsamples

# Benchmark execution:
n = 10_000_000

# Warm-up JIT compiler
numba_monte_carlo_pi(1000)

t0 = time.perf_counter()
res_numba = numba_monte_carlo_pi(n)
t_numba = time.perf_counter() - t0

print(f"Numba Parallel Execution: {t_numba:.4f}s (Result: {res_numba:.5f})")
```

---

## 5. Performance & Complexity Matrix

| Approach | Memory Per Float64 | CPU Cache Locality | Multithreading Speedup | Typical Acceleration |
|---|---|---|---|---|
| Python `list` Loop | 32 bytes (Pointer + PyFloat) | Dispersed (Heap Chasing) | Zero (GIL constrained) | $1.0\times$ (Baseline) |
| NumPy Vectorized | 8 bytes (Contiguous raw) | High (Streaming cache) | BLAS multi-threaded | $30\times - 80\times$ |
| Numba JIT Parallel | 8 bytes | Optimal (Registers) | Linear across cores | $100\times - 350\times$ |
'''

# =====================================================================
# 3. NumPy Numerical Computing & Memory Strides
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy"] = r'''# Chapter 3: NumPy Numerical Computing & Array Memory Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

A NumPy `ndarray` separates **Array Metadata** (shape, strides, dtype, flags) from the contiguous **Data Buffer**. This decoupling allows NumPy to perform operations like transposition, slicing, and reshaping in $O(1)$ constant time without duplicating array memory.

```
                NUMPY NDARRAY MEMORY ARCHITECTURE
    ┌────────────────────────────────────────────────────────┐
    │ ndarray Metadata Header:                               │
    │   • dtype: float64 (8 bytes per item)                  │
    │   • shape: (2, 3) ──► 2 rows, 3 columns                │
    │   • strides: (24, 8) ──► Bytes to advance per axis!   │
    └───────────────────────────┬────────────────────────────┘
                                │ Points to Raw C-Buffer
                                ▼
    ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
    │ Byte 0-7 │ Byte 8-15│Byte 16-23│Byte 24-31│Byte 32-39│Byte 40-47│
    │  [0, 0]  │  [0, 1]  │  [0, 2]  │  [1, 0]  │  [1, 1]  │  [1, 2]  │
    └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. Architectural Flowchart: Broadcasting Rules Engine

NumPy broadcasts arrays of mismatched shapes across element-wise operations using a deterministic alignment protocol:

```
                     BROADCASTING COMPATIBILITY RESOLUTION
                     
       Array A: Shape (5, 1, 32)
       Array B: Shape    (4, 32)
                            │
                            ▼
       Step 1: Right-Align Dimensions
               Array A:  5  x  1  x  32
               Array B:  1  x  4  x  32  (Prepend 1 to shorter array)
                            │
                            ▼
       Step 2: Compare Dimensions from Right to Left
               Dim 3: 32 vs 32  ──► Equal? YES! Match.
               Dim 2:  1 vs  4  ──► One is 1? YES! Broadcast 1 ➔ 4.
               Dim 1:  5 vs  1  ──► One is 1? YES! Broadcast 1 ➔ 5.
                            │
                            ▼
       Resulting Broadcast Output Shape: (5, 4, 32)
       (Achieved with ZERO data replication via stride 0 manipulation!)
```

---

## 3. Deep Theoretical Foundations

### 1. Memory Strides & Zero-Copy Views
The `strides` attribute is a tuple specifying the number of bytes to step in memory to advance by one index along each dimension.
For a 2D array of shape $(M, N)$ and data type size $S$ bytes:
- **C-Contiguous (Row-Major):** Elements in a row are adjacent in memory.
  $$\text{Strides} = (N \times S, S)$$
- **Fortran-Contiguous (Column-Major):** Elements in a column are adjacent.
  $$\text{Strides} = (S, M \times S)$$
- **Transposition (`arr.T`):** Swapping axes simply swaps the stride tuple $(N \times S, S) \to (S, N \times S)$. No bytes are moved in memory; it executes instantaneously in $O(1)$ time.

### 2. The Stride 0 Trick
When broadcasting a dimension of size 1 across size $K$, NumPy sets that dimension's stride to **0 bytes**. Every index access along that dimension references the exact same physical memory address, consuming zero additional RAM.

---

## 4. Production Implementation: Memory-Mapped Arrays for Massive Datasets

```python
import numpy as np
from pathlib import Path

def process_huge_matrix(filepath: Path | str, rows: int = 100_000, cols: int = 256) -> np.ndarray:
    """Uses memmap to process multi-gigabyte matrices with minimal RAM footprint."""
    # 1. Create a binary memory-mapped array on disk (100,000 x 256 x 4 bytes ≈ 102 MB)
    mmap_arr = np.memmap(filepath, dtype='float32', mode='w+', shape=(rows, cols))
    
    # 2. Populate chunks incrementally without memory bloat
    chunk_size = 10_000
    for i in range(0, rows, chunk_size):
        mmap_arr[i : i + chunk_size] = np.random.randn(chunk_size, cols).astype('float32')
        
    mmap_arr.flush()  # Commit to storage
    
    # 3. Read specific submatrix with zero-copy slice
    read_view = np.memmap(filepath, dtype='float32', mode='r', shape=(rows, cols))
    top_embeddings = read_view[:5, :]  # Instantaneous slice
    return np.array(top_embeddings)
```

---

## 5. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Modifying a View Expecting an Isolated Copy
```python
original = np.zeros((3, 3))
view_slice = original[:2, :2]
view_slice[0, 0] = 999  # MODIFIES 'original[0, 0]' AS WELL!

# PRODUCTION FIX: Force copy if isolation is required:
isolated_copy = original[:2, :2].copy()
```

### Pitfall 2: Inadvertent Temporary Array Allocation
```python
# ALLOCATES THREE TEMPORARY ARRAYS IN MEMORY:
# result = 2 * A + 3 * B - C

# PRODUCTION IN-PLACE FIX:
A *= 2
A += (3 * B)
A -= C
```
'''

# =====================================================================
# 4. Linear Algebra for Machine Learning & Vector Spaces
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra"] = r'''# Chapter 4: Linear Algebra for Machine Learning & Vector Spaces
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
'''

# =====================================================================
# 5. Statistics Fundamentals & Sampling Distributions
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals"] = r'''# Chapter 5: Statistical Foundations & Sampling Distributions
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Statistics bridges the epistemic gap between finite, noisy observational samples and the true, unobserved population data-generating mechanism. In predictive modeling and experimentation:
- **Descriptive Statistics:** Condenses high-dimensional sample matrices into summary indicators of central location, scale, and shape.
- **Inferential Statistics:** Quantifies confidence bounds and tests hypotheses concerning unobserved population parameters.

```
                 POPULATION VS SAMPLE PARAMETER ESTIMATION
    POPULATION (Target Universe):
    • Size: N (Often infinite or unobservable)
    • True Mean: μ = (1/N) Σ Xᵢ
    • True Variance: σ² = (1/N) Σ (Xᵢ - μ)²
                         │
                         ▼ Random Sampling (Size n << N)
    SAMPLE (Observed Data):
    • Size: n
    • Sample Mean: X̄ = (1/n) Σ Xᵢ  (Unbiased Estimator of μ)
    • Sample Variance: s² = (1/(n-1)) Σ (Xᵢ - X̄)²  (Bessel's Correction!)
```

---

## 2. Architectural Flowchart: Central Limit Theorem Convergence

```
                 CENTRAL LIMIT THEOREM (CLT) CONVERGENCE
    [Non-Normal Raw Population: Skewed, Bimodal, or Uniform]
                             │
                             ▼ Draw k Repeated Random Samples of Size n (n ≥ 30)
    Sample 1: [x₁₁, x₁₂, ..., x₁ₙ] ──► Compute Sample Mean X̄₁
    Sample 2: [x₂₁, x₂₂, ..., x₂ₙ] ──► Compute Sample Mean X̄₂
    ...
    Sample k: [xₖ₁, xₖ₂, ..., xₖₙ] ──► Compute Sample Mean X̄ₖ
                             │
                             ▼
    [Distribution of Sample Means {X̄₁, X̄₂, ..., X̄ₖ}]
    • Converges strictly to a Gaussian Normal Distribution!
    • Center: μ_X̄ = μ (Population Mean)
    • Standard Error: SE = σ / √n (Dispersion shrinks with sample size!)
```

---

## 3. Deep Theoretical Foundations

### 1. Mathematical Moments & Shape Metrics
The geometry of any probability distribution is quantitatively governed by its mathematical moments:
- **1st Raw Moment (Mean $\mu$):** Center of gravity / expected value:
  $$\mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x f(x) \, dx$$
- **2nd Central Moment (Variance $\sigma^2$):** Spread around the center:
  $$\sigma^2 = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
- **3rd Standardized Moment (Skewness $\gamma_1$):** Direction and degree of asymmetry:
  $$\gamma_1 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right] = \frac{\mu_3}{\sigma^3}$$
  - $\gamma_1 = 0$: Symmetric distribution (e.g. Normal).
  - $\gamma_1 > 0$: Positive / Right-skewed (long tail toward higher values, e.g. income distributions).
  - $\gamma_1 < 0$: Negative / Left-skewed (long tail toward lower values).
- **4th Standardized Moment (Kurtosis $\beta_2$):** Tail heaviness and outlier propensity:
  $$\text{Excess Kurtosis} = \frac{\mu_4}{\sigma^4} - 3$$
  - Mesokurtic ($= 0$): Normal distribution tails.
  - Leptokurtic ($> 0$): Heavy tails with higher outlier risk (e.g. financial returns, t-distribution).
  - Platykurtic ($< 0$): Thin tails with few outliers (e.g. Uniform distribution).

### 2. Bessel's Correction & Degrees of Freedom
When estimating variance from a sample using the sample mean $\bar{X}$ instead of the true population mean $\mu$, the naive divisor $n$ systematically underestimates the true variance because the deviations $(X_i - \bar{X})$ are constrained to sum to zero ($\sum (X_i - \bar{X}) \equiv 0$). This loss of 1 degree of freedom is corrected by Bessel's correction:
$$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (X_i - \bar{X})^2, \quad \mathbb{E}[s^2] = \sigma^2 \text{ (Unbiased!)}$$

### 3. Welford's Algorithm for Numerically Stable Online Variance
Computing variance via the textbook formula $\sum X_i^2 - n \bar{X}^2$ suffers from catastrophic cancellation in floating-point arithmetic when numbers are large. Welford's algorithm computes variance in a single streaming pass with machine precision:
$$M_{1, n} = M_{1, n-1} + \frac{x_n - M_{1, n-1}}{n}$$
$$M_{2, n} = M_{2, n-1} + (x_n - M_{1, n-1})(x_n - M_{1, n})$$
$$s^2 = \frac{M_{2, n}}{n - 1}$$

---

## 4. Production Implementation: Robust Estimators & Online Streaming

```python
import numpy as np
from scipy import stats

class OnlineStatisticsTracker:
    """Welford's algorithm for numerically stable streaming statistics in O(1) memory."""
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return np.sqrt(self.variance)

def robust_scale_estimates(arr: np.ndarray) -> dict[str, float]:
    """Computes parametric and robust non-parametric scale estimates."""
    median = float(np.median(arr))
    # Median Absolute Deviation (MAD): robust to extreme outliers
    mad = float(stats.median_abs_deviation(arr, scale='normal'))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr = float(q75 - q25)
    
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": median,
        "mad_normal_scale": mad,
        "iqr": iqr,
        "skewness": float(stats.skew(arr)),
        "excess_kurtosis": float(stats.kurtosis(arr))
    }
```

---

## 5. Performance & Complexity Matrix

| Statistic | Time Complexity | Auxiliary Space | Robustness Breakdown Point |
|---|---|---|---|
| Sample Mean ($\bar{X}$) | $O(N)$ | $O(1)$ | $0\%$ (Single infinite outlier ruins estimate) |
| Sample Median | $O(N)$ (QuickSelect) | $O(1)$ in-place / $O(N)$ | $50\%$ (Up to half data can be corrupted) |
| Standard Deviation ($s$) | $O(N)$ | $O(1)$ | $0\%$ |
| Median Absolute Deviation | $O(N)$ | $O(N)$ | $50\%$ (Gold standard for noisy sensors) |
| Welford Online Accumulator | $O(1)$ per item | $O(1)$ constant RAM | $0\%$ (Streaming real-time) |
'''

# =====================================================================
# 6. Probability Theory & Parametric Distributions
# =====================================================================
C02_BASICS_P1["02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions"] = r'''# Chapter 6: Probability Theory & Parametric Distributions
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Probability theory provides the mathematical calculus of uncertainty. In statistical modeling:
- A random variable maps physical event outcomes to real numbers: $X: \Omega \to \mathbb{R}$.
- Parametric distributions compress infinite empirical measurements into compact analytical forms defined by a few governing parameters ($\mu, \sigma, \lambda, p$).
- Bayesian inference continuously updates prior probability beliefs with newly observed evidence.

```
                    BAYESIAN INFERENCE PIPELINE
            Prior Knowledge P(θ)  ×  Observed Likelihood P(D|θ)
    ──────────────────────────────────────────────────────────────────
                        Marginal Evidence P(D)
                                  │
                                  ▼
                    Posterior Distribution P(θ|D)
```

---

## 2. Core Distribution Taxonomies & Mathematical Properties

### 1. Discrete Parametric Distributions

| Distribution | Support ($k$) | Probability Mass Function (PMF) | Expected Value $\mathbb{E}[X]$ | Variance $\text{Var}(X)$ | Industrial AI Application |
|---|---|---|---|---|---|
| **Bernoulli** | $\{0, 1\}$ | $p^k (1 - p)^{1 - k}$ | $p$ | $p(1 - p)$ | Binary click-through prediction |
| **Binomial** | $\{0, \dots, n\}$ | $\binom{n}{k} p^k (1 - p)^{n - k}$ | $n p$ | $n p (1 - p)$ | Batch hardware defect counts |
| **Poisson** | $\{0, 1, 2, \dots\}$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Website query arrival rate per sec |
| **Geometric** | $\{1, 2, \dots\}$ | $(1 - p)^{k - 1} p$ | $\frac{1}{p}$ | $\frac{1 - p}{p^2}$ | Trials until first successful API call |

### 2. Continuous Parametric Distributions

| Distribution | Support ($x$) | Probability Density Function (PDF) | Expected Value $\mathbb{E}[X]$ | Variance $\text{Var}(X)$ | Industrial AI Application |
|---|---|---|---|---|---|
| **Gaussian (Normal)** | $(-\infty, \infty)$ | $\frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ | $\mu$ | $\sigma^2$ | Sensor noise, measurement errors |
| **Log-Normal** | $(0, \infty)$ | $\frac{1}{x \sigma \sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$ | $\exp\left(\mu + \frac{\sigma^2}{2}\right)$ | $(\exp(\sigma^2) - 1)\mathbb{E}[X]^2$ | Financial wealth, web page dwell time |
| **Exponential** | $[0, \infty)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | Time between server failure events |
| **Beta** | $[0, 1]$ | $\frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{\text{B}(\alpha, \beta)}$ | $\frac{\alpha}{\alpha + \beta}$ | $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$ | Prior beliefs over conversion rates |

---

## 3. Deep Theoretical Foundations

### 1. The Principle of Maximum Likelihood Estimation (MLE)
Given an observed dataset $D = \{x_1, x_2, \dots, x_n\}$ assumed i.i.d. from parameterized distribution $f(x \mid \theta)$, the likelihood function is:
$$L(\theta) = \prod_{i=1}^n f(x_i \mid \theta)$$
Maximizing the log-likelihood avoids numerical underflow and converts products to sums:
$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i \mid \theta)$$
Setting the gradient score vector to zero yields the MLE estimator:
$$\nabla_\theta \ell(\theta) = 0 \implies \hat{\theta}_{\text{MLE}}$$

### 2. Conjugate Priors & Analytical Bayesian Updating
In Bayesian statistics, if the posterior distribution $P(\theta \mid D)$ belongs to the same probability distribution family as the prior $P(\theta)$, the prior is termed **conjugate** to the likelihood.
- **Beta-Binomial Conjugacy:**
  - Prior: $\theta \sim \text{Beta}(\alpha, \beta)$
  - Likelihood: $k$ successes in $n$ trials $\sim \text{Binomial}(n, \theta)$
  - Analytical Posterior: $\theta \mid D \sim \text{Beta}(\alpha + k, \beta + (n - k))$
  This allows instant analytical real-time updates in Multi-Armed Bandits (Thompson Sampling) without expensive Markov Chain Monte Carlo (MCMC) simulations.

---

## 4. Production Implementation: Bayesian Conjugate Updating & Thompson Sampling

```python
import numpy as np

class BetaBinomialBandit:
    """Thompson Sampling multi-armed bandit using exact Beta-Binomial conjugacy."""
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Uninformative Uniform Prior: Beta(1, 1)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        """Samples from posterior distributions to balance exploration and exploitation."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, reward: int) -> None:
        """Instantaneous O(1) Bayesian conjugate parameter update."""
        if reward == 1:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1

    def expected_conversion_rates(self) -> np.ndarray:
        """Returns the posterior mean expectation for each arm."""
        return self.alpha / (self.alpha + self.beta)
```
'''

print(f"Loaded {len(C02_BASICS_P1)} textbook chapters for Course 2 (Part 1).")

