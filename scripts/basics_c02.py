"""
Comprehensive, high-depth Basics & Architecture Guides for Course 2:
Applied Data Science with Python (11 modules)
"""

C02_BASICS = {}

# 1. Intro Data Science
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science"] = """# Data Science Foundations & CRISP-DM Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Data Science is an iterative engineering process governed by the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** lifecycle:

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

## 🧭 Deep Theoretical Foundations

### 1. The Bias-Variance Tradeoff Formulation
For any predictive estimator $\hat{f}(x)$, total expected mean squared error decomposes into three irreducible components:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2_{\text{noise}}$$
- **High Bias (Underfitting):** Model cannot capture true underlying relationships (oversimplified assumptions).
- **High Variance (Overfitting):** Model memorizes sample noise and generalizes poorly to out-of-fold validation data.

### 2. Critical Fallacies in Industrial Data Science
- **Data Leakage:** Information from the target label or future test set inadvertently contaminating the training pipeline (e.g., fitting scalers or imputing missing values on the entire dataset prior to train-test splitting).
- **Survivorship & Selection Bias:** Training models exclusively on surviving records (e.g. active bank accounts or approved loans), producing biased predictions for at-risk cohorts.

---

## 💻 Production Implementation: Strict Leakage-Free Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

# 1. Partition Data BEFORE any Transformation to prevent Leakage
X = pd.DataFrame({'feature_a': [10.5, np.nan, 14.2, 8.9, 12.1, np.nan, 15.0],
                  'feature_b': [100, 250, 180, 90, 140, 220, 310]})
y = np.array([25, 45, 38, 20, 31, 40, 55])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Encapsulate Transformations inside a Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Fits ONLY on training data
    ('scaler', StandardScaler()),                  # Fits ONLY on training data
    ('model', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
test_predictions = pipeline.predict(X_test)
```

---

## 📐 Data Science Methodology Matrix

| Stage | Primary Deliverable | Core Risks | Validation Check |
|---|---|---|---|
| Exploratory Analysis (EDA) | Summary stats, correlation matrices | Outlier distortion, Simpson's Paradox | Multi-dimensional group aggregation |
| Preprocessing | Clean tensor matrices | Feature leakage, distribution drift | Fit transforms on Train split ONLY |
| Model Evaluation | Cross-validation metric distributions | Overfitting, metric misalignment | Stratified out-of-time splits |
"""

# 2. Python Essentials for Data Science
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials"] = """# High-Performance Python & Vectorization Essentials
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Standard Python loops incur heavy dynamic type-checking overhead. Data science computing achieves orders of magnitude speedups by transitioning to contiguous vector memory buffers.

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

## 🧭 Deep Theoretical Foundations

### 1. The Global Interpreter Lock (GIL) & Vector Workarounds
CPython's GIL prevents multiple native threads from executing Python bytecodes concurrently. However, vectorized numerical libraries (NumPy, SciPy, BLAS/LAPACK) release the GIL during low-level C/Fortran array computations, enabling true parallel multicore SIMD operations.

### 2. Iterator Protocols & Memory Streaming
For large datasets exceeding available physical RAM, generator pipelines (`yield`, `itertools`, and lazy mapping) execute in $O(1)$ auxiliary space, streaming chunks through transformation kernels.

---

## 💻 Production Implementation: Memory & Latency Profiling

```python
import time
import numpy as np

# Performance Benchmark: Native Python List vs Vectorized NumPy Array
size = 2_000_000
python_list = list(range(size))
numpy_array = np.arange(size, dtype=np.int64)

# Native Python Iteration
t0 = time.perf_counter()
py_result = [x * 2 + 1 for x in python_list]
t_py = time.perf_counter() - t0

# Vectorized Hardware Execution
t1 = time.perf_counter()
np_result = numpy_array * 2 + 1
t_np = time.perf_counter() - t1

print(f"Python Loop Time: {t_py:.4f}s")
print(f"NumPy Vector Time: {t_np:.4f}s (Speedup: {t_py / t_np:.1f}x)")
```

---

## 📐 Computational Matrix

| Paradigm | Memory Footprint | CPU Cache Locality | Parallelism Support |
|---|---|---|---|
| Python `list` | High (8 bytes pointer + 28 bytes `PyObject`) | Poor (Pointer chasing in heap) | GIL constrained |
| NumPy `ndarray` | Minimal (Raw contiguous C data buffer) | Optimal (Fills L1/L2 cache lines) | Multi-threaded BLAS |
"""

# 3. NumPy
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy"] = """# NumPy Numerical Computing & Array Memory Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

A NumPy array (`ndarray`) separates **Array Metadata** (shape, strides, dtype) from the contiguous **Data Buffer**.

```
                NUMPY NDARRAY MEMORY ARCHITECTURE
    ┌────────────────────────────────────────────────────────┐
    │ ndarray Metadata Header:                               │
    │   • dtype: float64 (8 bytes)                           │
    │   • shape: (2, 3)                                      │
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

## 🧭 Deep Theoretical Foundations

### 1. Memory Strides & Zero-Copy Views
The `strides` tuple defines how many bytes in memory must be skipped to jump to the next element along each dimension:
- Slicing (`arr[:, ::2]`) or transposing (`arr.T`) does not copy raw data—it merely alters the `strides` and `shape` metadata, executing in $O(1)$ time.
- Reshaping operations that cannot be expressed via stride manipulation force an explicit $O(n)$ memory copy.

### 2. Broadcasting Rules
Two dimensions are compatible for element-wise broadcasting when:
1. They are equal, or
2. One of them is 1.
If dimensions differ in length, NumPy prepends 1s to the shorter shape until both dimensions match.

---

## 💻 Production Implementation: High-Performance Operations

```python
import numpy as np

# 1. Broadcasting Matrix Operations
features = np.random.randn(1000, 5)     # 1000 samples, 5 features
mean_vector = np.mean(features, axis=0) # Shape: (5,) -> Broadcasts to (1000, 5)!
normalized = features - mean_vector

# 2. In-Place Operations to Prevent Memory Spikes
a = np.ones((5000, 5000), dtype=np.float32)
# a = a * 2   # BAD: Allocates 100MB temporary buffer
a *= 2        # GOOD: Modifies data buffer in-place!
```

---

## 📐 NumPy Operation Complexity Matrix

| Operation | Time Complexity | Memory Allocation |
|---|---|---|
| Array Indexing (`arr[i, j]`) | $O(1)$ | $O(0)$ (Returns scalar) |
| Slicing / Transpose (`arr.T`) | $O(1)$ | $O(0)$ (Zero-copy View) |
| Dot Product / GEMM ($M \times K \cdot K \times N$) | $O(M \cdot K \cdot N)$ | Allocates $M \times N$ result |
| Boolean Mask Filtering (`arr[arr > 0]`) | $O(n)$ | Allocates new contiguous array |
"""

# 4. Linear Algebra
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra"] = """# Linear Algebra for Machine Learning & Vector Spaces
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
Any real matrix $A \in \mathbb{R}^{m \times n}$ decomposes as $A = U \Sigma V^T$, where:
- $U$ contains orthogonal eigenvectors of $AA^T$.
- $V$ contains orthogonal eigenvectors of $A^T A$.
- $\Sigma$ contains ordered non-negative singular values $\sigma_i = \sqrt{\lambda_i}$.

---

## 💻 Production Implementation: PCA from First Principles

```python
import numpy as np

def compute_pca(X: np.ndarray, n_components: int = 2):
    \"\"\"Calculates PCA via Covariance Eigendecomposition.\"\"\"
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
"""

# 5. Statistics Fundamentals
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals"] = """# Statistical Foundations & Sampling Distributions
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 CENTRAL LIMIT THEOREM (CLT) CONVERGENCE
    [Non-Normal Raw Population] ──► Draw N Random Samples (n >= 30)
                                          │
                                    Compute Sample Mean X̄
                                          │
    [Distribution of Means X̄] ──► Converges to Gaussian Bell Curve!
                                  Mean = μ, Std Error = σ / √n
```

---

## 🧭 Deep Theoretical Foundations

### 1. The Central Limit Theorem (CLT)
Regardless of the underlying population distribution (skewed, uniform, multimodal), the distribution of sample means $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ approaches a Normal distribution as sample size $n \to \infty$:
$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

### 2. Statistical Moments
- **1st Moment (Mean):** Expected location $\mu = \mathbb{E}[X]$.
- **2nd Moment (Variance):** Dispersion $\sigma^2 = \mathbb{E}[(X - \mu)^2]$.
- **3rd Moment (Skewness):** Distribution asymmetry ($>0$ right-skewed, $<0$ left-skewed).
- **4th Moment (Kurtosis):** Heavy-tailedness and outlier concentration relative to Normal ($\kappa = 3$).
"""

# 6. Probability Distributions
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions"] = """# Probability Theory & Parametric Distributions
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Bayes' Theorem & Conditional Probability
$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$
Forms the probabilistic engine for Naive Bayes classifiers, Bayesian optimization, and Kalman filters.

### 2. Core Distribution Taxonomies
- **Binomial Distribution:** $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$ (Discrete binary success rate).
- **Poisson Distribution:** $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$ (Arrival rate in fixed intervals).
- **Normal (Gaussian) Distribution:** $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$.
"""

# 7. Advanced Statistics
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics"] = """# Inferential Statistics & Hypothesis Testing
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                    HYPOTHESIS TESTING DECISION TREE
       Define Null Hypothesis (H₀) & Alternative (H₁)
                           │
       Compute Test Statistic (t, z, F, or χ²)
                           │
                 Obtain p-value from CDF
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
          p < α (0.05)            p >= α (0.05)
       Reject Null (H₀)      Fail to Reject Null (H₀)
     Statistically Significant   Insufficient Evidence
```

---

## 🧭 Deep Theoretical Foundations

### 1. Type I vs Type II Errors
- **Type I Error ($\alpha$):** Rejecting null hypothesis $H_0$ when it was actually true (False Positive).
- **Type II Error ($\beta$):** Failing to reject $H_0$ when it was false (False Negative).
- **Statistical Power ($1 - \beta$):** Probability of correctly detecting a genuine effect.

### 2. ANOVA (Analysis of Variance)
Compares means across $\ge 3$ groups by assessing variance between groups relative to variance within groups:
$$F = \frac{\text{Mean Square Between (MSB)}}{\text{Mean Square Within (MSW)}}$$
"""

# 8. Pandas
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas"] = """# Pandas Data Manipulation & BlockManager Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Pandas organizes 2D tabular data in a **DataFrame** backed by an internal **BlockManager**.

```
                   PANDAS BLOCKMANAGER MEMORY LAYOUT
    ┌────────────────────────────────────────────────────────┐
    │ DataFrame (Columns: age, salary, name, score)          │
    │ Index: Int64Index / DatetimeIndex                      │
    └──────────────────────────┬─────────────────────────────┘
                               │ Grouped by Dtype into Blocks!
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
    ┌───────────┐         ┌───────────┐         ┌───────────┐
    │ FloatBlock│         │ IntBlock  │         │ ObjectBlk │
    │ (2D NumPy)│         │ (2D NumPy)│         │ (Strings) │
    │ • salary  │         │ • age     │         │ • name    │
    │ • score   │         └───────────┘         └───────────┘
    └───────────┘
```

---

## 🧭 Deep Theoretical Foundations

### 1. Vectorized Split-Apply-Combine Pattern
When invoking `df.groupby('category').agg(...)`, Pandas maps categories to integer grouping codes (`factorize`), performing contiguous aggregations in compiled Cython routines rather than traversing rows.

### 2. Index Alignment & Slicing (`.loc` vs `.iloc`)
- `.loc[label]`: Label-based indexing incorporating endpoint bounds.
- `.iloc[integer]`: Raw 0-indexed position-based access (half-open range $[a, b)$).
"""

# 9. Data Wrangling
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling"] = """# Data Wrangling, Imputation & Outlier Engineering
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 DATA CLEANING & IMPUTATION LIFECYCLE
       Raw Data ──► Detect Missingness Patterns:
                      ├── MCAR (Missing Completely at Random)
                      ├── MAR  (Missing at Random)
                      └── MNAR (Missing Not at Random)
                           │
       Screen Outliers via IQR Tukey Fences / Z-score
                           │
       Apply Domain Scalers (StandardScaler / RobustScaler)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Tukey's Fences for Outlier Detection
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
Points outside these bounds are flagged as distributional anomalies.
"""

# 10. Data Visualization
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization"] = """# Data Visualization & Matplotlib/Seaborn Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Matplotlib visual graphics follow an explicit hierarchical **Artist Tree**.

```
                    MATPLOTLIB ARTIST OBJECT TREE
    ┌────────────────────────────────────────────────────────┐
    │ Figure (The Canvas Container)                          │
    │  └── Axes (The Actual Plotting Area / Coordinate Space)│
    │       ├── XAxis & YAxis (Ticks, TickLabels, Scale)     │
    │       ├── Line2D / BarContainer (The Data Plots)       │
    │       ├── Legend & Title Text Objects                  │
    └────────────────────────────────────────────────────────┘
```

---

## 🧭 Deep Theoretical Foundations

### Figure vs Axes Mechanics
- `Figure`: The top-level window or file surface (`plt.figure()`).
- `Axes`: The coordinate system containing data lines, bars, contours, and coordinate transforms (`fig.subplots()`). Always use the object-oriented API (`ax.plot()`) rather than stateful `plt.plot()`.
"""

# 11. Regex, JSON & APIs
C02_BASICS["02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis"] = """# Regex Engines, JSON Streaming & Web API Extraction
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                  REST API EXTRACTION & BACKOFF PIPELINE
       Client Request ──► Send HTTPS GET / POST
                                │
                      Status 429 / 503?
                      ├── YES ──► Sleep = Base × (2 ^ attempt) + Jitter
                      │           Retry Request!
                      │
                      └── NO  ──► Parse JSON Payload via Streaming Parser
```

---

## 🧭 Deep Theoretical Foundations

### Regular Expression Engines & Catastrophic Backtracking
Python's `re` module uses a Non-Deterministic Finite Automaton (NFA). Pathological patterns like `(a+)+b` tested against `aaaaX` lead to $O(2^n)$ exponential backtracking complexity. Use atomic grouping, possessive quantifiers, or non-overlapping token anchors.
"""

print(f"Loaded {len(C02_BASICS)} comprehensive guides for Course 2.")
