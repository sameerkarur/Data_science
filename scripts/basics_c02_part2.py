"""
Textbook-Scale Architectural & Conceptual Guides for Course 2 (Part 2):
IITK AIML Core: Applied Data Science with Python
Modules:
  07_advanced_statistics
  08_pandas
  09_data_wrangling
  10_data_visualization (with matplotlib and seaborn)
  11_regex_json_apis
"""

C02_BASICS_P2 = {}

# =====================================================================
# 7. Advanced Inferential Statistics & Hypothesis Testing
# =====================================================================
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics"] = r'''# Chapter 7: Inferential Statistics & Hypothesis Testing
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Inferential statistics allows engineers to make rigorous decisions under uncertainty by quantifying whether an observed difference between groups is statistically genuine or merely an artifact of random sampling noise.

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

## 2. Deep Theoretical Foundations

### 1. Decision Theory: Type I vs Type II Errors
In any statistical test:
- **Type I Error ($\alpha$):** Rejecting $H_0$ when $H_0$ is true (False Positive rate). Typically set to $\alpha = 0.05$.
- **Type II Error ($\beta$):** Failing to reject $H_0$ when $H_1$ is true (False Negative rate).
- **Statistical Power ($1 - \beta$):** The probability of correctly detecting a genuine effect. In industrial A/B testing, target power is typically $80\% - 90\%$.

### 2. Welch's t-test (Unequal Variances)
Standard Student's t-test assumes homoscedasticity (equal variance $\sigma_1^2 = \sigma_2^2$). In real-world data, Welch's t-test does not assume equal variances:
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
With degrees of freedom approximated via the Welch-Satterthwaite equation:
$$\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

### 3. ANOVA (Analysis of Variance) & F-Ratio
Used to test equality of means across $K \ge 3$ groups simultaneously without inflating the family-wise error rate:
$$F = \frac{\text{Mean Square Between (MSB)}}{\text{Mean Square Within (MSW)}} = \frac{\frac{1}{K - 1}\sum_{i=1}^K n_i (\bar{X}_i - \bar{X})^2}{\frac{1}{N - K}\sum_{i=1}^K \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2}$$

### 4. Multiple Testing Corrections
Running $M$ independent hypothesis tests at significance level $\alpha = 0.05$ inflates the family-wise false positive probability to $1 - (1 - \alpha)^M$ (at $M = 20$, probability of $\ge 1$ false discovery exceeds $64\%$).
- **Bonferroni Correction:** Controls Family-Wise Error Rate (FWER) conservatively:
  $$\alpha_{\text{adjusted}} = \frac{\alpha}{M}$$
- **Benjamini-Hochberg (FDR):** Controls the False Discovery Rate (fraction of declared discoveries that are false) with higher statistical power:
  $$p_{(i)} \le \frac{i}{M} Q$$

---

## 3. Production Implementation: Automated A/B Testing Engine

```python
import numpy as np
from scipy import stats

def evaluate_ab_experiment(control_conversions: np.ndarray, 
                           treatment_conversions: np.ndarray, 
                           alpha: float = 0.05) -> dict[str, float | bool]:
    """Evaluates continuous KPI uplifts using Welch's t-test with effect size."""
    n_ctrl, n_treat = len(control_conversions), len(treatment_conversions)
    mean_ctrl, mean_treat = np.mean(control_conversions), np.mean(treatment_conversions)
    
    # Welch's two-sample t-test (robust to unequal variances)
    t_stat, p_val = stats.ttest_ind(treatment_conversions, control_conversions, equal_var=False)
    
    # Cohen's d effect size
    s_pooled = np.sqrt(((n_ctrl - 1)*np.var(control_conversions, ddof=1) + 
                        (n_treat - 1)*np.var(treatment_conversions, ddof=1)) / (n_ctrl + n_treat - 2))
    cohens_d = (mean_treat - mean_ctrl) / s_pooled
    
    relative_lift = (mean_treat - mean_ctrl) / mean_ctrl if mean_ctrl != 0 else 0.0
    
    return {
        "control_mean": float(mean_ctrl),
        "treatment_mean": float(mean_treat),
        "relative_lift_pct": float(relative_lift * 100),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "statistically_significant": bool(p_val < alpha)
    }
```
'''

# =====================================================================
# 8. Pandas Data Manipulation & BlockManager Architecture
# =====================================================================
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas"] = r'''# Chapter 8: Pandas Data Manipulation & BlockManager Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Pandas provides structured tabular data abstractions (`Series` and `DataFrame`). Behind its high-level API lies the **BlockManager**, an internal memory manager that consolidates columns of identical data types into contiguous 2D NumPy arrays.

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

## 2. Deep Theoretical Foundations

### 1. Vectorized Split-Apply-Combine Engine
When executing `df.groupby('cohort').agg({'revenue': 'sum'})`:
1. **Split:** Pandas generates a fast integer array of category codes (`factorize`), avoiding expensive dictionary lookups.
2. **Apply:** Compiled Cython kernels compute row sums along contiguous memory blocks.
3. **Combine:** The aggregated arrays are reassembled into an indexed DataFrame in $O(N)$ linear time.

### 2. Method Chaining & Copy-on-Write (CoW)
Modern Pandas (2.0+) implements **Copy-on-Write (CoW)**:
Modifying a slice or subset DataFrame does not immediately allocate memory. A deep copy of the underlying Block is only triggered at the precise moment a write/mutation occurs, eliminating accidental view modification bugs and reducing peak memory by up to 50%.

### 3. High-Performance Indexing: `.loc` vs `.iloc`
- `.iloc[row_idx, col_idx]`: Zero-overhead integer indexing mapped directly into NumPy pointer strides.
- `.loc[label, col_label]`: Label-based hash table lookup against the DataFrame's `Index` object.

---

## 3. Production Implementation: Memory Optimization Pipeline

```python
import pandas as pd
import numpy as np

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces DataFrame memory consumption by up to 75% via optimal dtype casting."""
    df_optimized = df.copy()
    start_mem = df.memory_usage(deep=True).sum() / (1024**2)
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimize numeric columns
        if np.issubdtype(col_type, np.integer):
            c_min, c_max = df_optimized[col].min(), df_optimized[col].max()
            if c_min >= 0:
                if c_max < 255: df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif c_max < 65535: df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif c_max < 4294967295: df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                if c_min > -128 and c_max < 127: df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > -32768 and c_max < 32767: df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > -2147483648 and c_max < 2147483647: df_optimized[col] = df_optimized[col].astype(np.int32)
                
        elif np.issubdtype(col_type, np.floating):
            df_optimized[col] = df_optimized[col].astype(np.float32)
            
        elif col_type == object:
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            # Convert low-cardinality strings to categorical
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
                
    end_mem = df_optimized.memory_usage(deep=True).sum() / (1024**2)
    print(f"Memory Footprint Reduced: {start_mem:.2f} MB ➔ {end_mem:.2f} MB (-{(1 - end_mem/start_mem)*100:.1f}%)")
    return df_optimized
```
'''

# =====================================================================
# 9. Data Wrangling, Imputation & Outlier Engineering
# =====================================================================
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling"] = r'''# Chapter 9: Data Wrangling, Imputation & Outlier Engineering
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Raw industrial data is noisy, incomplete, and filled with anomalies. Robust data wrangling requires diagnosing missingness mechanisms before applying imputation, and utilizing non-parametric outlier boundaries.

```
                 DATA CLEANING & IMPUTATION LIFECYCLE
       Raw Data ──► Detect Missingness Patterns:
                      ├── MCAR (Missing Completely at Random)
                      ├── MAR  (Missing at Random)
                      └── MNAR (Missing Not at Random)
                           │
       Screen Outliers via IQR Tukey Fences / Isolation Forest
                           │
       Apply Domain Scalers (StandardScaler / RobustScaler)
```

---

## 2. Deep Theoretical Foundations

### 1. Missingness Taxonomy (Little & Rubin)
- **MCAR (Missing Completely at Random):** Missingness is entirely independent of observed and unobserved data: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M)$. Deletion does not introduce bias.
- **MAR (Missing at Random):** Missingness depends systematically on observed features, but not on the missing value itself: $P(M \mid Y_{\text{obs}}, Y_{\text{mis}}) = P(M \mid Y_{\text{obs}})$. Resolved via conditional imputation (MICE / IterativeImputer).
- **MNAR (Missing Not at Random):** Missingness depends directly on the unobserved value (e.g. high-income respondents refusing to state income). Requires explicit missingness indicator flags.

### 2. Multivariate Imputation by Chained Equations (MICE)
Rather than simple mean or median imputation (which destroys feature variance and correlations), MICE models each variable with missing values as a function of all other variables in an iterative round-robin Gibbs sampler:
$$\hat{Y}_j^{(t)} \sim P(Y_j \mid Y_{-j}^{(t)}, \theta_j)$$
Iterating across 10–20 cycles converges to the true joint conditional distribution.

### 3. Isolation Forest for Multi-Dimensional Outlier Detection
Isolation Forest isolates anomalies by randomly selecting a feature and a random split value:
- Since anomalies occupy sparse, isolated regions of the feature space, they require significantly fewer recursive splits to be isolated.
- The anomaly score $s(x, n)$ is derived from the average tree path length $E(h(x))$:
  $$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}, \quad \text{where } c(n) = 2\ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$$
  An anomaly score $s \to 1$ indicates a definite outlier.

---

## 3. Production Implementation: Leakage-Safe Imputer & Outlier Filter

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest

def clean_tabular_dataset(df: pd.DataFrame, contamination: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
    """Applies MICE imputation and Isolation Forest outlier filtering."""
    df_clean = df.copy()
    
    # 1. MICE Iterative Multivariate Imputation
    imputer = IterativeImputer(max_iter=10, random_state=42)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # 2. Multi-Dimensional Anomaly Isolation
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = iso.fit_predict(df_clean[numeric_cols])
    is_inlier = pd.Series(outlier_labels == 1, index=df_clean.index)
    
    return df_clean[is_inlier], is_inlier
```
'''

# =====================================================================
# 10. Data Visualization & Matplotlib/Seaborn Architecture
# =====================================================================
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization"] = r'''# Chapter 10: Data Visualization & Visual Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Data visualization translates high-dimensional relationships into visual encodings (position, length, angle, hue, saturation). Matplotlib provides the foundational rendering canvas and Artist object hierarchy, while Seaborn provides statistical aggregation abstractions.

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

## 2. Deep Theoretical Foundations

### 1. The Grammar of Graphics (Wilkinson / Wickham)
Data graphics decompose into independent orthogonal layers:
1. **Data:** Raw tabular dataset.
2. **Aesthetic Mapping (`aes`):** Mapping variables to visual channels (X, Y, Color, Size, Shape).
3. **Geometric Objects (`geom`):** The physical marks (points, lines, bars, ribbons).
4. **Statistical Transformations (`stat`):** Binning, smoothing, quantile estimation.
5. **Coordinate Systems (`coord`):** Cartesian, logarithmic, polar.
6. **Faceting (`facet`):** Conditioning subplots across discrete categories.

### 2. Kernel Density Estimation (KDE) Mechanics
Seaborn density plots approximate continuous probability distributions non-parametrically using kernel smoothing:
$$\hat{f}_h(x) = \frac{1}{n h} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$
Where $K(u)$ is typically the standard Gaussian kernel $\frac{1}{\sqrt{2\pi}}e^{-u^2 / 2}$ and $h$ is the bandwidth smoothing factor determined via Silverman's rule of thumb.

---

## 3. Production Implementation: Publication-Grade Multi-Panel Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_publication_diagnostics(actual: np.ndarray, predicted: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    """Generates publication-grade model diagnostic dashboard with dark aesthetic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Panel 1: Prediction vs Ground Truth
    axes[0].scatter(actual, predicted, alpha=0.4, color='#38bdf8', edgecolors='none', s=20)
    ideal_line = [min(actual), max(actual)]
    axes[0].plot(ideal_line, ideal_line, color='#f43f5e', linestyle='--', linewidth=1.5, label='Identity (y=x)')
    axes[0].set_title("Parity Plot: Actual vs Predicted", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Ground Truth Target", fontsize=10)
    axes[0].set_ylabel("Model Prediction", fontsize=10)
    axes[0].legend(frameon=True, facecolor='#1e293b', edgecolor='#334155')
    axes[0].grid(True, linestyle=':', alpha=0.3)
    
    # Panel 2: Residual Distribution with KDE
    sns.histplot(residuals, kde=True, ax=axes[1], color='#10b981', stat='density', bins=30)
    axes[1].axvline(0, color='#f43f5e', linestyle='--', linewidth=1.5, label='Zero Residual')
    axes[1].set_title("Residual Error Distribution (KDE)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Residual Error (Actual - Pred)", fontsize=10)
    axes[1].legend(frameon=True, facecolor='#1e293b', edgecolor='#334155')
    axes[1].grid(True, linestyle=':', alpha=0.3)
    
    fig.tight_layout()
    return fig
```
'''

# Submodules for visualization
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib"] = C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization"]
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn"] = C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization"]

# =====================================================================
# 11. Regex, JSON & Web APIs
# =====================================================================
C02_BASICS_P2["02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis"] = r'''# Chapter 11: Regular Expressions, JSON Streaming & Web APIs
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Modern data ingestion requires extracting unstructured textual signals via regular expressions, streaming semi-structured JSON payloads, and interfacing with distributed web microservices.

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

## 2. Deep Theoretical Foundations

### 1. Regular Expression Automata: DFA vs NFA
- **Deterministic Finite Automaton (DFA):** Processes each input character exactly once in $O(N)$ linear time. Does not support backreferences.
- **Non-Deterministic Finite Automaton (NFA - Python `re`):** Employs backtracking. While expressive, pathological nested patterns like `(a+)+$` evaluated on `aaaaX` lead to **catastrophic backtracking** with exponential $O(2^N)$ time complexity. Always anchor expressions and avoid ambiguous nested repetitions.

### 2. Distributed API Reliability & Jittered Exponential Backoff
When querying high-throughput endpoints, naive retries trigger thundering herd problems. Full jitter exponential backoff distributes retry load uniformly across client workers:
$$t_{\text{sleep}} = \text{Uniform}\left(0, \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}})\right)$$

---

## 3. Production Implementation: Resilient API Client with Streaming JSON

```python
import time
import random
import json
import urllib.request
import urllib.error
from typing import Any, Generator

class ResilientApiClient:
    """Production REST client featuring full-jitter exponential backoff."""
    def __init__(self, base_delay: float = 0.5, max_delay: float = 30.0, max_retries: int = 4):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries

    def fetch_with_retry(self, url: str) -> dict[str, Any]:
        attempt = 0
        while attempt <= self.max_retries:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'AIML-Production-Pipeline/1.0'})
                with urllib.request.urlopen(req, timeout=10.0) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    # Full Jitter backoff formula
                    ceiling = min(self.max_delay, self.base_delay * (2 ** attempt))
                    sleep_time = random.uniform(0, ceiling)
                    time.sleep(sleep_time)
                else:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(random.uniform(0, min(self.max_delay, self.base_delay * (2 ** attempt))))
                else:
                    raise
        raise RuntimeError(f"Max retries exceeded fetching: {url}")
```
'''

print(f"Loaded {len(C02_BASICS_P2)} textbook chapters for Course 2 (Part 2).")
