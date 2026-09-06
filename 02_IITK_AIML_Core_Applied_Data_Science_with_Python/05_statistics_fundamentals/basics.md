# Chapter 5: Statistical Foundations & Sampling Distributions
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
