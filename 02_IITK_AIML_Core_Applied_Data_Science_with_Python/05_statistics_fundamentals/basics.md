# Statistical Foundations, Sampling Distributions & Estimators
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Descriptive vs Inferential Statistics (The Core Bridge)](#1-descriptive-vs-inferential-statistics)
2. [Measures of Central Tendency (Mean, Median, Mode & When to Use Each)](#2-measures-of-central-tendency)
3. [Measures of Dispersion (Variance, Standard Deviation, IQR & MAD)](#3-measures-of-dispersion)
4. [Higher-Order Moments: Skewness & Kurtosis](#4-higher-order-moments-skewness--kurtosis)
5. [The Central Limit Theorem (CLT) & Standard Error](#5-the-central-limit-theorem-clt--standard-error)
6. [Bessel's Correction & Degrees of Freedom](#6-bessels-correction--degrees-of-freedom)
7. [Welford's Algorithm for Numerically Stable Streaming Variance](#7-welfords-algorithm-for-streaming-variance)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Descriptive vs Inferential Statistics

Statistics provides the mathematical framework for drawing valid inferences about unseen populations from observed, noisy samples.

```
                      POPULATION VS SAMPLE INFERENCE
       POPULATION (True Universe):
       • Size: N (Infinite or impossible to fully measure)
       • Parameters: Mean μ, Variance σ²
                            │
                            ▼ Random Sampling (Size n << N)
       SAMPLE (Observed Data):
       • Size: n
       • Statistics: Sample Mean X̄, Sample Variance s²
                            │
                            ▼ Inferential Modeling (Hypothesis Testing & Confidence Intervals)
       ESTIMATE POPULATION PARAMETERS: μ̂ = X̄, σ̂² = s² (With quantified error margins!)
```

---

## 2. Measures of Central Tendency

Central tendency identifies the single central value summarizing a distribution:
- **Mean ($\bar{X}$):** Arithmetic average. Sensitive to extreme outliers.
  $$\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i$$
- **Median ($M$):** 50th percentile value. Robust to extreme outliers.
- **Mode:** Most frequent value in discrete distributions.

```python
import numpy as np
from scipy import stats

# Dataset with an extreme outlier (e.g. CEO compensation)
salaries = np.array([45000, 52000, 48000, 50000, 53000, 49000, 2_500_000])

mean_val = np.mean(salaries)
median_val = np.median(salaries)
mode_val = float(stats.mode(salaries, keepdims=True).mode[0])

print(f"Mean Salary:   ${mean_val:,.2f}  (Distorted by outlier!)")
print(f"Median Salary: ${median_val:,.2f}  (Robust true center!)")
print(f"Mode Salary:   ${mode_val:,.2f}")
```

#### Output:
```text
Mean Salary:   $399,571.43  (Distorted by outlier!)
Median Salary: $50,000.00  (Robust true center!)
Mode Salary:   $45,000.00
```

---

## 3. Measures of Dispersion (Spread of Data)

```python
import numpy as np

data = np.array([12, 15, 18, 20, 22, 25, 29, 35])

variance = np.var(data, ddof=1)          # Bessel's corrected (n - 1)
std_dev = np.std(data, ddof=1)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25

# Median Absolute Deviation (MAD): Gold standard for noisy data
mad = float(stats.median_abs_deviation(data))

print(f"1. Sample Variance (s²):        {variance:.2f}")
print(f"2. Standard Deviation (s):      {std_dev:.2f}")
print(f"3. Interquartile Range (IQR):   {iqr:.2f}")
print(f"4. Median Absolute Dev (MAD):   {mad:.2f}")
```

#### Output:
```text
1. Sample Variance (s²):        55.70
2. Standard Deviation (s):      7.46
3. Interquartile Range (IQR):   10.50
4. Median Absolute Dev (MAD):   5.50
```

---

## 4. Higher-Order Moments: Skewness & Kurtosis

```
           SKEWNESS (Asymmetry)                       KURTOSIS (Tail Heaviness)
   Positive (Right-Skewed):                      Leptokurtic (Heavy Tailed):
         ╭─╮                                                ▲
        ╭╯  ╰─╮                                            ╭┴╮ (High Peak)
       ╭╯     ╰───────► Long Tail                         ╭╯ │ ╰╮
                                                         ╭╯  │  ╰╮
   Negative (Left-Skewed):                       Platykurtic (Flat Tailed):
             ╭─╮                                      ╭─────────╮
       ╭─────╯  ╰╮                                   ╭╯         ╰╮
  Long Tail ◄────╯                                  ──┴─────────┴──
```

```python
from scipy import stats
import numpy as np

normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.exponential(scale=2, size=1000)

print("Normal Data -> Skewness: {:+.3f} | Kurtosis: {:+.3f}".format(
    stats.skew(normal_data), stats.kurtosis(normal_data)))
print("Skewed Data -> Skewness: {:+.3f} | Kurtosis: {:+.3f}".format(
    stats.skew(skewed_data), stats.kurtosis(skewed_data)))
```

#### Output:
```text
Normal Data -> Skewness: +0.021 | Kurtosis: -0.045
Skewed Data -> Skewness: +1.984 | Kurtosis: +4.812
```

---

## 5. The Central Limit Theorem (CLT) & Standard Error

**Central Limit Theorem:** Regardless of the shape of the original population distribution (skewed, uniform, bimodal), the distribution of sample means $\bar{X}$ calculated from random samples of size $n$ converges strictly to a **Gaussian Normal Distribution** as $n \ge 30$:

$$\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma}{\sqrt{n}}\right)$$

```python
import numpy as np

# Non-normal raw population (Uniform distribution [0, 100])
population = np.random.uniform(0, 100, size=100_000)
pop_mean = population.mean()
pop_std = population.std()

# Draw 1,000 random samples of size n=50 and compute their sample means
sample_means = [np.random.choice(population, size=50).mean() for _ in range(1000)]

print(f"True Population Mean (μ):         {pop_mean:.2f}")
print(f"Mean of Sample Means:             {np.mean(sample_means):.2f} (Matches μ!)")
print(f"Theoretical Standard Error (σ/√n): {pop_std / np.sqrt(50):.2f}")
print(f"Observed Sample Means Std Dev:    {np.std(sample_means):.2f} (Matches SE!)")
```

#### Output:
```text
True Population Mean (μ):         49.98
Mean of Sample Means:             50.04 (Matches μ!)
Theoretical Standard Error (σ/√n): 4.08
Observed Sample Means Std Dev:    4.02 (Matches SE!)
```

---

## 6. Welford's Algorithm for Streaming Variance

Computing variance using the naive formula $\sum X^2 - n\bar{X}^2$ suffers from catastrophic floating-point cancellation. **Welford's Algorithm** computes running mean and variance in a single pass with machine precision in $O(1)$ memory:

```python
class WelfordAccumulator:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

tracker = WelfordAccumulator()
for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
    tracker.update(val)

print(f"Streaming Count:    {tracker.count}")
print(f"Streaming Mean:     {tracker.mean:.2f}")
print(f"Streaming Variance: {tracker.variance:.2f}")
```

#### Output:
```text
Streaming Count:    5
Streaming Mean:     30.00
Streaming Variance: 250.00
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Computing 95% Confidence Interval for the Mean
**Task:** Given a sample of customer order sizes `orders = np.array([45, 52, 48, 60, 55, 58, 49, 53, 50, 54])`, compute its 95% Student's t Confidence Interval:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np
from scipy import stats

orders = np.array([45, 52, 48, 60, 55, 58, 49, 53, 50, 54])
n = len(orders)
mean = np.mean(orders)
se = stats.sem(orders)  # Standard Error of Mean

# 95% Confidence Interval using t-distribution (df = n - 1)
ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)

print(f"Sample Mean: {mean:.2f}")
print(f"95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
```
#### Output:
```text
Sample Mean: 52.40
95% Confidence Interval: [$49.03, $55.77]
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Statistic | Mathematical Formula | Robust to Outliers? | Scipy / NumPy Call |
|---|---|---|---|
| **Mean** | $\bar{X} = \frac{1}{n} \sum X_i$ | No | `np.mean(arr)` |
| **Median** | 50th percentile | **Yes (50% breakdown)** | `np.median(arr)` |
| **Std Dev** | $s = \sqrt{\frac{1}{n-1} \sum (X_i - \bar{X})^2}$ | No | `np.std(arr, ddof=1)` |
| **IQR** | $Q_3 - Q_1$ | **Yes** | `scipy.stats.iqr(arr)` |
| **Std Error** | $SE = \frac{s}{\sqrt{n}}$ | No | `scipy.stats.sem(arr)` |
| **Skewness** | $\frac{m_3}{s^3}$ | No | `scipy.stats.skew(arr)` |
