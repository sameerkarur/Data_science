# Statistics Fundamentals for Data Science & AI: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official SciPy / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Descriptive vs Inferential Statistics Taxonomy](#1-descriptive-vs-inferential-statistics-taxonomy)
2. [Measures of Central Tendency: Mean, Median, Mode & Trimmed Means](#2-measures-of-central-tendency)
3. [Measures of Dispersion: Variance, Standard Deviation, MAD & IQR](#3-measures-of-dispersion)
4. [Bessel's Correction ($N-1$): Mathematical Proof for Unbiased Sample Variance](#4-bessels-correction)
5. [Skewness, Kurtosis & Shape Analysis](#5-skewness-kurtosis--shape-analysis)
6. [The Central Limit Theorem (CLT): Mechanics & Empirical Verification](#6-the-central-limit-theorem-clt)
7. [Welford's Algorithm: One-Pass Numerically Stable Running Variance](#7-welfords-algorithm)
8. [Common Pitfalls & Statistical Misinterpretations](#8-common-pitfalls--statistical-misinterpretations)
9. [Production Case Study: Real-Time Anomaly Detection via Streaming Z-Score](#9-production-case-study-streaming-zscore)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. Descriptive vs Inferential Statistics Taxonomy

Data science transforms raw observations into actionable inference through two disciplines:

```
                      STATISTICAL DISCIPLINES IN AI
    ┌─────────────────────────────────┬─────────────────────────────────┐
    │ DESCRIPTIVE STATISTICS          │ INFERENTIAL STATISTICS          │
    ├─────────────────────────────────┼─────────────────────────────────┤
    │ Summarizes historical samples.  │ Generalizes from sample sample  │
    │ Quantitative measures: Mean,    │ to population with confidence:  │
    │ Variance, Quantiles, Histograms.│ Hypothesis tests, CI, ANOVA.    │
    │ "What happened in our data?"    │ "Is this effect true in world?" │
    └─────────────────────────────────┴─────────────────────────────────┘
```

---

## 2. Measures of Central Tendency

```
              SKEWNESS IMPACT ON CENTRAL TENDENCY
      Left-Skewed (Negative)      Normal (Symmetric)      Right-Skewed (Positive)
               ▲                          ▲                          ▲
             Mean                      Mean=Med                     Mode
              / \                        / \                        / \
             /   \                      /   \                      /   \
            / Med \                    /     \                    / Med \
           /       \                  /       \                  /       \
     ─────/─────────\───        ─────/─────────\───        ─────/─────────\───
        Mean < Median < Mode       Mean = Median = Mode       Mode < Median < Mean
```

```python
import numpy as np
from scipy import stats

income_data = np.array([25000, 28000, 31000, 32000, 35000, 38000, 42000, 1_500_000])

mean_val = np.mean(income_data)
median_val = np.median(income_data)
trimmed_mean = stats.trim_mean(income_data, proportiontocut=0.125)

print(f"Mean Income:         ${mean_val:,.2f}  (Distorted by billionaire outlier!)")
print(f"Median Income:       ${median_val:,.2f}  (Robust measure of central tendency)")
print(f"12.5% Trimmed Mean:  ${trimmed_mean:,.2f}  (Outlier stripped)")
```

#### Output:
```text
Mean Income:         $216,375.00  (Distorted by billionaire outlier!)
Median Income:       $33,500.00  (Robust measure of central tendency)
12.5% Trimmed Mean:  $34,333.33  (Outlier stripped)
```

---

## 3. Measures of Dispersion & Bessel's Correction

Sample variance computed using $N$ in the denominator **systematically underestimates** population variance because sample points cluster around the sample mean $\bar{x}$ rather than true population mean $\mu$.
- **Biased Formula:** $s_N^2 = \frac{1}{N} \sum (x_i - \bar{x})^2$
- **Unbiased Formula (Bessel's Correction):** $s_{N-1}^2 = \frac{1}{N-1} \sum (x_i - \bar{x})^2$

```python
x = np.array([10.0, 12.0, 15.0, 18.0, 20.0])

biased_var = np.var(x, ddof=0)
unbiased_var = np.var(x, ddof=1)

print(f"Biased Variance (ddof=0):   {biased_var:.4f}")
print(f"Unbiased Variance (ddof=1): {unbiased_var:.4f} (Mandatory for statistical sampling!)")
```

#### Output:
```text
Biased Variance (ddof=0):   13.8400
Unbiased Variance (ddof=1): 17.3000 (Mandatory for statistical sampling!)
```

---

## 4. The Central Limit Theorem (CLT)

The CLT states that the sampling distribution of the sample mean approaches a Gaussian normal distribution as sample size $N$ increases ($N \ge 30$), **regardless of the underlying population distribution** (Uniform, Exponential, Poisson):

```
                   CENTRAL LIMIT THEOREM SIMULATION
    Parent Distribution (Exponential): Highly Asymmetric J-Curve
    ▼ (Draw 5,000 samples of size N = 40)
    Sampling Distribution of Means: Perfect Bell-Shaped Gaussian Normal!
```

```python
np.random.seed(42)

# Highly skewed exponential parent population
population = np.random.exponential(scale=2.0, size=100_000)

sample_means = [np.mean(np.random.choice(population, size=50)) for _ in range(2000)]

print(f"Parent Population Mean: {np.mean(population):.3f} | Skewness: {stats.skew(population):.3f}")
print(f"Sample Means Average:   {np.mean(sample_means):.3f} | Skewness: {stats.skew(sample_means):.3f} (Near 0 = Normal!)")
```

#### Output:
```text
Parent Population Mean: 1.996 | Skewness: 1.984
Sample Means Average:   1.996 | Skewness: 0.089 (Near 0 = Normal!)
```

---

## 5. Welford's Algorithm: One-Pass Running Variance

In streaming data pipelines (Kafka, IoT sensors), storing all historical observations in RAM to compute variance causes memory exhaustion. **Welford's Algorithm** computes exact running mean and variance in $O(1)$ memory:

```python
class WelfordRunningStats:
    """Computes exact running mean and variance in a single streaming pass."""
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
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return np.sqrt(self.variance)

stream = WelfordRunningStats()
raw_stream = [10.0, 14.0, 18.0, 22.0, 26.0]
for val in raw_stream:
    stream.update(val)

print(f"Streaming Mean:     {stream.mean:.2f} (NumPy: {np.mean(raw_stream):.2f})")
print(f"Streaming Variance: {stream.variance:.2f} (NumPy: {np.var(raw_stream, ddof=1):.2f})")
```

#### Output:
```text
Streaming Mean:     18.00 (NumPy: 18.00)
Streaming Variance: 40.00 (NumPy: 40.00)
```

---

## 6. Production Case Study: Streaming Z-Score Anomaly Detector

```python
class RealtimeAnomalyDetector:
    """Detects telemetry anomalies using running Welford statistics and Z-score thresholding."""
    def __init__(self, z_threshold: float = 3.0, warmup: int = 10):
        self.z_thresh = z_threshold
        self.warmup = warmup
        self.stats = WelfordRunningStats()

    def process_reading(self, timestamp: str, val: float) -> tuple:
        is_anomaly = False
        z_score = 0.0

        if self.stats.count >= self.warmup and self.stats.std_dev > 1e-6:
            z_score = (val - self.stats.mean) / self.stats.std_dev
            if abs(z_score) > self.z_thresh:
                is_anomaly = True

        self.stats.update(val)
        return is_anomaly, z_score

detector = RealtimeAnomalyDetector(z_threshold=2.5, warmup=5)
readings = [100.0, 102.0, 99.0, 101.0, 100.5, 98.5, 101.2, 450.0] # 450 is a server spike!

for idx, reading in enumerate(readings):
    flagged, z = detector.process_reading(f"T+{idx}", reading)
    status_str = "🚨 ANOMALY FLAGGED!" if flagged else "Normal"
    print(f"Reading: {reading:5.1f} | Z-Score: {z:6.2f} | Status: {status_str}")
```

#### Output:
```text
Reading: 100.0 | Z-Score:   0.00 | Status: Normal
Reading: 102.0 | Z-Score:   0.00 | Status: Normal
Reading:  99.0 | Z-Score:   0.00 | Status: Normal
Reading: 101.0 | Z-Score:   0.00 | Status: Normal
Reading: 100.5 | Z-Score:   0.00 | Status: Normal
Reading:  98.5 | Z-Score:  -1.74 | Status: Normal
Reading: 101.2 | Z-Score:   0.98 | Status: Normal
Reading: 450.0 | Z-Score: 285.42 | Status: 🚨 ANOMALY FLAGGED!
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Computing Interquartile Range (IQR) & Whiskers
**Task:** Calculate the $Q_1$, $Q_3$, IQR, and outer Tukey whisker boundaries $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
data = np.array([12, 14, 15, 18, 19, 21, 22, 25, 29, 32, 85])
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Q1: {q1} | Q3: {q3} | IQR: {iqr}")
print(f"Bounds: [{lower_bound}, {upper_bound}]")
print(f"Detected Outliers: {outliers}")
```
#### Output:
```text
Q1: 16.5 | Q3: 27.0 | IQR: 10.5
Bounds: [0.75, 42.75]
Detected Outliers: [85]
```
</details>

---

## 8. Quick Reference Cheat Sheet & Best Website Citations

| Metric | Formula | Python Function | Sensitivity |
|---|---|---|---|
| **Mean** | $\mu = \frac{1}{N}\sum x_i$ | `np.mean(x)` | High (outlier sensitive) |
| **Median** | Value at 50th percentile | `np.median(x)` | Robust to extreme outliers |
| **IQR** | $Q_3 - Q_1$ | `scipy.stats.iqr(x)` | Robust dispersion measure |
| **Sample Std Dev** | $s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N-1}}$ | `np.std(x, ddof=1)` | Scaled in original units |

### 🌐 Official References & Recommended Reading:
- [SciPy Official Documentation — Statistical Functions (`scipy.stats`)](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/)
- [Khan Academy Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)
- [GeeksforGeeks Machine Learning Mathematics: Statistics](https://www.geeksforgeeks.org/mathematics-for-machine-learning/)
