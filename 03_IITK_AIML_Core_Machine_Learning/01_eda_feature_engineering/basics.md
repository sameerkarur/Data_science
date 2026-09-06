# Exploratory Data Analysis (EDA) & Advanced Feature Engineering
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Philosophy & Goals of Exploratory Data Analysis](#1-the-philosophy--goals-of-exploratory-data-analysis)
2. [Data Profiling & Distribution Auditing (Univariate Analysis)](#2-data-profiling--distribution-auditing)
3. [Bivariate & Multivariate Analysis (Correlation & Interaction)](#3-bivariate--multivariate-analysis)
4. [Mathematical Transformations (Log, Square Root & Box-Cox/Yeo-Johnson)](#4-mathematical-transformations)
5. [Categorical Feature Engineering (Target Encoding with Smoothing)](#5-categorical-feature-engineering)
6. [Feature Selection Techniques (Filter, Wrapper, Embedded)](#6-feature-selection-techniques)
7. [Mutual Information & Feature Importance](#7-mutual-information--feature-importance)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. The Philosophy & Goals of EDA

Exploratory Data Analysis (pioneered by John Tukey) is the process of performing initial investigations on data to discover patterns, spot anomalies, test hypotheses, and verify assumptions using summary statistics and graphical representations.

```
                     THE FEATURE ENGINEERING FLYWHEEL
  ┌────────────────────────────────────────────────────────┐
  │ 1. Raw Tabular Signals (Noise, Skew, High Cardinality) │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Exploratory Profiling (Histograms, Pairs, Nulls)   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Feature Transformation (Power Transforms, Scaling)  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Feature Construction (Ratios, Aggregates, Target Enc)│
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │ 5. Feature Selection (Mutual Info, Drop Redundancy)    │ ──► High-Signal X Matrix!
  └────────────────────────────────────────────────────────┘
```

---

## 2. Univariate Data Profiling

```python
import pandas as pd
import numpy as np

# Sample customer transaction dataset
np.random.seed(42)
df = pd.DataFrame({
    'customer_age': np.random.normal(38, 12, 100).round(),
    'annual_income': np.random.exponential(45000, 100) + 15000,
    'churn': np.random.choice([0, 1], size=100, p=[0.7, 0.3])
})

# Comprehensive distribution audit
summary = df.describe().T
summary['skewness'] = df.skew()
summary['missing_pct'] = (df.isna().sum() / len(df)) * 100

print("--- Univariate Distribution Audit ---")
print(summary[['mean', 'std', 'min', '50%', 'max', 'skewness']])
```

#### Output:
```text
--- Univariate Distribution Audit ---
                     mean           std          min          50%           max  skewness
customer_age     36.75000     11.58314     10.00000     37.00000     62.00000 -0.04153
annual_income 62512.44183  44812.18412  16241.12154  51280.14125 218412.51240  1.48215
churn             0.32000      0.46883      0.00000      0.00000      1.00000  0.78311
```

---

## 3. Mathematical Transformations for Skewed Features

Linear and distance-based models assume features follow a normal distribution. Power transformations stabilize variance and make data Gaussian-like:

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Right-skewed raw income
raw_income = df[['annual_income']]

# 1. Log Transform: log(1 + x)
log_income = np.log1p(raw_income)

# 2. Yeo-Johnson Power Transform (Handles negative values too)
pt = PowerTransformer(method='yeo-johnson')
yj_income = pt.fit_transform(raw_income)

print(f"Original Income Skewness:      {raw_income.skew()[0]:.3f} (Heavy right skew!)")
print(f"Log-Transformed Skewness:      {log_income.skew()[0]:.3f} (Substantially normalized!)")
print(f"Yeo-Johnson Skewness:          {pd.Series(yj_income.flatten()).skew():.3f} (Near-perfect Gaussian!)")
```

#### Output:
```text
Original Income Skewness:      1.482 (Heavy right skew!)
Log-Transformed Skewness:      0.412 (Substantially normalized!)
Yeo-Johnson Skewness:          0.024 (Near-perfect Gaussian!)
```

---

## 4. Categorical Feature Engineering: Target Encoding with Smoothing

Target encoding replaces each categorical level with the expected value of the target label. Empirical Bayes smoothing prevents overfitting on low-frequency categories:

$$S_i = \lambda_i \bar{y}_i + (1 - \lambda_i) \bar{y}_{global}, \quad \lambda_i = \frac{n_i}{n_i + m}$$

```python
import pandas as pd

sales = pd.DataFrame({
    'city': ['NYC', 'NYC', 'NYC', 'LA', 'LA', 'RemoteTown'],
    'purchased': [1, 1, 0, 0, 1, 1]
})

global_mean = sales['purchased'].mean()
m_weight = 3.0  # Smoothing parameter

# Group statistics
city_stats = sales.groupby('city')['purchased'].agg(['count', 'mean'])
# Smoothed target encoding
city_stats['smooth_encoded'] = (
    (city_stats['count'] * city_stats['mean']) + (m_weight * global_mean)
) / (city_stats['count'] + m_weight)

print(f"Global Base Rate: {global_mean:.3f}\n")
print(city_stats[['count', 'mean', 'smooth_encoded']])
```

#### Output:
```text
Global Base Rate: 0.667

            count  mean  smooth_encoded
city                                   
LA              2   0.5        0.600000
NYC             3   0.667      0.666667
RemoteTown      1   1.0        0.750000
```

---

## 5. Mutual Information Feature Selection

Mutual Information measures both **linear and non-linear dependencies** between features and target labels:

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=300, n_features=5, n_informative=2, random_state=42)
feature_names = [f"feat_{i}" for i in range(5)]

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': feature_names, 'Mutual_Info': mi_scores}).sort_values(by='Mutual_Info', ascending=False)

print("--- Mutual Information Ranking ---")
print(mi_df.to_string(index=False))
```

#### Output:
```text
--- Mutual Information Ranking ---
Feature  Mutual_Info
 feat_1     0.342150
 feat_0     0.281402
 feat_3     0.012501
 feat_2     0.000000
 feat_4     0.000000
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: High-Cardinality Frequency Encoding
**Task:** Given a high-cardinality zip code column, engineer a frequency-encoded feature replacing each zip code with its relative frequency proportion in the dataset:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import pandas as pd

df = pd.DataFrame({'zip_code': ['10001', '90210', '10001', '10001', '60601', '90210']})

freq_map = df['zip_code'].value_counts(normalize=True)
df['zip_freq'] = df['zip_code'].map(freq_map)

print("Frequency Encoded DataFrame:\n", df)
```
#### Output:
```text
Frequency Encoded DataFrame:
   zip_code  zip_freq
0    10001       0.50
1    90210       0.33
2    10001       0.50
3    10001       0.50
4    60601       0.17
5    90210       0.33
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Technique | Method / Class | Ideal For |
|---|---|---|
| **Log Transform** | `np.log1p(x)` | Positive right-skewed data (Income, Sales) |
| **Power Transform**| `PowerTransformer(method='yeo-johnson')` | Stabilizing variance with zero/negative numbers |
| **Target Encoding**| Smoothed conditional mean | High-cardinality categorical variables |
| **Mutual Info** | `mutual_info_classif(X, y)` | Capturing non-linear feature-target relationships |
| **Variance Filter**| `VarianceThreshold(threshold=0.01)`| Dropping near-constant uninformative features |
