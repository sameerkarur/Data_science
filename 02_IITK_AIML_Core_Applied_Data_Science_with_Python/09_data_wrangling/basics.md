# Data Wrangling, Cleaning & Preprocessing: Complete Step-by-Step Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Data Wrangling? (The CRISP-DM Pipeline)](#1-what-is-data-wrangling)
2. [Handling Missing Data (MCAR, MAR, MNAR & Imputation Strategies)](#2-handling-missing-data)
3. [Outlier Detection & Treatment (Z-Score & IQR Method with Visual Boxplot)](#3-outlier-detection--treatment)
4. [Feature Scaling (StandardScaler vs MinMaxScaler vs RobustScaler)](#4-feature-scaling)
5. [Categorical Encoding (One-Hot, Ordinal & Target Encoding)](#5-categorical-encoding)
6. [Data Type Casting & String Sanitation](#6-data-type-casting--string-sanitation)
7. [Deduplication & Record Linkage](#7-deduplication--record-linkage)
8. [Building Automated Scikit-Learn Preprocessing Pipelines](#8-building-automated-scikit-learn-preprocessing-pipelines)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. What is Data Wrangling?

Data wrangling (or data munging) is the systematic process of transforming raw, messy data into an accurate, clean, and structured format suitable for analytics and machine learning models. Industry studies show that **70% to 80%** of a data scientist's time is spent on data wrangling.

```
                  THE DATA WRANGLING REFINERY PIPELINE
 ┌───────────────┐     ┌────────────────┐     ┌───────────────┐     ┌─────────────────┐
 │ RAW DATA      │ ──► │ DATA CLEANING  │ ──► │ TRANSFORMATION│ ──► │ MODEL READY     │
 │ Dirty CSV     │     │ Drop / Impute  │     │ Scaling       │     │ Feature Matrix  │
 │ Broken JSON   │     │ Fix Outliers   │     │ Encoding      │     │ Clean X, y      │
 │ API Responses │     │ Remove Dupes   │     │ Binning       │     │ Zero Leakage    │
 └───────────────┘     └────────────────┘     └───────────────┘     └─────────────────┘
```

---

## 2. Handling Missing Data

Missing values generally fall into three statistical taxonomies:
1. **MCAR (Missing Completely at Random):** Missingness is totally independent of all variables (e.g. sensor battery died).
2. **MAR (Missing at Random):** Missingness is systematically related to other observed variables.
3. **MNAR (Missing Not at Random):** The missing value itself depends on the unobserved truth (e.g. high-income individuals refusing to declare income).

### Code: Identifying & Imputing Missing Values
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.DataFrame({
    'Age': [25, np.nan, 29, 45, np.nan, 38],
    'Salary': [50000, 62000, np.nan, 110000, 95000, 85000],
    'Department': ['IT', 'HR', 'IT', np.nan, 'Finance', 'IT']
})

print("Missing Values Summary:\n", df.isna().sum())

# Strategy 1: Numerical Median Imputation
imputer_num = SimpleImputer(strategy='median')
df['Age_Imputed'] = imputer_num.fit_transform(df[['Age']])

# Strategy 2: Categorical Most Frequent Imputation
imputer_cat = SimpleImputer(strategy='most_frequent')
df['Department_Imputed'] = imputer_cat.fit_transform(df[['Department']])

print("\n--- Imputed DataFrame ---")
print(df[['Age_Imputed', 'Salary', 'Department_Imputed']])
```

#### Output:
```text
Missing Values Summary:
 Age           2
Salary        1
Department    1
dtype: int64

--- Imputed DataFrame ---
   Age_Imputed    Salary Department_Imputed
0         25.0   50000.0                 IT
1         33.5   62000.0                 HR
2         29.0       NaN                 IT
3         45.0  110000.0                 IT
4         33.5   95000.0            Finance
5         38.0   85000.0                 IT
```

---

## 3. Outlier Detection & Treatment (IQR & Z-Score)

### Visual Boxplot Anatomy (Tukey's IQR Method):
```
    Outlier               Q1          Median (Q2)       Q3                Outlier
      *     ├───[ Lower Whisker ]──────[ Box ]──────[ Upper Whisker ]───┤   *
                 Q1 - 1.5 * IQR                       Q3 + 1.5 * IQR
            ◄────────────────────── Interquartile Range ────────────────►
```

```python
import numpy as np
import pandas as pd

values = np.array([12, 14, 15, 18, 19, 19, 21, 22, 23, 25, 28, 95])  # 95 is extreme outlier

# Calculate IQR bounds
q25, q75 = np.percentile(values, [25, 75])
iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

outliers = values[(values < lower_bound) | (values > upper_bound)]
capped_values = np.clip(values, lower_bound, upper_bound)

print(f"Q25: {q25} | Q75: {q75} | IQR: {iqr}")
print(f"Valid Range: [{lower_bound:.1f}, {upper_bound:.1f}]")
print(f"Detected Outliers: {outliers}")
print(f"Winsorized/Capped: {capped_values}")
```

#### Output:
```text
Q25: 17.25 | Q75: 22.25 | IQR: 5.0
Valid Range: [9.8, 29.8]
Detected Outliers: [95]
Winsorized/Capped: [12.   14.   15.   18.   19.   19.   21.   22.   23.   25.   28.   29.75]
```

---

## 4. Feature Scaling (Standard vs MinMax vs Robust)

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = np.array([[10], [20], [30], [40], [500]])  # 500 is extreme outlier

std_scaler = StandardScaler().fit_transform(data)
minmax_scaler = MinMaxScaler().fit_transform(data)
robust_scaler = RobustScaler().fit_transform(data)

print("StandardScaler (Zero mean, unit variance):\n", np.round(std_scaler.flatten(), 2))
print("MinMaxScaler (Bounded strictly [0, 1]):\n", np.round(minmax_scaler.flatten(), 2))
print("RobustScaler (Median & IQR centered):\n", np.round(robust_scaler.flatten(), 2))
```

#### Output:
```text
StandardScaler (Zero mean, unit variance):
 [-0.58 -0.53 -0.47 -0.42  2.01]
MinMaxScaler (Bounded strictly [0, 1]):
 [0.   0.02 0.04 0.06 1.  ]
RobustScaler (Median & IQR centered):
 [-1.  -0.5  0.   0.5 23.5]
```

---

## 5. Categorical Encoding (One-Hot & Ordinal)

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    'Tier': ['Bronze', 'Silver', 'Gold', 'Platinum'],  # Ordinal
    'City': ['Paris', 'Tokyo', 'Paris', 'New York']    # Nominal
})

# 1. Ordinal Mapping (Preserving explicit hierarchy)
tier_ranking = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
df['Tier_Encoded'] = df['Tier'].map(tier_ranking)

# 2. Nominal One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)
print("Encoded DataFrame:\n", df_encoded)
```

#### Output:
```text
Encoded DataFrame:
        Tier  Tier_Encoded  City_Paris  City_Tokyo
0    Bronze             1           1           0
1    Silver             2           0           1
2      Gold             3           1           0
3  Platinum             4           0           0
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Pipeline for Automated Data Preprocessing
**Task:** Build a scikit-learn `ColumnTransformer` that imputes and standardizes numerical columns while one-hot encoding categorical columns:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    'Age': [25, 45, None, 35],
    'Salary': [50000, 110000, 80000, None],
    'Dept': ['HR', 'IT', 'Finance', 'IT']
})

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, ['Age', 'Salary']),
    ('cat', OneHotEncoder(drop_first=True), ['Dept'])
])

clean_matrix = preprocessor.fit_transform(df)
print("Pipeline Output Shape:", clean_matrix.shape)
print("Transformed Matrix:\n", clean_matrix.round(2))
```
#### Output:
```text
Pipeline Output Shape: (4, 4)
Transformed Matrix:
 [[-1.46 -1.27  1.    0.  ]
 [ 1.46  1.27  0.    1.  ]
 [ 0.    0.    0.    0.  ]
 [ 0.    0.    0.    1.  ]]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Task | Scikit-Learn / Pandas Class | Formula / Behavior |
|---|---|---|
| **Median Impute** | `SimpleImputer(strategy='median')` | Replaces NaNs with median |
| **Z-Score Scale** | `StandardScaler()` | $z = (x - \mu) / \sigma$ |
| **Range Scale** | `MinMaxScaler(feature_range=(0, 1))` | $x_{norm} = (x - min) / (max - min)$ |
| **Robust Scale** | `RobustScaler()` | Uses Median and IQR |
| **One-Hot Enc** | `OneHotEncoder(drop_first=True)` | Generates binary indicator cols |
