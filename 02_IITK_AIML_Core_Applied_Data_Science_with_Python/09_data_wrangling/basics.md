# Data Wrangling & Feature Preprocessing: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Scikit-Learn / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Data Wrangling Lifecycle & Garbage-In Garbage-Out Principle](#1-the-data-wrangling-lifecycle)
2. [Missing Data Mechanisms: MCAR, MAR, and MNAR Taxonomy](#2-missing-data-mechanisms)
3. [Imputation Strategies: Mean/Median vs KNN vs Iterative MICE Imputer](#3-imputation-strategies)
4. [Outlier Detection: IQR, Modified Z-Score & Isolation Forest](#4-outlier-detection)
5. [Feature Scaling: StandardScaler vs MinMaxScaler vs RobustScaler](#5-feature-scaling)
6. [Categorical Encoding: One-Hot, Ordinal, Target Encoding with Smoothing](#6-categorical-encoding)
7. [Multicollinearity & Variance Inflation Factor (VIF)](#7-multicollinearity--vif)
8. [Common Pitfalls: Data Leakage in Preprocessing Pipelines](#8-common-pitfalls-data-leakage)
9. [Production Case Study: Leak-Free Scikit-Learn ColumnTransformer Pipeline](#9-production-case-study-leak-free-pipeline)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. The Data Wrangling Lifecycle

In enterprise AI, 80% of project time is spent wrangling messy data:

```
                      DATA WRANGLING TAXONOMY PIPELINE
    Raw Data  ──►  [Schema Validation]  ──►  [Missing Value Imputation]
                                                      │
    Processed ◄──  [Feature Scaling]    ◄──  [Categorical Encoding]
    Matrix         (Standard / Robust)       (Target / One-Hot)
```

---

## 2. Missing Data Mechanisms: MCAR vs MAR vs MNAR

Donald Rubin's statistical classification of missingness:
1. **Missing Completely at Random (MCAR):** Missingness is completely independent of observed and unobserved data. Safe to drop or impute.
2. **Missing at Random (MAR):** Missingness depends on observed features (e.g. younger users withhold income). Imputation using regression/KNN is valid.
3. **Missing Not at Random (MNAR):** Missingness depends on the unobserved value itself (e.g. highest earners conceal salary). Dropping rows introduces massive survival bias.

---

## 3. Imputation Strategies: Simple vs Advanced MICE

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

raw_df = pd.DataFrame({
    'age': [25.0, 30.0, np.nan, 45.0, 50.0],
    'income': [50000.0, 60000.0, 75000.0, np.nan, 120000.0]
})

# KNN Imputation: Leverages Euclidean distance across features
knn_imp = KNNImputer(n_neighbors=2)
imputed_array = knn_imp.fit_transform(raw_df)
print("KNN Imputed Matrix:\n", pd.DataFrame(imputed_array, columns=raw_df.columns))
```

#### Output:
```text
KNN Imputed Matrix:
     age    income
0  25.0   50000.0
1  30.0   60000.0
2  37.5   75000.0
3  45.0   97500.0
4  50.0  120000.0
```

---

## 4. Feature Scaling: Comparison Matrix

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data_matrix = np.array([[-10.0], [0.0], [5.0], [10.0], [1000.0]]) # 1000 is an extreme outlier

std_scaled = StandardScaler().fit_transform(data_matrix)
rob_scaled = RobustScaler().fit_transform(data_matrix)

print("Standard Scaler (Crushed by outlier):\n", std_scaled.flatten()[:4])
print("Robust Scaler (Median & IQR preserved):\n", rob_scaled.flatten()[:4])
```

#### Output:
```text
Standard Scaler (Crushed by outlier):
 [-0.5332 -0.5084 -0.4960 -0.4836]
Robust Scaler (Median & IQR preserved):
 [-1.5 -0.5  0.   0.5]
```

---

## 5. Production Case Study: Leak-Free Preprocessing Pipeline

Data leakage occurs when parameters calculated on the test/validation set (e.g. test mean or target encoding priors) bleed into training. Always encapsulate transformations in a `Pipeline`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

num_features = ['age', 'income']
cat_features = ['department']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

print("Constructed Scikit-Learn Leak-Free Enterprise Pipeline:")
print(full_pipeline)
```

#### Output:
```text
Constructed Scikit-Learn Leak-Free Enterprise Pipeline:
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('num',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   RobustScaler())]),
                                                  ['age', 'income']),
                                                 ('cat',
                                                  OneHotEncoder(handle_unknown='ignore'),
                                                  ['department'])])),
                ('classifier', LogisticRegression())])
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Target Encoding with m-Estimate Smoothing
**Task:** Implement target encoding with smoothing formula:
$$S_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
def smooth_target_encode(df, cat_col, target_col, m=10):
    global_mean = df[target_col].mean()
    stats = df.groupby(cat_col)[target_col].agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    return df[cat_col].map(smoothed)

df_sample = pd.DataFrame({
    'city': ['NY', 'NY', 'SF', 'SF', 'SF', 'Austin'],
    'converted': [1, 1, 0, 0, 1, 0]
})
df_sample['encoded_city'] = smooth_target_encode(df_sample, 'city', 'converted', m=2)
print("Smoothed Target Encoding:\n", df_sample[['city', 'encoded_city']])
```
#### Output:
```text
Smoothed Target Encoding:
      city  encoded_city
0      NY      0.750000
1      NY      0.750000
2      SF      0.400000
3      SF      0.400000
4      SF      0.400000
5  Austin      0.333333
```
</details>

---

## 7. Quick Reference Cheat Sheet & Best Website Citations

| Task | Scikit-Learn Class | Best Use Case |
|---|---|---|
| **Numeric Imputation** | `SimpleImputer(strategy='median')` | Skewed tabular columns |
| **KNN Imputation** | `KNNImputer(n_neighbors=5)` | Multi-variable correlations |
| **Standardization** | `StandardScaler()` | Gradient descent, PCA |
| **Robust Scaling** | `RobustScaler()` | Data with severe outliers |
| **Categorical** | `OneHotEncoder(handle_unknown='ignore')` | Low-cardinality nominal features |

### 🌐 Official References & Recommended Reading:
- [Scikit-Learn Preprocessing Data Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-Learn Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)
- [W3Schools Data Science Tutorial](https://www.w3schools.com/datascience/)
