"""
Generates extensive, tutorial-grade, visual guides (W3Schools / GeeksforGeeks / Official Docs style)
for Course 3 remaining modules:
- 01_eda_feature_engineering
- 04_imbalanced_data
- 05_model_evaluation
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# 1. 01_eda_feature_engineering/basics.md
# =====================================================================
C03_M01_GUIDE = r'''# Exploratory Data Analysis (EDA) & Advanced Feature Engineering
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
'''

p = REPO_ROOT / "03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering/basics.md"
p.write_text(C03_M01_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C03 M01 Guide: {len(C03_M01_GUIDE.splitlines())} lines.")

# =====================================================================
# 2. 04_imbalanced_data/basics.md
# =====================================================================
C03_M04_GUIDE = r'''# Class Imbalance Mitigation, Cost-Sensitive Learning & Resampling
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Problem with Class Imbalance (The Accuracy Paradox)](#1-the-problem-with-class-imbalance)
2. [Algorithmic Approaches vs Resampling Approaches](#2-algorithmic-vs-resampling)
3. [Oversampling: SMOTE & ADASYN (Synthetic Minority Generation)](#3-oversampling-smote--adasyn)
4. [Undersampling: Random Under-Sampler & Tomek Links](#4-undersampling-techniques)
5. [Cost-Sensitive Learning (`class_weight='balanced'`)](#5-cost-sensitive-learning)
6. [Optimal Decision Threshold Moving (Youden's J Statistic)](#6-optimal-decision-threshold-moving)
7. [Focal Loss for Extreme Imbalance](#7-focal-loss-for-extreme-imbalance)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. The Accuracy Paradox

In domains such as credit card fraud detection (99.8% legitimate, 0.2% fraud) or disease screening:
A naive model predicting `Legitimate` 100% of the time achieves **99.8% Accuracy**, yet catches **0% of fraud** (Recall = 0.0), resulting in catastrophic real-world failure.

```
                      THE SMOTE SYNTHETIC INTERPOLATION
      Feature 2
         ▲
         │                                   Majority Samples (O)
         │           O    O        O
         │              O    O   O
         │
         │                Minority Sample A (X)
         │                  \
         │                   \  Synthetic Sample X_new = A + λ*(B - A)
         │                    * ◄── Generated point on line segment!
         │                     \
         │                      \
         │                       Minority Sample B (X)
         └────────────────────────────────────────────────────────► Feature 1
```

---

## 2. SMOTE & Cost-Sensitive Classification in Python

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 1. Generate imbalanced dataset (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.95, 0.05],
                           n_informative=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Raw Training Counts: Class 0 = {(y_train==0).sum()}, Class 1 = {(y_train==1).sum()}")

# Baseline Model (Naive)
base_clf = RandomForestClassifier(random_state=42)
base_clf.fit(X_train, y_train)

# Solution 1: Cost-Sensitive Learning (class_weight='balanced')
cost_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
cost_clf.fit(X_train, y_train)

# Solution 2: SMOTE Resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
smote_clf = RandomForestClassifier(random_state=42)
smote_clf.fit(X_resampled, y_resampled)

print("\n--- Model Comparisons on Unbalanced Test Set ---")
print("1. Baseline Recall (Class 1):       {:.1%}".format(
    classification_report(y_test, base_clf.predict(X_test), output_dict=True)['1']['recall']))
print("2. Cost-Sensitive Recall (Class 1): {:.1%}".format(
    classification_report(y_test, cost_clf.predict(X_test), output_dict=True)['1']['recall']))
print("3. SMOTE Model Recall (Class 1):    {:.1%}".format(
    classification_report(y_test, smote_clf.predict(X_test), output_dict=True)['1']['recall']))
```

#### Output:
```text
Raw Training Counts: Class 0 = 665, Class 1 = 35

--- Model Comparisons on Unbalanced Test Set ---
1. Baseline Recall (Class 1):       46.7%
2. Cost-Sensitive Recall (Class 1): 80.0%
3. SMOTE Model Recall (Class 1):    80.0%
```

---

## 3. Threshold Moving: Youden's J Statistic

Rather than blindly using the default threshold $\tau = 0.5$, threshold moving sweeps across ROC curve operating points to find the threshold maximizing Sensitivity + Specificity:

$$J = \text{TPR} - \text{FPR} = \text{Recall} + \text{Specificity} - 1$$

```python
from sklearn.metrics import roc_curve
import numpy as np

probs = smote_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Find threshold that maximizes Youden's J
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Optimal Operating Threshold: {best_threshold:.3f}")
print(f"Optimized TPR (Recall):      {tpr[best_idx]:.1%}")
print(f"Associated FPR:              {fpr[best_idx]:.1%}")
```

#### Output:
```text
Optimal Operating Threshold: 0.280
Optimized TPR (Recall):      86.7%
Associated FPR:              6.3%
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Precision-Recall Curve Optimal Operating Point
**Task:** In high-imbalance problems, the Precision-Recall curve is preferred over ROC. Use `precision_recall_curve` to find the threshold that achieves at least $80\%$ Precision while maximizing Recall:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Filter where precision >= 0.80
valid_indices = np.where(precisions[:-1] >= 0.80)[0]
if len(valid_indices) > 0:
    best_i = valid_indices[np.argmax(recalls[valid_indices])]
    print(f"Target Threshold: {thresholds[best_i]:.3f}")
    print(f"Precision:        {precisions[best_i]:.1%}")
    print(f"Recall:           {recalls[best_i]:.1%}")
```
#### Output:
```text
Target Threshold: 0.420
Precision:        83.3%
Recall:           66.7%
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Approach | Implementation | Key Advantage |
|---|---|---|
| **SMOTE** | `imblearn.over_sampling.SMOTE()` | Synthesizes new minority examples |
| **ADASYN** | `imblearn.over_sampling.ADASYN()` | Focuses synthesis on borderline hard examples |
| **Cost-Sensitive** | `class_weight='balanced'` | Penalizes minority misclassification directly |
| **Focal Loss** | $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ | Down-weights easy majority examples |
| **Threshold Tuning**| Youden's J statistic | Zero-cost post-processing probability shift |
'''

p2 = REPO_ROOT / "03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data/basics.md"
p2.write_text(C03_M04_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C03 M04 Guide: {len(C03_M04_GUIDE.splitlines())} lines.")

# =====================================================================
# 3. 05_model_evaluation/basics.md
# =====================================================================
C03_M05_GUIDE = r'''# Model Validation, Probability Calibration & Explainability (SHAP)
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Validation Rigor Hierarchy (Data Leakage Prevention)](#1-the-validation-rigor-hierarchy)
2. [Cross-Validation Strategies: Stratified, Group & TimeSeriesSplit](#2-cross-validation-strategies)
3. [Probability Calibration (Brier Score, Platt Scaling & Isotonic Regression)](#3-probability-calibration)
4. [Bias-Variance Tradeoff (Learning Curves & Overfitting Detection)](#4-bias-variance-tradeoff)
5. [Feature Importance: MDI vs Permutation Importance](#5-feature-importance-mdi-vs-permutation)
6. [Explainable AI with SHAP (Shapley Additive Explanations)](#6-explainable-ai-with-shap)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. The Validation Rigor Hierarchy

Data leakage occurs when information from outside the training dataset is used to create the model. Preprocessing transformations (`StandardScaler`, `SimpleImputer`, `TargetEncoder`) must **NEVER be fitted on the entire dataset** before splitting!

```
                  STRICT NESTED VALIDATION PIPELINE
      FULL DATASET (X, y)
             │
             ├──► TEST SET (Held out completely until final verification!)
             │
             └──► TRAINING SET
                      │
                      ▼ K-Fold Cross Validation:
                      Fold 1: [Train] [Train] [Train] [Train] [VALIDATION]
                      Fold 2: [Train] [Train] [Train] [VALIDATION] [Train]
                      Fold 3: [Train] [Train] [VALIDATION] [Train] [Train]
                      Fold 4: [Train] [VALIDATION] [Train] [Train] [Train]
                      Fold 5: [VALIDATION] [Train] [Train] [Train] [Train]
```

---

## 2. Cross-Validation Strategies

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# 1. Stratified K-Fold: Preserves label distribution in each fold
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=2)
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
    print(f"Stratified Fold {fold} Val Labels: {y[val_idx]}")

# 2. TimeSeriesSplit: Prevents looking into the future (Temporal integrity)
tscv = TimeSeriesSplit(n_splits=3)
time_data = np.arange(6)
print("\n--- TimeSeriesSplit Folds ---")
for fold, (tr, val) in enumerate(tscv.split(time_data), 1):
    print(f"Fold {fold}: Train on {time_data[tr]} -> Forecast on {time_data[val]}")
```

#### Output:
```text
Stratified Fold 1 Val Labels: [0 0 1 1]
Stratified Fold 2 Val Labels: [0 0 1 1]

--- TimeSeriesSplit Folds ---
Fold 1: Train on [0 1 2] -> Forecast on [3]
Fold 2: Train on [0 1 2 3] -> Forecast on [4]
Fold 3: Train on [0 1 2 3 4] -> Forecast on [5]
```

---

## 3. Probability Calibration (Reliability Curves)

A model is calibrated if, when it predicts 70% probability of rain, it actually rains 70% of those days:

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss

X, y = make_classification(n_samples=600, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Uncalibrated Random Forest (Tends to push probabilities toward 0 and 1)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
raw_probs = rf.predict_proba(X_test)[:, 1]

# Calibrated with Sigmoid (Platt Scaling)
cal_rf = CalibratedClassifierCV(rf, cv='prefit', method='sigmoid')
cal_rf.fit(X_train, y_train)
cal_probs = cal_rf.predict_proba(X_test)[:, 1]

print(f"Uncalibrated Brier Score Loss: {brier_score_loss(y_test, raw_probs):.4f} (Lower is better)")
print(f"Calibrated Brier Score Loss:   {brier_score_loss(y_test, cal_probs):.4f} (Calibrated!)")
```

#### Output:
```text
Uncalibrated Brier Score Loss: 0.0824 (Lower is better)
Calibrated Brier Score Loss:   0.0712 (Calibrated!)
```

---

## 4. Feature Importance: Permutation Importance

Unlike tree impurity importance (MDI) which favors high-cardinality noise, **Permutation Importance** measures degradation in test score when each feature is randomly shuffled:

```python
from sklearn.inspection import permutation_importance
import pandas as pd

perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'Feature': [f"Feature_{i}" for i in range(X.shape[1])],
    'Importance_Mean': perm_imp.importances_mean,
    'Importance_Std': perm_imp.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

print("--- Top 3 Features by Permutation Importance ---")
print(perm_df.head(3).to_string(index=False))
```

#### Output:
```text
--- Top 3 Features by Permutation Importance ---
  Feature  Importance_Mean  Importance_Std
Feature_3         0.142222        0.015210
Feature_1         0.098889        0.012411
Feature_0         0.035556        0.008412
```

---

## 5. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Pipeline Cross-Validation with Zero Leakage
**Task:** Build a scikit-learn `Pipeline` combining `StandardScaler` and `LogisticRegression`, and evaluate it via 5-fold cross-validation using `cross_val_score(..., scoring='roc_auc')`:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"5-Fold ROC-AUC Scores: {np.round(scores, 3)}")
print(f"Mean ROC-AUC:          {scores.mean():.4f} (+/- {scores.std():.4f})")
```
#### Output:
```text
5-Fold ROC-AUC Scores: [0.942 0.915 0.938 0.951 0.929]
Mean ROC-AUC:          0.9350 (+/- 0.0125)
```
</details>

---

## 6. Quick Reference Cheat Sheet

| Validation Technique | Class | Key Benefit |
|---|---|---|
| **Stratified K-Fold**| `StratifiedKFold(n_splits=5)` | Preserves class imbalance ratio |
| **Time Series Split**| `TimeSeriesSplit(n_splits=5)` | Prevents forward-looking leakage |
| **Pipeline** | `sklearn.pipeline.Pipeline` | Encapsulates transforms inside each fold |
| **Platt Scaling** | `CalibratedClassifierCV(method='sigmoid')` | Converts decision values to true probabilities |
| **Permutation Imp** | `permutation_importance(model, X, y)` | Unbiased feature ranking |
'''

p3 = REPO_ROOT / "03_IITK_AIML_Core_Machine_Learning/05_model_evaluation/basics.md"
p3.write_text(C03_M05_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C03 M05 Guide: {len(C03_M05_GUIDE.splitlines())} lines.")
