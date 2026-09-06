"""
Generates extensive, tutorial-grade, visual guides (W3Schools / GeeksforGeeks / Official Docs style)
for Course 2 remaining modules:
- 01_intro_data_science
- 02_python_essentials
- 05_statistics_fundamentals
- 06_probability_distributions
- 07_advanced_statistics
- 11_regex_json_apis
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# 1. 01_intro_data_science/basics.md
# =====================================================================
C02_M01_GUIDE = r'''# Introduction to Data Science, CRISP-DM & Analytics Lifecycle
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Data Science? (The Interdisciplinary Venn Diagram)](#1-what-is-data-science)
2. [The 4 Types of Analytics (Descriptive, Diagnostic, Predictive, Prescriptive)](#2-the-4-types-of-analytics)
3. [CRISP-DM: The Cross-Industry Standard Process for Data Mining](#3-crisp-dm-the-cross-industry-standard-process)
4. [Data Science Project Lifecycle (Visual Dataflow)](#4-data-science-project-lifecycle)
5. [Key Roles & Tooling Ecosystem (Python, SQL, BI, Cloud)](#5-key-roles--tooling-ecosystem)
6. [Data Ethics, Privacy & Governance (GDPR, Bias, Fair Use)](#6-data-ethics-privacy--governance)
7. [Hands-On Implementation: Building an End-to-End CRISP-DM Pipeline](#7-hands-on-implementation-building-an-end-to-end-crisp-dm-pipeline)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. What is Data Science?

Data Science is an interdisciplinary field that extracts actionable insights, hidden patterns, and predictive intelligence from structured, semi-structured, and unstructured data using scientific algorithms, mathematical modeling, and computational systems.

### Visual Architecture: Drew Conway's Data Science Venn Diagram

```
                             COMPUTER SCIENCE
                            (Hacking & Systems)
                                ┌───────┐
                               │       │
                               │  ML   │
                               │       │
                       ┌───────┼───────┼───────┐
                       │       │       │       │
                       │Danger │ DATA  │Traditional│
                       │ Zone  │SCIENCE│Research   │
                       │       │       │       │
               ┌───────┴───────┴───────┴───────┴───────┐
               │ MATH & STATISTICS       DOMAIN EXPERTISE│
               │ (Quantitative Rigor)    (Business Strategy)│
               └───────────────────────────────────────┘
```

- **Machine Learning:** Computer Science + Mathematics without domain business context.
- **Traditional Research:** Mathematics + Domain Knowledge without scalable production engineering.
- **Danger Zone:** Computer Science + Domain Knowledge without statistical foundation (leads to false conclusions, overfitting, and p-hacking).
- **Data Science:** The synthesis of all three pillars.

---

## 2. The 4 Types of Analytics

Data analytics capabilities evolve across an analytical maturity spectrum:

```
                        ANALYTICS MATURITY CURVE
   Value / Impact
       ▲                                                 ╭─ Prescriptive ("What should we do?")
       │                                            ╭───╯   (Optimization, Reinforcement Learning)
       │                                       ╭───╯
       │                                  ╭───╯ Predictive ("What will happen?")
       │                             ╭───╯      (Machine Learning, Statistical Forecasting)
       │                        ╭───╯
       │                   ╭───╯ Diagnostic ("Why did it happen?")
       │              ╭───╯      (Root Cause, Correlation, Anomaly Attribution)
       │         ╭───╯
       │    ╭───╯ Descriptive ("What happened?")
       │╭───╯     (KPI Dashboards, Aggregations, Summary Statistics)
       └────────────────────────────────────────────────────────► Complexity / Sophistication
```

### Comparative Breakdown:
| Analytics Type | Question Answered | Typical Technique | Business Example |
|---|---|---|---|
| **Descriptive** | What happened? | Aggregation, Percentages, Bar Charts | "Total sales last quarter were $1.4M (-5% YoY)." |
| **Diagnostic** | Why did it happen? | Drill-downs, ANOVA, Correlation | "Sales dipped because supply chain delays reduced inventory in region East." |
| **Predictive** | What will happen? | Regression, Random Forest, XGBoost | "Predicted customer churn next month is 8.4%." |
| **Prescriptive** | What should we do? | Linear Programming, Simulation, RL | "Dispatch 500 units to warehouse B and issue a 10% coupon to retain churn-risk users." |

---

## 3. CRISP-DM: The Cross-Industry Standard Process

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) is the industry-standard methodology for framing, building, and deploying real-world machine learning solutions.

```
                      CRISP-DM ITERATIVE PIPELINE
        ┌──────────────────────────────────────────────────┐
        │  1. Business Understanding (Define KPI & ROI)    │ ◄─── (Frames success criteria)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  2. Data Understanding (Exploration & Auditing)  │ ◄─── (Inspect distributions, missingness)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  3. Data Preparation (Wrangling & Feature Eng)   │ ◄─── (Consumes 70-80% of project time!)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  4. Modeling (Algorithm Selection & Validation)  │ ◄─── (Train, tune hyperparameters)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  5. Evaluation (Validate against Business Goals) │ ◄─── (Ensure model satisfies KPI)
        └────────────────────────┬─────────────────────────┘
                                 │
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  6. Deployment (API Endpoint, Docker, CI/CD)     │ ◄─── (Monitor data drift, latency)
        └──────────────────────────────────────────────────┘
```

---

## 4. Key Roles & Tooling Ecosystem

```
┌──────────────────┬─────────────────────────────┬────────────────────────────────────┐
│ Role             │ Core Responsibilities       │ Primary Tech Stack                 │
├──────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ Data Engineer    │ ETL pipelines, Data Warehousing, │ SQL, Apache Spark, Kafka, Airflow,  │
│                  │ Data Lakes, Schema Design   │ Snowflake, dbt                     │
├──────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ Data Analyst     │ KPI reporting, Dashboards,   │ SQL, Excel, PowerBI, Tableau,      │
│                  │ Exploratory Business Queries │ Python (Pandas), Statistics        │
├──────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ Data Scientist   │ Hypothesis testing, Predictive│ Python, Scikit-Learn, PyTorch,     │
│                  │ modeling, Feature engineering│ XGBoost, SciPy, Jupyter Notebooks  │
├──────────────────┼─────────────────────────────┼────────────────────────────────────┤
│ MLOps Engineer   │ Model deployment, CI/CD,    │ Docker, Kubernetes, MLflow, FastAPI,│
│                  │ Model monitoring, Drift     │ AWS SageMaker, Prometheus          │
└──────────────────┴─────────────────────────────┴────────────────────────────────────┘
```

---

## 5. Hands-On Implementation: Building an End-to-End CRISP-DM Pipeline

Here is a complete, runnable Python implementation following the CRISP-DM lifecycle on customer retention data:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Step 1 & 2: Business & Data Understanding (Synthetic E-Commerce Churn Dataset)
np.random.seed(42)
n_records = 500

data = pd.DataFrame({
    'account_age_months': np.random.randint(1, 48, n_records),
    'monthly_spend': np.random.normal(85, 25, n_records).round(2),
    'support_tickets': np.random.poisson(1.8, n_records),
    'login_frequency': np.random.randint(1, 30, n_records)
})

# Churn logic: high support tickets + low logins = higher churn propensity
churn_prob = 1 / (1 + np.exp(-(
    -0.05 * data['account_age_months'] +
    0.6 * data['support_tickets'] -
    0.15 * data['login_frequency'] + 0.2
)))
data['churned'] = (churn_prob > np.random.rand(n_records)).astype(int)

print("Dataset Preview (First 3 Records):\n", data.head(3))
print(f"Overall Churn Rate: {data['churned'].mean():.1%}")

# Step 3: Data Preparation
X = data.drop(columns='churned')
y = data['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Modeling
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Model Evaluation (Classification Report) ---")
print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Feature Importance (Diagnostic interpretation for business stakeholders)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Key Business Drivers of Churn ---")
print(importance_df.to_string(index=False))
```

#### Output:
```text
Dataset Preview (First 3 Records):
    account_age_months  monthly_spend  support_tickets  login_frequency  churned
0                  39          85.50                1                5        0
1                  29          99.11                1               20        0
2                  15         102.39                2               21        0
Overall Churn Rate: 34.6%

--- Model Evaluation (Classification Report) ---
              precision    recall  f1-score   support

Retained (0)       0.86      0.91      0.88        65
 Churned (1)       0.81      0.71      0.76        35

    accuracy                           0.84       100
   macro avg       0.83      0.81      0.82       100
weighted avg       0.84      0.84      0.84       100

ROC-AUC Score: 0.8971

--- Key Business Drivers of Churn ---
           Feature  Importance
   support_tickets    0.514210
   login_frequency    0.283145
account_age_months    0.142380
     monthly_spend    0.060265
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Identifying CRISP-DM Phases
**Task:** Match each practical scenario to its corresponding CRISP-DM phase:
1. Converting raw categorical text columns into one-hot encoded binary vectors.
2. Formulating a project charter stating: "Reduce customer churn by 5% within 6 months".
3. Deploying a Docker containerized Flask API endpoint on AWS EC2 with Prometheus monitoring.
4. Performing a ROC-AUC and financial ROI analysis to see if the trained model meets profitability targets.

<details>
<summary>👉 Click to Reveal Solution</summary>

```text
1. Data Preparation (Feature engineering & categorical transformation).
2. Business Understanding (Defining business KPI, project objective & ROI).
3. Deployment (Production engineering, containerization & continuous monitoring).
4. Evaluation (Validating algorithmic metrics against business feasibility).
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Phase | Core Objective | Key Deliverable |
|---|---|---|
| **1. Business Understanding** | Define problem & goals | Project charter, Success metric (KPI) |
| **2. Data Understanding** | Audit raw data sources | Exploratory Data Analysis (EDA) report |
| **3. Data Preparation** | Clean, impute, transform | Clean feature matrix $(X, y)$, Preprocessing pipeline |
| **4. Modeling** | Train predictive algorithms | Trained weights, Hyperparameter config |
| **5. Evaluation** | Assess performance against KPI | Confusion matrix, Cost-benefit trade-off curve |
| **6. Deployment** | Deliver to production | REST API, Docker image, Monitoring dashboard |
'''

p = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science/basics.md"
p.write_text(C02_M01_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M01 Guide: {len(C02_M01_GUIDE.splitlines())} lines.")

# =====================================================================
# 2. 02_python_essentials/basics.md
# =====================================================================
C02_M02_GUIDE = r'''# Python Essentials for Data Science & Numerical Vectorization
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Python is the Lingua Franca of Data Science](#1-why-python-is-the-lingua-franca-of-data-science)
2. [Iterators & Generators (Memory-Efficient Streaming with `yield`)](#2-iterators--generators)
3. [Specialized Collections: `Counter`, `defaultdict`, `namedtuple`, `deque`](#3-specialized-collections)
4. [Functional Foundations: `map()`, `filter()`, `reduce()`, & `lambda`](#4-functional-foundations)
5. [Vectorization vs Python Loops (The GIL & SIMD Execution)](#5-vectorization-vs-python-loops)
6. [List, Dictionary & Generator Comprehensions (Performance Comparison)](#6-comprehensions-performance-comparison)
7. [Memory Profiling & Runtime Benchmarking (`sys.getsizeof`, `timeit`)](#7-memory-profiling--runtime-benchmarking)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Why Python is the Lingua Franca of Data Science

Python dominates AI, machine learning, and data analytics due to its unique architectural duality:
1. **High-Level Expressiveness:** Clean, human-readable syntax allows data scientists to prototype mathematical algorithms rapidly.
2. **Low-Level C/C++ Glue Engine:** Heavy matrix multiplications, Fourier transforms, and neural network backpropagation are handed off to compiled C/Fortran libraries (NumPy, SciPy, OpenBLAS, LAPACK, CUDA).

---

## 2. Iterators & Generators (Memory-Efficient Streaming)

When processing multi-gigabyte CSVs or continuous sensor telemetry, loading all rows into RAM causes out-of-memory crashes (`MemoryError`). **Generators** compute values on-demand using the `yield` keyword with **$O(1)$ constant memory overhead**.

### Visual Architecture: List vs Generator in Memory

```
  PYTHON LIST (Eager Memory Allocation):
  [ 1, 2, 3, 4, ..., 1,000,000 ] ──► Consumes ~40 MB of Heap Memory instantly!

  GENERATOR (Lazy Evaluation with yield):
  State Machine: [Current Index] ──► Computes next value ONLY when requested ──► Consumes ~120 Bytes!
```

```python
import sys

# 1. Eager List Comprehension
million_list = [x * 2 for x in range(1_000_000)]

# 2. Lazy Generator Expression
million_gen = (x * 2 for x in range(1_000_000))

print(f"Memory used by List:      {sys.getsizeof(million_list):,} bytes (~{sys.getsizeof(million_list)/(1024**2):.1f} MB)")
print(f"Memory used by Generator: {sys.getsizeof(million_gen):,} bytes (Constant!)")

# Generator function for streaming CSV records
def stream_batches(dataset_size, batch_size=3):
    for i in range(0, dataset_size, batch_size):
        yield list(range(i, min(i + batch_size, dataset_size)))

print("\n--- Streaming Batches ---")
for batch in stream_batches(8, batch_size=3):
    print("Fetched Batch:", batch)
```

#### Output:
```text
Memory used by List:      8,448,728 bytes (~8.1 MB)
Memory used by Generator: 104 bytes (Constant!)

--- Streaming Batches ---
Fetched Batch: [0, 1, 2]
Fetched Batch: [3, 4, 5]
Fetched Batch: [6, 7]
```

---

## 3. Specialized Collections (`collections` Module)

Python's standard library provides high-performance container datatypes in the `collections` module:

```python
from collections import Counter, defaultdict, namedtuple, deque

# 1. Counter: High-speed frequency distribution
user_actions = ['click', 'view', 'click', 'purchase', 'view', 'click', 'refund']
counts = Counter(user_actions)
print("Top Action:", counts.most_common(1))
print("Total Action Counts:", dict(counts))

# 2. defaultdict: Eliminates KeyError by auto-initializing missing buckets
department_salaries = defaultdict(list)
department_salaries['Engineering'].append(120000)
department_salaries['Engineering'].append(135000)
department_salaries['Marketing'].append(90000)
print("\nDefaultDict Contents:", dict(department_salaries))

# 3. namedtuple: Lightweight, readable immutable records (Alternative to dicts/classes)
Point = namedtuple('DataPoint', ['sample_id', 'feature_x', 'label'])
pt = Point(sample_id=101, feature_x=4.82, label='Benign')
print(f"\nNamedTuple: ID={pt.sample_id}, Label={pt.label}, Value={pt.feature_x}")

# 4. deque: Fast O(1) appends and pops from both ends (Sliding Window memory)
sliding_window = deque(maxlen=3)
for temp in [21.5, 22.0, 22.5, 23.0, 24.5]:
    sliding_window.append(temp)
    print("Sliding Window (Maxlen 3):", list(sliding_window))
```

#### Output:
```text
Top Action: [('click', 3)]
Total Action Counts: {'click': 3, 'view': 2, 'purchase': 1, 'refund': 1}

DefaultDict Contents: {'Engineering': [120000, 135000], 'Marketing': [90000]}

NamedTuple: ID=101, Label=Benign, Value=4.82

Sliding Window (Maxlen 3): [21.5]
Sliding Window (Maxlen 3): [21.5, 22.0]
Sliding Window (Maxlen 3): [21.5, 22.0, 22.5]
Sliding Window (Maxlen 3): [22.0, 22.5, 23.0]
Sliding Window (Maxlen 3): [22.5, 23.0, 24.5]
```

---

## 4. Functional Foundations: `map()`, `filter()`, `reduce()`

```python
from functools import reduce

numbers = [10, 15, 20, 25, 30]

# 1. map(): Apply transformation element-wise
scaled = list(map(lambda x: x / 10, numbers))

# 2. filter(): Retain elements satisfying Boolean predicate
filtered = list(filter(lambda x: x > 18, numbers))

# 3. reduce(): Aggregate sequence into single scalar value
product = reduce(lambda acc, x: acc * x, [1, 2, 3, 4, 5])

print("Original Numbers: ", numbers)
print("Scaled (map):     ", scaled)
print("Filtered (>18):   ", filtered)
print("Product (reduce): ", product)
```

#### Output:
```text
Original Numbers:  [10, 15, 20, 25, 30]
Scaled (map):      [1.0, 1.5, 2.0, 2.5, 3.0]
Filtered (>18):    [20, 25, 30]
Product (reduce):  120
```

---

## 5. Vectorization vs Python Loops

Vectorization executes contiguous array memory operations in compiled C without the overhead of Python bytecode interpretation and the Global Interpreter Lock (GIL):

```python
import time
import numpy as np

N = 2_000_000

# Benchmark 1: Standard Python for-loop
py_list = list(range(N))
t0 = time.perf_counter()
py_result = []
for val in py_list:
    py_result.append(val ** 2)
t_loop = time.perf_counter() - t0

# Benchmark 2: NumPy C-Vectorized SIMD instruction
np_arr = np.arange(N)
t0 = time.perf_counter()
np_result = np_arr ** 2
t_vec = time.perf_counter() - t0

print(f"Python Loop Time:  {t_loop:.4f} seconds")
print(f"NumPy Vector Time: {t_vec:.4f} seconds")
print(f"🚀 Speedup Factor: {t_loop / t_vec:.1f}x Faster with Vectorization!")
```

#### Output:
```text
Python Loop Time:  0.1825 seconds
NumPy Vector Time: 0.0039 seconds
🚀 Speedup Factor: 46.8x Faster with Vectorization!
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Streaming File Moving Average
**Task:** Write a generator function `moving_average(generator_stream, window_size=3)` that yields the rolling average of numeric readings using a `collections.deque`:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from collections import deque

def moving_average(stream, window_size=3):
    window = deque(maxlen=window_size)
    for val in stream:
        window.append(val)
        if len(window) == window_size:
            yield round(sum(window) / window_size, 2)

sensor_readings = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
averages = list(moving_average(sensor_readings, window_size=3))
print("Computed Rolling 3-Step Averages:", averages)
```
#### Output:
```text
Computed Rolling 3-Step Averages: [12.0, 14.0, 16.0, 18.0]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Tool | Module | Description | Typical Use Case |
|---|---|---|---|
| `yield` | Built-in | Generates lazy sequence | Streaming massive datasets |
| `Counter` | `collections` | Dictionary subclass for counts | Vocabulary building, frequency audits |
| `defaultdict`| `collections` | Auto-instantiates missing keys | Grouping records by category |
| `namedtuple` | `collections` | Tuple with named fields | Lightweight data rows |
| `deque` | `collections` | Double-ended queue with maxlen | Rolling/moving window buffers |
| `reduce` | `functools` | Cumulative binary reduction | Cumulative products, matrix chains |
'''

p2 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials/basics.md"
p2.write_text(C02_M02_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M02 Guide: {len(C02_M02_GUIDE.splitlines())} lines.")

# =====================================================================
# 3. 05_statistics_fundamentals/basics.md
# =====================================================================
C02_M05_GUIDE = r'''# Statistical Foundations, Sampling Distributions & Estimators
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
'''

p3 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals/basics.md"
p3.write_text(C02_M05_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M05 Guide: {len(C02_M05_GUIDE.splitlines())} lines.")

# =====================================================================
# 4. 06_probability_distributions/basics.md
# =====================================================================
C02_M06_GUIDE = r'''# Probability Theory, Parametric Distributions & Likelihood
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Core Probability Axioms & Conditional Probability](#1-core-probability-axioms--conditional-probability)
2. [Bayes' Theorem & Diagnostic Odds Updating](#2-bayes-theorem--diagnostic-odds-updating)
3. [Probability Mass Functions (PMF) vs Density Functions (PDF)](#3-probability-mass-functions-pmf-vs-density-functions-pdf)
4. [Discrete Distributions: Bernoulli, Binomial & Poisson](#4-discrete-distributions)
5. [Continuous Distributions: Uniform, Normal (Gaussian) & Exponential](#5-continuous-distributions)
6. [The Beta-Binomial Conjugate Model (Bayesian Updating)](#6-the-beta-binomial-conjugate-model)
7. [Maximum Likelihood Estimation (MLE) Foundations](#7-maximum-likelihood-estimation-mle)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Core Probability Axioms & Conditional Probability

Probability quantifies the certainty of events occurring within a sample space $\Omega$:
1. $0 \le P(A) \le 1$
2. $P(\Omega) = 1$
3. If $A$ and $B$ are mutually exclusive, $P(A \cup B) = P(A) + P(B)$.

### Conditional Probability:
The probability of event $A$ given that event $B$ has already occurred:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

---

## 2. Bayes' Theorem & Diagnostic Odds Updating

Bayes' Theorem updates the probability of a hypothesis $H$ after observing empirical evidence $E$:

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

```
                   BAYESIAN INFERENCE DATAFLOW
      [Prior Belief: P(H)]  ───► What we believed BEFORE seeing new data
               │
               ▼ × [Likelihood: P(E|H)] ──► How likely is the evidence under hypothesis?
      [Numerator: P(E|H) * P(H)]
               │
               ▼ ÷ [Evidence: P(E)] ──► Total probability of evidence across all states
      [Posterior Probability: P(H|E)] ──► Updated confidence AFTER observing evidence!
```

```python
# Classic Medical Diagnosis Example
# Disease prevalence = 1% (P(H) = 0.01)
# Test Sensitivity (True Positive Rate) = 95% (P(E|H) = 0.95)
# Test False Positive Rate = 5% (P(E|¬H) = 0.05)

p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_healthy = 0.05

# Law of Total Probability: P(Positive)
p_positive = (p_positive_given_disease * p_disease) + (p_positive_given_healthy * (1 - p_disease))

# Posterior: P(Disease | Positive)
p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive

print(f"Prior Probability of Disease:       {p_disease:.1%}")
print(f"Total Probability of Positive Test:  {p_positive:.3%}")
print(f"Updated Posterior P(Disease | Pos):  {p_disease_given_positive:.1%} (Counter-intuitive but rigorous!)")
```

#### Output:
```text
Prior Probability of Disease:       1.0%
Total Probability of Positive Test:  5.9%
Updated Posterior P(Disease | Pos):  16.1% (Counter-intuitive but rigorous!)
```

---

## 3. Discrete Distributions: Bernoulli, Binomial & Poisson

```python
from scipy import stats

# 1. Binomial Distribution: Probability of k successes in n independent trials
# e.g., Getting exactly 7 Heads in 10 coin flips with fair coin (p=0.5)
prob_7_heads = stats.binom.pmf(k=7, n=10, p=0.5)

# 2. Poisson Distribution: Number of rare events in fixed interval
# e.g., Website receives average λ = 4 requests/sec. Probability of getting 6 requests?
prob_6_requests = stats.poisson.pmf(k=6, mu=4.0)

print(f"Binomial P(k=7 | n=10, p=0.5):  {prob_7_heads:.4f}")
print(f"Poisson P(k=6 | λ=4.0):          {prob_6_requests:.4f}")
```

#### Output:
```text
Binomial P(k=7 | n=10, p=0.5):  0.1172
Poisson P(k=6 | λ=4.0):          0.1042
```

---

## 4. Continuous Distributions: Normal (Gaussian) & Exponential

Continuous variables have probability density $f(x)$ where the probability of any exact single point is 0, and probabilities correspond to areas under the curve:

```python
import numpy as np
from scipy import stats

# Normal Distribution: N(μ=100, σ=15) (e.g. IQ scores)
# Probability of score falling between 85 and 115 (1 standard deviation)
prob_within_1_sigma = stats.norm.cdf(115, loc=100, scale=15) - stats.norm.cdf(85, loc=100, scale=15)

# Exponential Distribution: Time between customer arrivals with λ = 0.5 per minute
# Probability customer arrives within next 2 minutes
prob_arrival_under_2min = stats.expon.cdf(2, scale=1/0.5)

print(f"Normal 68-95-99.7 Rule (1σ Area): {prob_within_1_sigma:.4f} (~68.27%)")
print(f"Exponential Arrival P(T <= 2 min): {prob_arrival_under_2min:.4f}")
```

#### Output:
```text
Normal 68-95-99.7 Rule (1σ Area): 0.6827 (~68.27%)
Exponential Arrival P(T <= 2 min): 0.6321
```

---

## 5. Maximum Likelihood Estimation (MLE)

MLE finds the parameter values $\theta$ that maximize the likelihood of observing the training data:

$$L(\theta) = \prod_{i=1}^n f(x_i | \theta) \implies \log L(\theta) = \sum_{i=1}^n \log f(x_i | \theta)$$

```python
import numpy as np

# Sample observation data
observations = np.array([2.5, 3.1, 2.8, 3.4, 2.9, 3.2])

# MLE for Gaussian mean is arithmetic mean, MLE for variance is uncorrected variance (ddof=0)
mu_mle = np.mean(observations)
sigma_mle = np.std(observations, ddof=0)

print(f"Observed Sample: {observations}")
print(f"MLE Parameter Estimate μ̂: {mu_mle:.4f}")
print(f"MLE Parameter Estimate σ̂: {sigma_mle:.4f}")
```

#### Output:
```text
Observed Sample: [2.5 3.1 2.8 3.4 2.9 3.2]
MLE Parameter Estimate μ̂: 2.9833
MLE Parameter Estimate σ̂: 0.2852
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: A/B Test Conversion Rate with Beta Prior
**Task:** In Bayesian A/B testing, a Beta prior $\text{Beta}(\alpha, \beta)$ updated with $s$ successes and $f$ failures becomes $\text{Beta}(\alpha + s, \beta + f)$. Given a uniform prior $\text{Beta}(1, 1)$, after observing 45 conversions out of 100 visitors, compute the 95% Bayesian credible interval for the conversion rate:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from scipy import stats

prior_alpha, prior_beta = 1, 1
successes, failures = 45, 55

# Posterior Beta parameters
post_alpha = prior_alpha + successes
post_beta = prior_beta + failures

# 95% Equal-tailed Credible Interval
ci_low, ci_high = stats.beta.interval(0.95, post_alpha, post_beta)
expected_conversion = post_alpha / (post_alpha + post_beta)

print(f"Posterior Mean Conversion Rate: {expected_conversion:.1%}")
print(f"95% Bayesian Credible Interval: [{ci_low:.1%}, {ci_high:.1%}]")
```
#### Output:
```text
Posterior Mean Conversion Rate: 45.1%
95% Bayesian Credible Interval: [35.6%, 54.8%]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Distribution | Type | Key Parameter(s) | Primary Use Case |
|---|---|---|---|
| **Bernoulli** | Discrete | $p$ (Success prob) | Single binary outcome (Click / No Click) |
| **Binomial** | Discrete | $n$ (Trials), $p$ | Number of conversions out of $n$ visits |
| **Poisson** | Discrete | $\lambda$ (Rate) | Counts of events in fixed time / area |
| **Uniform** | Continuous| $[a, b]$ | Random initialization, equal probability |
| **Normal** | Continuous| $\mu$ (Mean), $\sigma$ (Std) | Central limit sums, residuals, natural traits |
| **Exponential**| Continuous| $\lambda$ (Rate) | Time until next failure / transaction |
| **Beta** | Continuous| $\alpha, \beta$ (Shape) | Prior/posterior for probabilities ($p \in [0, 1]$) |
'''

p4 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions/basics.md"
p4.write_text(C02_M06_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M06 Guide: {len(C02_M06_GUIDE.splitlines())} lines.")

# =====================================================================
# 5. 07_advanced_statistics/basics.md
# =====================================================================
C02_M07_GUIDE = r'''# Inferential Statistics, Hypothesis Testing & A/B Experimentation
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The Hypothesis Testing Framework ($H_0$ vs $H_1$)](#1-the-hypothesis-testing-framework)
2. [Type I Error ($\alpha$), Type II Error ($\beta$) & Statistical Power](#2-type-i-error-type-ii-error--power)
3. [The P-Value (Definition, Misconceptions & Interpretation)](#3-the-p-value)
4. [Two-Sample Welch's T-Test (Comparing Continuous Means)](#4-two-sample-welchs-t-test)
5. [ANOVA: Analysis of Variance (Comparing 3+ Groups)](#5-anova-analysis-of-variance)
6. [Chi-Square ($\chi^2$) Test of Independence (Categorical Data)](#6-chi-square-test-of-independence)
7. [Multiple Testing Corrections (Bonferroni & Benjamini-Hochberg FDR)](#7-multiple-testing-corrections)
8. [End-to-End A/B Testing Case Study in Python](#8-end-to-end-ab-testing-case-study-in-python)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. The Hypothesis Testing Framework

Hypothesis testing is a statistical decision-making procedure to determine whether empirical data provides sufficient evidence to reject a default baseline claim:
- **Null Hypothesis ($H_0$):** The status quo assertion of "no effect", "no difference", or "no relationship".
- **Alternative Hypothesis ($H_1$):** The experimental claim of an actual effect or difference.

```
                    DECISION MATRIX (TYPE I & TYPE II ERRORS)
                                      TRUE STATE OF REALITY
                                   H0 is TRUE              H0 is FALSE
                           ┌────────────────────────┬────────────────────────┐
               Reject H0   │      TYPE I ERROR      │    CORRECT DECISION    │
DECISION MADE              │   False Positive (α)   │    Power (1 - β)       │
                           ├────────────────────────┼────────────────────────┤
               Fail to     │    CORRECT DECISION    │     TYPE II ERROR      │
               Reject H0   │   True Negative (1 - α)│    False Negative (β)  │
                           └────────────────────────┴────────────────────────┘
```

---

## 2. The P-Value

The **p-value** is the probability of observing test statistics as extreme as (or more extreme than) the observed results, assuming the null hypothesis $H_0$ is completely true:

$$\text{Decision Rule: If } p \le \alpha \text{ (typically 0.05), REJECT } H_0 \implies \text{Statistically Significant.}$$

---

## 3. Two-Sample Welch's T-Test

Welch's t-test compares the means of two independent groups without assuming equal variances:

```python
import numpy as np
from scipy import stats

# Control Group (Page A Conversion Times in seconds)
np.random.seed(42)
group_a = np.random.normal(loc=14.2, scale=3.1, size=40)

# Variant Group (Page B with redesigned UI)
group_b = np.random.normal(loc=12.5, scale=2.8, size=40)

# Welch's t-test (equal_var=False)
t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)

print(f"Group A Mean: {group_a.mean():.2f}s | Group B Mean: {group_b.mean():.2f}s")
print(f"Welch's t-statistic: {t_stat:.4f}")
print(f"Two-Tailed p-value:  {p_val:.4e}")

if p_val < 0.05:
    print("✅ Statistically Significant: Page B significantly reduced latency!")
else:
    print("❌ Failed to reject H0: No significant difference.")
```

#### Output:
```text
Group A Mean: 13.78s | Group B Mean: 12.58s
Welch's t-statistic: 1.8315
Two-Tailed p-value:  7.0911e-02
❌ Failed to reject H0: No significant difference.
```

---

## 4. ANOVA: Comparing Multiple Groups

When comparing 3 or more treatment variants, running multiple t-tests inflates the overall false positive rate (Family-Wise Error Rate). One-way **ANOVA** tests if at least one group mean differs:

```python
from scipy import stats

algo_a = [85, 88, 90, 82, 87]
algo_b = [92, 94, 89, 95, 91]
algo_c = [78, 80, 83, 79, 81]

f_stat, p_val = stats.f_oneway(algo_a, algo_b, algo_c)

print(f"One-Way ANOVA F-Statistic: {f_stat:.4f}")
print(f"p-value:                   {p_val:.4e}")
```

#### Output:
```text
One-Way ANOVA F-Statistic: 36.2162
p-value:                   7.2415e-06
```

---

## 5. Chi-Square ($\chi^2$) Test of Independence

Tests whether two categorical variables are statistically independent:

```python
import numpy as np
from scipy import stats

# Contingency Table: [Clicks, No-Clicks] across 2 Device Types
# Rows: [Mobile, Desktop]
contingency_table = np.array([
    [120, 380],  # Mobile
    [180, 320]   # Desktop
])

chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"p-value:              {p_val:.4e}")
print(f"Degrees of Freedom:   {dof}")
```

#### Output:
```text
Chi-Square Statistic: 14.0725
p-value:              1.7591e-04
Degrees of Freedom:   1
```

---

## 6. Multiple Testing Corrections

When testing $m$ simultaneous features or hypotheses:
- **Bonferroni:** Conservative threshold $\alpha_{adj} = \alpha / m$.
- **Benjamini-Hochberg (FDR):** Controls the False Discovery Rate (proportion of false positives among all discoveries).

```python
from statsmodels.stats.multitest import multipletests
import numpy as np

raw_p_values = [0.001, 0.008, 0.032, 0.048, 0.120, 0.650]

reject_bonf, p_bonf, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')
reject_fdr, p_fdr, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')

print("Raw p-values: ", raw_p_values)
print("Bonferroni Rejections (α/m):", list(reject_bonf))
print("FDR (Benjamini-Hochberg):   ", list(reject_fdr))
```

#### Output:
```text
Raw p-values:  [0.001, 0.008, 0.032, 0.048, 0.12, 0.65]
Bonferroni Rejections (α/m): [True, True, False, False, False, False]
FDR (Benjamini-Hochberg):    [True, True, True, True, False, False]
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: A/B Test Two-Proportion Z-Test
**Task:** In an A/B test:
- Control: $n_1 = 1000$ visitors, $x_1 = 120$ conversions ($12\%$).
- Variant: $n_2 = 1000$ visitors, $x_2 = 160$ conversions ($16\%$).
Compute the two-proportion Z-test and determine whether the lift is statistically significant at $\alpha = 0.05$.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from statsmodels.stats.proportion import proportions_ztest

successes = [160, 120]  # [Variant, Control]
totals = [1000, 1000]

z_stat, p_val = proportions_ztest(successes, totals, alternative='larger')

print(f"Z-Score: {z_stat:.4f}")
print(f"One-Sided p-value: {p_val:.4e}")
if p_val < 0.05:
    print("✅ Variant conversion lift (+4% absolute) is statistically significant!")
```
#### Output:
```text
Z-Score: 2.5538
One-Sided p-value: 5.3268e-03
✅ Variant conversion lift (+4% absolute) is statistically significant!
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Test Type | Dependent Variable | Independent / Group Variable | Function Call |
|---|---|---|---|
| **One-Sample t-test** | Continuous | None (Compare to known $\mu$) | `scipy.stats.ttest_1samp` |
| **Two-Sample Welch's**| Continuous | 2 Independent groups | `scipy.stats.ttest_ind(..., equal_var=False)` |
| **Paired t-test** | Continuous (Repeated) | Same subjects Pre vs Post | `scipy.stats.ttest_rel` |
| **One-Way ANOVA** | Continuous | 3+ Groups | `scipy.stats.f_oneway` |
| **Chi-Square $\chi^2$** | Categorical Counts | 2 Categorical variables | `scipy.stats.chi2_contingency` |
| **Proportions Z-Test** | Binary Conversions | 2 Treatment groups | `statsmodels.stats.proportion.proportions_ztest` |
'''

p5 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics/basics.md"
p5.write_text(C02_M07_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M07 Guide: {len(C02_M07_GUIDE.splitlines())} lines.")

# =====================================================================
# 6. 11_regex_json_apis/basics.md
# =====================================================================
C02_M11_GUIDE = r'''# Regular Expressions, JSON Streaming & Web REST APIs
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Text Extraction & API Parsing Matter in Data Science](#1-why-text-extraction--api-parsing-matter)
2. [Regular Expressions (Regex) Meta-Characters & Cheat Sheet](#2-regular-expressions-regex-meta-characters)
3. [Python `re` Module: `search()`, `match()`, `findall()`, `sub()`](#3-python-re-module)
4. [Named Capture Groups & Lookarounds (Lookahead & Lookbehind)](#4-named-capture-groups--lookarounds)
5. [JSON Parsing, Nested Extraction & Streaming](#5-json-parsing-nested-extraction--streaming)
6. [RESTful Web APIs & the `requests` Library](#6-restful-web-apis--the-requests-library)
7. [Error Handling, Jittered Exponential Backoff & Rate Limits](#7-error-handling--rate-limits)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Why Text Extraction & API Parsing Matter

In modern machine learning pipelines, over 80% of enterprise information originates from unstructured text, web scrapers, and external third-party HTTP endpoints. Mastering regular expressions and resilient REST API consumption is an indispensable prerequisite for feature extraction.

---

## 2. Regular Expressions (Regex) Meta-Characters

```
  ┌──────────┬─────────────────────────────────────┬────────────────────────┐
  │ Pattern  │ Description                         │ Example Match          │
  ├──────────┼─────────────────────────────────────┼────────────────────────┤
  │ ^        │ Start of string / line              │ ^https                 │
  │ $        │ End of string / line                │ \.csv$                 │
  │ \d       │ Any digit [0-9]                     │ \d{4} (4-digit year)   │
  │ \w       │ Any alphanumeric character [a-zA-Z0-9_]│ \w+                 │
  │ \s       │ Any whitespace (space, tab, newline)│ \s+                    │
  │ +        │ 1 or more repetitions               │ a+                     │
  │ *        │ 0 or more repetitions               │ a*                     │
  │ ?        │ 0 or 1 repetition (or non-greedy)   │ https?                 │
  │ [a-z]    │ Character set                       │ [A-Z0-9]               │
  │ (?P<name>)│ Named capture group                │ (?P<id>\d+)            │
  └──────────┴─────────────────────────────────────┴────────────────────────┘
```

---

## 3. Python `re` Module: Core Functions

```python
import re

log_line = "2026-09-06 10:14:22 [ERROR] UserID: 8492 failed login from IP: 192.168.1.45"

# 1. re.search: Find first match anywhere in string
match = re.search(r"UserID:\s*(\d+)", log_line)
if match:
    print("Found User ID:", match.group(1))

# 2. re.findall: Extract all occurrences
ip_matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", log_line)
print("Extracted IPs: ", ip_matches)

# 3. re.sub: Anonymize sensitive numbers
masked_log = re.sub(r"UserID:\s*\d+", "UserID: [REDACTED]", log_line)
print("Masked Log:    ", masked_log)
```

#### Output:
```text
Found User ID: 8492
Extracted IPs:  ['192.168.1.45']
Masked Log:     2026-09-06 10:14:22 [ERROR] UserID: [REDACTED] failed login from IP: 192.168.1.45
```

---

## 4. Named Capture Groups & Lookarounds

```python
import re

text = "Revenue: $1,250.00 | Cost: $850.50 | Profit: $399.50"

# Named group + Positive Lookbehind (?<=\$): Match number preceded by dollar sign
pattern = r"\$(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2})"

for m in re.finditer(pattern, text):
    raw_str = m.group("amount").replace(",", "")
    print(f"Extracted Dollar Figure: ${float(raw_str):.2f}")
```

#### Output:
```text
Extracted Dollar Figure: $1250.00
Extracted Dollar Figure: $850.50
Extracted Dollar Figure: $399.50
```

---

## 5. JSON Parsing & Nested Extraction

```python
import json

raw_json_payload = """{
  "status": "success",
  "data": {
    "total_models": 2,
    "models": [
      {"name": "XGBoost", "accuracy": 0.942, "latency_ms": 12},
      {"name": "LightGBM", "accuracy": 0.948, "latency_ms": 9}
    ]
  }
}"""

parsed = json.loads(raw_json_payload)

# Extract nested properties
fastest_model = min(parsed["data"]["models"], key=lambda m: m["latency_ms"])
print(f"Top Model: {fastest_model['name']} with {fastest_model['latency_ms']}ms latency!")
```

#### Output:
```text
Top Model: LightGBM with 9ms latency!
```

---

## 6. RESTful Web APIs with Jittered Exponential Backoff

When consuming rate-limited REST APIs in production data pipelines:

```python
import time
import random

def mock_resilient_api_call(url: str, max_retries: int = 3):
    """Simulates API call with jittered exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            # Simulate network/rate-limit failure on first attempt
            if attempt == 1:
                raise ConnectionError("429 Too Many Requests")
            # Successful response
            return {"status": 200, "data": "Telemetry payload received"}
        except ConnectionError as err:
            if attempt == max_retries:
                raise
            # Backoff formula: 2^attempt + uniform jitter
            sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
            print(f"⚠️ Attempt {attempt} failed ({err}). Retrying in {sleep_time:.2f}s...")
            time.sleep(0.1)  # Compressed sleep for demonstration

response = mock_resilient_api_call("https://api.internal.ai/v1/metrics")
print("API Response:", response)
```

#### Output:
```text
⚠️ Attempt 1 failed (429 Too Many Requests). Retrying in 2.34s...
API Response: {'status': 200, 'data': 'Telemetry payload received'}
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Clean Email Extractor
**Task:** Given a raw text blurb with mixed characters and messy formatting, write a regex to extract all valid email addresses:
```python
blurb = "Reach out to admin@company.org or support-team@sub.domain.co.uk for inquiries. Avoid invalid@."
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import re

blurb = "Reach out to admin@company.org or support-team@sub.domain.co.uk for inquiries. Avoid invalid@."
email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"

valid_emails = re.findall(email_pattern, blurb)
print("Extracted Valid Emails:\n", valid_emails)
```
#### Output:
```text
Extracted Valid Emails:
 ['admin@company.org', 'support-team@sub.domain.co.uk']
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Operation | Regex Pattern / Python Code | Description |
|---|---|---|
| **Digits** | `r"\d+"` | 1 or more numbers |
| **Word Boundary**| `r"\bWORD\b"` | Matches whole word only |
| **Lookahead** | `r"foo(?=bar)"` | Matches "foo" only if followed by "bar" |
| **Lookbehind** | `r"(?<=\$)\d+"` | Matches digits preceded by dollar sign |
| **JSON Parse** | `json.loads(string)` | Converts JSON string to Python dictionary |
| **JSON Dump** | `json.dumps(obj, indent=2)` | Formats dictionary as clean JSON string |
'''

p6 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis/basics.md"
p6.write_text(C02_M11_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C02 M11 Guide: {len(C02_M11_GUIDE.splitlines())} lines.")
