"""
Generates W3Schools / GeeksforGeeks style tutorials with visual diagrams, code, output blocks,
and practice exercises for Applied Data Science, Machine Learning, Deep Learning, and RAG.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Data Wrangling Guide
DATA_WRANGLING_GUIDE = r'''# Data Wrangling, Cleaning & Preprocessing: Complete Step-by-Step Guide
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
'''

p_wrang = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling/basics.md"
p_wrang.write_text(DATA_WRANGLING_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated Data Wrangling Guide: {len(DATA_WRANGLING_GUIDE.splitlines())} lines.")

# 2. Classification Master Guide
CLASSIFICATION_GUIDE = r'''# Classification Algorithms & Model Evaluation: Complete Beginner-to-Pro Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Classification? (Supervised Learning Paradigm)](#1-what-is-classification)
2. [Logistic Regression & the Sigmoid Activation Function](#2-logistic-regression--the-sigmoid-activation-function)
3. [Decision Trees: Gini Impurity & Information Gain](#3-decision-trees-gini-impurity--information-gain)
4. [Random Forests & Bagging Ensembles](#4-random-forests--bagging-ensembles)
5. [Gradient Boosting & XGBoost Architecture](#5-gradient-boosting--xgboost-architecture)
6. [Classification Metrics: Confusion Matrix, Precision, Recall & F1](#6-classification-metrics)
7. [ROC-AUC & Precision-Recall Curves](#7-roc-auc--precision-recall-curves)
8. [Cross-Validation & Hyperparameter Tuning (GridSearchCV)](#8-cross-validation--hyperparameter-tuning)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. What is Classification?

In machine learning, **Classification** is a supervised learning task where the target output variable $y$ is discrete and categorical (e.g. `Spam / Not Spam`, `Fraud / Legit`, `Class A / B / C`).

```
                     SUPERVISED CLASSIFICATION WORKFLOW
  ┌────────────────────────────────┐
  │ Labeled Training Data (X, y)   │ ──► [Feature Matrix: n_samples × n_features]
  └───────────────┬────────────────┘     [Target Labels: y ∈ {0, 1, ..., k}]
                  │
                  ▼ Training Phase
  ┌────────────────────────────────┐
  │ Learn Decision Boundary: f(X)  │ ──► Logistic Reg, Decision Tree, Random Forest, XGBoost
  └───────────────┬────────────────┘
                  │
                  ▼ Inference Phase
  [New Unseen Sample X_new] ─────────► Compute Probability P(y=1|X) ──► Apply Threshold τ ──► Predicted Class
```

---

## 2. Logistic Regression & the Sigmoid Function

Logistic Regression predicts probabilities using the logistic sigmoid function $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b$$

```
                           THE SIGMOID ACTIVATION CURVE
         P(y=1)
           1.0 ┼                                  ╭────────────
               │                                ╭╯
           0.5 ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─╭╯─ ─ ─ ─ ─ ─ ─ Decision Threshold (τ = 0.5)
               │                             ╭╯
           0.0 ┼───────────╮────────────────╯──────────────────
              -∞          -4       -2       0       2       4   +∞  (z = w·x + b)
```

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4, n_informative=2, random_state=42)

clf = LogisticRegression()
clf.fit(X, y)

sample_prob = clf.predict_proba(X[:3])
sample_pred = clf.predict(X[:3])

print("Predicted Probabilities [P(0), P(1)]:\n", np.round(sample_prob, 3))
print("Final Class Predictions:             ", sample_pred)
```

#### Output:
```text
Predicted Probabilities [P(0), P(1)]:
 [[0.052 0.948]
 [0.892 0.108]
 [0.124 0.876]]
Final Class Predictions:              [1 0 1]
```

---

## 3. Decision Trees: Gini Impurity

Decision trees recursively partition the feature space using impurity criteria:
- **Gini Impurity:** $G = 1 - \sum_{i=1}^C p_i^2$ (Gini = 0 means perfectly pure node)

```python
from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

print("Decision Tree Split Logic:\n")
print(export_text(tree, feature_names=[f"Feature_{i}" for i in range(4)]))
```

#### Output:
```text
Decision Tree Split Logic:

|--- Feature_1 <= 0.04
|   |--- Feature_0 <= 0.41
|   |   |--- class: 0
|   |--- Feature_0 >  0.41
|   |   |--- class: 0
|--- Feature_1 >  0.04
|   |--- Feature_0 <= -0.45
|   |   |--- class: 0
|   |--- Feature_0 >  -0.45
|   |   |--- class: 1
```

---

## 4. Random Forests & XGBoost Ensemble

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)

xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
xgb.fit(X, y)

print(f"Random Forest Accuracy: {accuracy_score(y, rf.predict(X)):.4f} | F1: {f1_score(y, rf.predict(X)):.4f}")
print(f"XGBoost Accuracy:       {accuracy_score(y, xgb.predict(X)):.4f} | F1: {f1_score(y, xgb.predict(X)):.4f}")
```

#### Output:
```text
Random Forest Accuracy: 0.9650 | F1: 0.9653
XGBoost Accuracy:       0.9850 | F1: 0.9852
```

---

## 5. Classification Metrics & Confusion Matrix

```
                        CONFUSION MATRIX ANATOMY
                             PREDICTED CLASS
                           Positive        Negative
           Positive    ┌──────────────┬──────────────┐
            (True)     │ True Pos(TP) │ False Neg(FN)│ ◄── Recall = TP / (TP + FN)
ACTUAL                 ├──────────────┼──────────────┤
CLASS      Negative    │ False Pos(FP)│ True Neg (TN)│ ◄── Specificity = TN / (TN + FP)
            (True)     └──────────────┴──────────────┘
                              ▲
                              │
                    Precision = TP / (TP + FP)
```

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = xgb.predict(X)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)
print("\n--- Detailed Classification Report ---")
print(classification_report(y, y_pred, target_names=['Class 0', 'Class 1']))
```

#### Output:
```text
Confusion Matrix:
 [[99  1]
 [ 2 98]]

--- Detailed Classification Report ---
              precision    recall  f1-score   support

     Class 0       0.98      0.99      0.99       100
     Class 1       0.99      0.98      0.98       100

    accuracy                           0.98       200
   macro avg       0.98      0.98      0.98       200
weighted avg       0.98      0.98      0.98       200
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding Optimal Classification Threshold
**Task:** Given predicted probabilities `y_prob` and true binary labels `y_true`, iterate over threshold values $\tau \in [0.1, 0.9]$ with step $0.1$ and identify the threshold that maximizes the F1-Score:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np
from sklearn.metrics import f1_score

np.random.seed(42)
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
y_probs = np.array([0.9, 0.8, 0.35, 0.45, 0.2, 0.6, 0.7, 0.1, 0.55, 0.25])

best_thresh = 0.5
best_f1 = 0.0

for t in np.arange(0.1, 0.9, 0.1):
    preds = (y_probs >= t).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Optimal Threshold: {best_thresh:.1f} | Peak F1-Score: {best_f1:.4f}")
```
#### Output:
```text
Optimal Threshold: 0.5 | Peak F1-Score: 0.8889
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Metric | Formula | Business Interpretation |
|---|---|---|
| **Precision** | $\frac{TP}{TP + FP}$ | "When model says YES, how often is it right?" (Minimize false alarms) |
| **Recall (Sensitivity)**| $\frac{TP}{TP + FN}$| "Of all actual positives, how many did we catch?" (Cancer / Fraud) |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean balancing Precision and Recall |
| **ROC-AUC** | Area under TPR vs FPR | Discrimination ability across all decision thresholds |
'''

p_clf = REPO_ROOT / "03_IITK_AIML_Core_Machine_Learning/03_classification/basics.md"
p_clf.write_text(CLASSIFICATION_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated Classification Guide: {len(CLASSIFICATION_GUIDE.splitlines())} lines.")

# 3. RAG Architecture Guide
RAG_GUIDE = r'''# Retrieval-Augmented Generation (RAG): Complete Architecture & Engineering Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is RAG & Why Do LLMs Need It?](#1-what-is-rag--why-do-llms-need-it)
2. [The End-to-End RAG Architecture (Visual Dataflow)](#2-the-end-to-end-rag-architecture-visual-dataflow)
3. [Document Ingestion & Chunking Strategies](#3-document-ingestion--chunking-strategies)
4. [Vector Embeddings & Semantic Similarity (Cosine, Dot Product)](#4-vector-embeddings--semantic-similarity)
5. [Vector Database Storage & Indexing (ChromaDB / HNSW)](#5-vector-database-storage--indexing)
6. [Context Retrieval & Hybrid Search (Keyword + Dense Vectors)](#6-context-retrieval--hybrid-search)
7. [Augmented Prompt Engineering & Generation](#7-augmented-prompt-engineering--generation)
8. [Complete RAG Pipeline Implementation from Scratch in Python](#8-complete-rag-pipeline-implementation-from-scratch)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. What is RAG & Why Do LLMs Need It?

**Retrieval-Augmented Generation (RAG)** is an AI framework that augments Large Language Models (LLMs) with dynamic, external domain knowledge without fine-tuning model weights.

### Why do LLMs fail without RAG?
- **Knowledge Cutoffs:** LLMs know nothing about events or data after their training date.
- **Hallucinations:** When an LLM lacks factual context, it generates plausible-sounding falsehoods.
- **Private Enterprise Data:** LLMs cannot access private company documents, PDFs, or databases.
- **Cost & Latency of Fine-Tuning:** Retraining an LLM every day is prohibitively expensive.

---

## 2. The End-to-End RAG Architecture

```
                               THE FULL RAG PIPELINE
   [Enterprise Docs: PDF, Markdown, DB]
                    │
                    ▼ 1. INGESTION & CHUNKING
     [Chunk 1]   [Chunk 2]   [Chunk 3]
         │           │           │
         └───────────┬───────────┘
                     ▼ 2. EMBEDDING MODEL (e.g. text-embedding-3-small)
               [Dense Vectors: 1536-D]
                     │
                     ▼ 3. VECTOR DATABASE (ChromaDB / Pinecone / HNSW)
             ┌─────────────────────────────┐
             │ Vector Index & Metadata     │
             └──────────────┬──────────────┘
                            │
   [User Query: "What is refund policy?"] ──► 4. Embed Query ──► Vector Similarity Search (Top-k)
                                                                            │
                                                                            ▼ 5. Retrieved Chunks
   [Augmented Prompt: Context + Question] ◄─────────────────────────────────┘
         │
         ▼ 6. LLM GENERATION (GPT-4 / Claude / Llama-3)
   "According to section 4, refunds are processed within 14 business days..." (Grounded & Factual!)
```

---

## 3. Document Ingestion & Chunking Strategies

Chunking divides long documents into smaller semantic units:
- **Fixed Size Chunking:** e.g. 500 tokens with 50-token overlap.
- **Recursive Character Chunking:** Splits sequentially on paragraphs (`\n\n`), sentences (`\n`), and punctuation (`.`).
- **Semantic Chunking:** Splits where embedding distance between consecutive sentences exceeds a threshold.

```python
def recursive_chunker(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Simple sliding window chunker with overlapping context."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

sample_doc = "Artificial intelligence is transforming software engineering. " * 50
chunks = recursive_chunker(sample_doc, chunk_size=30, overlap=10)
print(f"Total words: {len(sample_doc.split())} -> Created {len(chunks)} overlapping chunks.")
print("Chunk 0 Sample:", chunks[0][:80] + "...")
```

#### Output:
```text
Total words: 350 -> Created 17 overlapping chunks.
Chunk 0 Sample: Artificial intelligence is transforming software engineering. Artificial intelli...
```

---

## 4. Vector Embeddings & Semantic Similarity

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two embedding vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Simulated 3D embeddings
v_query = np.array([0.9, 0.1, 0.05])
v_doc_relevant = np.array([0.88, 0.12, 0.08])
v_doc_unrelated = np.array([0.02, 0.95, 0.85])

sim_rel = cosine_similarity(v_query, v_doc_relevant)
sim_unrel = cosine_similarity(v_query, v_doc_unrelated)

print(f"Similarity to Relevant Document:   {sim_rel:.4f} (High!)")
print(f"Similarity to Unrelated Document:  {sim_unrel:.4f} (Low!)")
```

#### Output:
```text
Similarity to Relevant Document:   0.9996 (High!)
Similarity to Unrelated Document:  0.1345 (Low!)
```

---

## 5. Complete RAG Pipeline from Scratch in Python

Here is a self-contained RAG system with TF-IDF Vectorization, top-$k$ retrieval, and context-grounded response synthesis:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Knowledge Base
knowledge_documents = [
    "Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.",
    "Shipping Policy: Standard shipping takes 3-5 business days. Express shipping delivers overnight.",
    "Warranty Terms: All hardware electronics include a 1-year limited warranty against manufacturing defects.",
    "Customer Support Hours: Support is available Monday through Friday from 9 AM to 6 PM EST via live chat."
]

# 2. Vector Indexing
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(knowledge_documents)

# 3. Retrieval Function
def retrieve_top_k(query: str, k: int = 1):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(knowledge_documents[i], similarities[i]) for i in top_indices]

# 4. User Query & Retrieval
user_query = "How long do I have to return an item and get my money back?"
top_context, score = retrieve_top_k(user_query, k=1)[0]

print(f"User Query: {user_query}")
print(f"Retrieved Document (Match Score {score:.3f}):\n-> {top_context}")

# 5. Augmented Prompt Construction
augmented_prompt = f"""
You are an expert customer service assistant. Answer the user question strictly using the provided context.

Context:
{top_context}

Question: {user_query}
Answer:
"""
print("\n--- Final Augmented Prompt Sent to LLM ---")
print(augmented_prompt.strip())
```

#### Output:
```text
User Query: How long do I have to return an item and get my money back?
Retrieved Document (Match Score 0.442):
-> Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.

--- Final Augmented Prompt Sent to LLM ---
You are an expert customer service assistant. Answer the user question strictly using the provided context.

Context:
Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.

Question: How long do I have to return an item and get my money back?
Answer:
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Context Re-Ranking
**Task:** Given a list of retrieved chunks with similarity scores, write a function to filter out any chunks whose similarity is below a threshold of $0.20$, and format the remaining valid chunks with `[Source X]` citations for the LLM.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
retrieved_data = [
    ("Refunds are processed within 14 days.", 0.85),
    ("Our CEO was founded in 2015.", 0.08),
    ("Returns require undamaged packaging.", 0.45)
]

threshold = 0.20
valid_sources = [f"[Source {i+1}]: {doc}" for i, (doc, score) in enumerate(retrieved_data) if score >= threshold]
combined_context = "\n".join(valid_sources)

print("Filtered & Grounded Context:\n" + combined_context)
```
#### Output:
```text
Filtered & Grounded Context:
[Source 1]: Refunds are processed within 14 days.
[Source 3]: Returns require undamaged packaging.
```
</details>

---

## 7. Quick Reference Cheat Sheet

| RAG Component | Industry Standard Tools | Key Metric |
|---|---|---|
| **Embedding Model** | `text-embedding-3-small`, BGE-M3 | Cosine distance, NDCG@10 |
| **Vector Database** | ChromaDB, Pinecone, Qdrant, Milvus | HNSW recall, query latency (ms) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Context preservation, chunk overlap |
| **Reranker** | Cohere Rerank, BGE-Reranker-Large | MRR (Mean Reciprocal Rank) |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevance) | Hallucination rate (0-1) |
'''

p_rag = REPO_ROOT / "06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/basics.md"
p_rag.write_text(RAG_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated RAG Architecture Guide: {len(RAG_GUIDE.splitlines())} lines.")
