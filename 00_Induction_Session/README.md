# Professional Certificate Course in AI & Machine Learning
**E&ICT Academy, IIT Kanpur**  
*Program Induction, Foundational Architecture & Engineering Roadmap*

---

## 📌 Executive Overview

Welcome to the **Professional Certificate Program in Generative AI and Machine Learning**, offered by the **Electronics and ICT (E&ICT) Academy at the Indian Institute of Technology Kanpur (IIT Kanpur)**. 

This repository houses the complete, self-contained educational repository, containing:
- **7 Core Courses** covering Python Foundations, Applied Data Science, Classical Machine Learning, Deep Learning, Generative AI, and Advanced RAG Systems.
- **3 Comprehensive Industrial Capstones** (Autonomous Driving Object Localization, Multi-Store Retail Demand Forecasting, and Cultural Heritage Preservation AI).
- **15 Version 2 Next-Generation Projects** re-engineered with event-sourcing, Bayesian optimization, TabNet, and multi-agent swarms.
- **1,650+ Interactive Practice Problems** executable in Google Colab with zero local setup.
- **990+ Curated Technical Interview Flashcards** with verified model derivations.

---

## 🧭 Program Architecture & Curriculum Roadmap

```
Data_science/
├── 00_Induction_Session/                                                     ← Program Orientation & Prerequisites Roadmap
├── 01_IITK_AIML_Foundations_Programming_Refresher/                           ← Python Memory Model, Data Structures & OOP
├── 02_IITK_AIML_Core_Applied_Data_Science_with_Python/                       ← NumPy, Pandas, Linear Algebra & Inferential Stats
├── 03_IITK_AIML_Core_Machine_Learning/                                       ← Supervised, Unsupervised & Ensemble Algorithms
├── 04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/                 ← Neural Networks, CNNs, Optimization & Grad-CAM
├── 05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/ ← Prompt Systems, Function Calling & Fine-Tuning
├── 06_IITK_AIML_Advanced_Generative_AI/                                      ← RAG Architectures, ChromaDB & Diffusion Vision AI
├── 07_IITK_AIML_Capstone/                                                    ← Production Systems (Vision, Forecasting, NLP)
├── projects_version2/                                                        ← 15 Alternative Modern Architecture Implementations
└── Electives (01–05)/                                                        ← Vision, NLP, RL, Azure AI & IITK Masterclasses
```

---

## 📐 Foundational Mathematical & Algorithmic Prerequisites

To build intuition for modern artificial intelligence systems, the curriculum builds upon four foundational pillars:

### 1. Linear Algebra & Vector Spaces
- **Vector Operations:** Dot products, cosine similarity, projections, and $L_1$/$L_2$ norms.
- **Matrix Decompositions:** Eigenvalues, eigenvectors, and Singular Value Decomposition (SVD) underpinning PCA and latent semantic analysis.
- **Dimensionality:** Vector embeddings, projection matrices, and high-dimensional manifolds.

### 2. Multivariable Calculus & Optimization
- **Gradient Systems:** Partial derivatives, the gradient vector $\nabla f$, and the Jacobian/Hessian matrices.
- **Backpropagation:** The multivariate chain rule for propagating error signals through deep neural layers.
- **Optimizers:** First-order methods (SGD, Momentum, RMSprop, Adam, AdamW) and learning rate scheduling dynamics.

### 3. Probability & Inferential Statistics
- **Probability Spaces:** Bayes' Theorem, joint and conditional probability distributions.
- **Distributions:** Gaussian, Bernoulli, Binomial, Poisson, and heavy-tailed Power Law distributions.
- **Inferential Testing:** Central Limit Theorem (CLT), confidence intervals, hypothesis testing ($p$-values, Type I/II errors, and ANOVA).

### 4. Algorithmic Complexity & Systems
- **Asymptotic Analysis:** Big-O time and space complexity for search, sorting, and tensor transformations.
- **Vectorization vs. Iteration:** SIMD hardware acceleration, broadcasting semantics, and cache-locality in memory buffers.

---

## 🛠️ Computing Environment & Development Toolchain

The learning materials are designed for dual-mode execution: **Cloud Zero-Config** (via Google Colab) and **Local Production Workstations**.

### 1. Cloud-First GPU Execution (Recommended)
Every practice notebook and project in this repository includes an embedded **Open in Colab** badge and a cloud setup cell that clones required datasets dynamically:
- Free access to NVIDIA T4 / A100 Tensor Core GPUs.
- Zero local dependencies or installation conflicts.

### 2. Local Environment Setup
For offline development and production experimentation:

```bash
# Clone the repository
git clone https://github.com/sameerkarur/Data_science.git
cd Data_science

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install foundational scientific dependencies
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

---

## 🎯 Pedagogical Framework & Deliverables Standards

### The STAR Project Methodology
Every engineering deliverable in this curriculum is documented according to the **STAR** framework:
- **Situation:** Industry domain, business dilemma, data provenance, and operational constraints.
- **Task:** Mathematical objective, optimization target, and quantifiable success metrics.
- **Action:** Data wrangling, feature engineering, hypothesis testing, model selection, and hyperparameter tuning.
- **Result:** Benchmark metrics (RMSE, $R^2$, ROC-AUC, F1-Score), business ROI, latency profiles, and failure modes.

### Dual-Architecture Philosophy (Version 1 vs. Version 2)
To foster engineering depth beyond typical academic coursework:
- **Version 1 (Foundational Baseline):** Clean, standard pedagogical implementations demonstrating textbook data science workflows.
- **Version 2 (`projects_version2/`):** Enterprise-grade alternative paradigms featuring event-sourced SQLite engines, DAG schedulers, Bayesian hyperparameter optimization, TabNet sparse attention, and multi-agent narrative swarms.

---

## ⭐ How to Use This Academy & Support the Project

1. **Interactive Learning Academy:** Launch `index.html` locally or through GitHub Pages to access the interactive single-page studio.
2. **Technical Interview Flashcards:** Use the **Interview Hub** tab to review 990+ high-yield questions with collapsible derivations and local mastery tracking.
3. **Star the Repository:** If this open-source curriculum and codebase help your AI/ML journey, please **star the repository on GitHub** (`sameerkarur/Data_science`) to support further open educational materials!
