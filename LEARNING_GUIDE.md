# Master AI/ML & Generative AI Learning Guide
**Professional Certificate Program — E&ICT Academy, IIT Kanpur (Dec 2025 – Nov 2026)**  
*Compiled & Authored by Sameer Karur*

---

## 📌 Table of Contents
1. [Program Architecture & LMS Course Mapping](#1-program-architecture--lms-course-mapping)
2. [Theoretical Foundations & Core Mathematics](#2-theoretical-foundations--core-mathematics)
3. [Course-by-Course Deep Dives & Industry Applications](#3-course-by-course-deep-dives--industry-applications)
   - [Course 00: Induction Session](#course-00-induction-session)
   - [Course 01: Foundations — Programming Refresher](#course-01-foundations--programming-refresher)
   - [Course 02: Core — Applied Data Science with Python](#course-02-core--applied-data-science-with-python)
   - [Course 03: Core — Machine Learning](#course-03-core--machine-learning)
   - [Course 04: Core — Deep Learning with Keras & TensorFlow](#course-04-core--deep-learning-with-keras--tensorflow)
   - [Course 05: Core — Essentials of Generative AI, Prompt Engineering & ChatGPT](#course-05-core--essentials-of-generative-ai-prompt-engineering--chatgpt)
   - [Course 06: Advanced Generative AI](#course-06-advanced-generative-ai)
   - [Course 07: Capstone (Autonomous Driving, Sales Forecasting, Heritage AI)](#course-07-capstone)
   - [Electives 01–05: Specializations & Masterclasses](#electives-0105-specializations--masterclasses)
4. [Master Formula & Mathematical Cheatsheet](#4-master-formula--mathematical-cheatsheet)
5. [Top 100 AI/ML Technical Interview Questions & Model Answers](#5-top-100-aiml-technical-interview-questions--model-answers)
6. [Project Portfolio Dual-Architecture: Version 1 vs Version 2](#6-project-portfolio-dual-architecture-version-1-vs-version-2)

---

# 1. Program Architecture & LMS Course Mapping

The curriculum is structured across 8 mandatory modules (Induction + 7 core courses) followed by 5 specialized electives:

```
Data_science/
├── 00_Induction_Session/
├── 01_IITK_AIML_Foundations_Programming_Refresher/
├── 02_IITK_AIML_Core_Applied_Data_Science_with_Python/
├── 03_IITK_AIML_Core_Machine_Learning/
├── 04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/
├── 05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/
├── 06_IITK_AIML_Advanced_Generative_AI/
├── 07_IITK_AIML_Capstone/
│   ├── project1_autonomous_driving/
│   ├── project2_sales_forecasting/
│   └── project3_preserving_heritage/
├── Elective_01_ADL_and_Computer_Vision/
├── Elective_02_NLP_and_Speech_Recognition/
├── Elective_03_Reinforcement_Learning/
├── Elective_04_Microsoft_Azure_AI_Fundamentals/
├── Elective_05_Academic_Masterclass_by_IIT_Kanpur/
├── projects_version2/                   ← Next-Gen Alternative Architectures
└── datasets/shared/                     ← Common Data Store
```

---

# 2. Theoretical Foundations & Core Mathematics

### 2.1 Linear Algebra for AI/ML
* **Vector Spaces & Inner Products:** Dot product measures projection and alignment:
  $$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta) = \sum_{i=1}^n u_i v_i$$
  In embeddings and RAG, Cosine Similarity normalizes magnitude to focus purely on directional semantics:
  $$\text{CosineSimilarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
* **Matrix Decompositions:**
  - **Singular Value Decomposition (SVD):** Decomposes any matrix $A \in \mathbb{R}^{m \times n}$ into:
    $$A = U \Sigma V^T$$
    where $U$ contains left singular vectors, $\Sigma$ contains singular values (energy), and $V^T$ contains right singular vectors. Used directly in recommendation systems (collaborative filtering) and dimensionality reduction.
  - **Principal Component Analysis (PCA):** Eigendecomposition of covariance matrix $C = \frac{1}{n} X^T X = Q \Lambda Q^T$. Projects data onto orthogonal axes of maximum variance, eliminating multicollinearity and reducing noise.

### 2.2 Probability & Statistical Inference
* **Bayes Theorem:**
  $$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$
  Foundational to Naive Bayes classifiers, Bayesian optimization for hyperparameter tuning, and probabilistic generative models.
* **Central Limit Theorem (CLT):** The distribution of sample means approximates a normal distribution as sample size $n \to \infty$, regardless of the underlying population distribution.
* **Hypothesis Testing:**
  - Null Hypothesis ($H_0$) vs Alternative ($H_1$).
  - Two-sample $t$-test (comparing means of two continuous groups).
  - Chi-Square Test ($\chi^2$) for independence of categorical variables.
  - ANOVA (Analysis of Variance) for multiple group comparisons.
  - Significance level $\alpha$ (typically 0.05); reject $H_0$ if $p$-value $< \alpha$.

### 2.3 Optimization & Calculus
* **Gradient Descent:**
  $$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$
* **Stochastic & Mini-batch Gradient Descent:** Approximates full gradient with a batch $\mathcal{B} \subset \mathcal{D}$ of size $32, 64, 128$.
* **Adam Optimizer:** Combines Momentum (first moment $m_t$) and RMSProp (second raw moment $v_t$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

# 3. Course-by-Course Deep Dives & Industry Applications

## Course 00: Induction Session
* **Core Topics:** Program orientation, Python ecosystem, cloud environments, version control (Git/GitHub), high-performance computing expectations.
* **Key Learning:** Setting up isolated virtual environments (`venv`, `conda`), understanding dependencies, adhering to reproducibility standards.

## Course 01: Foundations — Programming Refresher
* **Python Memory Model:** CPython reference counting, garbage collection cycles, object mutability vs immutability (tuples/strings vs lists/dicts).
* **Advanced Constructs:** List/dict/set comprehensions, generators (`yield` for lazy memory streaming), `itertools`, `functools` (`lru_cache`, `partial`), decorators (`@wraps`), context managers (`with` / `__enter__`, `__exit__`).
* **Object-Oriented Programming (OOP):** Encapsulation, inheritance, polymorphism, abstract base classes (`abc.ABC`), dataclasses, and solid design patterns (Factory, Strategy, Observer).
* **Exception Handling & File I/O:** Robust file streams, atomic writes, JSON/CSV parsing, custom exception hierarchies.

## Course 02: Core — Applied Data Science with Python
* **NumPy Vectorization:** Broadcasting semantics, stride tricks, SIMD CPU acceleration, memory views vs copies.
* **Pandas Analytics:** MultiIndex pivoting, split-apply-combine (`groupby`), window functions (`rolling`, `expanding`), categoricals for memory compression, datetime indexing.
* **Data Visualization & Storytelling:** Matplotlib figure-axes architecture, Seaborn statistical distributions (kdeplots, heatmaps, pairplots), visual encoding best practices.
* **Feature Engineering:** Handling missing values (MICE, iterative imputer, domain heuristics), outlier detection (Tukey's IQR, Z-score, Isolation Forest), feature scaling (MinMaxScaler, StandardScaler, RobustScaler).

## Course 03: Core — Machine Learning
* **Supervised Regression & Classification:**
  - Cost functions: Mean Squared Error (MSE), Cross-Entropy Loss (Log Loss).
  - Regularization: L1 (Lasso) promotes sparsity; L2 (Ridge) shrinks weights; ElasticNet balances both.
* **Tree-Based Ensembles:**
  - Random Forest: Bootstrap aggregating (bagging), feature sub-sampling, out-of-bag (OOB) error estimation.
  - Gradient Boosted Decision Trees (GBDT / XGBoost / LightGBM): Sequentially fitting pseudo-residuals of loss function, shrinkage learning rate, regularization on leaf counts.
* **Unsupervised Clustering:**
  - K-Means: Lloyd's algorithm, Voronoi cells, inertia minimization, Elbow method, Silhouette Coefficient $s = \frac{b - a}{\max(a, b)}$.
  - Hierarchical Clustering: Agglomerative bottom-up tree generation with Ward, Complete, or Average linkage.
* **Handling Extreme Imbalance:**
  - Synthetic Minority Over-sampling Technique (SMOTE): Interpolating $k$-nearest neighbors in feature space.
  - Focal Loss and Cost-sensitive learning: Weighting minority class misclassifications higher.
* **Model Evaluation Metrics:** Precision, Recall, $F_\beta$ score, ROC curve, Area Under ROC (ROC-AUC), Precision-Recall AUC (PR-AUC for skewed data).

## Course 04: Core — Deep Learning with Keras & TensorFlow
* **Neural Architecture Foundations:** Forward propagation, computational graphs, automatic differentiation, backpropagation chain rule.
* **Activation Functions:** Non-linear mappings: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, Tanh $\tanh(z)$, ReLU $\max(0, z)$, Leaky ReLU, GELU (Gaussian Error Linear Unit).
* **Preventing Overfitting in Deep Networks:** Dropout (Bernoulli masking during training), Batch Normalization (stabilizing internal covariate shift), Weight decay, Early Stopping with model checkpointing.
* **Computer Vision (CNNs):**
  - Convolutional layers: Feature map extraction, spatial locality, parameter sharing.
  - Max Pooling / Average Pooling: Translational invariance and dimension reduction.
  - Transfer Learning: Freezing base feature extractors (e.g. MobileNetV2, ResNet50), training custom classification heads, progressive fine-tuning of top layers with low learning rates.

## Course 05: Core — Essentials of Generative AI, Prompt Engineering & ChatGPT
* **Transformer Architecture:**
  - Self-Attention Mechanism:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - Multi-Head Attention: Allows model to jointly attend to information from different representation subspaces.
* **Prompt Engineering Frameworks:**
  - Zero-shot vs Few-shot (in-context demonstrations).
  - Chain-of-Thought (CoT): "Let's think step by step" to stimulate multi-hop reasoning.
  - Role Prompting & System Framing: Directing persona, constraints, and tone.
  - Output Formatting: JSON schema enforcement, Markdown tables, structured extraction.
* **LLM Agents & Tool Use:** ReAct (Reason + Act) loop: Thought $\to$ Action $\to$ Observation $\to$ Final Answer.

## Course 06: Advanced Generative AI
* **Practice Question Banks & Labs:**
  - `01_rag_architectures/`: 50 practice problems + solutions + 30 interview Q&A on chunking, embeddings, context injection, and RAG evaluation.
  - `02_vector_databases_chroma/`: 50 practice problems + solutions + 30 interview Q&A on HNSW, ANN search, quantization, and ChromaDB persistence.
  - `03_multimodal_generative_models/`: 50 practice problems + solutions + 30 interview Q&A on CLIP, diffusion denoising, DALL-E, and image composition.
* **Retrieval-Augmented Generation (RAG):**
  - Document Ingestion: PyPDFLoader, chunking strategies with overlap (`RecursiveCharacterTextSplitter`).
  - Dense Vector Databases: ChromaDB, FAISS, cosine distance indexing.
  - Retrieval & Augmentation: Context injection into grounded system prompts, zero-hallucination guardrails.
  - Interactive UI: Gradio chat interface with streaming responses.
* **Creative Multi-Modal AI:**
  - OpenAI DALL-E / Images API integration.
  - System prompt chaining: Generating marketing pitch copy $\to$ synthesizing image generation prompts $\to$ automated image fetching & rendering.

## Course 07: Capstone (Autonomous Driving, Sales Forecasting, Heritage AI)
* **Capstone 1: Autonomous Driving:**
  - Image cropping & annotation parsing from Pascal VOC XMLs.
  - Transfer learning MobileNetV2 vehicle crop classifier (89.9% accuracy).
  - Safety analytics on Tesla fatal crash records (victim demographics, collision types, Autopilot roles).
* **Capstone 2: Sales Forecasting:**
  - High-frequency daily retail transaction analytics across restaurants and menu items.
  - Time-series feature engineering (calendar lags, rolling averages, seasonality indicators).
  - XGBoost champion model achieving $R^2 = 0.948$ and RMSE 57.90 on forward holdout periods.
* **Capstone 3: Preserving Cultural Heritage:**
  - Deep Learning landmark image classifier across 10 global monuments.
  - Collaborative filtering tourism recommendation engine using SVD matrix factorization.

## Electives 01–05: Specializations & Masterclasses
* **Elective 01 — ADL & Computer Vision:** Modern detection backbones (YOLOv8, Faster R-CNN), semantic segmentation (U-Net), Vision Transformers (ViT).
* **Elective 02 — NLP & Speech Recognition:** Subword tokenization (BPE, WordPiece), sequence-to-sequence models, Whisper speech-to-text, CTC loss.
* **Elective 03 — Reinforcement Learning:** Markov Decision Processes (MDPs), Bellman optimality equation, Q-Learning, Deep Q-Networks (DQN), Policy Gradient methods (PPO).
* **Elective 04 — Microsoft Azure AI Fundamentals:** Cognitive Services (Vision, Speech, Language, Decision), Azure Machine Learning Studio workspace, MLOps lifecycle.
* **Elective 05 — Academic Masterclass by IIT Kanpur:** Cutting-edge research frontiers, scalable AI ethics, explainability (XAI), and domain-specific foundation models.

---

# 4. Master Formula & Mathematical Cheatsheet

| Concept | Mathematical Formulation | Primary Application |
|---|---|---|
| **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | Vector Embeddings, RAG retrieval |
| **Cross-Entropy Loss** | $-\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})$ | Multi-class classification, Deep Learning |
| **Mean Squared Error** | $\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$ | Regression, Forecasting |
| **Ridge Regression (L2)** | $\mathcal{L}_{L2} = \text{MSE} + \lambda \sum_{j=1}^p \theta_j^2$ | Weight shrinkage, multicollinearity |
| **Lasso Regression (L1)** | $\mathcal{L}_{L1} = \text{MSE} + \lambda \sum_{j=1}^p \|\theta_j\|$ | Feature selection, sparse coefficients |
| **Self-Attention** | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | Transformers, LLMs, Vision Transformers |
| **CNN Output Dimension** | $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$ | Convolutional layer spatial sizing |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Imbalanced classification |
| **Silhouette Coefficient** | $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$ | Cluster separation evaluation |
| **SVD Matrix Factorization** | $R \approx U \cdot \Sigma \cdot V^T$ | Collaborative filtering, dimensionality reduction |

---

# 5. Top 100 AI/ML Technical Interview Questions & Model Answers

*(Categorized by domain for high-yield technical preparation)*

### Part A: Python & Software Engineering (Q1–Q15)
1. **Q: How does CPython manage memory, and what causes memory leaks?**  
   *A:* CPython primarily uses reference counting supplemented by a generational cyclic garbage collector for self-referential cycles. Memory leaks occur when global lists/dictionaries accumulate references, circular references with overridden `__del__` methods prevent cleanup, or C-extensions allocate heap memory without freeing.
2. **Q: Explain the difference between `deepcopy` and shallow `copy`.**  
   *A:* A shallow copy (`copy.copy`) constructs a new compound object and inserts references to the original child objects. A deep copy (`copy.deepcopy`) recursively clones all nested objects, ensuring mutations to the clone do not affect the original.
3. **Q: How do Python generators optimize memory when streaming data?**  
   *A:* Generators use the `yield` statement to suspend execution state and evaluate lazily on-demand via the iterator protocol (`__next__`). They require $O(1)$ memory compared to $O(N)$ for full list instantiation.
4. **Q: What is the Python Global Interpreter Lock (GIL)?**  
   *A:* The GIL is a mutex that allows only one native thread to execute Python bytecode at a time, preventing race conditions in CPython's reference counting. For CPU-bound parallel workloads, use `multiprocessing` or native C/NumPy libraries that release the GIL.
5. **Q: What are `*args` and `**kwargs`?**  
   *A:* `*args` unpacks positional arguments into a tuple; `**kwargs` unpacks keyword arguments into a dictionary.

### Part B: Statistics, Probability & Data Science (Q16–Q35)
16. **Q: What is the difference between Type I and Type II errors?**  
    *A:* Type I error ($\alpha$) is a false positive (rejecting a true null hypothesis). Type II error ($\beta$) is a false negative (failing to reject a false null hypothesis). Statistical power is $1 - \beta$.
17. **Q: When would you use Pearson vs Spearman correlation?**  
    *A:* Pearson evaluates linear relationships between continuous, normally distributed variables. Spearman evaluates monotonic relationships on ranked values and is robust to outliers and non-linear monotonic trends.
18. **Q: Explain the Bias-Variance Tradeoff.**  
    *A:* Total Error = $\text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$. High bias indicates underfitting (oversimplified model missing true patterns); high variance indicates overfitting (model memorizing training noise). Regularization and ensembles balance the two.
19. **Q: What is Multicollinearity and how do you detect and fix it?**  
    *A:* It occurs when predictor variables are highly correlated, inflating coefficient variance. Detected via Variance Inflation Factor ($\text{VIF} > 5 \text{ or } 10$) or correlation heatmaps. Fixed via PCA, L1 Lasso regularization, or dropping redundant collinear features.
20. **Q: Why is ROC-AUC misleading on heavily imbalanced datasets?**  
    *A:* The True Negative Rate in False Positive Rate ($\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$) can be overwhelmingly large, making FPR look tiny even when hundreds of false positives occur. Use the Precision-Recall Curve (PR-AUC) instead.

### Part C: Classical Machine Learning (Q36–Q55)
36. **Q: How does Random Forest differ from Gradient Boosting?**  
    *A:* Random Forest uses Bagging: trains independent deep trees in parallel on bootstrap samples and averages predictions (reduces variance). Gradient Boosting uses Boosting: trains shallow trees sequentially to predict negative gradients/residuals of previous trees (reduces bias).
37. **Q: How does K-Means choose initial centroids in K-Means++?**  
    *A:* K-Means++ picks the first centroid uniformly at random, then samples subsequent centroids with probability proportional to the squared Euclidean distance $D(x)^2$ from the closest existing centroid, accelerating convergence and avoiding poor local minima.
38. **Q: Explain how SMOTE works.**  
    *A:* For each minority sample, SMOTE finds its $k$-nearest minority neighbors, randomly selects one neighbor, and generates synthetic points along the connecting line segment: $x_{new} = x + \lambda (x_{neighbor} - x)$ where $\lambda \in [0, 1]$.
39. **Q: What is the objective function of XGBoost?**  
    *A:* $\text{Obj} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^t \Omega(f_k)$, where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$. It uses second-order Taylor expansion (gradients $g_i$ and Hessians $h_i$) for rapid convergence.
40. **Q: What is the purpose of cross-validation (e.g. Stratified K-Fold)?**  
    *A:* It prevents data leakage and over-optimistic evaluation by partitioning data into $K$ folds while preserving class balance, evaluating generalized out-of-fold performance.

### Part D: Deep Learning & Computer Vision (Q56–Q75)
56. **Q: What causes the Vanishing Gradient problem and how is it mitigated?**  
    *A:* Deep networks using saturating activations (Sigmoid/Tanh) produce derivatives $< 0.25$, multiplying through backpropagation layers until earlier layer updates vanish. Mitigated by ReLU/GELU activations, Residual connections (ResNet skips), He/Glorot weight initialization, and Batch Normalization.
57. **Q: Why does Batch Normalization accelerate training?**  
    *A:* It normalizes layer inputs across the mini-batch to zero mean and unit variance, smoothing the optimization landscape, allowing higher learning rates, and providing mild regularization.
58. **Q: What is the difference between feature extraction and fine-tuning in Transfer Learning?**  
    *A:* In feature extraction, all pre-trained convolutional weights are frozen and only the new classification head is trained. In fine-tuning, top convolutional layers are unfrozen and trained alongside the head with a very small learning rate ($\approx 10^{-5}$).
59. **Q: How does Convolution work in 2D images?**  
    *A:* A kernel/filter matrix slides across the input tensor, performing element-wise multiplications and summing them to produce a spatial feature map. Stride controls step size; padding controls boundary retention.
60. **Q: What is IoU (Intersection over Union) in Object Detection?**  
    *A:* $\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$ between the predicted bounding box and the ground truth box. An IoU $\ge 0.5$ is typically considered a true positive.

### Part E: Generative AI, LLMs & Advanced Topics (Q76–Q100)
76. **Q: Why divide by $\sqrt{d_k}$ in Scaled Dot-Product Attention?**  
    *A:* For large projection dimensions $d_k$, dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ preserves unit variance and gradient stability.
77. **Q: Explain the difference between Dense Retrieval and Sparse Retrieval in RAG.**  
    *A:* Sparse retrieval (e.g. BM25) matches exact keywords and lexical tokens using inverted indices. Dense retrieval encodes queries and passages into continuous semantic vector embeddings via neural encoders (e.g. text-embedding-ada-002), matching conceptual meaning even with zero keyword overlap. Hybrid RAG combines both via Reciprocal Rank Fusion (RRF).
78. **Q: What is Reciprocal Rank Fusion (RRF)?**  
    *A:* $\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$, where $r_m(d)$ is the rank of document $d$ in system $m$, and $k$ is a smoothing constant (typically 60). It normalizes disparate ranking scales across dense and sparse retrievers without score calibration.
79. **Q: How do you prevent hallucinations in enterprise RAG systems?**  
    *A:* (1) Strict prompt grounding ("Answer ONLY using the provided context; if not present, state 'I do not have this information'"). (2) Source attribution / citation verification. (3) Embedding distance thresholding. (4) Re-ranking with cross-encoders. (5) Self-critique / hallucination evaluation models (e.g. Ragas / TruLens).
80. **Q: What is the difference between Fine-Tuning and RAG?**  
    *A:* Fine-Tuning updates the internal weights of the model to learn new styles, formats, or domain grammar. RAG provides external, up-to-date, retrievable dynamic factual context without altering model parameters.
81. **Q: What is ReAct prompt engineering?**  
    *A:* An agent framework that interleaves reasoning ("Thought: I need to check the inventory table") with action execution ("Action: query_db('SELECT ...')") and observation consumption before finalizing answers.
82. **Q: What is the temperature parameter in LLM decoding?**  
    *A:* It scales logit probabilities before softmax: $P(y_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$. Low temperature ($T \to 0$) yields deterministic, greedy responses; high temperature ($T \to 1.0$) increases creative entropy.
83. **Q: Explain Quantization in LLMs (e.g. 4-bit / 8-bit).**  
    *A:* Reduces weight precision from FP32/FP16 to INT8 or INT4 (e.g. AWQ, GPTQ, GGUF), shrinking VRAM footprints by 50–75% while retaining over 98% of model capabilities.
84. **Q: What is SVD in recommendation systems?**  
    *A:* Decomposes the user-item interaction matrix into latent user preference factors and latent item attribute factors, predicting missing ratings via inner products.

---

# 6. Project Portfolio Dual-Architecture: Version 1 vs Version 2

Every project completed in this program has been implemented in two distinct architectural paradigms:
* **Version 1 (Core LMS Baseline):** Implemented in the respective course directory, focusing on fundamental problem statements and LMS submission packages.
* **Version 2 (`projects_version2/`):** Enterprise-grade, production-oriented alternative technical implementations featuring advanced design patterns, alternative algorithmic families, explainability layers, and modern architectural paradigms.

| Project # | Domain & Curriculum | Version 1 Baseline Architecture | Version 2 Advanced Alternative Architecture |
|---|---|---|---|
| **01** | Python: Expense Tracker | CLI script with CSV file persistence | Object-Oriented Event-Sourced SQLite engine with budget alerting & analytical reporting |
| **02** | Python: Task Manager | Basic JSON store with SHA256 hashing | Priority-Queue DAG task dependency scheduler with Role-Based Access Control (RBAC) |
| **03** | Data Science: Sales Analysis | Aggregated Pandas groupby & Matplotlib charts | Multi-dimensional Cohort Analysis, RFM Customer Segmentation & Price Elasticity Modeling |
| **04** | Data Science: Marketing Campaigns | Exploratory data analysis & response plots | Uplift Modeling, Propensity Score Matching (PSM) & Multi-Touch Attribution Simulator |
| **05** | ML: Song Cohorts (Spotify) | Standard K-Means clustering (4 audio features) | Density-Based HDBSCAN + Agglomerative Clustering with PCA/t-SNE & Silhouette Optimization |
| **06** | ML: Employee Turnover | Random Forest baseline with basic SMOTE | Explainable Gradient Boosting (XGBoost) with Bayesian Optimization & SHAP Attribution |
| **07** | DL: Lending Club Loan Analysis | Standard Keras Dense MLP classifier | Deep Tabular ResNet with categorical entity embeddings & Focal Loss for severe default risk |
| **08** | DL: Home Loan Risk Analysis | Multilayer Perceptron on tabular sample | TabNet-style Sparse Attentive Architecture with Feature Masking & Calibrated Probabilities |
| **09** | GenAI: Interactive Storytelling | Single prompt iterative story loop | Multi-Agent Narrative State Graph with Dynamic Inventory, Memory Store & Branching Tree |
| **10** | GenAI: Virtual PM Consultant | Single persona system prompt | Multi-Role Autonomous PMO Swarm (Agile Coach, Risk Officer, Architect) with Jira export |
| **11** | Adv GenAI: Nestlé HR Assistant | Standard LangChain + ChromaDB RetrievalQA | Hybrid Dense-Sparse RAG (BM25 + Dense RRF) with Cross-Encoder Re-ranking & Guardrails |
| **12** | Adv GenAI: Netflix Creative Studio | Direct DALL-E image prompt generation | Multi-Modal Compositional Pipeline with Style Transfer, Multi-Aspect Generation & PIL layout |
| **13** | Capstone 1: Autonomous Perception | MobileNetV2 vehicle crop classifier + EDA | EfficientNet-B0 Backbone with Grad-CAM Visual Interpretability & Spatio-Temporal Risk Model |
| **14** | Capstone 2: Sales Demand Forecast | Tabular XGBoost with calendar lag features | Hierarchical Probabilistic Ensemble (CatBoost + Ridge Stacking) with Rolling Forecast Intervals |
| **15** | Capstone 3: Heritage Tourism AI | MobileNetV2 landmark CNN + SVD Matrix Fact. | Deep ResNet50 Landmark Vision + Bayesian Personalized Ranking (BPR) Graph Recommender |

---
*For hands-on execution and replication of all Version 2 projects, navigate to `projects_version2/`.*
