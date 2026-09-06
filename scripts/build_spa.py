"""
Builds the production Single Page Application index.html with:
1. Interactive 33-Chapter Curriculum Textbook Table of Contents
2. Marked Markdown Parser with automated GitHub slugs and compact anchor IDs
3. Global link interceptor to prevent raw file displays (intercepts anchors, .md, .ipynb, and GitHub URLs)
4. Smooth in-modal anchor scrolling and chapter navigation (Previous/Next chapter)
5. 33-Topic Technical Interview Flashcards Hub
6. Single-instance Learning Journey navbar tab with academic attribution
7. Zero stubs, 100% production ready
"""

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# 33-CHAPTER MASTER CURRICULUM TEXTBOOK REGISTRY
# =====================================================================
CHAPTERS = [
    # Part I: Python Foundations & Engineering Architecture (Course 1)
    {
        "num": 1,
        "part": "Part I · Python Foundations & Architecture",
        "part_code": "python",
        "title": "Variables, Types & CPython Memory Architecture",
        "path": "01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes/basics.md",
        "colab": "01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes/01_variables_datatypes_practice.ipynb",
        "topicId": "c01_m01",
        "desc": "CPython memory internals, PyObject pointers, reference counting, small integer caching, type hints, bitwise ops, and memory profiling.",
    },
    {
        "num": 2,
        "part": "Part I · Python Foundations & Architecture",
        "part_code": "python",
        "title": "Control Flow, Pattern Matching & Functional Paradigms",
        "path": "01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions/basics.md",
        "colab": "01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions/02_control_flow_functions_practice.ipynb",
        "topicId": "c01_m02",
        "desc": "Structural pattern matching (match/case), for...else, generators & itertools, LEGB scope, closures, decorators, and retry engines.",
    },
    {
        "num": 3,
        "part": "Part I · Python Foundations & Architecture",
        "part_code": "python",
        "title": "Advanced Data Structures & Algorithmic Complexity",
        "path": "01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures/basics.md",
        "colab": "01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures/03_data_structures_practice.ipynb",
        "topicId": "c01_m03",
        "desc": "List resizing mathematics, compact dict hash tables, deque, Counter, defaultdict, algorithmic Big-O proofs, and high-throughput LRU cache.",
    },
    {
        "num": 4,
        "part": "Part I · Python Foundations & Architecture",
        "part_code": "python",
        "title": "Object-Oriented Architecture, Metaclasses & Design Patterns",
        "path": "01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules/basics.md",
        "colab": "01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules/04_oop_modules_practice.ipynb",
        "topicId": "c01_m04",
        "desc": "OOP pillars, __new__ vs __init__, MRO & C3 linearization, descriptor protocol, __slots__ memory optimization, ABCs, and Scikit-Learn BaseEstimator.",
    },
    {
        "num": 5,
        "part": "Part I · Python Foundations & Architecture",
        "part_code": "python",
        "title": "Robust File I/O, Serialization & Exception Resilience",
        "path": "01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions/basics.md",
        "colab": "01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions/05_file_io_exceptions_practice.ipynb",
        "topicId": "c01_m05",
        "desc": "OS buffering, atomic file replacement, JSON/Pickle security, custom exception hierarchies, explicit chaining, and write-ahead logging (WAL).",
    },

    # Part II: Applied Mathematics & Data Science with Python (Course 2)
    {
        "num": 6,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "The Data Science Paradigm & CRISP-DM Framework",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science/01_intro_data_science_practice.ipynb",
        "topicId": "c02_m01",
        "desc": "CRISP-DM lifecycle, hypothesis formulation, problem framing, data governance, ethics, and enterprise production tradeoffs.",
    },
    {
        "num": 7,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Python Data Science Ecosystem & Virtual Environments",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials/02_python_essentials_practice.ipynb",
        "topicId": "c02_m02",
        "desc": "Virtualenv, conda, package dependency resolution, deterministic builds, Jupyter kernels, memory profiling, and modular scripting.",
    },
    {
        "num": 8,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Vectorized Numerical Computing with NumPy",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy/numpy_practice.ipynb",
        "topicId": "c02_m03",
        "desc": "730-line Master Guide: ndarray strides, contiguity, SIMD vectorization, broadcasting rules, advanced fancy indexing, and fast Euclidean matrices.",
    },
    {
        "num": 9,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Linear Algebra, Spectral Theory & SVD for Machine Learning",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra/04_linear_algebra_practice.ipynb",
        "topicId": "c02_m04",
        "desc": "Vector norms (L1, L2, Linf), cosine similarity, rank, eigenvalues/eigenvectors, PCA via eigendecomposition, SVD derivation, and latent semantic search.",
    },
    {
        "num": 10,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Descriptive & Inferential Statistics from First Principles",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals/05_statistics_fundamentals_practice.ipynb",
        "topicId": "c02_m05",
        "desc": "Central tendency, dispersion, Bessel's correction proof, skewness/kurtosis, Central Limit Theorem mechanics, and Welford's running variance.",
    },
    {
        "num": 11,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Probability Theory, Distributions & Bayesian Inference",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions/06_probability_distributions_practice.ipynb",
        "topicId": "c02_m06",
        "desc": "Probability axioms, Bayes' theorem, discrete distributions (Binomial, Poisson), continuous (Normal, Exponential), MLE derivation, and capacity planning.",
    },
    {
        "num": 12,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Statistical Hypothesis Testing, Power & Enterprise A/B Testing",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics/07_advanced_statistics_practice.ipynb",
        "topicId": "c02_m07",
        "desc": "H0 vs H1, Type I/II errors, p-value misconceptions, Welch's t-test, ANOVA, Chi-Square, Bonferroni corrections, MDE sample sizing, and A/B test decision engine.",
    },
    {
        "num": 13,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "High-Performance Data Manipulation with Pandas",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas/pandas_practice.ipynb",
        "topicId": "c02_m08",
        "desc": "575-line Master Guide: BlockManager, Series/DataFrame internals, MultiIndex slicing, split-apply-combine, window functions, and categorical memory compression.",
    },
    {
        "num": 14,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Data Wrangling, Cleaning & Preprocessing Pipelines",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling/09_data_wrangling_practice.ipynb",
        "topicId": "c02_m09",
        "desc": "MCAR/MAR/MNAR mechanisms, KNN & MICE imputation, Tukey's IQR & Isolation Forests, RobustScaler, Target encoding with smoothing, and ColumnTransformer.",
    },
    {
        "num": 15,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "Data Visualization Architecture (Matplotlib & Seaborn)",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib/matplotlib_practice.ipynb",
        "topicId": "c02_m010",
        "desc": "Figure-Axes object hierarchy, transforms, statistical plots (KDE, violin, heatmaps), multi-facet grids, visual encoding, and publication-ready charts.",
    },
    {
        "num": 16,
        "part": "Part II · Applied Mathematics & Data Science",
        "part_code": "ds",
        "title": "String Processing, Regex Internals & Web APIs",
        "path": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis/basics.md",
        "colab": "02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis/11_regex_json_apis_practice.ipynb",
        "topicId": "c02_m11",
        "desc": "Regex NFA engines, lookarounds, named capture, streaming JSON (ijson), REST APIs, exponential backoff, ReDoS prevention, and PII redaction pipeline.",
    },

    # Part III: Machine Learning Engineering & Statistical Learning (Course 3)
    {
        "num": 17,
        "part": "Part III · Machine Learning Engineering",
        "part_code": "ml",
        "title": "Exploratory Data Analysis & Feature Selection Pipelines",
        "path": "03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering/basics.md",
        "colab": "03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering/01_eda_feature_engineering_practice.ipynb",
        "topicId": "c03_m01",
        "desc": "Tukey EDA philosophy, Box-Cox & Yeo-Johnson transformations, Mutual Information vs ANOVA, PCA/t-SNE/UMAP, target leakage, and LTV pipelines.",
    },
    {
        "num": 18,
        "part": "Part III · Machine Learning Engineering",
        "part_code": "ml",
        "title": "Unsupervised Clustering & Density Estimation",
        "path": "03_IITK_AIML_Core_Machine_Learning/02_clustering/basics.md",
        "colab": "03_IITK_AIML_Core_Machine_Learning/02_clustering/02_clustering_practice.ipynb",
        "topicId": "c03_m02",
        "desc": "K-Means++ initialization, Lloyd's convergence proof, Silhouette & Davies-Bouldin metrics, DBSCAN density reachability, and Hierarchical clustering.",
    },
    {
        "num": 19,
        "part": "Part III · Machine Learning Engineering",
        "part_code": "ml",
        "title": "Supervised Classification & Decision Forests (XGBoost/LightGBM)",
        "path": "03_IITK_AIML_Core_Machine_Learning/03_classification/basics.md",
        "colab": "03_IITK_AIML_Core_Machine_Learning/03_classification/03_classification_practice.ipynb",
        "topicId": "c03_m03",
        "desc": "Logistic loss & Newton-Raphson, Gini impurity & CART, Random Forest OOB error, Gradient Boosting pseudo-residuals, XGBoost math, and ROC/PR analysis.",
    },
    {
        "num": 20,
        "part": "Part III · Machine Learning Engineering",
        "part_code": "ml",
        "title": "Extreme Class Imbalance Mitigation (SMOTE, ADASYN & Focal Loss)",
        "path": "03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data/basics.md",
        "colab": "03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data/04_imbalanced_data_practice.ipynb",
        "topicId": "c03_m04",
        "desc": "Accuracy paradox, SMOTE/ADASYN geometry, cost-sensitive loss matrices, Youden's J optimal threshold tuning, Focal Loss math, and credit fraud engine.",
    },
    {
        "num": 21,
        "part": "Part III · Machine Learning Engineering",
        "part_code": "ml",
        "title": "Model Validation, Probability Calibration & Explainable AI (SHAP)",
        "path": "03_IITK_AIML_Core_Machine_Learning/05_model_evaluation/basics.md",
        "colab": "03_IITK_AIML_Core_Machine_Learning/05_model_evaluation/05_model_evaluation_practice.ipynb",
        "topicId": "c03_m05",
        "desc": "Stratified vs Purged Time-Series CV, Platt scaling & Isotonic regression, Shapley game theory axioms, TreeSHAP, PDP/ICE plots, and financial model audit.",
    },

    # Part IV: Deep Learning with TensorFlow & Keras (Course 4)
    {
        "num": 22,
        "part": "Part IV · Deep Learning with TensorFlow & Keras",
        "part_code": "dl",
        "title": "Neural Network Architecture & Backpropagation Calculus",
        "path": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics/basics.md",
        "colab": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics/01_neural_network_basics_practice.ipynb",
        "topicId": "c04_m01",
        "desc": "965-line Master Textbook: McCulloch-Pitts, XOR proof, Universal Approximation Theorem, 10 activation functions (GELU, Swish), multivariate chain rule backprop, Xavier/He init, and pure NumPy NN.",
    },
    {
        "num": 23,
        "part": "Part IV · Deep Learning with TensorFlow & Keras",
        "part_code": "dl",
        "title": "TensorFlow 2.x & Keras 3 Framework Architecture",
        "path": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow/basics.md",
        "colab": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow/02_keras_tensorflow_practice.ipynb",
        "topicId": "c04_m02",
        "desc": "Eager execution vs @tf.function autograph, tf.GradientTape, Keras 3 Functional & Subclassing APIs, custom training loops, callbacks, and multi-task learning.",
    },
    {
        "num": 24,
        "part": "Part IV · Deep Learning with TensorFlow & Keras",
        "part_code": "dl",
        "title": "Production Data Pipelines (tf.data), Normalization & Regularization",
        "path": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance/basics.md",
        "colab": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance/03_preprocessing_imbalance_practice.ipynb",
        "topicId": "c04_m03",
        "desc": "ETL input pipelines (prefetch, interleave, AUTOTUNE), Batch vs Layer Normalization math, Dropout Bernoulli dynamics, weight decay, and streaming datasets.",
    },
    {
        "num": 25,
        "part": "Part IV · Deep Learning with TensorFlow & Keras",
        "part_code": "dl",
        "title": "Deep Learning Evaluation, Learning Rate Schedules & Grad-CAM",
        "path": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl/basics.md",
        "colab": "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl/04_model_evaluation_dl_practice.ipynb",
        "topicId": "c04_m04",
        "desc": "Loss curve diagnosis, Cosine Annealing & One-Cycle schedules, Grad-CAM mathematical formulation, layer feature visualization, and SavedModel/ONNX exports.",
    },

    # Part V: Generative AI, Prompt Engineering & ChatGPT (Course 5)
    {
        "num": 26,
        "part": "Part V · Generative AI & Prompt Engineering",
        "part_code": "genai",
        "title": "Enterprise Prompt Engineering, In-Context Learning & Guardrails",
        "path": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering/basics.md",
        "colab": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering/01_prompt_engineering_practice.ipynb",
        "topicId": "c05_m01",
        "desc": "Prompt anatomy, zero/few-shot ICL, Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), ReAct agent loop, prompt injection defense, and automated evaluation.",
    },
    {
        "num": 27,
        "part": "Part V · Generative AI & Prompt Engineering",
        "part_code": "genai",
        "title": "Chat Completion APIs, Tool Calling & Structured Outputs",
        "path": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications/basics.md",
        "colab": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications/02_chatgpt_applications_practice.ipynb",
        "topicId": "c05_m02",
        "desc": "BPE token economics, JSON Schema function calling, Pydantic structured outputs, Server-Sent Events (SSE) streaming, conversation memory buffers, and SQL generator.",
    },
    {
        "num": 28,
        "part": "Part V · Generative AI & Prompt Engineering",
        "part_code": "genai",
        "title": "Parameter-Efficient Fine-Tuning (PEFT), LoRA & Quantization",
        "path": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization/basics.md",
        "colab": "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization/03_genai_optimization_practice.ipynb",
        "topicId": "c05_m03",
        "desc": "LoRA matrix decomposition math W = W0 + (alpha/r)(B*A), rank/alpha dynamics, QLoRA NF4 quantization, paged optimizers, decoding sampling (temp, top-p, top-k), and adapter merging.",
    },

    # Part VI: Advanced Generative AI & Multimodal Models (Course 6)
    {
        "num": 29,
        "part": "Part VI · Advanced Generative AI & Multimodal",
        "part_code": "rag",
        "title": "Enterprise Retrieval-Augmented Generation (RAG) Architecture",
        "path": "06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/basics.md",
        "colab": "06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/01_rag_architectures_practice.ipynb",
        "topicId": "c06_m01",
        "desc": "Multi-stage RAG, recursive & semantic chunking, dense embeddings, Maximal Marginal Relevance (MMR), Cross-Encoder reranking, HyDE expansion, and Ragas evaluation.",
    },
    {
        "num": 30,
        "part": "Part VI · Advanced Generative AI & Multimodal",
        "part_code": "rag",
        "title": "Vector Databases, Approximate Nearest Neighbors & ChromaDB",
        "path": "06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma/basics.md",
        "colab": "06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma/02_vector_databases_chroma_practice.ipynb",
        "topicId": "c06_m02",
        "desc": "Relational vs Vector stores, Euclidean/Cosine/Dot product metrics, HNSW graph architecture (M, efConstruction, efSearch), IVFFlat & Product Quantization, and multi-tenant ChromaDB.",
    },
    {
        "num": 31,
        "part": "Part VI · Advanced Generative AI & Multimodal",
        "part_code": "rag",
        "title": "Multimodal Vision-Language (CLIP) & Latent Diffusion Models",
        "path": "06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models/basics.md",
        "colab": "06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models/03_multimodal_generative_models_practice.ipynb",
        "topicId": "c06_m03",
        "desc": "OpenAI CLIP contrastive dual encoder, DDPM forward noising, Latent Diffusion Models (LDMs), U-Net cross-attention noise predictor, Classifier-Free Guidance (CFG), and ControlNet.",
    },

    # Part VII: Industrial Capstones & Applied Systems (Course 7)
    {
        "num": 32,
        "part": "Part VII · Industrial Capstones & Systems",
        "part_code": "capstone",
        "title": "Autonomous Driving Perception: Multi-Task Object Detection & Bounding Box Localization",
        "path": "07_IITK_AIML_Capstone/project1_autonomous_driving/WRITEUP.md",
        "colab": "07_IITK_AIML_Capstone/project1_autonomous_driving/autonomous_driving_perception_capstone.ipynb",
        "topicId": "cap_p1",
        "desc": "MobileNetV2 feature backbone, dual classification & bounding box regression heads, CIoU loss, Grad-CAM attention maps, and real-time inference.",
    },
    {
        "num": 33,
        "part": "Part VII · Industrial Capstones & Systems",
        "part_code": "capstone",
        "title": "365-Day Hierarchical Demand Forecasting: Fourier Dynamics & XGBoost Ensembles",
        "path": "07_IITK_AIML_Capstone/project2_sales_forecasting/WRITEUP.md",
        "colab": "07_IITK_AIML_Capstone/project2_sales_forecasting/sales_forecasting_capstone.ipynb",
        "topicId": "cap_p2",
        "desc": "Hierarchical time-series reconciliation (Bottom-Up, Top-Down), Fourier harmonic seasonality, XGBoost + Linear Stacking ensembles, and WAPE/RMSLE evaluation.",
    },
]

# Build JSON string for injecting into frontend script
CHAPTERS_JSON = json.dumps(CHAPTERS)

def generate_interview_hub_html() -> str:
    """Generates the interactive 33-topic interview question vault."""
    cards_html = []
    for ch in CHAPTERS:
        # Determine question count
        q_count = 30
        cards_html.append(f"""        <div class="interview-card" data-domain="{ch['part_code']}">
          <div class="interview-card-header">
            <span class="domain-tag domain-{ch['part_code']}">{ch['part'].split('·')[1].strip() if '·' in ch['part'] else ch['part']}</span>
            <span class="qa-count-badge">30 Q&A</span>
          </div>
          <h3 class="interview-card-title">{ch['title']}</h3>
          <p class="interview-card-desc">{ch['desc']}</p>
          <div class="interview-card-footer">
            <span class="mastery-pill" id="mastery-badge-{ch['topicId']}">0 / 30 Mastered</span>
            <button class="btn-launch-flashcards" onclick="openStudio('{ch['topicId']}', 'qa')">
              Launch Interview Vault ↗
            </button>
          </div>
        </div>""")
    
    cards_str = "\n".join(cards_html)
    return f"""    <!-- DEDICATED TECHNICAL INTERVIEW VAULT VIEW -->
    <div id="section-interview-hub" style="display:none">
      <div class="hub-hero">
        <span class="hub-hero-badge">🎯 Comprehensive Technical Interview Vault</span>
        <h2 class="hub-hero-title">Staff-Level Machine Learning & AI Question Bank</h2>
        <p class="hub-hero-sub">
          Master 990+ rigorous technical interview questions, algorithmic derivations, system design tradeoffs, and code implementations across all 33 curriculum specializations.
        </p>

        <!-- Topic Filter Pills -->
        <div class="hub-filter-pills">
          <button class="hub-pill active" onclick="filterInterviewHub('all', this)">All Categories (33 Modules)</button>
          <button class="hub-pill" onclick="filterInterviewHub('python', this)">Python Foundations (150 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('ds', this)">Data Science & Stats (330 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('ml', this)">Machine Learning (150 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('dl', this)">Deep Learning (120 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('genai', this)">Generative AI (90 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('rag', this)">Advanced GenAI & RAG (90 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('capstone', this)">Industrial Capstones (60 Q)</button>
        </div>

        <!-- Master Curriculum Guide Banner -->
        <div class="hub-master-card" onclick="openStandaloneDoc('LEARNING_GUIDE.md', 'Master AI/ML Learning Guide')">
          <div class="hub-master-left">
            <span class="hub-master-badge">Comprehensive Guide</span>
            <h4>Master AI/ML & Generative AI Learning Guide (LEARNING_GUIDE.md)</h4>
            <p>Theory foundations, mathematical cheatsheet, STAR project write-ups, and the Top 100 System Architecture Q&A.</p>
          </div>
          <button class="btn-hub-master">Open Master Guide ↗</button>
        </div>
      </div>

      <!-- 33 Topic Cards Grid -->
      <div class="interview-grid" id="interview-cards-grid">
{cards_str}
      </div>
    </div>"""

def generate_guides_view_html() -> str:
    """Generates the dedicated Book Table of Contents & Master Guides view."""
    # Build 33 chapter cards for the Master Textbook
    chapter_cards = []
    for ch in CHAPTERS:
        clean_title_esc = ch['title'].replace("'", "\\'")
        chapter_cards.append(f"""        <div class="book-chapter-card" data-part="{ch['part_code']}">
          <div class="book-card-top">
            <span class="book-part-tag">{ch['part']}</span>
            <span class="book-chap-pill">Chapter {ch['num']}</span>
          </div>
          <h4 class="book-card-title">{ch['title']}</h4>
          <p class="book-card-desc">{ch['desc']}</p>
          <div class="book-card-actions">
            <button class="btn-book-read" onclick="openStandaloneDoc('{ch['path']}', 'Chapter {ch['num']}: {clean_title_esc}')">
              📖 Read Chapter Handbook
            </button>
            <a class="btn-book-colab" href="https://colab.research.google.com/github/sameerkarur/Data_science/blob/main/{ch['colab']}" target="_blank" rel="noopener noreferrer" title="Practice in Colab">
              🚀 Colab
            </a>
            <button class="btn-book-qa" onclick="openStudio('{ch['topicId']}', 'qa')" title="Practice Interview Flashcards">
              🎯 Q&A
            </button>
          </div>
        </div>""")

    chapters_html_str = "\n".join(chapter_cards)

    return f"""    <!-- DEDICATED MASTER GUIDES & TEXTBOOK VIEW -->
    <div id="section-guides-group" style="display:none">
      <div class="hub-hero">
        <span class="hub-hero-badge">📚 Comprehensive Curriculum Textbook & Architectural Manuals</span>
        <h2 class="hub-hero-title">AI/ML & Generative AI Master Textbook</h2>
        <p class="hub-hero-sub">
          Authoritative academic engineering handbooks covering foundational algorithms to advanced frontier systems. Each chapter features step-by-step mathematical derivations, ASCII system architectures, and hands-on exercises.
        </p>

        <!-- Chapter Filter Bar -->
        <div class="book-filter-bar">
          <input type="text" id="book-search-input" class="book-search-field" placeholder="🔍 Search chapters, mathematical formulas & architectures (e.g. Backprop, LoRA, SVD, HNSW)..." oninput="searchBookChapters(this.value)">
          <div class="hub-filter-pills" style="margin-top: 0.75rem;">
            <button class="hub-pill active" onclick="filterBookPart('all', this)">All 33 Chapters</button>
            <button class="hub-pill" onclick="filterBookPart('python', this)">Python Foundations (Ch 1–5)</button>
            <button class="hub-pill" onclick="filterBookPart('ds', this)">Applied Data Science (Ch 6–16)</button>
            <button class="hub-pill" onclick="filterBookPart('ml', this)">Machine Learning (Ch 17–21)</button>
            <button class="hub-pill" onclick="filterBookPart('dl', this)">Deep Learning & Keras (Ch 22–25)</button>
            <button class="hub-pill" onclick="filterBookPart('genai', this)">Generative AI (Ch 26–28)</button>
            <button class="hub-pill" onclick="filterBookPart('rag', this)">Advanced GenAI & RAG (Ch 29–31)</button>
            <button class="hub-pill" onclick="filterBookPart('capstone', this)">Industrial Capstones (Ch 32–33)</button>
          </div>
        </div>
      </div>

      <!-- 33 Chapter Textbook Cards -->
      <div class="book-toc-grid" id="book-chapters-grid">
{chapters_html_str}
      </div>

      <!-- REPOSITORY MANUALS & META GUIDES -->
      <div style="margin-top: 3.5rem; margin-bottom: 1.5rem;">
        <h3 style="color:#f8fafc; font-size:1.35rem; font-weight:700; border-left:4px solid #38bdf8; padding-left:0.75rem;">
          🏛️ System Architecture Manuals & Program Documentation
        </h3>
      </div>

      <div class="guides-grid">
        <div class="guide-card" onclick="openStandaloneDoc('LEARNING_GUIDE.md', 'Master AI/ML Learning Guide')">
          <div class="guide-card-icon">📘</div>
          <h3 class="guide-card-title">Master AI/ML Learning Guide</h3>
          <p class="guide-card-desc">Complete curriculum theory, mathematical formulas, course-by-course deep dives, and system architecture.</p>
          <div class="guide-card-meta">LEARNING_GUIDE.md · 88KB</div>
          <button class="btn-guide-launch">Read Master Guide ↗</button>
        </div>

        <div class="guide-card" onclick="openStandaloneDoc('PRACTICE_GUIDE.md', 'Practice Guide & Curriculum Overview')">
          <div class="guide-card-icon">⚡</div>
          <h3 class="guide-card-title">Practice & Execution Guide</h3>
          <p class="guide-card-desc">Step-by-step roadmap for all 1,650+ practice problems, solutions notebooks, and local setup environments.</p>
          <div class="guide-card-meta">PRACTICE_GUIDE.md · 24KB</div>
          <button class="btn-guide-launch">Read Practice Guide ↗</button>
        </div>

        <div class="guide-card" onclick="openStandaloneDoc('projects_version2/README.md', 'Version 2 Advanced Projects Suite')">
          <div class="guide-card-icon">🚀</div>
          <h3 class="guide-card-title">Version 2 Projects Executive Brief</h3>
          <p class="guide-card-desc">Comprehensive index and architectural documentation of the 15 re-engineered advanced portfolio projects.</p>
          <div class="guide-card-meta">projects_version2/README.md · 16KB</div>
          <button class="btn-guide-launch">Explore V2 Suite ↗</button>
        </div>

        <div class="guide-card" onclick="openStandaloneDoc('README.md', 'Repository Overview & System Architecture')">
          <div class="guide-card-icon">🌟</div>
          <h3 class="guide-card-title">Repository Documentation</h3>
          <p class="guide-card-desc">Full repository index, course mapping, quickstart instructions, and GitHub Pages deployment details.</p>
          <div class="guide-card-meta">README.md · 12KB</div>
          <button class="btn-guide-launch">Read README ↗</button>
        </div>

        <div class="guide-card" onclick="openStandaloneDoc('LEARNING_JOURNEY.md', 'Learning Journey, Experience & Simplilearn Attribution')">
          <div class="guide-card-icon">🎓</div>
          <h3 class="guide-card-title">Learning Journey & Attribution</h3>
          <p class="guide-card-desc">Personal coursework experience, Simplilearn and IIT Kanpur acknowledgments, engineering philosophy, and copyright fair-use terms.</p>
          <div class="guide-card-meta">LEARNING_JOURNEY.md · Educational Fair Use</div>
          <button class="btn-guide-launch">Read Learning Journey ↗</button>
        </div>
      </div>
    </div>"""

# OpenCareerAI Modern Design Stylesheet Additions
extra_css = """
    /* ==========================================
       OPENTRAIN / OPENCAREERAI MODERN DESIGN SYSTEM
       ========================================== */
    :root {
      --bg-main: #090d16;
      --bg-card: #111827;
      --border-card: #1f2937;
      --accent-cyan: #38bdf8;
      --accent-emerald: #10b981;
      --accent-indigo: #6366f1;
    }

    /* HIGH-CONTRAST ACCESSIBLE LINK & CONTENT STYLES */
    .stat-card {
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .stat-card:hover {
      border-color: #38bdf8 !important;
      transform: translateY(-2px);
      box-shadow: 0 4px 16px -2px rgba(56, 189, 248, 0.25);
    }
    .doc-viewer a,
    .doc-viewer a:visited,
    .standalone-dialog a,
    .studio-dialog a {
      color: #38bdf8 !important;
      text-decoration: underline;
      text-decoration-color: rgba(56, 189, 248, 0.45);
      text-underline-offset: 3px;
      font-weight: 600;
      transition: all 0.2s ease;
    }
    .doc-viewer a:hover,
    .standalone-dialog a:hover,
    .studio-dialog a:hover {
      color: #7dd3fc !important;
      text-decoration-color: #38bdf8;
      text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }
    .doc-viewer table a,
    .doc-viewer table a:visited {
      color: #38bdf8 !important;
      font-weight: 600;
      text-decoration: underline;
    }
    .doc-viewer table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.5rem 0;
      border: 1px solid #334155;
      border-radius: 8px;
      overflow: hidden;
      background: rgba(15, 23, 42, 0.7);
    }
    .doc-viewer th {
      background: #1e293b;
      color: #38bdf8;
      font-weight: 700;
      font-size: 0.9rem;
      padding: 0.85rem 1rem;
      border: 1px solid #334155;
      text-align: left;
    }
    .doc-viewer td {
      padding: 0.75rem 1rem;
      border: 1px solid #334155;
      color: #e2e8f0;
      font-size: 0.88rem;
    }
    .doc-viewer tr:nth-child(even) {
      background: rgba(30, 41, 59, 0.4);
    }
    .doc-viewer pre {
      background: #090d16 !important;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 1.25rem;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.88rem;
      line-height: 1.6;
      color: #e2e8f0;
      margin: 1.5rem 0;
    }
    .doc-viewer code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.12);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.88em;
    }
    .doc-viewer pre code {
      background: transparent;
      padding: 0;
      color: inherit;
      font-size: inherit;
    }
    .doc-viewer h1 {
      color: #f8fafc;
      font-size: 1.85rem;
      margin-top: 2rem;
      margin-bottom: 1rem;
      border-bottom: 2px solid #334155;
      padding-bottom: 0.5rem;
      scroll-margin-top: 1.5rem;
    }
    .doc-viewer h2 {
      color: #38bdf8;
      font-size: 1.45rem;
      margin-top: 1.75rem;
      margin-bottom: 0.75rem;
      scroll-margin-top: 1.5rem;
    }
    .doc-viewer h3 {
      color: #e2e8f0;
      font-size: 1.2rem;
      margin-top: 1.5rem;
      margin-bottom: 0.5rem;
      scroll-margin-top: 1.5rem;
    }
    .doc-viewer h4 {
      color: #cbd5e1;
      font-size: 1.05rem;
      margin-top: 1.25rem;
      margin-bottom: 0.5rem;
      scroll-margin-top: 1.5rem;
    }
    .doc-viewer details {
      margin: 1.5rem 0;
      padding: 1rem 1.25rem;
      background: rgba(17, 24, 39, 0.8);
      border: 1px solid #334155;
      border-radius: 8px;
    }
    .doc-viewer summary {
      cursor: pointer;
      font-weight: 600;
      color: #38bdf8;
      outline: none;
      user-select: none;
    }
    .doc-viewer blockquote {
      border-left: 4px solid #38bdf8;
      background: rgba(56, 189, 248, 0.08);
      padding: 0.75rem 1.25rem;
      margin: 1.5rem 0;
      border-radius: 0 8px 8px 0;
      color: #cbd5e1;
    }

    /* BOOK TOC & CHAPTER GRID STYLES */
    .book-filter-bar {
      margin-top: 1.5rem;
      max-width: 850px;
    }
    .book-search-field {
      width: 100%;
      background: #111827;
      border: 1px solid #374151;
      border-radius: 8px;
      padding: 0.85rem 1.25rem;
      color: #f3f4f6;
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s ease;
      box-sizing: border-box;
    }
    .book-search-field:focus {
      border-color: #38bdf8;
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
    }
    .book-toc-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.25rem;
      margin-top: 1.5rem;
    }
    .book-chapter-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
    }
    .book-chapter-card:hover {
      border-color: #38bdf8;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px -2px rgba(56, 189, 248, 0.2);
    }
    .book-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .book-part-tag {
      font-size: 0.72rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #94a3b8;
    }
    .book-chap-pill {
      font-size: 0.75rem;
      font-weight: 700;
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
    }
    .book-card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #f8fafc;
      margin: 0 0 0.5rem 0;
      line-height: 1.4;
    }
    .book-card-desc {
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 1.25rem;
      flex-grow: 1;
    }
    .book-card-actions {
      display: flex;
      gap: 0.5rem;
      align-items: center;
    }
    .btn-book-read {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 0.5rem 0.85rem;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      flex-grow: 1;
      text-align: center;
      transition: all 0.2s ease;
    }
    .btn-book-read:hover {
      background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
      box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);
    }
    .btn-book-colab {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.35);
      color: #fbbf24;
      border-radius: 6px;
      padding: 0.5rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 600;
      text-decoration: none !important;
      display: inline-flex;
      align-items: center;
      transition: all 0.2s ease;
    }
    .btn-book-colab:hover {
      background: rgba(245, 158, 11, 0.25);
      border-color: #fbbf24;
    }
    .btn-book-qa {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      border-radius: 6px;
      padding: 0.5rem 0.75rem;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-book-qa:hover {
      background: rgba(16, 185, 129, 0.25);
      border-color: #34d399;
    }

    /* IN-MODAL CHAPTER NAVIGATION BAR */
    .book-chapter-nav-bar {
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .btn-nav-chap {
      background: #1e293b;
      border: 1px solid #334155;
      color: #38bdf8;
      border-radius: 8px;
      padding: 0.65rem 1.15rem;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-nav-chap:hover {
      background: #334155;
      border-color: #38bdf8;
      color: #7dd3fc;
    }
    .btn-nav-top {
      background: transparent;
      border: 1px solid #475569;
      color: #94a3b8;
      border-radius: 8px;
      padding: 0.65rem 1rem;
      font-size: 0.88rem;
      cursor: pointer;
    }
    .btn-nav-top:hover {
      border-color: #cbd5e1;
      color: #f1f5f9;
    }

    /* INTERVIEW HUB STYLES */
    .hub-hero {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
    }
    .hub-hero-badge {
      display: inline-block;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: #38bdf8;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      margin-bottom: 0.75rem;
      letter-spacing: 0.5px;
    }
    .hub-hero-title {
      font-size: 1.85rem;
      font-weight: 800;
      color: #f8fafc;
      margin: 0 0 0.5rem 0;
      letter-spacing: -0.5px;
    }
    .hub-hero-sub {
      font-size: 0.95rem;
      color: #94a3b8;
      margin: 0 0 1.25rem 0;
      max-width: 800px;
      line-height: 1.6;
    }
    .hub-filter-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }
    .hub-pill {
      background: #111827;
      border: 1px solid #374151;
      color: #9ca3af;
      padding: 0.45rem 0.95rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .hub-pill:hover {
      border-color: #38bdf8;
      color: #f3f4f6;
    }
    .hub-pill.active {
      background: #0284c7;
      border-color: #38bdf8;
      color: #ffffff;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    .hub-master-card {
      margin-top: 1.5rem;
      background: linear-gradient(135deg, rgba(2, 132, 199, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .hub-master-card:hover {
      border-color: #38bdf8;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px -2px rgba(56, 189, 248, 0.3);
    }
    .hub-master-badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.15);
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      margin-bottom: 0.35rem;
    }
    .hub-master-left h4 {
      color: #f8fafc;
      font-size: 1.1rem;
      font-weight: 700;
      margin: 0 0 0.25rem 0;
    }
    .hub-master-left p {
      color: #cbd5e1;
      font-size: 0.85rem;
      margin: 0;
    }
    .btn-hub-master {
      background: #0284c7;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 0.65rem 1.15rem;
      font-size: 0.85rem;
      font-weight: 600;
      white-space: nowrap;
      cursor: pointer;
    }
    .interview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .interview-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
    }
    .interview-card:hover {
      border-color: #38bdf8;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px -2px rgba(56, 189, 248, 0.2);
    }
    .interview-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .domain-tag {
      font-size: 0.72rem;
      text-transform: uppercase;
      font-weight: 700;
      letter-spacing: 0.5px;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
    }
    .domain-python { background: rgba(56, 189, 248, 0.15); color: #38bdf8; }
    .domain-ds { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .domain-ml { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
    .domain-dl { background: rgba(239, 68, 68, 0.15); color: #f87171; }
    .domain-genai { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .domain-rag { background: rgba(236, 72, 153, 0.15); color: #f472b6; }
    .domain-capstone { background: rgba(99, 102, 241, 0.15); color: #818cf8; }
    .qa-count-badge {
      font-size: 0.75rem;
      color: #94a3b8;
      font-weight: 600;
    }
    .interview-card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #f8fafc;
      margin: 0 0 0.5rem 0;
      line-height: 1.4;
    }
    .interview-card-desc {
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 1.25rem;
      flex-grow: 1;
    }
    .interview-card-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 0.75rem;
      border-top: 1px solid #1f2937;
    }
    .mastery-pill {
      font-size: 0.75rem;
      font-weight: 600;
      color: #cbd5e1;
      background: #1f2937;
      padding: 0.2rem 0.55rem;
      border-radius: 9999px;
      border: 1px solid #374151;
    }
    .btn-launch-flashcards {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 0.45rem 0.85rem;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-launch-flashcards:hover {
      background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
      box-shadow: 0 0 12px rgba(14, 165, 233, 0.4);
    }
    .guides-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .guide-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 1.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
    }
    .guide-card:hover {
      border-color: #38bdf8;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px -2px rgba(56, 189, 248, 0.2);
    }
    .guide-card-icon {
      font-size: 2rem;
      margin-bottom: 0.75rem;
    }
    .guide-card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: #f8fafc;
      margin: 0 0 0.5rem 0;
    }
    .guide-card-desc {
      font-size: 0.88rem;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 1.25rem;
      flex-grow: 1;
    }
    .guide-card-meta {
      font-size: 0.78rem;
      color: #64748b;
      margin-bottom: 0.75rem;
      font-family: monospace;
    }
    .btn-guide-launch {
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 0.5rem 1rem;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-guide-launch:hover {
      background: #0284c7;
      color: #fff;
      border-color: #38bdf8;
    }
    .app-attribution-footer {
      margin-top: 4rem;
      padding: 2.5rem 1.5rem;
      background: #090d16;
      border-top: 1px solid #1f2937;
    }
    .attribution-card {
      max-width: 1100px;
      margin: 0 auto;
      background: rgba(17, 24, 39, 0.6);
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 2rem;
      text-align: center;
    }
    .attribution-badge {
      display: inline-block;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.25rem 0.85rem;
      border-radius: 9999px;
      margin-bottom: 1rem;
    }
    .attribution-title {
      color: #f8fafc;
      font-size: 1.3rem;
      font-weight: 700;
      margin: 0 0 0.75rem 0;
    }
    .attribution-desc {
      color: #cbd5e1;
      font-size: 0.92rem;
      line-height: 1.6;
      margin-bottom: 0.75rem;
      max-width: 850px;
      margin-left: auto;
      margin-right: auto;
    }
    .attribution-sub {
      color: #94a3b8;
      font-size: 0.82rem;
      line-height: 1.6;
      margin-bottom: 1.5rem;
      max-width: 850px;
      margin-left: auto;
      margin-right: auto;
    }
    .attribution-actions {
      display: flex;
      justify-content: center;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .btn-attribution-modal {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 0.65rem 1.25rem;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn-attribution-modal:hover {
      background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
      box-shadow: 0 0 16px rgba(14, 165, 233, 0.4);
    }
    .nav-github-link {
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 0.65rem 1.25rem;
      font-size: 0.9rem;
      font-weight: 600;
      text-decoration: none !important;
      display: inline-flex;
      align-items: center;
      transition: all 0.2s ease;
    }
    .nav-github-link:hover {
      border-color: #38bdf8;
      color: #38bdf8;
    }
"""

updated_switch_nav = """function switchNavSection(sec, btn) {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      if (btn) btn.classList.add('active');

      const coursesSec = document.getElementById('section-courses-group');
      const capstonesSec = document.getElementById('section-capstones-group');
      const v2Sec = document.getElementById('section-v2-group');
      const electivesSec = document.getElementById('section-electives-group');
      const interviewSec = document.getElementById('section-interview-hub');
      const guidesSec = document.getElementById('section-guides-group');

      if (coursesSec) coursesSec.style.display = 'none';
      if (capstonesSec) capstonesSec.style.display = 'none';
      if (v2Sec) v2Sec.style.display = 'none';
      if (electivesSec) electivesSec.style.display = 'none';
      if (interviewSec) interviewSec.style.display = 'none';
      if (guidesSec) guidesSec.style.display = 'none';

      if (sec === 'all') {
        if (coursesSec) coursesSec.style.display = '';
        if (capstonesSec) capstonesSec.style.display = '';
        if (v2Sec) v2Sec.style.display = '';
        if (electivesSec) electivesSec.style.display = '';
      } else if (sec === 'courses') {
        if (coursesSec) coursesSec.style.display = '';
      } else if (sec === 'v2') {
        if (v2Sec) v2Sec.style.display = '';
      } else if (sec === 'capstones') {
        if (capstonesSec) capstonesSec.style.display = '';
      } else if (sec === 'interview') {
        if (interviewSec) {
          interviewSec.style.display = '';
          updateAllMasteryBadges();
        }
        window.scrollTo({ top: document.querySelector('.controls-panel')?.offsetTop || 350, behavior: 'smooth' });
      } else if (sec === 'theory') {
        if (guidesSec) guidesSec.style.display = '';
        window.scrollTo({ top: document.querySelector('.controls-panel')?.offsetTop || 350, behavior: 'smooth' });
      }
    }

    function filterInterviewHub(domainClass, btn) {
      document.querySelectorAll('.hub-filter-pills .hub-pill').forEach(p => p.classList.remove('active'));
      if (btn) btn.classList.add('active');
      const cards = document.querySelectorAll('.interview-card');
      cards.forEach(c => {
        if (domainClass === 'all' || c.getAttribute('data-domain') === domainClass) {
          c.style.display = '';
        } else {
          c.style.display = 'none';
        }
      });
    }

    function filterBookPart(partCode, btn) {
      btn.parentElement.querySelectorAll('.hub-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const cards = document.querySelectorAll('.book-chapter-card');
      cards.forEach(c => {
        if (partCode === 'all' || c.getAttribute('data-part') === partCode) {
          c.style.display = '';
        } else {
          c.style.display = 'none';
        }
      });
    }

    function searchBookChapters(query) {
      const q = (query || '').toLowerCase().trim();
      const cards = document.querySelectorAll('.book-chapter-card');
      cards.forEach(c => {
        const text = c.textContent.toLowerCase();
        if (!q || text.includes(q)) {
          c.style.display = '';
        } else {
          c.style.display = 'none';
        }
      });
    }

    function updateAllMasteryBadges() {
      const stored = JSON.parse(localStorage.getItem('aiml_mastered_q') || '{}');
      document.querySelectorAll('.interview-card').forEach(card => {
        const idBadge = card.querySelector('.mastery-pill');
        if (!idBadge || !idBadge.id) return;
        const topicKey = idBadge.id.replace('mastery-badge-', '');
        let count = 0;
        Object.keys(stored).forEach(k => {
          if (k.includes(topicKey) && stored[k]) count++;
        });
        idBadge.innerText = `${count} / 30 Mastered`;
        if (count > 0) {
          idBadge.style.background = 'rgba(16, 185, 129, 0.2)';
          idBadge.style.borderColor = '#10b981';
        }
      });
    }
"""

marked_parser_logic = f"""
    // Injected Curriculum Registry
    const CURRICULUM_CHAPTERS = {CHAPTERS_JSON};

    // Configure Marked with GFM, line breaks, and standard anchor slug generation
    if (typeof marked !== 'undefined') {{
      marked.setOptions({{
        gfm: true,
        breaks: true
      }});

      const renderer = new marked.Renderer();
      renderer.heading = function({{ text, depth, raw }}) {{
        const cleanText = text.replace(/<[^>]*>/g, '').toLowerCase().trim();
        const slug = cleanText
          .replace(/[^\\w\\s-]/g, '')
          .replace(/\\s+/g, '-');
        const compactSlug = cleanText.replace(/[^a-z0-9]/g, '');
        return '<h' + depth + ' id="' + slug + '" data-slug="' + slug + '" data-compact="' + compactSlug + '">' + text + '</h' + depth + '>\\n';
      }};
      marked.use({{ renderer }});
    }}

    function parseMarkdown(md) {{
      if (!md) return '';
      if (typeof marked !== 'undefined' && marked.parse) {{
        try {{
          return '<div class="doc-viewer">' + marked.parse(md) + '</div>';
        }} catch(e) {{
          console.error("Marked parse error:", e);
        }}
      }}
      return '<div class="doc-viewer"><pre>' + escapeHtml(md) + '</pre></div>';
    }}

    // Global Link Interceptor: Completely eliminates raw file redirects
    document.addEventListener('click', function(e) {{
      const link = e.target.closest('a');
      if (!link) return;
      let href = link.getAttribute('href');
      if (!href) return;

      href = href.trim();

      // CASE 1: In-Page Anchor Links (#anchor)
      if (href.startsWith('#')) {{
        e.preventDefault();
        e.stopPropagation();
        if (href === '#' || href === '') return;
        const targetId = href.substring(1);
        scrollToAnchorInModal(targetId);
        return;
      }}

      // CASE 2: Raw GitHub or GitHub blob URLs pointing to this repo
      if (href.includes('githubusercontent.com') || href.includes('github.com')) {{
        if (href.includes('index.html#')) {{
          e.preventDefault();
          e.stopPropagation();
          const targetId = href.split('index.html#')[1];
          scrollToAnchorInModal(targetId);
          return;
        }}
        if (href.endsWith('.md') || href.includes('.md#')) {{
          const match = href.match(/(?:main|master)\\/(.*\\.md(?:#.*)?)$/);
          if (match) {{
            e.preventDefault();
            e.stopPropagation();
            const parts = match[1].split('#');
            openStandaloneDoc(parts[0], getCleanTitleFromPath(parts[0]), parts[1]);
            return;
          }}
        }}
        if (href.endsWith('.ipynb')) {{
          const match = href.match(/(?:main|master)\\/(.*\\.ipynb)$/);
          if (match) {{
            e.preventDefault();
            e.stopPropagation();
            window.open('https://colab.research.google.com/github/' + GITHUB_REPO + '/blob/' + BRANCH + '/' + match[1], '_blank', 'noopener,noreferrer');
            return;
          }}
        }}
      }}

      // CASE 3: Relative Markdown Links (*.md)
      if (href.endsWith('.md') || href.includes('.md#')) {{
        e.preventDefault();
        e.stopPropagation();
        const parts = href.split('#');
        const docPath = normalizeRepoPath(parts[0]);
        openStandaloneDoc(docPath, getCleanTitleFromPath(docPath), parts[1]);
        return;
      }}

      // CASE 4: Relative Jupyter Notebook Links (*.ipynb)
      if (href.endsWith('.ipynb')) {{
        e.preventDefault();
        e.stopPropagation();
        const nbPath = normalizeRepoPath(href);
        window.open('https://colab.research.google.com/github/' + GITHUB_REPO + '/blob/' + BRANCH + '/' + nbPath, '_blank', 'noopener,noreferrer');
        return;
      }}

      // CASE 5: External links - ensure target=_blank
      if (href.startsWith('http://') || href.startsWith('https://')) {{
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }}
    }});

    function scrollToAnchorInModal(targetId) {{
      if (!targetId) return;
      const cleanTarget = decodeURIComponent(targetId).toLowerCase().replace(/[^a-z0-9]+/g, '');
      
      const studioModal = document.getElementById('studio-modal');
      const standaloneModal = document.getElementById('standalone-modal');
      let container = null;

      if (standaloneModal && standaloneModal.style.display !== 'none' && standaloneModal.style.display !== '') {{
        container = document.getElementById('standalone-body');
      }} else if (studioModal && studioModal.style.display !== 'none' && studioModal.style.display !== '') {{
        container = document.getElementById('studio-body');
      }} else {{
        container = document.documentElement || document.body;
      }}

      if (!container) return;

      let targetElem = null;
      try {{
        targetElem = container.querySelector('#' + CSS.escape(targetId)) ||
                     container.querySelector('[name="' + CSS.escape(targetId) + '"]');
      }} catch(e) {{}}

      if (!targetElem) {{
        const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6, [data-slug]');
        for (let i = 0; i < headings.length; i++) {{
          const h = headings[i];
          const hSlug = (h.getAttribute('data-slug') || h.id || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
          const hCompact = (h.getAttribute('data-compact') || '').toLowerCase();
          const hText = (h.textContent || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
          
          if (hSlug === cleanTarget || hCompact === cleanTarget || hText === cleanTarget ||
              (cleanTarget.length > 4 && (hSlug.includes(cleanTarget) || cleanTarget.includes(hSlug) || hText.includes(cleanTarget) || cleanTarget.includes(hText)))) {{
            targetElem = h;
            break;
          }}
        }}
      }}

      if (targetElem) {{
        if (container === document.documentElement || container === document.body) {{
          targetElem.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }} else {{
          container.scrollTo({{
            top: targetElem.offsetTop - 24,
            behavior: 'smooth'
          }});
        }}
      }}
    }}

    function normalizeRepoPath(path) {{
      if (!path) return '';
      let p = path.replace(/^(\\.\\/|\\/)/, '');
      while (p.startsWith('../')) {{
        p = p.substring(3);
      }}
      return p;
    }}

    function getCleanTitleFromPath(path) {{
      if (!path) return 'Document Reader';
      // Match against known chapters
      const chap = CURRICULUM_CHAPTERS.find(c => c.path === path || path.endsWith(c.path));
      if (chap) return 'Chapter ' + chap.num + ': ' + chap.title;
      const filename = path.split('/').pop().replace('.md', '');
      return filename.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
    }}

    // Standalone Document Reader with Chapter Navigation & Anchor Scrolling
    let currentStandaloneDocPath = '';
    function openStandaloneDoc(path, title, anchor) {{
      const modal = document.getElementById('standalone-modal');
      const titleElem = document.getElementById('standalone-title');
      const pathElem = document.getElementById('standalone-path');
      const ghBtn = document.getElementById('standalone-gh-btn');
      const body = document.getElementById('standalone-body');

      currentStandaloneDocPath = path;
      titleElem.textContent = title || getCleanTitleFromPath(path);
      pathElem.textContent = path;
      ghBtn.href = 'https://github.com/' + GITHUB_REPO + '/blob/' + BRANCH + '/' + path;

      body.innerHTML = `
        <div style="text-align:center; padding:3rem; color:var(--muted)">
          <div style="font-size:2rem; margin-bottom:0.5rem">⚡</div>
          <h4>Loading Document...</h4>
        </div>
      `;
      modal.style.display = 'flex';

      fetchFile(path).then(text => {{
        currentStandaloneText = text;
        let renderedHtml = parseMarkdown(text);

        // Check if this document is part of the 33-chapter curriculum
        const chapIdx = CURRICULUM_CHAPTERS.findIndex(c => c.path === path || path.endsWith(c.path));
        if (chapIdx !== -1) {{
          const curr = CURRICULUM_CHAPTERS[chapIdx];
          const prev = chapIdx > 0 ? CURRICULUM_CHAPTERS[chapIdx - 1] : null;
          const next = chapIdx < CURRICULUM_CHAPTERS.length - 1 ? CURRICULUM_CHAPTERS[chapIdx + 1] : null;

          renderedHtml += `
            <div class="book-chapter-nav-bar">
              <div>
                ${{prev ? `<button class="btn-nav-chap" onclick="navigateChapter(${{chapIdx - 1}})">← Ch ${{prev.num}}: ${{escapeHtml(prev.title)}}</button>` : `<span style="color:#64748b; font-size:0.85rem">Start of Curriculum</span>`}}
              </div>
              <button class="btn-nav-top" onclick="document.getElementById('standalone-body').scrollTo({{ top: 0, behavior: 'smooth' }})">Top of Chapter ↑</button>
              <div>
                ${{next ? `<button class="btn-nav-chap" onclick="navigateChapter(${{chapIdx + 1}})">Ch ${{next.num}}: ${{escapeHtml(next.title)}} →</button>` : `<span style="color:#64748b; font-size:0.85rem">End of Curriculum</span>`}}
              </div>
            </div>
          `;
        }}

        body.innerHTML = renderedHtml;

        if (anchor) {{
          setTimeout(() => {{
            scrollToAnchorInModal(anchor);
          }}, 180);
        }} else {{
          body.scrollTo({{ top: 0, behavior: 'instant' }});
        }}
      }}).catch(err => {{
        body.innerHTML = '<div class="qa-card"><div class="qa-card-body">Could not load document (' + escapeHtml(path) + '). View on <a href="' + ghBtn.href + '" target="_blank" style="color:#60a5fa">GitHub</a>.</div></div>';
      }});
    }}

    function navigateChapter(index) {{
      if (index < 0 || index >= CURRICULUM_CHAPTERS.length) return;
      const target = CURRICULUM_CHAPTERS[index];
      openStandaloneDoc(target.path, 'Chapter ' + target.num + ': ' + target.title);
    }}
"""

def build():
    # 1. Get base pristine template 47563f8:index.html to prevent duplicate injections
    head_html = subprocess.check_output(
        ["git", "show", "47563f8:index.html"],
        text=True,
        cwd=str(REPO_ROOT),
    )

    # 2. Read marked.umd.js
    marked_code = (REPO_ROOT / "assets" / "js" / "marked.umd.js").read_text(encoding="utf-8")

    interview_hub_html = generate_interview_hub_html()
    guides_view_html = generate_guides_view_html()

    # Add CSS
    head_html = head_html.replace('</style>', extra_css + '\n  </style>')

    # Replace domain button with GitHub repository link
    head_html = re.sub(
        r'<button class="btn-domain-info"[^>]*>.*?</button>',
        '<a href="https://github.com/sameerkarur/Data_science" target="_blank" class="nav-github-link">⭐ GitHub Repository ↗</a>',
        head_html,
    )

    # Update stat: 300+ -> 990+
    head_html = head_html.replace(
        '300+</span><span class="stat-lbl">Interview Flashcards',
        '990+</span><span class="stat-lbl">Interview Flashcards',
    )

    # Clean Colab button
    clean_colab_btn = """          <a id="studio-colab-link" class="btn-colab-launch" href="#" target="_blank" rel="noopener noreferrer" title="Run in Google Colab">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style="vertical-align:middle"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 15h-2v-6h2v6zm4 0h-2v-6h2v6z" fill="#f59e0b"/></svg>
            <span>Run in Google Colab</span>
          </a>"""
    head_html = re.sub(
        r'<a id="studio-colab-link"[^>]*>[\s\S]*?</a>',
        clean_colab_btn,
        head_html,
        count=1,
    )

    # Remove domain modal
    head_html = re.sub(
        r'<!-- CUSTOM DOMAIN & FREE HOSTING MODAL -->\s*<div id="domain-modal"[\s\S]*?</div>\s*</div>',
        '',
        head_html,
    )

    # Insert Interview Hub & Master Textbook Guides
    head_html = head_html.replace(
        '</main>',
        f'{interview_hub_html}\n{guides_view_html}\n  </main>',
    )

    # Enhance stat cards to be interactive
    head_html = head_html.replace(
        '<div class="stat-card"><span class="stat-val">1,650+</span><span class="stat-lbl">Practice Problems</span></div>',
        '<div class="stat-card" onclick="switchNavSection(\'courses\', document.querySelectorAll(\'.nav-tab\')[1])" title="View all course practice problems"><span class="stat-val">1,650+</span><span class="stat-lbl">Practice Problems</span></div>',
    )
    head_html = head_html.replace(
        '<div class="stat-card"><span class="stat-val">990+</span><span class="stat-lbl">Interview Flashcards</span></div>',
        '<div class="stat-card" onclick="switchNavSection(\'interview\', document.querySelectorAll(\'.nav-tab\')[4])" title="Open 33-module technical interview vault"><span class="stat-val">990+</span><span class="stat-lbl">Interview Flashcards</span></div>',
    )
    head_html = head_html.replace(
        '<div class="stat-card"><span class="stat-val">15</span><span class="stat-lbl">Next-Gen V2 Projects</span></div>',
        '<div class="stat-card" onclick="switchNavSection(\'v2\', document.querySelectorAll(\'.nav-tab\')[2])" title="Explore Version 2 alternative architectures"><span class="stat-val">15</span><span class="stat-lbl">Next-Gen V2 Projects</span></div>',
    )
    head_html = head_html.replace(
        '<div class="stat-card"><span class="stat-val">3</span><span class="stat-lbl">Capstone Deliverables</span></div>',
        '<div class="stat-card" onclick="switchNavSection(\'capstones\', document.querySelectorAll(\'.nav-tab\')[3])" title="Explore Capstone industrial projects"><span class="stat-val">3</span><span class="stat-lbl">Capstone Deliverables</span></div>',
    )

    # Remove openDomainModal and closeDomainModal if present
    head_html = re.sub(
        r'function openDomainModal\(\)\s*\{[\s\S]*?function closeDomainModal\([^\)]*\)\s*\{[\s\S]*?\n    \}',
        '',
        head_html,
    )
    head_html = head_html.replace('closeDomainModal();', '')

    # Fix relative project links
    head_html = head_html.replace(
        'href="01_IITK_AIML_Foundations_Programming_Refresher/projects/Personal_Expense_Tracker/" target="_blank">View Project Folder</a>',
        'href="https://github.com/sameerkarur/Data_science/tree/main/01_IITK_AIML_Foundations_Programming_Refresher/projects/Personal_Expense_Tracker" target="_blank">View on GitHub</a>'
    )
    head_html = head_html.replace(
        'href="01_IITK_AIML_Foundations_Programming_Refresher/projects/Task_Manager/" target="_blank">View Project Folder</a>',
        'href="https://github.com/sameerkarur/Data_science/tree/main/01_IITK_AIML_Foundations_Programming_Refresher/projects/Task_Manager" target="_blank">View on GitHub</a>'
    )

    # Add Learning Journey tab to navigation (Single instance)
    head_html = head_html.replace(
        '<button class="nav-tab" onclick="switchNavSection(\'theory\', this)">📖 Master Guides</button>',
        '<button class="nav-tab" onclick="switchNavSection(\'theory\', this)">📖 Master Guides & Textbook</button>\n      <button class="nav-tab" onclick="openStandaloneDoc(\'LEARNING_JOURNEY.md\', \'Learning Journey, Experience & Simplilearn Attribution\')">🎓 Learning Journey</button>'
    )

    # Update title and header branding
    head_html = head_html.replace(
        '<title>AI/ML Practice Academy — Sameer Karur</title>',
        '<title>AI/ML Practice Academy — Machine Learning & Generative AI Studio</title>'
    )
    head_html = head_html.replace(
        '<span>IITK AIML Professional Certificate · Sameer Karur</span>',
        '<span>IITK AIML Professional Certificate · Comprehensive Learning Studio</span>'
    )

    # Replace footer with comprehensive attribution and gratitude block
    attribution_footer = """  <!-- CURRICULUM ATTRIBUTION & FAIR USE FOOTER -->
  <footer class="app-attribution-footer">
    <div class="attribution-card">
      <div class="attribution-badge">🎓 Academic Collaboration & Coursework Synthesis</div>
      <h4 class="attribution-title">Learning Journey, Educational Synthesis & Simplilearn Attribution</h4>
      <p class="attribution-desc">
        This interactive studio, 1,650+ practice problems, 990+ interview flashcards, 33-chapter textbook handbooks, and dual-architecture project implementations were independently synthesized and engineered as a self-study mastery resource while completing the <strong>Professional Certificate Program in Generative AI and Machine Learning</strong>, delivered by <strong>Simplilearn</strong> in academic collaboration with <strong>E&ICT Academy, IIT Kanpur</strong>.
      </p>
      <p class="attribution-sub">
        Special thanks and profound gratitude to <strong>Simplilearn</strong> for providing the structured curriculum, live faculty mentorship, and industrial problem statements, and to <strong>IIT Kanpur</strong> for foundational theoretical guidance. All curriculum titles, program syllabi, and trademarks belong to their respective holders. Published strictly under <strong>Educational Fair Use</strong>.
      </p>
      <div class="attribution-actions">
        <button class="btn-attribution-modal" onclick="openStandaloneDoc('LEARNING_JOURNEY.md', 'Learning Journey, Experience & Simplilearn Attribution')">
          📖 Read Full Learning Journey & Experience Writeup ↗
        </button>
        <a href="https://github.com/sameerkarur/Data_science" target="_blank" class="nav-github-link">
          ⭐ Star on GitHub (Support More Free Educational Guides)
        </a>
      </div>
    </div>
  </footer>"""

    head_html = re.sub(
        r'<footer>[\s\S]*?</footer>',
        attribution_footer,
        head_html,
        count=1,
    )

    # Replace switchNavSection in script
    old_switch_re = re.compile(
        r'function switchNavSection\(sec, btn\)\s*\{[\s\S]*?\n    \}',
        re.DOTALL,
    )
    assert old_switch_re.search(head_html), "Could not find old switchNavSection in head_html!"
    head_html = old_switch_re.sub(
        lambda m: updated_switch_nav.strip(),
        head_html,
    )

    # Replace parseMarkdown in script
    old_parse_re = re.compile(
        r'function parseMarkdown\(md\)\s*\{[\s\S]*?\}\s*function parseInline',
        re.DOTALL,
    )
    head_html = old_parse_re.sub(
        lambda m: marked_parser_logic + '\n    function parseInline',
        head_html,
    )

    # Replace openStandaloneDoc in script to avoid duplicate declaration
    old_open_doc_re = re.compile(
        r'function openStandaloneDoc\(path, title\)\s*\{[\s\S]*?\n    \}',
        re.DOTALL,
    )
    head_html = old_open_doc_re.sub('', head_html)

    # Ensure robust local-first + cache-busted fetch in fetchFile
    old_fetch_re = re.compile(
        r'function fetchFile\(path\)\s*\{[\s\S]*?\n    \}',
        re.DOTALL,
    )
    new_fetch_func = """function fetchFile(path) {
      // 1. Try local/relative fetch first (immediate & always freshest when served locally or on GitHub Pages)
      return fetch(path, { cache: 'no-cache' }).then(res => {
        if (!res.ok) throw new Error('Relative fetch failed');
        return res.text();
      }).catch(() => {
        // 2. Fallback to raw github with cache-busting timestamp
        const rawUrl = `https://raw.githubusercontent.com/${GITHUB_REPO}/${BRANCH}/${path}?v=${Date.now()}`;
        return fetch(rawUrl).then(res => {
          if (!res.ok) throw new Error('Raw fetch failed');
          return res.text();
        });
      });
    }"""
    head_html = old_fetch_re.sub(new_fetch_func, head_html)

    # Prepend marked_code at start of script
    head_html = head_html.replace(
        '<script>',
        '<script>\n' + marked_code + '\n',
    )

    # Verify script syntax with node
    script_match = re.search(r'<script>(.*?)</script>', head_html, re.DOTALL)
    if not script_match:
        raise RuntimeError("No <script> tag found in generated HTML")

    test_js_path = Path("/tmp/test_assembled.js")
    test_js_path.write_text(script_match.group(1), encoding="utf-8")

    res = subprocess.run(["node", "-c", str(test_js_path)], capture_output=True, text=True)
    if res.returncode != 0:
        print("❌ JavaScript syntax error in assembled index.html:\n", res.stderr)
        return False

    print("🎉 JavaScript syntax is 100% PERFECTLY VALID in assembled index.html!")

    # Write out index.html
    (REPO_ROOT / "index.html").write_text(head_html, encoding="utf-8")
    print(f"✅ Successfully wrote {len(head_html)} bytes to {REPO_ROOT / 'index.html'}!")
    return True

if __name__ == "__main__":
    build()
