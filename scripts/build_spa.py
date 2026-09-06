"""
Self-contained build engine for AI/ML Practice Academy Single Page Application (index.html).
Features:
- Pure offline bundled marked.umd.js from OpenCareerAI.
- 33 interactive interview flashcard topic decks with 990 verified technical questions.
- Master learning guides view and dual-architecture project portfolio.
- Zero extraneous custom domain buttons or modals.
- Clean Google Colab execution buttons.
- Node.js syntax-verified JavaScript.
"""

import subprocess
import re
import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. 33 Curated Interview Topics
INTERVIEW_TOPICS = [
    # Course 1: Python Foundations
    ("Variables, Memory & Data Types", "01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes", "Python Foundations", "python", "30 High-Yield Questions", "Deep memory model, mutable vs immutable, interned strings, walrus operator, and object layout."),
    ("Control Flow, Functions & Scopes", "01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions", "Python Foundations", "python", "30 High-Yield Questions", "LEGB scope hierarchy, closures, decorators, generator internals, and pass-by-object-reference."),
    ("Data Structures & Algorithms", "01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures", "Python Foundations", "python", "30 High-Yield Questions", "List array over-allocation, hash table collision resolution, deques, heaps, and Timsort."),
    ("OOP & Modular Architecture", "01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules", "Python Foundations", "python", "30 High-Yield Questions", "C3 MRO linearization, metaclasses, descriptors, __slots__, and circular import management."),
    ("File I/O, Exceptions & Robust Systems", "01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions", "Python Foundations", "python", "30 High-Yield Questions", "Context managers (__enter__/__exit__), memory mapping (mmap), exception chaining, and atomic I/O."),
    
    # Course 2: Applied Data Science
    ("Data Science Principles & Lifecycle", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science", "Applied Data Science", "ds", "30 High-Yield Questions", "CRISP-DM lifecycle, data leakage prevention, survivorship bias, and metric alignment."),
    ("Python Essentials for Data Science", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials", "Applied Data Science", "ds", "30 High-Yield Questions", "Vectorization mechanics, SIMD parallelization, iterator streaming, and memory profiling."),
    ("NumPy Numerical Computing", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy", "Applied Data Science", "ds", "30 High-Yield Questions", "Strides, memory layouts (C vs Fortran), broadcasting arithmetic rules, and memory views vs copies."),
    ("Linear Algebra for Machine Learning", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra", "Applied Data Science", "ds", "30 High-Yield Questions", "Eigenvalues, eigenvectors, SVD decomposition, geometric transformations, and condition numbers."),
    ("Statistical Foundations", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals", "Applied Data Science", "ds", "30 High-Yield Questions", "Central Limit Theorem, sampling distributions, skewness, kurtosis, and robust estimators."),
    ("Probability & Distributions", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions", "Applied Data Science", "ds", "30 High-Yield Questions", "Bayes' theorem, probability mass vs density functions, Poisson, Gaussian, and log-normal models."),
    ("Advanced Inferential Statistics", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics", "Applied Data Science", "ds", "30 High-Yield Questions", "Hypothesis testing, p-value misinterpretations, Type I/II errors, ANOVA, and bootstrap confidence intervals."),
    ("Pandas High-Performance Manipulation", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas", "Applied Data Science", "ds", "30 High-Yield Questions", "BlockManager memory internals, index alignment, merge vs join strategies, categorical performance."),
    ("Data Wrangling & Cleaning", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling", "Applied Data Science", "ds", "30 High-Yield Questions", "Missing data mechanisms (MCAR/MAR/MNAR), outlier detection (IQR, Isolation Forests), transformations."),
    ("Data Visualization Architecture", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization", "Applied Data Science", "ds", "30 High-Yield Questions", "Matplotlib Artist hierarchy, perception-driven color palettes, Seaborn statistical aggregation."),
    ("Matplotlib Deep-Dive", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib", "Applied Data Science", "ds", "30 High-Yield Questions", "Figure vs Axes objects, custom tick formatters, twin axes, and vector graphic rendering engines."),
    ("Seaborn Statistical Graphics", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn", "Applied Data Science", "ds", "30 High-Yield Questions", "FacetGrid, PairGrid architectures, KDE estimations, and distribution visualization."),
    ("Regex, JSON & Web API Extraction", "02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis", "Applied Data Science", "ds", "30 High-Yield Questions", "Backtracking algorithms, lookarounds, streaming JSON parsers, and exponential backoff retry patterns."),

    # Course 3: Machine Learning
    ("EDA & Advanced Feature Engineering", "03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering", "Machine Learning", "ml", "30 High-Yield Questions", "Target encoding with smoothing, interaction features, curse of dimensionality, and permutation importance."),
    ("Clustering & Unsupervised Learning", "03_IITK_AIML_Core_Machine_Learning/02_clustering", "Machine Learning", "ml", "30 High-Yield Questions", "K-Means++ seeding, Voronoi partitions, Silhouette analysis, HDBSCAN density clustering."),
    ("Classification Algorithms", "03_IITK_AIML_Core_Machine_Learning/03_classification", "Machine Learning", "ml", "30 High-Yield Questions", "Logistic regression odds ratios, SVM margin maximization, Random Forest bagging vs XGBoost gradient boosting."),
    ("Class Imbalance Mitigation", "03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data", "Machine Learning", "ml", "30 High-Yield Questions", "SMOTE synthetic generation, Focal Loss, cost-sensitive matrix optimization, PR-AUC vs ROC-AUC."),
    ("Model Evaluation & Validation Rigor", "03_IITK_AIML_Core_Machine_Learning/05_model_evaluation", "Machine Learning", "ml", "30 High-Yield Questions", "Stratified K-Fold, Purged Time-Series split, Brier Score calibration, and bias-variance decomposition."),

    # Course 4: Deep Learning
    ("Neural Network Fundamentals", "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics", "Deep Learning", "dl", "30 High-Yield Questions", "Backpropagation chain rule, vanishing/exploding gradients, Xavier/He weight initialization, activation functions."),
    ("Keras & TensorFlow Architecture", "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow", "Deep Learning", "dl", "30 High-Yield Questions", "Sequential vs Functional vs Subclassing APIs, custom GradientTape training loops, tf.data pipeline optimization."),
    ("Deep Learning Data Pipelines & Imbalance", "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance", "Deep Learning", "dl", "30 High-Yield Questions", "Data augmentation invariance, class-weighted cross-entropy, mixed-precision (FP16/BF16) training."),
    ("Deep Learning Evaluation & Interpretability", "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl", "Deep Learning", "dl", "30 High-Yield Questions", "Grad-CAM visual explanations, saliency maps, Monte Carlo Dropout uncertainty estimation, ROC/PR curves."),

    # Course 5: Essentials of Generative AI
    ("Prompt Engineering Foundations", "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering", "Generative AI", "genai", "30 High-Yield Questions", "Zero-shot, Few-shot, Chain-of-Thought (CoT), Tree of Thoughts (ToT), prompt injection defenses."),
    ("ChatGPT & Enterprise LLM Applications", "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications", "Generative AI", "genai", "30 High-Yield Questions", "Tool use / Function calling, structured JSON output enforcement, temperature vs top_p, context window limits."),
    ("GenAI Optimization & Fine-Tuning", "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization", "Generative AI", "genai", "30 High-Yield Questions", "LoRA / QLoRA low-rank adaptation, PEFT architectures, quantization (INT8/INT4), RLHF and DPO alignments."),

    # Course 6: Advanced Generative AI & RAG
    ("RAG Architectures & Retrieval Engineering", "06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures", "Advanced GenAI & RAG", "rag", "30 High-Yield Questions", "Dense vs sparse retrieval, hybrid search with Reciprocal Rank Fusion (RRF), Cross-Encoder re-ranking, HyDE."),
    ("Vector Databases & High-Dimensional Indexing", "06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma", "Advanced GenAI & RAG", "rag", "30 High-Yield Questions", "HNSW graph search, IVFFlat inverted indices, cosine similarity vs dot product, metadata filtering."),
    ("Multimodal Generative Models", "06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models", "Advanced GenAI & RAG", "rag", "30 High-Yield Questions", "Latent Diffusion Models, CLIP joint embeddings, Classifier-Free Guidance (CFG), ControlNet conditioning.")
]

def interview_hub_card(title: str, folder_rel: str, domain_label: str, domain_class: str, q_badge: str, desc: str) -> str:
    p_path = f"{folder_rel}/practice.ipynb"
    s_path = f"{folder_rel}/solutions.ipynb"
    b_path = f"{folder_rel}/basics.md"
    q_path = f"{folder_rel}/interview_qa.md"
    topic_key = folder_rel.replace('/', '_').replace('-', '_')
    escaped_title = html.escape(title)
    escaped_desc = html.escape(desc)

    return f"""          <div class="interview-card" data-domain="{domain_class}">
            <div class="interview-card-top">
              <span class="domain-tag domain-{domain_class}">{domain_label}</span>
              <span class="q-count-badge">{q_badge}</span>
            </div>
            <h4 class="interview-card-title">{escaped_title}</h4>
            <p class="interview-card-desc">{escaped_desc}</p>
            <div class="interview-card-meta">
              <span class="mastery-pill" id="mastery-badge-{topic_key}">0 / 30 Mastered</span>
            </div>
            <div class="interview-card-actions">
              <button class="btn-hub-launch" onclick="openStudio('{p_path}', '{escaped_title}', '{s_path}', '{b_path}', '{q_path}', 'qa')">
                <span>🎯</span> Open Flashcards Studio
              </button>
              <button class="btn-hub-lab" onclick="openStudio('{p_path}', '{escaped_title}', '{s_path}', '{b_path}', '{q_path}', 'notebook')">
                💻 Run Lab Notebook
              </button>
            </div>
          </div>"""

def generate_interview_hub_html() -> str:
    cards = []
    for title, folder_rel, domain_label, domain_class, q_badge, desc in INTERVIEW_TOPICS:
        cards.append(interview_hub_card(title, folder_rel, domain_label, domain_class, q_badge, desc))
    
    cards_str = "\n".join(cards)
    return f"""    <!-- DEDICATED FULL INTERVIEW HUB VIEW -->
    <div id="section-interview-hub" style="display:none">
      <div class="hub-hero">
        <span class="hub-hero-badge">💼 FAANG & Tier-1 AI Tech Interview Preparation</span>
        <h2 class="hub-hero-title">Technical Interview Flashcard Vault</h2>
        <p class="hub-hero-sub">
          Master 990+ high-yield algorithmic, mathematical, and architectural questions curated across the entire curriculum.
          Practice with instant answer reveals, full conceptual derivations, and persistent mastery tracking.
        </p>

        <!-- Topic Filter Pills -->
        <div class="hub-filter-pills">
          <button class="hub-pill active" onclick="filterInterviewHub('all', this)">All Categories (33 Modules)</button>
          <button class="hub-pill" onclick="filterInterviewHub('python', this)">Python Foundations (150 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('ds', this)">Data Science & Stats (390 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('ml', this)">Machine Learning (150 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('dl', this)">Deep Learning (120 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('genai', this)">Generative AI (90 Q)</button>
          <button class="hub-pill" onclick="filterInterviewHub('rag', this)">Advanced GenAI & RAG (90 Q)</button>
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
    return """    <!-- DEDICATED MASTER GUIDES VIEW -->
    <div id="section-guides-group" style="display:none">
      <div class="hub-hero">
        <span class="hub-hero-badge">📚 Architectural Reference & Deep Learning Manuals</span>
        <h2 class="hub-hero-title">Curriculum Master Guides & Cheatsheets</h2>
        <p class="hub-hero-sub">
          Authoritative technical specifications, comprehensive mathematical formulations, and engineering guides.
        </p>
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
      </div>
    </div>"""

# 2. OpenCareerAI Modern Design Stylesheet Additions
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

    .nav-github-link {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.85rem;
      border-radius: 0.5rem;
      background: #1e293b;
      color: #f8fafc;
      font-size: 0.82rem;
      font-weight: 600;
      text-decoration: none;
      border: 1px solid #334155;
      transition: all 0.2s;
    }
    .nav-github-link:hover {
      background: #334155;
      border-color: #38bdf8;
      color: #38bdf8;
    }

    /* Hub Hero */
    .hub-hero {
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
      border: 1px solid #1e293b;
      border-radius: 1rem;
      padding: 2.25rem 2rem;
      margin-bottom: 2rem;
      text-align: center;
    }
    .hub-hero-badge {
      display: inline-block;
      padding: 0.3rem 0.85rem;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.1);
      color: #38bdf8;
      font-size: 0.82rem;
      font-weight: 600;
      border: 1px solid rgba(56, 189, 248, 0.25);
      margin-bottom: 0.85rem;
    }
    .hub-hero-title {
      font-size: 2rem;
      font-weight: 800;
      color: #f8fafc;
      margin-bottom: 0.5rem;
      letter-spacing: -0.025em;
    }
    .hub-hero-sub {
      color: #94a3b8;
      font-size: 0.98rem;
      max-width: 780px;
      margin: 0 auto 1.75rem auto;
      line-height: 1.6;
    }
    .hub-filter-pills {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 0.5rem;
      margin-bottom: 1.75rem;
    }
    .hub-pill {
      padding: 0.45rem 0.95rem;
      border-radius: 0.5rem;
      background: #1e293b;
      color: #94a3b8;
      font-size: 0.82rem;
      font-weight: 600;
      border: 1px solid #334155;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .hub-pill:hover {
      background: #334155;
      color: #f8fafc;
    }
    .hub-pill.active {
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border-color: rgba(56, 189, 248, 0.45);
      box-shadow: 0 0 12px -2px rgba(56, 189, 248, 0.2);
    }

    .hub-master-card {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.5rem;
      text-align: left;
      background: #0f172a;
      border: 1px solid rgba(245, 158, 11, 0.35);
      border-radius: 0.875rem;
      padding: 1.35rem 1.75rem;
      cursor: pointer;
      transition: all 0.2s ease;
      box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.4);
    }
    .hub-master-card:hover {
      border-color: #f59e0b;
      transform: translateY(-2px);
      box-shadow: 0 8px 24px -4px rgba(245, 158, 11, 0.2);
    }
    .hub-master-badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #fbbf24;
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.25);
      padding: 0.2rem 0.5rem;
      border-radius: 0.375rem;
      margin-bottom: 0.35rem;
    }
    .hub-master-left h4 {
      font-size: 1.1rem;
      color: #f8fafc;
      margin-bottom: 0.25rem;
    }
    .hub-master-left p {
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.5;
      margin: 0;
    }
    .btn-hub-master {
      flex-shrink: 0;
      padding: 0.65rem 1.15rem;
      border-radius: 0.5rem;
      background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
      color: #fff;
      font-size: 0.85rem;
      font-weight: 700;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-hub-master:hover {
      filter: brightness(1.15);
    }

    /* Interview Grid */
    .interview-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1.25rem;
      margin-top: 1rem;
    }
    .interview-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 0.875rem;
      padding: 1.35rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
      position: relative;
    }
    .interview-card:hover {
      border-color: #38bdf8;
      transform: translateY(-3px);
      box-shadow: 0 12px 28px -6px rgba(56, 189, 248, 0.15);
    }
    .interview-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .domain-tag {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 0.2rem 0.5rem;
      border-radius: 0.375rem;
    }
    .domain-python { background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.25); }
    .domain-ds { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
    .domain-ml { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
    .domain-dl { background: rgba(236, 72, 153, 0.12); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.25); }
    .domain-genai { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.25); }
    .domain-rag { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }

    .q-count-badge {
      font-size: 0.75rem;
      font-weight: 600;
      color: #94a3b8;
      background: #1e293b;
      padding: 0.2rem 0.5rem;
      border-radius: 0.375rem;
    }
    .interview-card-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 0.5rem;
      line-height: 1.35;
    }
    .interview-card-desc {
      font-size: 0.85rem;
      color: #94a3b8;
      line-height: 1.5;
      margin-bottom: 1rem;
      flex-grow: 1;
    }
    .interview-card-meta {
      margin-bottom: 0.85rem;
    }
    .mastery-pill {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.2rem 0.6rem;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.08);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .interview-card-actions {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .btn-hub-launch {
      width: 100%;
      padding: 0.6rem 0.85rem;
      border-radius: 0.5rem;
      background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
      color: #fff;
      font-size: 0.85rem;
      font-weight: 600;
      border: none;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
    }
    .btn-hub-launch:hover {
      filter: brightness(1.15);
      box-shadow: 0 4px 14px -2px rgba(37, 99, 235, 0.4);
    }
    .btn-hub-lab {
      width: 100%;
      padding: 0.45rem 0.85rem;
      border-radius: 0.5rem;
      background: transparent;
      color: #94a3b8;
      font-size: 0.8rem;
      font-weight: 600;
      border: 1px solid #334155;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-hub-lab:hover {
      background: #1e293b;
      color: #f8fafc;
      border-color: #64748b;
    }

    /* Master Guides Grid */
    .guides-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 1.5rem;
      margin-top: 1rem;
    }
    .guide-card {
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 1rem;
      padding: 1.75rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .guide-card:hover {
      border-color: #38bdf8;
      transform: translateY(-3px);
      box-shadow: 0 12px 28px -6px rgba(56, 189, 248, 0.15);
    }
    .guide-card-icon {
      font-size: 2rem;
      margin-bottom: 0.75rem;
    }
    .guide-card-title {
      font-size: 1.2rem;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 0.5rem;
    }
    .guide-card-desc {
      font-size: 0.88rem;
      color: #94a3b8;
      line-height: 1.55;
      margin-bottom: 1.25rem;
      flex-grow: 1;
    }
    .guide-card-meta {
      font-size: 0.75rem;
      font-family: var(--font-mono, monospace);
      color: #64748b;
      margin-bottom: 1rem;
    }
    .btn-guide-launch {
      padding: 0.65rem 1rem;
      border-radius: 0.5rem;
      background: #1e293b;
      color: #38bdf8;
      font-size: 0.85rem;
      font-weight: 600;
      border: 1px solid rgba(56, 189, 248, 0.25);
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-guide-launch:hover {
      background: rgba(56, 189, 248, 0.15);
      border-color: #38bdf8;
    }
"""

updated_switch_nav = """
    // Navigation & Section Filter
    function switchNavSection(sec, btn) {
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
      document.querySelectorAll('.hub-pill').forEach(p => p.classList.remove('active'));
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

marked_parser_logic = """
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        gfm: true,
        breaks: true
      });
    }

    function parseMarkdown(md) {
      if (!md) return '';
      if (typeof marked !== 'undefined' && marked.parse) {
        try {
          return `<div class="doc-viewer">${marked.parse(md)}</div>`;
        } catch(e) {
          console.error("Marked parse error:", e);
        }
      }
      return `<div class="doc-viewer"><pre>${escapeHtml(md)}</pre></div>`;
    }
"""

def build():
    # 1. Get HEAD:index.html
    head_html = subprocess.check_output(
        ["git", "show", "HEAD:index.html"],
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

    # Insert Interview Hub & Guides
    head_html = head_html.replace(
        '</main>',
        f'{interview_hub_html}\n{guides_view_html}\n  </main>',
    )

    # Replace switchNavSection in script
    old_switch_re = re.compile(
        r'function switchNavSection\(sec, btn\)\s*\{[\s\S]*?\}\s*function handleGlobalSearch',
        re.DOTALL,
    )
    head_html = old_switch_re.sub(
        lambda m: updated_switch_nav + '\n\n    function handleGlobalSearch',
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
