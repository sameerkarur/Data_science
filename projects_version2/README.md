# Projects Version 2: Next-Generation Alternative Implementations
**Advanced Architectures, Alternative Algorithmic Paradigms & Production Patterns**  
*Professional Certificate Course in Generative AI and Machine Learning — E&ICT Academy, IIT Kanpur*  
*Author: Sameer Karur*

---

## 🎯 Executive Overview

While the standard program deliverables (Version 1) in each course folder address the foundational curriculum requirements, **Projects Version 2** demonstrates mastery through **alternative, modern, and production-ready architectural paradigms**.

Every single project from Course 1 through Course 7 Capstone has been re-architected with:
1. **Alternative Algorithmic Families:** (e.g., HDBSCAN vs K-Means, Tabular ResNet/TabNet vs standard MLPs, Hybrid Dense-Sparse RRF vs naive vector RAG, BPR Graph Ranking vs SVD).
2. **Modern Software Engineering Patterns:** Event-sourcing, Priority-Queue DAG schedulers, Multi-Agent StateGraphs, and Swarm intelligence.
3. **Production Robustness & Explainability:** SHAP feature attribution, Grad-CAM visual heatmaps, probabilistic forecast intervals, and hallucination guardrails.

---

## 🗺️ Project Index (15 Alternative Architectures)

| # | Project Directory | Domain & Problem | Version 1 Baseline | Version 2 Advanced Paradigm |
|---|---|---|---|---|
| **01** | [`01_Python_Expense_Tracker_V2/`](01_Python_Expense_Tracker_V2/) | Personal Finance Engine | Basic CLI script + CSV | Event-Sourced SQLite Engine + Real-time Budget Velocity Alerts |
| **02** | [`02_Python_Task_Manager_V2/`](02_Python_Task_Manager_V2/) | Task Scheduler & Auth | JSON store + SHA256 | Priority-Queue DAG Dependency Scheduler + Salted HMAC RBAC |
| **03** | [`03_DS_Sales_Analysis_Cohort_RFM_V2/`](03_DS_Sales_Analysis_Cohort_RFM_V2/) | Retail Sales Intelligence | Pandas groupby + Bar plots | Multi-Cohort Retention + RFM Segmentation + Price Elasticity |
| **04** | [`04_DS_Marketing_Campaign_Uplift_Attribution_V2/`](04_DS_Marketing_Campaign_Uplift_Attribution_V2/) | Marketing Analytics | Descriptive response plots | Two-Model Uplift Modeling + Markov Chain Multi-Touch Attribution |
| **05** | [`05_ML_Spotify_Cohorts_HDBSCAN_PCA_V2/`](05_ML_Spotify_Cohorts_HDBSCAN_PCA_V2/) | Audio & Music Clustering | K-Means (4 features) | Density-Based HDBSCAN + PCA/t-SNE + Mood Playlist Generator |
| **06** | [`06_ML_Employee_Attrition_Explainable_Boosting_V2/`](06_ML_Employee_Attrition_Explainable_Boosting_V2/) | Human Resources Retention | Random Forest + SMOTE | Explainable XGBoost + Bayesian Tuning + SHAP Attribution + Survival |
| **07** | [`07_DL_Lending_Club_Tabular_ResNet_V2/`](07_DL_Lending_Club_Tabular_ResNet_V2/) | Credit Risk Default Prediction | Sequential Dense Keras MLP | Deep Tabular ResNet + Entity Embeddings + Focal Loss + MC Dropout |
| **08** | [`08_DL_Home_Loan_Risk_TabNet_V2/`](08_DL_Home_Loan_Risk_TabNet_V2/) | Mortgage Risk Classification | Basic Multilayer Perceptron | TabNet Attentive Network with Feature Masking & Calibrated Probs |
| **09** | [`09_GenAI_Storytelling_MultiAgent_StateGraph_V2/`](09_GenAI_Storytelling_MultiAgent_StateGraph_V2/) | Interactive Branching Fiction | Single-prompt iterative loop | Multi-Agent StateGraph Engine + Memory Vector Store + Tension Metric |
| **10** | [`10_GenAI_Virtual_PMO_Swarm_Simulation_V2/`](10_GenAI_Virtual_PMO_Swarm_Simulation_V2/) | Autonomous Agile PMO | Single persona system prompt | Multi-Role PMO Swarm (Coach, Risk, Architect) + Monte Carlo Burndowns |
| **11** | [`11_AdvGenAI_Hybrid_Dense_Sparse_RAG_V2/`](11_AdvGenAI_Hybrid_Dense_Sparse_RAG_V2/) | Enterprise Policy Q&A | Dense Vector Search (Chroma) | Hybrid BM25 + Dense RRF + Cross-Encoder Re-Ranker + Guardrails |
| **12** | [`12_AdvGenAI_Multimodal_Creative_Studio_V2/`](12_AdvGenAI_Multimodal_Creative_Studio_V2/) | Ad Campaign Creative Studio | Single DALL-E prompt call | Multi-Aspect Ratio Pipeline + Style Transfer + Typography Composition |
| **13** | [`13_Capstone1_Autonomous_Perception_ViT_EfficientNet_V2/`](13_Capstone1_Autonomous_Perception_ViT_EfficientNet_V2/) | Vehicle Perception & Safety | MobileNetV2 + Basic EDA | EfficientNet-B0 Backbone + Grad-CAM Heatmaps + Spatio-Temporal Risk |
| **14** | [`14_Capstone2_Hierarchical_Probabilistic_Demand_Forecasting_V2/`](14_Capstone2_Hierarchical_Probabilistic_Demand_Forecasting_V2/) | Restaurant Demand Forecasting | Single-model XGBoost | Hierarchical Stacking Ensemble (CatBoost + LightGBM + Ridge) + Intervals |
| **15** | [`15_Capstone3_Dual_Vision_BPR_Recommender_V2/`](15_Capstone3_Dual_Vision_BPR_Recommender_V2/) | Cultural Heritage Tourism | MobileNetV2 + SVD Collab | ResNet50 Landmark Vision + Bayesian Personalized Ranking (BPR) Graph |

---

## 🚀 Execution & Environment

All projects are designed to execute seamlessly with the shared workspace dependencies:

```bash
# Set library path for OpenMP / XGBoost / TensorFlow
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Run any project's executable implementation using the workspace virtualenv:
/Users/sameerkarur/Documents/Git/Data_science/.venv_dl/bin/python <project_folder>/run_v2.py
```

Each project folder contains:
1. `run_v2.py` / `.ipynb` — fully executable standalone Python implementation with self-contained dataset fallbacks or direct links to `datasets/shared/`.
2. `README.md` — deep architectural comparison, mathematical formulation, benchmark metrics, and production notes.
