# Elective 04: Cloud AI & Enterprise ML Engineering (Azure AI)
**E&ICT Academy, IIT Kanpur — Specialization Syllabus & Engineering Guide**

---

## 📌 Domain Overview

Enterprise AI production systems demand robust cloud infrastructure, reproducible continuous delivery (MLOps) pipelines, and stringent governance frameworks. This elective explores the architecture, orchestration, and security frameworks required to engineer, deploy, and monitor production machine learning and generative AI workloads on Microsoft Azure.

---

## 🧭 Specialization Architecture & Curriculum Roadmap

### 1. Azure AI Platform & Infrastructure Architecture
- **Azure Machine Learning (Azure ML) Workspaces:** Hub and project resource model, unified role-based access control (RBAC), and virtual network (VNet) isolation.
- **Compute Infrastructure:** Low-priority vs. dedicated compute instances, multi-node GPU clusters (NVIDIA A100/H100), and automated autoscaling triggers.
- **Data Asset Management:** Cloud datastores (Azure Blob, Data Lake Gen2), versioned datasets, and feature store architectures with Feast.

### 2. Cognitive Services & Foundation Model Orchestration
- **Azure OpenAI Service:** Enterprise deployment of GPT-4o, embeddings, and multimodal models with private endpoints and SOC2/HIPAA compliance.
- **Azure AI Search (formerly Cognitive Search):** Hybrid search architecture integrating BM25 keyword matching, vector dense retrieval, semantic re-ranking, and index partitioning.
- **Computer Vision & Speech Services:** Azure AI Custom Vision for transfer learning, Document Intelligence (OCR & layout analysis), and Neural Speech pipelines.

### 3. Production MLOps Pipelines & CI/CD
- **Automated Workflows:** Azure ML Pipelines defined in YAML and Python SDK v2 for automated data preprocessing, distributed training, and validation gates.
- **Experiment Tracking & Governance:** MLflow native integration logging hyperparameter sweeps, metric curves, model artifacts, and environment containers.
- **Deployment Strategies:** Managed Online Endpoints (MOE) with blue/green deployment, canary traffic routing, and auto-rollback thresholds.

### 4. Responsible AI, Safety & Compliance
- **Content Safety Guardrails:** Azure AI Content Safety API filtering hate speech, self-harm, sexual, and violence vectors in real-time streaming prompts.
- **Model Explainability & Fairlearn:** SHAP/LIME integration, disparity metric assessment, and demographic parity evaluation.
- **Data Privacy & Governance:** Customer-managed encryption keys (CMEK), zero-data-retention agreements for LLM inference, and audit logging with Azure Monitor.

---

## 📐 Enterprise Architecture Reference: Cloud RAG Pipeline

```
Client App (React / Mobile)
        │
        ▼  [HTTPS / API Gateway]
Azure API Management (APIM)
        │
        ▼
FastAPI Backend (Azure Container Apps)
  ├── 1. Vector Search Query ──► Azure AI Search (Hybrid BM25 + HNSW Vector Index)
  ├── 2. Top-K Chunks ◄────────┘
  ├── 3. Guardrails Inspection ─► Azure AI Content Safety
  └── 4. Grounded Prompt ──────► Azure OpenAI Service (GPT-4o Managed Private Endpoint)
        │
        ▼  [Streamed Response]
End User Client
```

---

## 💻 Recommended Applied Projects & Production Blueprints

1. **Enterprise Financial Document Analyzer:** Azure Document Intelligence extracting multi-page tabular balance sheets with Azure OpenAI structured synthesis.
2. **Production MLOps Regression Pipeline:** End-to-end GitHub Actions pipeline retraining and validating XGBoost models on Azure ML compute clusters.
3. **Enterprise Zero-Trust Knowledge Copilot:** Full RAG implementation combining Azure AI Search, Azure OpenAI, and entra ID role-based document access controls.
