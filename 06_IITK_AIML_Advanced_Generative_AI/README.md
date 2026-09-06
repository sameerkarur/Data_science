# Course 06: Advanced Generative AI & Retrieval-Augmented Generation
**Professional Certificate Program — E&ICT Academy, IIT Kanpur**

---

## 📌 Course Overview

Course 06 bridges foundational generative modeling with production-grade enterprise AI systems. It focuses on the end-to-end design, implementation, and optimization of **Retrieval-Augmented Generation (RAG)**, **High-Dimensional Vector Databases**, and **Multimodal Generative Vision Models**.

---

## 🧭 Practice Modules & Interview Question Banks

| Subtopic | Practice Notebook | Complete Solutions | Concepts & Reference | Interview Q&A Bank |
|---|---|---|---|---|
| **01. RAG Architectures & Retrieval Engineering** | [`01_rag_architectures/practice.ipynb`](01_rag_architectures/practice.ipynb) | [`01_rag_architectures/solutions.ipynb`](01_rag_architectures/solutions.ipynb) | [`01_rag_architectures/basics.md`](01_rag_architectures/basics.md) | [`01_rag_architectures/interview_qa.md`](01_rag_architectures/interview_qa.md) (30 Q&A) |
| **02. Vector Databases & ChromaDB** | [`02_vector_databases_chroma/practice.ipynb`](02_vector_databases_chroma/practice.ipynb) | [`02_vector_databases_chroma/solutions.ipynb`](02_vector_databases_chroma/solutions.ipynb) | [`02_vector_databases_chroma/basics.md`](02_vector_databases_chroma/basics.md) | [`02_vector_databases_chroma/interview_qa.md`](02_vector_databases_chroma/interview_qa.md) (30 Q&A) |
| **03. Multimodal Generative Models & Vision AI** | [`03_multimodal_generative_models/practice.ipynb`](03_multimodal_generative_models/practice.ipynb) | [`03_multimodal_generative_models/solutions.ipynb`](03_multimodal_generative_models/solutions.ipynb) | [`03_multimodal_generative_models/basics.md`](03_multimodal_generative_models/basics.md) | [`03_multimodal_generative_models/interview_qa.md`](03_multimodal_generative_models/interview_qa.md) (30 Q&A) |

---

## 🚀 Course-End Deliverable Projects

### 1. Enterprise Policy RAG Assistant (Nestlé HR Copilot)
- **Problem Statement:** Large multinational enterprise policies span hundreds of pages of complex, legally binding PDF documents. Standard keyword search fails on nuanced natural language queries, while raw LLMs hallucinate inaccurate policy details.
- **Architecture:** PyPDF extraction, semantic chunking with overlap via `RecursiveCharacterTextSplitter`, dense OpenAI embeddings, local persistence with ChromaDB, and prompt-constrained conversational generation.
- **Interactive UI:** Gradio streaming chat interface with grounded source citation.
- **Directory:** `project1_hr_assistant/`
- **Writeup:** [`project1_hr_assistant/WRITEUP.md`](project1_hr_assistant/WRITEUP.md)

### 2. Autonomous Creative Campaign Generator (Netflix Design Suite)
- **Problem Statement:** Visual design iterations for international entertainment streaming campaigns require rapid concept exploration across diverse genres (noir, action, romance, sci-fi) while maintaining high stylistic consistency.
- **Architecture:** Generative Vision API orchestration, prompt engineering templates with negative constraint injection, aspect ratio control, and real-time image rendering.
- **Interactive UI:** Gradio parameter studio with live image download and prompt history logging.
- **Directory:** `project2_designs/`
- **Writeup:** [`project2_designs/WRITEUP.md`](project2_designs/WRITEUP.md)

---

## 🛠️ Execution & Practice Environment

### Cloud Execution (Google Colab)
Click any **Colab** badge across the curriculum notebooks to launch instant, free GPU-accelerated cloud sessions. Notebooks automatically initialize prerequisites without requiring local configuration.

### Local Execution Setup

```bash
cd 06_IITK_AIML_Advanced_Generative_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run interactive Jupyter lab
jupyter notebook
```

Set your API credentials in your environment:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
