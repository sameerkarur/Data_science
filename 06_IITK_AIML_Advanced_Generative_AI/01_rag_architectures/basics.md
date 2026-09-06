# Enterprise RAG Architectures & Retrieval Engineering
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Enterprise Retrieval-Augmented Generation (RAG) grounds generative language models on verifiable proprietary document knowledge.

```
                    ENTERPRISE HYBRID RAG DATA PIPELINE
    Raw Documents (PDF/HTML) ──► Recursive Chunking + Overlap ──► Embedding Model
                                                                        │
                                                                 Vector Database
                                                                        │
    User Query ──► 1. Dense Semantic Search (Cosine / HNSW) ─────┐      │
               ──► 2. Sparse Lexical Search (BM25 Keywords) ───┼──────┘
                                                                ▼
                                              Reciprocal Rank Fusion (RRF)
                                                                ▼
                                                Cross-Encoder Re-Ranker
                                                                ▼
                                                   Top-K Context Chunks
                                                                ▼
                                                 Prompt + Context ──► LLM Output
```

---

## 🧭 Deep Theoretical Foundations

### 1. Hybrid Search & Reciprocal Rank Fusion (RRF)
Dense embeddings capture abstract semantic concepts but struggle with exact part numbers, acronyms, or rare terms. Sparse BM25 excels at keyword matching. RRF unifies both candidate rankings without score normalization:
$$	ext{RRF\_Score}(d) = \sum_{m \in \{	ext{Dense}, 	ext{Sparse}\}} rac{1}{k + r_m(d)}, \quad 	ext{where } k pprox 60$$

### 2. Bi-Encoder vs Cross-Encoder Architecture
- **Bi-Encoder (Retrieval):** Embeds queries and documents into separate vector spaces independently. Computes similarity via fast inner product search ($O(1)$ per vector in HNSW graphs). High throughput, moderate precision.
- **Cross-Encoder (Re-Ranking):** Concatenates `[CLS] Query [SEP] Document` into a single transformer, allowing cross-attention between every query token and document token. Yields state-of-the-art precision, applied to the top 20–50 candidate chunks.
