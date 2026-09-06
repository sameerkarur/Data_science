# Project 11 (V2): Hybrid Dense-Sparse RAG & Guardrails
**Next-Generation Advanced Generative AI Architecture**  
*Curriculum: Advanced Generative AI*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Retrieval Architecture**| Purely dense embeddings (ChromaDB Vector Store) | Hybrid Search: BM25 Lexical + Dense Semantic Vectors |
| **Fusion Algorithm** | Single vector similarity threshold | Reciprocal Rank Fusion (RRF) with constant $k=60$ |
| **Acronym / Keyword Handling**| Often degrades on exact corporate policy codes | BM25 guarantees exact keyword matching while vectors capture intent |
| **Hallucination Prevention**| Unchecked LLM completion | Guardrail Thresholding: Rejects queries lacking grounded supporting evidence |
| **Explainability** | Black-box answer | Full citation attribution and per-retriever rank transparency |

---

## 🚀 How to Run

```bash
python projects_version2/11_AdvGenAI_Hybrid_Dense_Sparse_RAG_V2/hybrid_dense_sparse_rag_v2.py
```
