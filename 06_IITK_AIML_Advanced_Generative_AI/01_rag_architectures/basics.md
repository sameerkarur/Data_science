# Chapter 1: Enterprise RAG Architectures & Retrieval Engineering
**Comprehensive Textbook Guide — Advanced Generative AI**

---

## 1. Executive Overview & Mental Models

Retrieval-Augmented Generation (RAG) grounds generative language models in external, authoritative knowledge stores. By separating the non-parametric memory (external vector indexes) from parametric memory (pre-trained model weights), RAG eliminates hallucinations, enables continuous knowledge updates without fine-tuning, and provides verifiable citation trails.

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

## 2. Deep Theoretical Foundations

### 1. Hybrid Search & Reciprocal Rank Fusion (RRF)
Dense embeddings capture broad semantic intent but often fail on exact keyword queries (acronyms, model numbers, IDs). BM25 excels at keyword matching but misses semantic paraphrases. RRF combines the ranked lists from both retrievers without requiring calibration of their raw score scales:
$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where:
- $M$ is the set of retrieval models (Dense and Sparse).
- $r_m(d)$ is the rank of document $d$ in system $m$.
- $k \approx 60$ is a smoothing constant that prevents high-ranking outliers from dominating the fused ranking.

### 2. Bi-Encoder vs Cross-Encoder Architecture
- **Bi-Encoder (Retrieval Stage):** Embeds queries and documents independently into vector representations:
  $$\text{sim}(q, d) = \cos(E(q), E(d))$$
  Can pre-compute document vectors offline; searches in sub-millisecond logarithmic time via HNSW indexes.
- **Cross-Encoder (Re-Ranking Stage):** Passes the query and document jointly into a single transformer:
  $$s(q, d) = \text{Transformer}(\text{"[CLS] " } + q + \text{ " [SEP] "} + d)$$
  Computes full cross-attention between every query token and document token. Provides state-of-the-art ranking precision; applied over the top $20-50$ candidates from the Bi-Encoder.

### 3. The RAG Triad Evaluation Metrics (Ragas)
1. **Context Relevance:** Fraction of retrieved context chunks that are directly relevant to the user query.
2. **Faithfulness (Groundedness):** Degree to which the generated answer can be strictly inferred from the context (measures hallucination rate).
3. **Answer Relevance:** Extent to which the generated answer directly addresses the user question.

---

## 3. Production Implementation: Hybrid Dense-Sparse RAG Pipeline

```python
from typing import Any

def reciprocal_rank_fusion(dense_ranks: list[str], sparse_ranks: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Fuses dense and sparse candidate rankings without score normalization."""
    rrf_scores: dict[str, float] = {}
    
    # Accumulate dense reciprocal ranks
    for rank, doc_id in enumerate(dense_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Accumulate sparse reciprocal ranks
    for rank, doc_id in enumerate(sparse_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Sort descending by fused score
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs

def build_grounded_rag_prompt(query: str, retrieved_contexts: list[str]) -> str:
    """Constructs prompt with citation anchoring and injection boundaries."""
    formatted_context = "\n\n".join([f"[[Document {i+1}]]\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
    return f"""You are an enterprise AI assistant. Answer the user query strictly and exclusively using the provided context documents.
If the answer cannot be determined with certainty from the context, state "I cannot answer based on the provided documents."

<context_documents>
{formatted_context}
</context_documents>

<user_query>
{query}
</user_query>

Provide your answer with document citation brackets like [[Document 1]]:"""
```
