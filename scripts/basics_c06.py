"""
Textbook-Scale Architectural & Conceptual Guides for Course 6:
IITK AIML Core: Advanced Generative AI
Modules:
  01_rag_architectures
  02_vector_databases_chroma
  03_multimodal_generative_models
"""

C06_BASICS = {}

# =====================================================================
# 1. Enterprise RAG Architectures & Retrieval Engineering
# =====================================================================
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures"] = r'''# Chapter 1: Enterprise RAG Architectures & Retrieval Engineering
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
'''

# =====================================================================
# 2. Vector Databases & Chroma
# =====================================================================
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma"] = r'''# Chapter 2: Vector Databases & High-Dimensional HNSW Indexing
**Comprehensive Textbook Guide — Advanced Generative AI**

---

## 1. Executive Overview & Mental Models

Vector databases are specialized storage engines designed to store, manage, and query dense vector embeddings. Traditional relational databases (B-Trees) cannot index multi-dimensional vectors due to the **Curse of Dimensionality**. Vector databases solve this through **Approximate Nearest Neighbor (ANN)** graph algorithms.

```
                 HIERARCHICAL NAVIGABLE SMALL WORLD (HNSW)
    Layer 2 (Sparse Highway):     (•) ────────────────────────► (•)
                                   │                             │
    Layer 1 (Medium Density):     (•) ────────► (•) ──────────► (•)
                                   │             │               │
    Layer 0 (Dense Ground Layer): (•) ─► (•) ─► (•) ─► (•) ─► (•)
    Query searches greedily from top layer down, achieving O(log N) lookup!
```

---

## 2. Deep Theoretical Foundations

### 1. Hierarchical Navigable Small World (HNSW) Theory
HNSW models high-dimensional spaces as a multi-layer geometric graph:
- **Skip-List Metaphor:** The top layer contains a very sparse graph with long-range edges for coarse routing. Each lower layer increases density exponentially, until the bottom layer (Layer 0) contains all vectors with short-range neighbor edges.
- **Greedy Search:** Search begins at the top entry point, moving to the neighbor closest to the query vector. When a local minimum is reached in Layer $l$, search transitions down to Layer $l-1$ at the same coordinate.
- **Time Complexity:** Achieves $O(\log N)$ search time with recall rates exceeding $98\%$.

### 2. Distance Metrics Geometry
- **Cosine Distance:** Invariant to vector length; evaluates semantic angle:
  $$D_{\text{Cosine}}(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Inner Product (Dot Product):** Reflects both angle and magnitude. If vectors are $L_2$-normalized ($\|u\|_2 = 1$), Dot Product is monotonically equivalent to Cosine Distance.
- **Euclidean ($L_2$) Distance:** Geometric physical distance in Euclidean space:
  $$D_{L2}(u, v) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$

---

## 3. Production Implementation: ChromaDB Vector Store with Metadata Filtering

```python
import numpy as np

class InMemoryVectorStore:
    """Minimal vector store demonstrating normalized cosine similarity and metadata filtering."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.doc_ids: list[str] = []
        self.metadata: list[dict] = []
        self.embeddings: list[np.ndarray] = []

    def add_records(self, doc_ids: list[str], vectors: np.ndarray, meta: list[dict]) -> None:
        for d_id, vec, m in zip(doc_ids, vectors, meta):
            norm_vec = vec / (np.linalg.norm(vec) + 1e-10)  # L2 normalization
            self.doc_ids.append(d_id)
            self.embeddings.append(norm_vec)
            self.metadata.append(m)

    def query(self, query_vec: np.ndarray, top_k: int = 5, where: dict | None = None) -> list[dict]:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix = np.array(self.embeddings)
        scores = np.dot(matrix, query_norm)
        
        ranked_indices = np.argsort(scores)[::-1]
        results = []
        for idx in ranked_indices:
            if where:
                # Apply metadata predicate filter
                m = self.metadata[idx]
                if not all(m.get(k) == v for k, v in where.items()):
                    continue
            results.append({
                "id": self.doc_ids[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx]
            })
            if len(results) >= top_k:
                break
        return results
```
'''

# =====================================================================
# 3. Multimodal Generative Models, CLIP & Diffusion
# =====================================================================
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models"] = r'''# Chapter 3: Multimodal Generative AI, CLIP & Diffusion Models
**Comprehensive Textbook Guide — Advanced Generative AI**

---

## 1. Executive Overview & Mental Models

Multimodal generative AI models bridge different perceptual modalities (vision, text, audio). Contrastive models (CLIP) align images and text into a shared latent manifold, while Latent Diffusion Models (LDMs) synthesize high-fidelity images by iteratively inverting a continuous Gaussian noise process.

```
                   LATENT DIFFUSION MODEL (STABLE DIFFUSION)
    Text Prompt ──► Text Encoder (CLIP) ──► Conditioning Vector (c)
                                                   │
                                                   ▼
    Gaussian Noise z_T ──► [U-Net Denoising Iterations with Cross-Attention] ──► Latent z_0
                                                                                   │
                                                         Decoder (VAE) ◄──────────┘
                                                               │
                                                               ▼
                                                      Output 1024x1024 Image
```

---

## 2. Deep Theoretical Foundations

### 1. Contrastive Language-Image Pretraining (CLIP - Radford et al.)
CLIP trains an image encoder $f(I)$ and text encoder $g(T)$ jointly across a mini-batch of $N$ image-text pairs:
$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right)$$
Where the image-to-text contrastive cross-entropy loss is:
$$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(f(I_i), g(T_i)) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(f(I_i), g(T_j)) / \tau)}$$
Maximizing the cosine similarity of true $(I_i, T_i)$ pairs while penalizing all $N^2 - N$ mismatched pairs in the batch.

### 2. Classifier-Free Guidance (CFG - Ho & Salimans)
During diffusion denoising, CFG balances prompt adherence against generation sample diversity. At each denoising step $t$, the predicted noise vector $\hat{\epsilon}_\theta$ combines unconditional and conditional predictions:
$$\hat{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot \left( \epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset) \right)$$
Where:
- $c$ is the text prompt conditioning embedding.
- $\emptyset$ is the null/empty prompt embedding.
- $s \ge 1$ is the guidance scale. Setting $s > 1$ amplifies the direction pointing toward the conditioned text prompt, sharpening visual fidelity.

---

## 3. Production Implementation: Zero-Shot Multimodal Classification via CLIP Embeddings

```python
import numpy as np

def zero_shot_clip_predict(image_embedding: np.ndarray, 
                           text_candidate_embeddings: np.ndarray, 
                           class_labels: list[str], 
                           temperature: float = 0.01) -> dict[str, float]:
    """Computes zero-shot classification probabilities from normalized CLIP embeddings."""
    # Ensure L2 unit norms
    img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-10)
    txt_norm = text_candidate_embeddings / (np.linalg.norm(text_candidate_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarities
    logits = np.dot(txt_norm, img_norm) / temperature
    # Stable Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return {label: float(prob) for label, prob in zip(class_labels, probs)}
```
'''

print(f"Loaded {len(C06_BASICS)} textbook chapters for Course 6.")
