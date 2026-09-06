"""
Comprehensive, high-depth Basics & Architecture Guides for Course 6:
Advanced Generative AI & Retrieval Engineering (3 modules)
"""

C06_BASICS = {}

# 1. RAG Architectures
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures"] = """# Enterprise RAG Architectures & Retrieval Engineering
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
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{k + r_m(d)}, \quad \text{where } k \approx 60$$

### 2. Bi-Encoder vs Cross-Encoder Architecture
- **Bi-Encoder (Retrieval):** Embeds queries and documents into separate vector spaces independently. Computes similarity via fast inner product search ($O(1)$ per vector in HNSW graphs). High throughput, moderate precision.
- **Cross-Encoder (Re-Ranking):** Concatenates `[CLS] Query [SEP] Document` into a single transformer, allowing cross-attention between every query token and document token. Yields state-of-the-art precision, applied to the top 20–50 candidate chunks.
"""

# 2. Vector Databases & Chroma
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma"] = """# Vector Databases & High-Dimensional HNSW Indexing
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Approximate Nearest Neighbors (ANN) & HNSW
Exhaustive vector distance calculation across millions of embeddings requires $O(N \cdot d)$ floating-point operations. HNSW organizes vectors into multi-layer skip-list graphs where search proceeds logarithmically $O(\log N)$ with $>98\%$ recall.

### 2. Distance Metrics Comparison
- **Cosine Distance:** $1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ (Normalized semantic orientation, invariant to document length).
- **Dot Product:** $u \cdot v$ (Includes magnitude; requires $L_2$-normalized vectors to match cosine).
- **Euclidean ($L_2$) Distance:** $\|u - v\|_2 = \sqrt{\sum (u_i - v_i)^2}$ (Spatial geometric proximity).
"""

# 3. Multimodal Generative Models
C06_BASICS["06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models"] = """# Multimodal Generative AI, CLIP & Diffusion Models
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Contrastive Language-Image Pretraining (CLIP)
CLIP trains an image encoder (ViT) and text encoder (Transformer) jointly using a symmetric contrastive cross-entropy loss:
$$\mathcal{L} = \frac{1}{2}\left( \mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}} \right)$$
Maximizing the cosine similarity of true $(I_i, T_i)$ pairs while penalizing all $N^2 - N$ mismatched pairs in the batch.

### 2. Classifier-Free Guidance (CFG) in Diffusion
Controls adherence to the prompt vs visual creativity:
$$\hat{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot \left( \epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset) \right)$$
Where $s \ge 1$ is the guidance scale, amplifying the conditioned prompt vector direction away from unconditional noise $\emptyset$.
"""

print(f"Loaded {len(C06_BASICS)} comprehensive guides for Course 6.")
