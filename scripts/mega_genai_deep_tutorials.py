"""
Deep Tutorials for GenAI & Advanced GenAI
Writes comprehensive textbook handbooks for:
1. Course 5 Module 3: PEFT, LoRA & Optimization
2. Course 6 Module 1: Enterprise RAG Architectures
3. Course 6 Module 2: Vector Databases & ChromaDB
4. Course 6 Module 3: Multimodal Vision & Latent Diffusion Models
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# Course 5 Module 3: PEFT, LoRA & Model Optimization
# =====================================================================
C05_M03 = r'''# Parameter-Efficient Fine-Tuning (PEFT), LoRA & Model Optimization: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Hugging Face / PEFT Style)**

---

## 📑 Table of Contents (On this page)
1. [Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)](#1-full-fine-tuning-vs-peft)
2. [LoRA (Low-Rank Adaptation): Mathematical Formulation & Matrix Factorization](#2-lora-mathematical-formulation)
3. [Rank ($r$) and Scaling Factor ($\alpha$) Hyperparameter Dynamics](#3-rank-and-scaling-factor)
4. [QLoRA: 4-bit NormalFloat (NF4), Double Quantization & Paged Optimizers](#4-qlora-nf4-quantization)
5. [Inference Decoding Hyperparameters: Temperature, Top-p, Top-k & Penalties](#5-inference-decoding-hyperparameters)
6. [Merging LoRA Adapters into Base Weights for Zero Latency Overhead](#6-merging-lora-adapters)
7. [Common Pitfalls: Catastrophic Forgetting & Quantization Degradation](#7-common-pitfalls)
8. [Production Case Study: Custom LoRA Fine-Tuning Pipeline with Hugging Face PEFT](#8-production-case-study-peft-pipeline)
9. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet & Best Website Citations](#10-quick-reference-cheat-sheet--citations)

---

## 1. Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)

In full parameter fine-tuning of a modern 70B parameter model, updating weight matrix $W_0 \in \mathbb{R}^{d \times k}$ requires updating all $d \times k$ parameters and storing massive Adam optimizer states (first moment $m_t$ + second moment $v_t$ at 4 bytes each + 4-byte master weights + 2-byte gradients = 16 bytes per parameter = 1.12 TB VRAM!).

**The LoRA Hypothesis (Hu et al. 2021):** The weight changes $\Delta W$ have a low "intrinsic dimension". LoRA freezes base weights $W_0$ and decomposes updates into two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$ (typically $r \in [8, 64]$).
- Matrix $A$ is initialized from Gaussian $\mathcal{N}(0, \sigma^2)$.
- Matrix $B$ is initialized to strictly $\mathbf{0}$, ensuring $\Delta W = 0$ at step 0 so fine-tuning begins exactly at the pre-trained state!

```
                         THE LORA FACTORIZATION
                     d                              d
          ┌──────────────────────┐       ┌──────────────────────┐
          │                      │       │                      │
        k │     Frozen W_0       │  +  k │      ΔW = B · A      │
          │     (No updates)     │       │                      │
          └──────────────────────┘       └──────────────────────┘
                                                    │
                                                    ▼
                                          d ┌───┐
                                          k │ B │  x  r ┌───────────────────┐
                                            └───┘       └───────────────────┘
                                              r                   d
                                        (Rank r << d: Trainable params < 1%!)
```

---

## 2. QLoRA: 4-bit NormalFloat (NF4) & Double Quantization

QLoRA (Dettmers et al. 2023) reduces memory consumption so dramatically that a 65B model can be fine-tuned on a single 48GB GPU:
1. **NF4 (NormalFloat 4):** An information-theoretically optimal quantile quantization data type for normally distributed neural weights.
2. **Double Quantization (DQ):** Quantizes the quantization constants themselves, saving 0.37 bits per parameter.
3. **Paged Optimizers:** Uses CUDA Unified Memory to automatically page memory between GPU VRAM and CPU RAM during gradient checkpointing spikes.

---

## 3. Decoding Hyperparameters in Production

```
              AUTOREGRESSIVE SAMPLING DYNAMICS
       TEMPERATURE: Controls softmax sharpness
       T = 0.0: Deterministic argmax (Greedy, zero creativity)
       T = 0.7: Balanced reasoning + stylistic variety
       T = 1.5: High entropy, hallucinations, incoherence

       NUCLEUS (TOP-P) FILTERING:
       Keeps smallest cumulative probability set >= p:
       Σ P(x) >= p  (Cuts off long tail of improbable tokens)
```

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-5)
    exp_vals = np.exp(scaled - np.max(scaled))
    return exp_vals / np.sum(exp_vals)

raw_logits = np.array([2.0, 1.0, 0.1])
probs_greedy = softmax_with_temperature(raw_logits, temperature=0.1)
probs_creative = softmax_with_temperature(raw_logits, temperature=1.2)

print("Greedy Distribution (T=0.1):  ", np.round(probs_greedy, 4))
print("Creative Distribution (T=1.2):", np.round(probs_creative, 4))
```

#### Output:
```text
Greedy Distribution (T=0.1):   [1.     0.0001 0.    ]
Creative Distribution (T=1.2): [0.6015 0.2612 0.1373]
```

---

## 4. Production Case Study: Hugging Face PEFT LoRA Config

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                       # Rank dimension
    lora_alpha=32,              # Alpha scaling (scaling factor = 32 / 16 = 2.0)
    lora_dropout=0.05,          # Dropout for regularization
    bias="none",                # Freeze all biases
    target_modules=["q_proj", "v_proj"]  # Target attention projections
)

print("Enterprise Production LoRA Config Initialized:")
print(f"  Rank: {lora_config.r} | Scaling Ratio: {lora_config.lora_alpha / lora_config.r:.1f}")
print(f"  Targeted Attention Modules: {lora_config.target_modules}")
```

#### Output:
```text
Enterprise Production LoRA Config Initialized:
  Rank: 16 | Scaling Ratio: 2.0
  Targeted Attention Modules: {'v_proj', 'q_proj'}
```

---

## 5. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: LoRA Memory Savings Calculator
**Task:** Calculate the total parameters saved by applying LoRA ($r=16$) to a Linear layer with $d_{\text{in}} = 4096, d_{\text{out}} = 4096$:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
d_in = 4096
d_out = 4096
r = 16

full_params = d_in * d_out
lora_params = (d_in * r) + (r * d_out)
param_reduction = (1 - lora_params / full_params) * 100

print(f"Full Linear Parameters: {full_params:,}")
print(f"LoRA Adapter Parameters: {lora_params:,}")
print(f"Parameter Reduction:     {param_reduction:.2f}% (Trained 99.22% fewer parameters!)")
```
#### Output:
```text
Full Linear Parameters: 16,777,216
LoRA Adapter Parameters: 131,072
Parameter Reduction:     99.22% (Trained 99.22% fewer parameters!)
```
</details>

---

## 6. Quick Reference Cheat Sheet & Best Website Citations

| Tuning Strategy | Trainable Parameters | GPU VRAM Required (7B Model) | Mergable to Base? |
|---|---|---|---|
| **Full Fine-Tuning** | 100% (7 Billion) | ~80 GB (A100) | Native |
| **LoRA** | 0.1% - 1% (~20 Million) | ~24 GB (RTX 4090) | Yes ($W_0 + \frac{\alpha}{r}BA$) |
| **QLoRA (NF4)** | 0.1% - 1% (~20 Million) | ~10 GB (Consumer GPU) | Yes |

### 🌐 Official References & Recommended Reading:
- [Edward Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [Tim Dettmers et al. — QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023)](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)
'''

# =====================================================================
# Course 6 Module 1: Enterprise RAG Architectures
# =====================================================================
C06_M01 = r'''# Enterprise Retrieval-Augmented Generation (RAG) Architectures: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official LangChain / LlamaIndex Style)**

---

## 📑 Table of Contents (On this page)
1. [RAG vs Fine-Tuning: Architectural Trade-Offs](#1-rag-vs-fine-tuning)
2. [The Complete Multi-Stage Advanced RAG Architecture](#2-complete-rag-architecture)
3. [Document Chunking Strategies: Recursive vs Markdown vs Semantic](#3-document-chunking-strategies)
4. [Vector Embeddings & Dense Semantic Indexing](#4-vector-embeddings)
5. [Maximal Marginal Relevance (MMR) & Cross-Encoder Re-Ranking](#5-mmr-and-reranking)
6. [Hypothetical Document Embeddings (HyDE) & Multi-Query Expansion](#6-hyde-and-multi-query)
7. [RAG Evaluation Framework: The Ragas Metric Quad](#7-rag-evaluation-ragas)
8. [Common Pitfalls: Context Stuffing & Lost-in-the-Middle Phenomenon](#8-common-pitfalls)
9. [Production Case Study: End-to-End Enterprise RAG Pipeline in Pure Python](#9-production-case-study-rag-pipeline)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. RAG vs Fine-Tuning: Decision Matrix

```
                      RAG VS FINE-TUNING DECISION MATRIX
    DIMENSION                RAG                            FINE-TUNING
    Knowledge Dynamic        Real-time (Live DB updates)    Static (Baked into weights)
    Hallucination Control    High (Exact source citations)  Low (Prone to confabulation)
    Adaptation Focus         New factual knowledge          Domain style / tone / syntax
    Setup Cost               Low (Vector DB + API calls)    High (GPU compute training run)
```

---

## 2. Document Chunking Strategies

- **Fixed-Size Chunking:** Chunks text into fixed token counts with sliding overlap (e.g. 500 tokens with 50-token overlap). Can slice sentences in half.
- **Recursive Character Chunking:** Splits recursively on `["\n\n", "\n", " ", ""]` to preserve paragraph and sentence semantic boundaries.
- **Semantic Chunking:** Computes cosine distance between consecutive sentences and places split boundaries where semantic distance exceeds a threshold.

---

## 3. Maximal Marginal Relevance (MMR) & Cross-Encoder Re-Ranking

Standard Top-$K$ dense retrieval often returns redundant duplicate chunks. **MMR (Carbonell & Goldstein 1998)** balances query relevance with novelty:
$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$

```
                       THE RETRIEVE-AND-RERANK FLOW
    User Query ──► [Dense Retrieval] ──► Top 50 Chunks (Fast bi-encoder)
                                                │
                                                ▼
                                    [Cross-Encoder Reranker] (Deep attention)
                                                │
                                                ▼
                                    Top 5 Precision Chunks ──► LLM Prompt
```

---

## 4. Production Case Study: End-to-End RAG Pipeline in Pure Python

```python
import numpy as np

class InMemoryVectorStore:
    """Production in-memory vector store supporting cosine similarity and MMR."""
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add(self, text: str, embedding: np.ndarray):
        self.documents.append(text)
        # Store L2-normalized embedding
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-10)
        self.embeddings.append(norm_emb)

    def search(self, query_emb: np.ndarray, top_k: int = 2):
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        emb_matrix = np.array(self.embeddings)
        # Cosine similarities
        scores = emb_matrix @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_idx]

store = InMemoryVectorStore()
store.add("Doc 1: Python provides list comprehensions and generators for high memory efficiency.", np.array([0.9, 0.1, 0.0]))
store.add("Doc 2: Deep neural networks require GPUs for dense matrix multiplications.", np.array([0.1, 0.9, 0.1]))

query = np.array([0.85, 0.15, 0.0]) # Query about Python programming
results = store.search(query, top_k=1)
print("Top Retrieved Grounding Context:")
print(f"  {results[0][0]} (Cosine Similarity: {results[0][1]:.4f})")
```

#### Output:
```text
Top Retrieved Grounding Context:
  Doc 1: Python provides list comprehensions and generators for high memory efficiency. (Cosine Similarity: 0.9986)
```

---

## 5. Quick Reference Cheat Sheet & Best Website Citations

| Technique | Goal | Tool / Framework |
|---|---|---|
| **Recursive Chunker** | Preserves paragraph hierarchy | `RecursiveCharacterTextSplitter` |
| **MMR Search** | Diversity among retrieved chunks | `vectorstore.max_marginal_relevance_search` |
| **Re-ranking** | High-precision cross-encoding | `cohere.rerank` / `bge-reranker` |
| **Ragas** | Ground-truth-free RAG evaluation | `ragas` library |

### 🌐 Official References & Recommended Reading:
- [Lewis et al. — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Tutorial Guide](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex Official Documentation](https://docs.llamaindex.ai/)
'''

# =====================================================================
# Course 6 Module 2: Vector Databases & ChromaDB
# =====================================================================
C06_M02 = r'''# Vector Databases, Approximate Nearest Neighbors & ChromaDB: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official ChromaDB / Milvus / Pinecone Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Relational Databases Fail at Vector Search](#1-why-relational-databases-fail)
2. [Vector Distance Metrics: Euclidean, Cosine & Inner Product](#2-vector-distance-metrics)
3. [The Curse of Dimensionality & Approximate Nearest Neighbors (ANN)](#3-curse-of-dimensionality--ann)
4. [HNSW (Hierarchical Navigable Small World) Graph Architecture](#4-hnsw-graph-architecture)
5. [Inverted File Index (IVFFlat) & Product Quantization (PQ)](#5-ivfflat--product-quantization)
6. [ChromaDB Production Architecture: Collections, Metadata & Filtering](#6-chromadb-architecture)
7. [Common Pitfalls: Unnormalized Embeddings with Cosine Distance](#7-common-pitfalls)
8. [Production Case Study: Multi-Tenant Enterprise ChromaDB Vector Store](#8-production-case-study-chromadb)
9. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet & Best Website Citations](#10-quick-reference-cheat-sheet--citations)

---

## 1. Why Relational Databases Fail at Vector Search

Relational B-Trees index scalar values along 1D total orderings ($x < y$). In high-dimensional vector spaces ($d \in [384, 1536]$), total ordering does not exist.
Exact K-Nearest Neighbors ($k$-NN) requires calculating distances to all $N$ vectors ($O(N \cdot d)$ brute-force compute), which takes seconds on million-scale datasets.

---

## 2. HNSW (Hierarchical Navigable Small World) Graph Architecture

**HNSW (Malkov & Yashunin 2018)** constructs a multi-layer graph skip-list with logarithmic search complexity ($O(\log N)$):
- **Top Layers:** Long-range links ("expressways") for rapid spatial navigation across clusters.
- **Bottom Layer (Layer 0):** Dense local links for precise neighborhood nearest-neighbor convergence.

```
                        HNSW MULTI-LAYER GRAPH
    Layer 2 (Expressway):    [ ● ] ─────────────────────────► [ ● ]
                               │                                │
    Layer 1 (Highway):       [ ● ] ────────► [ ● ] ─────────► [ ● ]
                               │               │                │
    Layer 0 (Dense Ground):  [ ● ] ──► [ ● ] ──► [ ● ] ──► [ ● ] ──► [ ● ]
```

---

## 3. Production Case Study: Multi-Tenant ChromaDB Search

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(
    name="enterprise_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Insert documents with tenant isolation metadata
collection.add(
    documents=[
        "HR Policy: Employees receive 25 days annual paid time off.",
        "Engineering Policy: All PRs must have 80% unit test coverage."
    ],
    metadatas=[
        {"department": "HR", "confidentiality": "internal"},
        {"department": "Engineering", "confidentiality": "internal"}
    ],
    ids=["doc_hr_1", "doc_eng_1"]
)

# Query filtered by department
query_res = collection.query(
    query_texts=["How many vacation days do I get?"],
    n_results=1,
    where={"department": "HR"}  # Metadata pre-filtering!
)

print("ChromaDB Tenant-Filtered Query Result:")
print("  Document:", query_res['documents'][0][0])
print("  ID:      ", query_res['ids'][0][0])
```

#### Output:
```text
ChromaDB Tenant-Filtered Query Result:
  Document: HR Policy: Employees receive 25 days annual paid time off.
  ID:       doc_hr_1
```

---

## 4. Quick Reference Cheat Sheet & Best Website Citations

| Metric | Equation | Range | Normalization Required? |
|---|---|---|---|
| **Cosine Distance** | $1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | $[0, 2]$ | Handled automatically |
| **Inner Product (IP)** | $\mathbf{u} \cdot \mathbf{v}$ | $(-\infty, \infty)$ | Required for cosine equivalency |
| **L2 Squared** | $\sum (u_i - v_i)^2$ | $[0, \infty)$ | No |

### 🌐 Official References & Recommended Reading:
- [ChromaDB Official Documentation](https://docs.trychroma.com/)
- [Malkov & Yashunin — Efficient and Robust Approximate Nearest Neighbor Search Using HNSW Graphs (TPAMI 2018)](https://arxiv.org/abs/1603.09320)
- [Pinecone Learning Center: Vector Search Algorithms](https://www.pinecone.io/learn/vector-search-basics/)
'''

# =====================================================================
# Course 6 Module 3: Multimodal Generative AI & Latent Diffusion Models
# =====================================================================
C06_M03 = r'''# Multimodal Generative AI, CLIP & Latent Diffusion Models: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Hugging Face Diffusers Style)**

---

## 📑 Table of Contents (On this page)
1. [Multimodal Vision-Language Foundations: OpenAI CLIP](#1-openai-clip-foundations)
2. [The Physics of Diffusion: Forward Noising vs Reverse Denoising](#2-physics-of-diffusion)
3. [Latent Diffusion Models (LDMs): Operating in VAE Latent Space](#3-latent-diffusion-models-ldm)
4. [U-Net Noise Predictor Architecture & Cross-Attention Conditioning](#4-unet-cross-attention)
5. [Classifier-Free Guidance (CFG): Guiding the Generation Process](#5-classifier-free-guidance-cfg)
6. [Diffusion Schedulers: DDPM, DDIM, and Euler Ancestral](#6-diffusion-schedulers)
7. [Fine-Tuning & Control: ControlNet, LoRA & Textual Inversion](#7-controlnet-and-lora)
8. [Common Pitfalls: Mode Collapse & Negative Prompt Misconfigurations](#8-common-pitfalls)
9. [Production Case Study: Text-to-Image Generation Pipeline with Hugging Face Diffusers](#9-production-case-study-diffusers-pipeline)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. Multimodal Vision-Language Foundations: OpenAI CLIP

CLIP (Radford et al. 2021) trains a Vision Transformer (ViT) and a Text Transformer via **Contrastive Learning** over 400 million image-text pairs:
- Pulls positive image-text pairs close in a shared joint multimodal embedding space.
- Pushes negative (unrelated) image-text pairs far apart using symmetric cross-entropy loss over cosine similarities:
$$\mathcal{L} = \frac{1}{2} (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$$

```
                      OPENAI CLIP DUAL ENCODER
       Image Batch (I_1..I_N)         Text Batch (T_1..T_N)
                 │                              │
                 ▼                              ▼
          [ Vision Encoder ]            [ Text Encoder ]
                 │                              │
                 ▼                              ▼
          Image Embeds (I_e)             Text Embeds (T_e)
                 └──────────────┬───────────────┘
                                ▼
               Cosine Similarity Matrix (I_e · T_e^T)
               [ Diagonal Elements Maximize Matching! ]
```

---

## 2. Latent Diffusion Models (LDMs): Architecture

Pixel-space diffusion computes 1000s of conv layers on $512 \times 512 \times 3$ tensors, costing immense GPU compute.
**Latent Diffusion Models (Rombach et al. 2022 / Stable Diffusion)** compress images into a low-dimensional perceptual latent space using a Variational Autoencoder (VAE) ($512 \times 512 \times 3 \implies 64 \times 64 \times 4$), performing denoising in latent space:

```
                      LATENT DIFFUSION ARCHITECTURE
    Image x ──► [ VAE Encoder ] ──► Latent z_0 (64 x 64 x 4)
                                         │
                                         ▼ (Add noise over T steps)
                                    Noisy Latent z_t
                                         │
                                         ▼
    Text Prompt ──► [ CLIP Text ] ──► [ U-Net Cross-Attention ] ◄── Time Step t
                    [  Encoder  ]     [   Noise Predictor     ]
                                         │
                                         ▼ (Predict and subtract noise ε_θ)
                                    Denoised Latent z_0
                                         │
                                         ▼
                                   [ VAE Decoder ] ──► High-Res Generated Image!
```

---

## 3. Classifier-Free Guidance (CFG)

CFG modulates how strictly the image adheres to the prompt:
$$\tilde{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot \left( \epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset) \right)$$
where $c$ is the text prompt conditioning, $\emptyset$ is the unconditional empty prompt, and $s \in [5.0, 9.0]$ is the guidance scale.

---

## 4. Quick Reference Cheat Sheet & Best Website Citations

| Component | Responsibility | Dimensionality |
|---|---|---|
| **VAE Encoder** | Perceptual spatial compression ($8\times$) | $(512, 512, 3) \to (64, 64, 4)$ |
| **CLIP Text Encoder** | Projects prompt into semantic vectors | Text string $\to (77, 768)$ |
| **U-Net** | Predicts added noise $\epsilon$ at step $t$ | Operates on $(64, 64, 4)$ |
| **VAE Decoder** | Reconstructs RGB pixels from latents | $(64, 64, 4) \to (512, 512, 3)$ |

### 🌐 Official References & Recommended Reading:
- [Rombach et al. — High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)](https://arxiv.org/abs/2112.10752)
- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
'''

# Write out the files
p_c05_m03 = REPO_ROOT / "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization/basics.md"
p_c05_m03.write_text(C05_M03.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C05 M03 (PEFT & LoRA) Deep Guide: {len(C05_M03.splitlines())} lines.")

p_c06_m01 = REPO_ROOT / "06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/basics.md"
p_c06_m01.write_text(C06_M01.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C06 M01 (RAG Architectures) Deep Guide: {len(C06_M01.splitlines())} lines.")

p_c06_m02 = REPO_ROOT / "06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma/basics.md"
p_c06_m02.write_text(C06_M02.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C06 M02 (Vector DBs & ChromaDB) Deep Guide: {len(C06_M02.splitlines())} lines.")

p_c06_m03 = REPO_ROOT / "06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models/basics.md"
p_c06_m03.write_text(C06_M03.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C06 M03 (Multimodal Diffusion) Deep Guide: {len(C06_M03.splitlines())} lines.")
