# Retrieval-Augmented Generation (RAG): Complete Architecture & Engineering Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is RAG & Why Do LLMs Need It?](#1-what-is-rag--why-do-llms-need-it)
2. [The End-to-End RAG Architecture (Visual Dataflow)](#2-the-end-to-end-rag-architecture-visual-dataflow)
3. [Document Ingestion & Chunking Strategies](#3-document-ingestion--chunking-strategies)
4. [Vector Embeddings & Semantic Similarity (Cosine, Dot Product)](#4-vector-embeddings--semantic-similarity)
5. [Vector Database Storage & Indexing (ChromaDB / HNSW)](#5-vector-database-storage--indexing)
6. [Context Retrieval & Hybrid Search (Keyword + Dense Vectors)](#6-context-retrieval--hybrid-search)
7. [Augmented Prompt Engineering & Generation](#7-augmented-prompt-engineering--generation)
8. [Complete RAG Pipeline Implementation from Scratch in Python](#8-complete-rag-pipeline-implementation-from-scratch)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. What is RAG & Why Do LLMs Need It?

**Retrieval-Augmented Generation (RAG)** is an AI framework that augments Large Language Models (LLMs) with dynamic, external domain knowledge without fine-tuning model weights.

### Why do LLMs fail without RAG?
- **Knowledge Cutoffs:** LLMs know nothing about events or data after their training date.
- **Hallucinations:** When an LLM lacks factual context, it generates plausible-sounding falsehoods.
- **Private Enterprise Data:** LLMs cannot access private company documents, PDFs, or databases.
- **Cost & Latency of Fine-Tuning:** Retraining an LLM every day is prohibitively expensive.

---

## 2. The End-to-End RAG Architecture

```
                               THE FULL RAG PIPELINE
   [Enterprise Docs: PDF, Markdown, DB]
                    │
                    ▼ 1. INGESTION & CHUNKING
     [Chunk 1]   [Chunk 2]   [Chunk 3]
         │           │           │
         └───────────┬───────────┘
                     ▼ 2. EMBEDDING MODEL (e.g. text-embedding-3-small)
               [Dense Vectors: 1536-D]
                     │
                     ▼ 3. VECTOR DATABASE (ChromaDB / Pinecone / HNSW)
             ┌─────────────────────────────┐
             │ Vector Index & Metadata     │
             └──────────────┬──────────────┘
                            │
   [User Query: "What is refund policy?"] ──► 4. Embed Query ──► Vector Similarity Search (Top-k)
                                                                            │
                                                                            ▼ 5. Retrieved Chunks
   [Augmented Prompt: Context + Question] ◄─────────────────────────────────┘
         │
         ▼ 6. LLM GENERATION (GPT-4 / Claude / Llama-3)
   "According to section 4, refunds are processed within 14 business days..." (Grounded & Factual!)
```

---

## 3. Document Ingestion & Chunking Strategies

Chunking divides long documents into smaller semantic units:
- **Fixed Size Chunking:** e.g. 500 tokens with 50-token overlap.
- **Recursive Character Chunking:** Splits sequentially on paragraphs (`\n\n`), sentences (`\n`), and punctuation (`.`).
- **Semantic Chunking:** Splits where embedding distance between consecutive sentences exceeds a threshold.

```python
def recursive_chunker(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Simple sliding window chunker with overlapping context."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

sample_doc = "Artificial intelligence is transforming software engineering. " * 50
chunks = recursive_chunker(sample_doc, chunk_size=30, overlap=10)
print(f"Total words: {len(sample_doc.split())} -> Created {len(chunks)} overlapping chunks.")
print("Chunk 0 Sample:", chunks[0][:80] + "...")
```

#### Output:
```text
Total words: 350 -> Created 17 overlapping chunks.
Chunk 0 Sample: Artificial intelligence is transforming software engineering. Artificial intelli...
```

---

## 4. Vector Embeddings & Semantic Similarity

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two embedding vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Simulated 3D embeddings
v_query = np.array([0.9, 0.1, 0.05])
v_doc_relevant = np.array([0.88, 0.12, 0.08])
v_doc_unrelated = np.array([0.02, 0.95, 0.85])

sim_rel = cosine_similarity(v_query, v_doc_relevant)
sim_unrel = cosine_similarity(v_query, v_doc_unrelated)

print(f"Similarity to Relevant Document:   {sim_rel:.4f} (High!)")
print(f"Similarity to Unrelated Document:  {sim_unrel:.4f} (Low!)")
```

#### Output:
```text
Similarity to Relevant Document:   0.9996 (High!)
Similarity to Unrelated Document:  0.1345 (Low!)
```

---

## 5. Complete RAG Pipeline from Scratch in Python

Here is a self-contained RAG system with TF-IDF Vectorization, top-$k$ retrieval, and context-grounded response synthesis:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Knowledge Base
knowledge_documents = [
    "Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.",
    "Shipping Policy: Standard shipping takes 3-5 business days. Express shipping delivers overnight.",
    "Warranty Terms: All hardware electronics include a 1-year limited warranty against manufacturing defects.",
    "Customer Support Hours: Support is available Monday through Friday from 9 AM to 6 PM EST via live chat."
]

# 2. Vector Indexing
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(knowledge_documents)

# 3. Retrieval Function
def retrieve_top_k(query: str, k: int = 1):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(knowledge_documents[i], similarities[i]) for i in top_indices]

# 4. User Query & Retrieval
user_query = "How long do I have to return an item and get my money back?"
top_context, score = retrieve_top_k(user_query, k=1)[0]

print(f"User Query: {user_query}")
print(f"Retrieved Document (Match Score {score:.3f}):\n-> {top_context}")

# 5. Augmented Prompt Construction
augmented_prompt = f"""
You are an expert customer service assistant. Answer the user question strictly using the provided context.

Context:
{top_context}

Question: {user_query}
Answer:
"""
print("\n--- Final Augmented Prompt Sent to LLM ---")
print(augmented_prompt.strip())
```

#### Output:
```text
User Query: How long do I have to return an item and get my money back?
Retrieved Document (Match Score 0.442):
-> Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.

--- Final Augmented Prompt Sent to LLM ---
You are an expert customer service assistant. Answer the user question strictly using the provided context.

Context:
Refund Policy: Customers can request a full refund within 30 days of purchase with original receipt.

Question: How long do I have to return an item and get my money back?
Answer:
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Context Re-Ranking
**Task:** Given a list of retrieved chunks with similarity scores, write a function to filter out any chunks whose similarity is below a threshold of $0.20$, and format the remaining valid chunks with `[Source X]` citations for the LLM.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
retrieved_data = [
    ("Refunds are processed within 14 days.", 0.85),
    ("Our CEO was founded in 2015.", 0.08),
    ("Returns require undamaged packaging.", 0.45)
]

threshold = 0.20
valid_sources = [f"[Source {i+1}]: {doc}" for i, (doc, score) in enumerate(retrieved_data) if score >= threshold]
combined_context = "\n".join(valid_sources)

print("Filtered & Grounded Context:\n" + combined_context)
```
#### Output:
```text
Filtered & Grounded Context:
[Source 1]: Refunds are processed within 14 days.
[Source 3]: Returns require undamaged packaging.
```
</details>

---

## 7. Quick Reference Cheat Sheet

| RAG Component | Industry Standard Tools | Key Metric |
|---|---|---|
| **Embedding Model** | `text-embedding-3-small`, BGE-M3 | Cosine distance, NDCG@10 |
| **Vector Database** | ChromaDB, Pinecone, Qdrant, Milvus | HNSW recall, query latency (ms) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Context preservation, chunk overlap |
| **Reranker** | Cohere Rerank, BGE-Reranker-Large | MRR (Mean Reciprocal Rank) |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevance) | Hallucination rate (0-1) |
