# Chapter 2: Vector Databases & High-Dimensional HNSW Indexing
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
