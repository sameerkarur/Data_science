# Vector Databases & ChromaDB — Practice Problems (50+)

## Setup Instructions
Open `practice.ipynb` in Jupyter Notebook or VS Code to run interactive code cells.

### Q1. What is a Vector Database? Write a Python dictionary listing 4 popular production vector stores.

```python
# Your code here
```

### Q2. Simulate creating a collection in a vector database schema.

```python
# Your code here
```

### Q3. Compute the Euclidean L2 norm of a vector [3.0, 4.0] using numpy.

```python
# Your code here
```

### Q4. Normalize a batch of 3 vectors so their L2 norms equal 1.0.

```python
# Your code here
```

### Q5. Demonstrate Cosine Distance ($1 - \text{CosineSimilarity}$) calculation.

```python
# Your code here
```

### Q6. Simulate inserting 3 documents with IDs, vector embeddings, and metadata into a list.

```python
# Your code here
```

### Q7. Implement Exact Nearest Neighbors (K-NN brute force) search on a small vector array.

```python
# Your code here
```

### Q8. Explain the difference between Exact K-NN and Approximate Nearest Neighbors (ANN).

```python
# Your code here
```

### Q9. Explain HNSW (Hierarchical Navigable Small World) index graphs intuitively.

```python
# Your code here
```

### Q10. Simulate Metadata Filtering: Retrieve only vectors whose metadata has year == 2026.

```python
# Your code here
```

### Q11. Calculate the memory required to store 1,000,000 768-dimensional float32 embeddings in RAM.

```python
# Your code here
```

### Q12. Calculate memory savings when using Scalar Quantization (INT8) on the same 1M vectors.

```python
# Your code here
```

### Q13. Demonstrate how to generate deterministic character n-gram pseudo-embeddings in Python.

```python
# Your code here
```

### Q14. Simulate deleting a vector by ID from a local storage dictionary.

```python
# Your code here
```

### Q15. Implement an Upsert operation (update if exists, else insert).

```python
# Your code here
```

### Q16. Demonstrate batch embedding ingestion (processing in batches of 2).

```python
# Your code here
```

### Q17. Explain the difference between Inverted File Index (IVF) and Flat index in FAISS.

```python
# Your code here
```

### Q18. Demonstrate Euclidean Distance to Cosine Similarity conversion for normalized vectors.

```python
# Your code here
```

### Q19. Simulate Top-K query returning document IDs and similarity scores sorted descending.

```python
# Your code here
```

### Q20. Explain Product Quantization (PQ) for vector compression.

```python
# Your code here
```

### Q21. Calculate the dot product of two orthogonal vectors and print interpretation.

```python
# Your code here
```

### Q22. Simulate a namespace partitioning pattern (e.g. multi-tenant isolation).

```python
# Your code here
```

### Q23. Demonstrate how to update metadata of an existing vector without re-embedding.

```python
# Your code here
```

### Q24. Calculate dimensionality reduction compression ratio from 1536 to 256 using Matryoshka embeddings.

```python
# Your code here
```

### Q25. Simulate hybrid query execution (filtering metadata tag then running vector search).

```python
# Your code here
```

### Q26. Explain the difference between ChromaDB persistent SQLite mode vs ephemeral client.

```python
# Your code here
```

### Q27. Simulate writing a collection export to JSON.

```python
# Your code here
```

### Q28. Demonstrate loading collection back from JSON.

```python
# Your code here
```

### Q29. Calculate cosine similarity across a 1x3 query vector and a 4x3 matrix in numpy.

```python
# Your code here
```

### Q30. Explain what an Embedding Drift is in production vector search.

```python
# Your code here
```

### Q31. Simulate a vector distance threshold filter keeping only docs with distance < 0.3.

```python
# Your code here
```

### Q32. Explain the M parameter in HNSW indexing.

```python
# Your code here
```

### Q33. Explain the efConstruction parameter in HNSW.

```python
# Your code here
```

### Q34. Demonstrate how to count total vectors in a nested dictionary store.

```python
# Your code here
```

### Q35. Simulate vector deduplication (detecting duplicate vectors with cosine similarity > 0.999).

```python
# Your code here
```

### Q36. Explain Inner Product metric vs L2 distance.

```python
# Your code here
```

### Q37. Demonstrate how to store and retrieve binary embeddings using bitwise XOR operations (Hamming distance).

```python
# Your code here
```

### Q38. Calculate the index construction time complexity for brute force flat index vs IVF.

```python
# Your code here
```

### Q39. Simulate query latency benchmarking in milliseconds.

```python
# Your code here
```

### Q40. Demonstrate key-value storage association between vector ID and raw text chunk.

```python
# Your code here
```

### Q41. Explain the role of WAL (Write-Ahead Logging) in persistent vector databases.

```python
# Your code here
```

### Q42. Simulate vector truncation: slice 1536 dim embedding down to first 512 dimensions.

```python
# Your code here
```

### Q43. Explain why Matryoshka Representation Learning (MRL) allows vector slicing without retraining.

```python
# Your code here
```

### Q44. Demonstrate how to filter vectors using multiple logical criteria (AND / OR).

```python
# Your code here
```

### Q45. Explain Sharding in distributed vector databases like Milvus or Qdrant.

```python
# Your code here
```

### Q46. Simulate computing average vector embedding (centroid) of 3 cluster vectors.

```python
# Your code here
```

### Q47. Explain the difference between dense embeddings and sparse vectors.

```python
# Your code here
```

### Q48. Demonstrate checking whether a collection exists before creation.

```python
# Your code here
```

### Q49. Simulate building a simple cosine similarity distance lookup table.

```python
# Your code here
```

### Q50. Summarize the complete lifecycle of a vector in a database in Python print statements.

```python
# Your code here
```
