# Interview Q&A — Vector Databases & ChromaDB

> **30 High-Yield Questions & Model Answers** for AI/ML and GenAI Technical Interviews.

### Q1. What is an Approximate Nearest Neighbor (ANN) search and why is it necessary?

**Answer:** Exact K-NN search requires computing distance against every stored vector, which scales as O(N) and becomes impractical for millions of high-dimensional vectors. ANN uses hierarchical graph or tree indices (e.g. HNSW, IVF) to query in O(log N) time with over 95-99% recall accuracy.

### Q2. Explain the internal mechanics of HNSW (Hierarchical Navigable Small World).

**Answer:** HNSW constructs multi-layered proximity graphs. The top layers contain sparse nodes with long-distance links for rapid geometric navigation across the vector space. As the search descends through layers, the graph becomes progressively denser until the exact local neighborhood is identified.

### Q3. How does Cosine Similarity differ from Euclidean (L2) distance in high dimensions?

**Answer:** Cosine similarity measures the cosine of the angle between two vectors, completely ignoring their magnitude (ideal for text semantics where text length shouldn't distort meaning). Euclidean distance measures absolute straight-line spatial distance. When vectors are L2-normalized, Cosine distance and squared Euclidean distance are monotonically equivalent.

### Q4. What is Vector Quantization (Product Quantization & Scalar Quantization)?

**Answer:** Quantization compresses vectors to fit into limited RAM. Scalar Quantization reduces 32-bit floats to 8-bit integers (4x memory reduction). Product Quantization divides the vector into sub-vectors and replaces each sub-vector with a codebook centroid index, achieving 8x-16x compression.

### Q5. What are Matryoshka Embeddings (MRL)?

**Answer:** Embeddings trained so that earlier dimensions contain the most significant semantic information. This enables slicing high-dimensional embeddings (e.g., 1536 down to 512 or 256) with negligible loss in retrieval accuracy, saving massive storage and search latency.

### Q6. Explain Pre-filtering vs Post-filtering in vector database queries.

**Answer:** Pre-filtering filters records by metadata before performing vector search (guarantees filter match, but can fragment ANN graph traversal). Post-filtering performs ANN search first, then discards non-matching metadata records (can return fewer than K results if many top hits are filtered out).

### Q7. How does ChromaDB achieve local persistence?

**Answer:** ChromaDB stores vector indices in memory using native HNSW C++ bindings and records document texts, embeddings, and metadata into a persistent local SQLite/DuckDB file on disk.

### Q8. What is an IVF (Inverted File) index in FAISS?

**Answer:** IVF partitions the vector space into Voronoi cells using K-Means clustering. At query time, the system compares the query only against the centroids of the nearest cells and searches inside those specific inverted lists, skipping the vast majority of the database.

### Q9. How do you handle multi-tenant isolation in a vector database?

**Answer:** Via namespace isolation (distinct sub-indexes per tenant), separate collections, or strict metadata filtering on tenant_id enforced in the query pipeline.

### Q10. What is the impact of embedding dimension on search latency?

**Answer:** Distance computations scale linearly with vector dimension D (O(D)). Higher dimensions require more memory bandwidth and CPU/GPU SIMD cycles, increasing latency.

### Q11. How does a vector database handle deletes and updates?

**Answer:** Most graph-based indices (like HNSW) perform soft deletes (tombstoning) because removing a node breaks graph connectivity. Re-indexing or compaction is scheduled periodically to prune tombstones.

### Q12. What is the difference between In-Memory FAISS and ChromaDB?

**Answer:** FAISS is a specialized C++ computation library for similarity search without metadata storage or server management. ChromaDB is a full vector database wrapping FAISS/hnswlib with document storage, metadata filtering, and client APIs.

### Q13. Explain the trade-offs of using GPU acceleration for vector search.

**Answer:** GPUs provide massive SIMD parallelism, speeding up brute force flat search by 20x-50x. However, GPU VRAM is expensive and limited compared to host RAM, making hybrid CPU-GPU architectures standard for massive datasets.

### Q14. What is embedding normalization and why is it standard practice?

**Answer:** Normalizing vectors to unit length (L2 norm = 1) allows replacing expensive cosine similarity calculations with simple vector dot products, greatly accelerating distance computations.

### Q15. How do you select the best distance metric for a given embedding model?

**Answer:** Always follow the training objective of the embedding model: models trained with Cosine loss should use Cosine distance; models trained with inner product (MIPS) should use Dot Product.

### Q16. What causes the Curse of Dimensionality in vector spaces?

**Answer:** As dimensionality increases, the volume of the space grows exponentially and distances between random pairs of points concentrate around the mean, making discrimination harder without specialized ANN structures.

### Q17. How do you benchmark vector database performance?

**Answer:** Using QPS (Queries Per Second), Latency percentiles (P50, P95, P99), Index build time, and Recall@K against a brute-force ground truth.

### Q18. What is hybrid search in vector databases?

**Answer:** Combining dense semantic vector search with sparse keyword search (e.g. BM25) and fusing the ranked results using Reciprocal Rank Fusion (RRF) or weighted score summation.

### Q19. Can you store unstructured media (audio, video, images) directly in vector databases?

**Answer:** Vector databases store the high-dimensional embedding vectors representing the media, alongside metadata URLs/pointers to the raw binary assets stored in object stores (e.g. AWS S3).

### Q20. What is an Embedding Ingestion Pipeline?

**Answer:** An ETL pipeline that ingests raw documents, cleans and chunks text, calls embedding models in parallel batches, and writes vector records with metadata into the vector database.

### Q21. How does database sharding work in vector search?

**Answer:** Collections are partitioned into shards across cluster nodes. A coordinator node scatters the query vector to all shards, gathers local top-K results, and merges them into a global top-K ranking.

### Q22. What is WAL (Write-Ahead Log) in vector databases?

**Answer:** A logging mechanism that writes mutations to disk before applying them to in-memory index graphs, guaranteeing data recovery if a crash occurs during graph rebalancing.

### Q23. What is the difference between dense and sparse vector representations?

**Answer:** Dense vectors have continuous real-valued floats across every dimension (capturing latent concepts). Sparse vectors have mostly zeros, with non-zero values corresponding directly to vocabulary token indices.

### Q24. How does metadata indexing work in vector stores?

**Answer:** Metadata is indexed using B-Trees, inverted indexes, or bitmap indexes alongside the vector graph to enable sub-millisecond filtering on categorical and numerical fields.

### Q25. What are the dangers of re-indexing in production?

**Answer:** Re-indexing consumes significant CPU/RAM and can cause latency spikes. It should be performed on a blue/green shadow collection and swapped atomically once complete.

### Q26. Explain the role of the similarity threshold in preventing hallucinations.

**Answer:** Setting a minimum similarity cutoff (e.g. Cosine Similarity >= 0.75) ensures that out-of-domain or irrelevant queries retrieve zero chunks rather than low-quality noise.

### Q27. How do you handle embedding model version upgrades?

**Answer:** Different embedding models produce non-comparable vector spaces. Upgrading models requires spinning up a new collection, re-embedding the entire corpus, and switching query traffic.

### Q28. What is the difference between Cosine Distance and Cosine Similarity?

**Answer:** Cosine Similarity ranges from -1 to 1 (1 = identical). Cosine Distance is 1 - CosineSimilarity, ranging from 0 to 2 (0 = identical). Databases minimize distance.

### Q29. What is zero-shot classification using vector databases?

**Answer:** Embed candidate class label descriptions into the vector space. Embed the input sample. The nearest neighbor class vector represents the zero-shot predicted class.

### Q30. Summarize the single biggest advantage of vector databases in enterprise AI.

**Answer:** They bridge the gap between unstructured human language and deterministic compute, allowing LLMs to search millions of proprietary documents with sub-second latency.
