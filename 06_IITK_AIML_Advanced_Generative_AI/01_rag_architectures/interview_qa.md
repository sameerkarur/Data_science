# Interview Q&A — RAG Architectures & Retrieval Engineering

> **30 High-Yield Questions & Model Answers** for AI/ML and GenAI Technical Interviews.

### Q1. What is RAG and why is it preferred over fine-tuning for enterprise QA?

**Answer:** RAG provides real-time retrieval from external knowledge bases without model retraining, offers dynamic access control, prevents catastrophic forgetting, and provides verifiable source citations.

### Q2. Explain Dense vs Sparse retrieval and why Hybrid Search is best.

**Answer:** Dense captures semantic intent via neural vector embeddings. Sparse (BM25) guarantees exact keyword and acronym matching. Hybrid search combines both via Reciprocal Rank Fusion (RRF) for superior recall.

### Q3. What is the role of chunk size and overlap in document chunking?

**Answer:** Chunk size determines the context window resolution. Overlap prevents loss of semantic coherence across chunk boundaries. Too small misses macro context; too large dilutes embedding relevance.

### Q4. How does Reciprocal Rank Fusion (RRF) work mathematically?

**Answer:** RRF scores each document as sum(1 / (k + rank_i)) across ranking systems i, with k typically set to 60. It normalizes disparate ranking scores without needing probability calibration.

### Q5. What causes hallucinations in RAG and how do you prevent them?

**Answer:** Hallucinations happen when retrieved context is irrelevant or when LLMs fabricate details. Mitigate with strict negative constraints, minimum similarity thresholds, re-ranking, and citation checks.

### Q6. Explain the difference between Bi-Encoders and Cross-Encoders.

**Answer:** Bi-Encoders embed query and documents independently into vectors for fast search. Cross-Encoders process query and document jointly through attention layers for high accuracy re-ranking.

### Q7. What is Lost-in-the-Middle and how do you resolve it?

**Answer:** LLMs prioritize text at the beginning and end of long contexts. Resolved by re-ordering retrieved passages so highest-scoring chunks appear at the start and end of the prompt.

### Q8. Explain Parent-Document Retrieval.

**Answer:** Small child chunks are embedded and indexed for precise search, but the larger parent document/section is passed to the LLM to provide complete context.

### Q9. What are the core evaluation metrics in the Ragas framework?

**Answer:** Faithfulness (groundedness in context), Answer Relevance (alignment with user question), Context Precision (signal-to-noise ratio in retrieved chunks), and Context Recall.

### Q10. What is Hypothetical Document Embeddings (HyDE)?

**Answer:** HyDE prompts an LLM to generate a hypothetical answer to the query, then embeds that hypothetical passage to search the vector database, improving dense retrieval matching.

### Q11. How do you handle multi-modal inputs in a modern RAG system?

**Answer:** Extract and embed text using text encoders, extract image captions/OCR, and use joint multimodal embedding spaces (e.g. CLIP) to retrieve both text passages and visual diagrams.

### Q12. What is MMR (Maximal Marginal Relevance)?

**Answer:** An algorithm that selects chunks maximizing relevance to the query while penalizing similarity to already-selected chunks, maximizing informational diversity.

### Q13. How do vector databases handle metadata filtering?

**Answer:** They use pre-filtering (filtering metadata before ANN graph search) or post-filtering (filtering results after vector retrieval) to enforce tenant isolation and category scopes.

### Q14. What is Self-RAG?

**Answer:** An adaptive framework where models output special reflection tokens determining whether retrieval is needed, assessing retrieval relevance, and critiquing answer faithfulness.

### Q15. How do embedding models represent domain-specific acronyms?

**Answer:** General models often struggle with proprietary jargon. Mitigate via hybrid BM25 search, domain fine-tuning, or prepending an internal glossary during ingestion.

### Q16. Explain the difference between Cosine Similarity and Dot Product.

**Answer:** Cosine similarity normalizes vectors to unit length, measuring purely angular alignment. Dot product also scales with vector magnitude. For normalized vectors, both are identical.

### Q17. What is Query Routing in Agentic RAG?

**Answer:** A classification step where a router agent assesses query complexity and routes it to vector search, SQL databases, web search, or direct conversational LLM generation.

### Q18. How do you prevent prompt injection in RAG pipelines?

**Answer:** Sanitize inputs, use delimiter tags (e.g. XML tags), validate context boundaries, and run secondary safety classifier models before synthesizing outputs.

### Q19. What is ColBERT and why is it called late interaction?

**Answer:** ColBERT encodes query and doc tokens separately, then computes token-level MaxSim at search time. It balances the speed of bi-encoders with the accuracy of cross-encoders.

### Q20. How does Context Compression work in LangChain?

**Answer:** A small secondary model scans retrieved chunks and extracts only the relevant sentences, eliminating irrelevant fluff before prompting the main LLM.

### Q21. What is an Inverted Index?

**Answer:** A data structure mapping every distinct word/token to the list of documents and positions where it appears, powering BM25 search.

### Q22. How do you monitor drift in a production RAG system?

**Answer:** Track query embedding drift, user feedback thumbs up/down, average similarity scores of retrieved passages, and proportion of queries triggering safety/fallback guardrails.

### Q23. What is chunk size trade-off for technical vs narrative text?

**Answer:** Technical code/API docs benefit from smaller chunks (200-400 tokens) for precise syntax matching. Legal/policy docs need larger chunks (800-1200 tokens) to capture multi-clause logic.

### Q24. Explain the trade-off between In-Memory vs Persistent Vector Stores.

**Answer:** In-memory (e.g. FAISS CPU) provides sub-millisecond retrieval for small collections but is volatile. Persistent stores (Chroma, Pinecone, Milvus) handle billions of vectors with ACID replication.

### Q25. What is Graph RAG?

**Answer:** Combines knowledge graphs (entities and relationships) with vector search to answer complex multi-hop queries that traverse interconnected documents.

### Q26. What is semantic caching in LLM infrastructure?

**Answer:** Storing previous queries and responses in a vector cache. If a new query has cosine similarity > 0.95 with a cached query, return the cached answer immediately, saving latency and cost.

### Q27. How do you implement Role-Based Access Control (RBAC) in RAG?

**Answer:** Attach security permission tags to each chunk's metadata during ingestion and filter vector queries with user role tokens so users never retrieve unauthorized documents.

### Q28. What is the cold-start problem in RAG?

**Answer:** When a new database has few documents or un-indexed files, leading to low recall. Mitigated by web search fallback or clear system disclaimers.

### Q29. What is LLM-as-a-Judge?

**Answer:** Using a high-capability LLM (e.g. GPT-4) guided by precise rubric prompts to automatically evaluate the accuracy, tone, and groundedness of smaller production models.

### Q30. Summarize the single biggest failure mode in enterprise RAG systems.

**Answer:** Retrieval failure: fetching irrelevant chunks. If the right information is not in the retrieved context, the LLM cannot synthesize an accurate factual answer.
