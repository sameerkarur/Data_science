# RAG Architectures & Retrieval Engineering — Practice Problems (50+)

## Setup Instructions
Open `practice.ipynb` in Jupyter Notebook or VS Code to run interactive code cells.

### Q1. What is Retrieval-Augmented Generation (RAG)? Write a dictionary defining its 3 core stages: Ingestion, Retrieval, Synthesis.

```python
# Your code here
```

### Q2. Simulate chunking a long document into fixed 200-character windows with a 50-character stride.

```python
# Your code here
```

### Q3. Explain why chunk overlap is critical in document ingestion.

```python
# Your code here
```

### Q4. Implement a simple word-count based token estimator (approx 0.75 words per token).

```python
# Your code here
```

### Q5. Construct a structured prompt template inserting retrieved context into a zero-hallucination instruction.

```python
# Your code here
```

### Q6. Demonstrate character-level recursive splitting heuristics (paragraphs '\n\n' -> sentences '\n' -> spaces ' ').

```python
# Your code here
```

### Q7. Calculate Cosine Similarity between two 3D vector embeddings manually using math or numpy.

```python
# Your code here
```

### Q8. Demonstrate Euclidean distance between the same two vectors and contrast with Cosine similarity.

```python
# Your code here
```

### Q9. Simulate an In-Memory Document Store with document IDs, metadata, and text passages.

```python
# Your code here
```

### Q10. Implement a simple Keyword (Lexical) Search filtering passages containing any query term.

```python
# Your code here
```

### Q11. Simulate Reciprocal Rank Fusion (RRF) scoring for a document ranked #2 in BM25 and #4 in Dense retrieval.

```python
# Your code here
```

### Q12. Explain the difference between Dense Retrieval and Sparse Retrieval.

```python
# Your code here
```

### Q13. Create a simulated Metadata Filter filtering chunks where department == 'Engineering'.

```python
# Your code here
```

### Q14. Implement a simple Top-K selection taking an array of similarity scores and returning top 3 indices.

```python
# Your code here
```

### Q15. Simulate a Cross-Encoder Re-Ranker scoring query-passage pairs.

```python
# Your code here
```

### Q16. Explain the 'Lost in the Middle' phenomenon in LLM context windows.

```python
# Your code here
```

### Q17. Demonstrate how to place the highest-scored document at the top of the prompt to avoid lost-in-the-middle.

```python
# Your code here
```

### Q18. Simulate a Citation Attribution check: verify if the answer text contains substrings from the source document.

```python
# Your code here
```

### Q19. Define Faithfulness metric in RAG evaluation (Ragas framework).

```python
# Your code here
```

### Q20. Define Answer Relevance metric in RAG evaluation.

```python
# Your code here
```

### Q21. Define Context Recall metric in RAG evaluation.

```python
# Your code here
```

### Q22. Construct a query transformation function for hypothetical document embeddings (HyDE).

```python
# Your code here
```

### Q23. Simulate multi-query expansion (generating 3 alternative rephrasings of a user query).

```python
# Your code here
```

### Q24. Implement a basic semantic cache dictionary hashing queries to prior answers.

```python
# Your code here
```

### Q25. Calculate memory savings when using FP16 instead of FP32 embeddings for 100,000 vectors of dim 1536.

```python
# Your code here
```

### Q26. Simulate Parent-Child document chunking (retrieving small child chunk, injecting larger parent chunk into prompt).

```python
# Your code here
```

### Q27. Implement a guardrail checking if query is attempting prompt injection (e.g., 'ignore previous instructions').

```python
# Your code here
```

### Q28. Demonstrate how to parse an answer into structured JSON schema using standard library json.

```python
# Your code here
```

### Q29. Simulate Context Window Token Overflow detection given a max context budget of 4096 tokens.

```python
# Your code here
```

### Q30. Format a conversational RAG prompt incorporating chat history.

```python
# Your code here
```

### Q31. Write a simple MMR (Maximal Marginal Relevance) formula simulator balancing relevancy and novelty.

```python
# Your code here
```

### Q32. Demonstrate chunking by sentence count (grouping every 3 sentences together).

```python
# Your code here
```

### Q33. Simulate an Extractive QA span extractor finding substring bounds.

```python
# Your code here
```

### Q34. Explain the difference between Self-RAG and traditional RAG.

```python
# Your code here
```

### Q35. Implement a simple relevance score threshold filter (rejecting docs with similarity < 0.70).

```python
# Your code here
```

### Q36. Demonstrate query routing (classifying if query requires RAG vs standard LLM chat).

```python
# Your code here
```

### Q37. Implement a Markdown table formatter for retrieved policy comparison.

```python
# Your code here
```

### Q38. Calculate the storage footprint of 50,000 document metadata records in Python dictionary.

```python
# Your code here
```

### Q39. Demonstrate how to strip boilerplate headers and footers from raw scraped text.

```python
# Your code here
```

### Q40. Simulate asynchronous retrieval latency comparison (sequential vs parallel simulated).

```python
# Your code here
```

### Q41. Explain the concept of ColBERT late interaction token-level RAG retrieval.

```python
# Your code here
```

### Q42. Construct a zero-shot grading prompt for LLM-as-a-judge checking answer correctness.

```python
# Your code here
```

### Q43. Implement token truncation keeping only first N words to avoid exceeding LLM context length.

```python
# Your code here
```

### Q44. Simulate vector normalization (unit vector scaling: v / ||v||).

```python
# Your code here
```

### Q45. Explain why normalized vectors allow using Dot Product as a fast substitute for Cosine Similarity.

```python
# Your code here
```

### Q46. Implement an automated Fallback handler when vector retrieval returns empty results.

```python
# Your code here
```

### Q47. Construct a structured JSON prompt template enforcing typed outputs for RAG information extraction.

```python
# Your code here
```

### Q48. Demonstrate date-based metadata filtering on policy documents.

```python
# Your code here
```

### Q49. Simulate Cross-Lingual RAG query translation step.

```python
# Your code here
```

### Q50. Summarize the end-to-end RAG architecture in a complete runnable Python function.

```python
# Your code here
```
