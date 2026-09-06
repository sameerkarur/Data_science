# Interview Q&A — GenAI Optimization & Fine-Tuning Strategies

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the trade-offs between Fine-Tuning, Prompt Engineering, and RAG.

**Answer:** Prompt Engineering: zero training cost, instant iteration, but limited by context window and requires strong models. RAG: grounds models in external dynamic enterprise data with verifiable source citations without weight updates, ideal for evolving knowledge. Fine-Tuning: updates model weights to adapt tone, style, specific output syntax, or specialized domain jargon, but expensive to train, prone to catastrophic forgetting, and cannot access private real-time data unless paired with RAG.

### Q2. What is PEFT (Parameter-Efficient Fine-Tuning) and why is it preferred over full fine-tuning?

**Answer:** Full fine-tuning updates all billions of parameters, requiring massive VRAM and storing a separate multi-gigabyte checkpoint for every specialized task. PEFT freezes 99%+ of the base model weights and trains a tiny fraction of parameters (adapters), slashing VRAM by 80% and enabling modular adapter swapping on a single shared base model.

### Q3. Explain LoRA (Low-Rank Adaptation) mathematically.

**Answer:** LoRA freezes the pre-trained weight matrix W_0 in R^{d x k} and models weight updates as the low-rank decomposition ΔW = B * A, where A in R^{r x k} is initialized with Gaussian noise and B in R^{d x r} is initialized with zeros, with rank r << min(d, k). Forward pass: h = W_0 x + (α / r) B A x. When r = 8, trainable parameters drop by 10,000x.

### Q4. What is QLoRA (Quantized Low-Rank Adaptation)?

**Answer:** QLoRA quantizes the frozen base model to 4-bit NormalFloat (NF4) precision, introduces Double Quantization (quantizing quantization constants), and uses Paged Optimizers to manage memory spikes. This enables fine-tuning a 70B parameter model on a single 48GB GPU without performance degradation.

### Q5. Explain Prefix Tuning and Prompt Tuning.

**Answer:** Prompt Tuning prepends a small sequence of continuous learnable embedding vectors (virtual tokens) to the input prompt, training only these soft prompt embeddings while freezing all transformer weights. Prefix Tuning prepends learnable key and value parameter vectors to every attention layer throughout the network.

### Q6. What is RLHF (Reinforcement Learning from Human Feedback)?

**Answer:** 1. Train a base language model with Supervised Fine-Tuning (SFT) on high-quality demonstration data. 2. Train a Reward Model by collecting human pairwise preference rankings (Response A > Response B) and training a scalar reward scoring network using Bradley-Terry preference loss. 3. Fine-tune the SFT model using Proximal Policy Optimization (PPO) to maximize reward while penalizing divergence from the base model via KL penalty.

### Q7. Explain DPO (Direct Preference Optimization) and why it replaces PPO in modern alignment.

**Answer:** DPO mathematically proves that the constrained RLHF objective can be solved in closed form without training a separate reward model or executing unstable PPO reinforcement learning loops. It optimizes the policy model directly on pairwise preference data (y_win, y_lose) via binary cross-entropy on implicit log-ratio rewards, delivering greater stability and faster convergence.

### Q8. What is Quantization in Large Language Models (FP16, INT8, INT4, AWQ, GPTQ)?

**Answer:** Quantization converts high-precision floating-point weights into lower-precision integers, reducing VRAM and increasing memory bandwidth throughput. GPTQ performs post-training layer-wise quantization using second-order Hessian information. AWQ (Activation-aware Weight Quantization) protects the 1% most salient weight channels based on activation magnitudes, retaining near-FP16 accuracy at 4-bit sizes.

### Q9. What is KV Caching and why is it essential for autoregressive LLM inference?

**Answer:** In autoregressive generation, each new token requires computing attention against all previous tokens. Without KV caching, Keys and Values for all past tokens must be recomputed on every step (O(N²) redundant operations). The KV Cache stores computed Key and Value projection matrices in GPU VRAM across generation steps, reducing computation to O(N) per new token.

### Q10. Explain PagedAttention and how vLLM revolutionizes LLM serving throughput.

**Answer:** Standard KV caches require contiguous physical VRAM allocations based on maximum sequence length, wasting up to 60-80% of VRAM due to internal fragmentation and over-allocation. PagedAttention partitions KV caches into non-contiguous virtual memory blocks (pages) mapped dynamically like OS virtual memory, eliminating fragmentation, enabling memory sharing across parallel requests, and increasing throughput by 2x–4x.

### Q11. What is Speculative Decoding?

**Answer:** A small, ultra-fast draft model generates K speculative candidate tokens in rapid sequence. The large target model processes all K candidate tokens in parallel in a single forward pass, verifying and accepting valid tokens. Because verifying is done in parallel via matrix multiplication, generation speed increases 2x–3x without altering output probability distributions.

### Q12. Explain FlashAttention (FlashAttention-1, 2, and 3).

**Answer:** Standard attention materializes the massive N x N attention matrix in slow GPU High Bandwidth Memory (HBM), resulting in memory bandwidth bottlenecks. FlashAttention uses tiling to compute softmax incrementally without writing intermediate N x N attention matrices to HBM, keeping operations within ultra-fast GPU SRAM and achieving 2x–4x speedups while reducing memory from O(N²) to O(N).

### Q13. What is Prompt Caching and how does it reduce token costs?

**Answer:** Prompt caching identifies identical prompt prefixes across requests (such as large system prompts, few-shot exemplars, or long reference documents). The server stores the pre-computed KV cache states on GPU memory or disk; future requests sharing the exact prefix reuse the cached KV states, cutting TTFT latency by 80% and token costs by 50%.

### Q14. Differentiate between Time-To-First-Token (TTFT) and Tokens-Per-Second (TPS).

**Answer:** TTFT is the latency from when a user submits a prompt until the first generated token appears, dominated by the prompt prefill phase (processing the input context). TPS (generation throughput) is the rate at which subsequent output tokens are generated per second, constrained by memory bandwidth during autoregressive decoding.

### Q15. What is Continuous Batching (Iteration-Level Batching)?

**Answer:** Traditional batching waits for all requests in a batch to finish generation before processing new requests, wasting GPU compute when sequences finish early. Continuous batching dynamically inserts new incoming requests and removes completed requests on every single iteration step, maximizing GPU utilization.

### Q16. Explain Model Distillation for LLMs.

**Answer:** A large frontier model (e.g. GPT-4o) generates reasoning demonstrations, synthetic training data, or output logits used to train a smaller model (e.g. 8B parameter model). The smaller student model learns to approximate the capabilities of the larger teacher model within a specific domain at a fraction of the inference cost.

### Q17. What is Model Merging (MergeKit: SLERP, DARE, TIES)?

**Answer:** Model merging combines the weights of multiple specialized fine-tuned models of identical architecture into a single unified model without further training. SLERP (Spherical Linear Interpolation) interpolates weights along high-dimensional spherical arcs. TIES and DARE resolve parameter interference by pruning redundant updates and harmonizing directional conflicts.

### Q18. How do you evaluate GenAI models using G-Eval?

**Answer:** G-Eval is a framework that uses LLMs (like GPT-4) with Chain-of-Thought prompting and form-based scoring rubrics to evaluate complex generation quality (coherence, groundedness, consistency), scoring with high correlation to human expert judgments.

### Q19. What is RAG Triad of Metrics (Truffles / TruLens / Ragas)?

**Answer:** 1. Context Relevance: is the retrieved context relevant to the user query? 2. Groundedness (Faithfulness): is the generated answer strictly derived from and supported by the retrieved context? 3. Answer Relevance: does the generated response directly answer the user's initial question?

### Q20. Explain Semantic Router for prompt optimization.

**Answer:** A semantic router embeds incoming user queries and routes them to specialized small models, cached responses, or deterministic APIs based on vector similarity thresholds, avoiding expensive frontier LLM calls for routine or deterministic intents.

### Q21. What is Tensor Parallelism (TP) vs Pipeline Parallelism (PP)?

**Answer:** Tensor Parallelism splits individual weight matrices (e.g. attention heads, MLP projections) horizontally or vertically across GPUs within a single machine using high-speed NVLink. Pipeline Parallelism partitions layers sequentially across different machines, pipelining mini-batches through layer stages.

### Q22. Explain the Roofline Model for LLM inference optimization.

**Answer:** The Roofline model plots operational computational performance (GFLOPs/s) against Arithmetic Intensity (FLOPs per byte of memory transferred). It reveals whether an inference workload is compute-bound (saturating GPU SM cores) or memory-bandwidth-bound (waiting for weights to transfer from HBM).

### Q23. What is Chunked Prefill?

**Answer:** In LLM serving, long prefill requests can block short decoding requests, causing massive latency spikes. Chunked prefill breaks long prompt contexts into smaller chunks processed across multiple iteration steps interleaved with decoding passes, smoothing tail latency.

### Q24. How do Context Window Extension techniques work (RoPE Scaling: Linear vs YaRN)?

**Answer:** Rotary Position Embeddings (RoPE) represent position via complex rotation. Extending context beyond pre-trained limits requires scaling position frequencies. Linear scaling divides positions uniformly by factor s, but dilutes high-frequency components. YaRN (Yet another RoPE extensioN) applies non-uniform frequency interpolation, preserving high-frequency resolution for local tokens while stretching low frequencies for long-distance attention.

### Q25. What is GQA (Grouped-Query Attention) vs MQA (Multi-Query Attention)?

**Answer:** Multi-Head Attention (MHA) maintains independent Key and Value heads for every Query head, consuming vast KV cache VRAM. Multi-Query Attention (MQA) shares a single K and V head across all Query heads, slashing VRAM but risking quality drops. Grouped-Query Attention (GQA) groups query heads (e.g. 8 query heads share 1 KV head), balancing memory reduction with representational quality (standard in LLaMA 3).

### Q26. Explain Token Pruning and Token Merging in Transformers.

**Answer:** Token pruning removes uninformative or redundant tokens from intermediate layers based on attention weight scores. Token merging dynamically blends similar token vectors using cosine similarity, reducing sequence length inside deep layers without retraining.

### Q27. What is Triton (OpenAI) and how does it optimize custom GPU kernels?

**Answer:** Triton is a Python-based language and compiler for writing highly optimized GPU kernels without writing complex C++/CUDA. It automatically handles memory coalescing, shared memory caching, and thread block synchronization, accelerating custom attention and quantization operations.

### Q28. Explain Cost-Latency-Quality Pareto Frontier in GenAI system design.

**Answer:** System design requires navigating the Pareto trade-off: Frontier models offer highest quality but high latency and cost; SLMs (Small Language Models) or quantized models offer sub-100ms latency and minimal cost with lower reasoning capacity. Production architectures use routing and cascade tiers to achieve near-frontier quality at minimal cost.

### Q29. What is Synthetic Data Generation for fine-tuning?

**Answer:** Using high-capability frontier models to generate structured question-answer pairs, edge case scenarios, and reasoning traces from raw unstructured data. Filtering data using heuristic checks and reward models yields curated datasets that train specialized, compact open-source models.

### Q30. How do you prevent Data Contamination during LLM evaluation?

**Answer:** Ensure benchmark evaluation datasets are strictly held out from pre-training and fine-tuning corpora. Techniques: (1) synthetic dynamic benchmarks, (2) Canary strings / GUIDs placed in benchmark data to detect if training scrapers ingested test sets, and (3) private enterprise golden sets.
