# Interview Q&A — ChatGPT Applications & Tool Orchestration

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain OpenAI Function Calling (Tool Use) mechanics.

**Answer:** You provide the LLM with a list of function signatures in JSON Schema format (function name, description, parameter types). If the user query requires external data, the model halts natural language generation and outputs a structured JSON object containing the function name and extracted arguments: '{"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}'. Your application executes the function and returns the JSON result back to the model in a 'tool' message, which the model synthesizes into a final answer.

### Q2. What is the difference between parallel function calling and sequential function calling?

**Answer:** Parallel function calling allows the model to output multiple independent tool calls in a single turn (e.g. calling 'get_weather(city="London")' and 'get_weather(city="Paris")' simultaneously). Sequential function calling executes one call, feeds the result back, and lets the model decide the next call, necessary when parameter inputs depend on previous tool outputs.

### Q3. Explain the difference between Stateless APIs and Stateful Chat interfaces.

**Answer:** The OpenAI API is completely stateless: it retains zero memory of previous requests between distinct HTTP calls. To maintain stateful conversational chat, the client application must store the entire history of 'user', 'assistant', and 'system' messages and resubmit the complete conversation history with every new query.

### Q4. What are the primary conversational memory patterns in LangChain (Buffer, Summary, Entity)?

**Answer:** ConversationBufferMemory: stores every raw message verbatim; simple, but rapidly exceeds context limits. ConversationSummaryMemory: uses an LLM to continuously summarize past dialogue into a running synopsis, conserving tokens. ConversationEntityMemory: extracts and updates state about specific entities (people, projects, preferences) mentioned throughout the chat.

### Q5. Explain Server-Sent Events (SSE) and Streaming API responses.

**Answer:** Instead of waiting several seconds for the model to generate the complete response before returning a single large JSON payload, SSE establishes an open HTTP connection where the server pushes generated tokens as an asynchronous stream of text chunks ('data: {"choices": [{"delta": {"content": "hello"}}]}') the moment they are generated, drastically reducing Time-To-First-Token (TTFT) perceived latency.

### Q6. How do System Prompt Guardrails prevent jailbreaks in commercial ChatGPT bots?

**Answer:** Techniques: (1) Strong role definitions with unambiguous constraints, (2) Explicit refusal protocols for sensitive domains (PII, legal, medical advice), (3) Input preprocessing with moderation APIs, and (4) Secondary checker LLMs that audit draft responses before delivering to users.

### Q7. What is the OpenAI Moderation API and what categories does it check?

**Answer:** The Moderation API is a free endpoint that checks text against OpenAI's safety policies, returning scores across categories: hate, hate/threatening, harassment, self-harm, sexual, sexual/minors, and violence.

### Q8. Explain RAG (Retrieval-Augmented Generation) in ChatGPT enterprise applications.

**Answer:** RAG grounds the LLM in proprietary internal knowledge bases: user query -> embed query -> vector search top-k relevant document chunks -> inject chunks into system/user prompt -> generate factual answer with source citations, preventing hallucinations without model retraining.

### Q9. What is Hallucination in Large Language Models and why does it occur?

**Answer:** Hallucination is the generation of fluent, syntactically correct, but factually false or ungrounded assertions. It occurs because LLMs are statistical next-token probability predictors trained to optimize fluency and plausibility, not factual truth; they lack an internal epistemological truth-verification engine.

### Q10. How do you design a robust Multi-Agent System (e.g. AutoGen, CrewAI, LangGraph)?

**Answer:** Multi-agent systems partition complex tasks among specialized personas (e.g. Researcher, Coder, Reviewer). They coordinate via structured communication topologies (hierarchical supervisor, sequential pipeline, or stateful DAG graphs), utilizing shared memory and tool execution capabilities.

### Q11. Explain LangGraph and Stateful Multi-Agent Orchestration.

**Answer:** LangGraph models agent workflows as cyclical state graphs where nodes represent agent actions or tool executions, edges define routing logic (conditional branches based on LLM output), and a centralized state object tracks execution history with human-in-the-loop checkpoints.

### Q12. What is Human-in-the-Loop (HITL) in LLM applications?

**Answer:** HITL pauses agent execution before high-stakes actions (e.g. sending financial transactions, deleting databases, executing arbitrary shell code) to request explicit human confirmation or guidance through an approval interface.

### Q13. Explain OpenAI Assistants API (Threads, Runs, Vector Stores).

**Answer:** The Assistants API provides a managed stateful backend: 'Assistant' defines model, instructions, and tools. 'Thread' represents a persistent conversation session storing message history automatically. 'Run' executes the Assistant on a Thread, handling retrieval and code execution natively.

### Q14. How does the Code Interpreter (Advanced Data Analysis) tool work in ChatGPT?

**Answer:** It provides the model with a sandboxed, stateful Python runtime environment. The model generates Python code to parse uploaded files, perform data cleaning, execute complex mathematical computations, and generate Matplotlib/Seaborn plots returned directly to the user.

### Q15. What is Semantic Caching and how does GPTCache reduce API costs and latency?

**Answer:** Semantic caching embeds incoming user queries and computes cosine similarity against a vector database of past cached queries. If similarity exceeds a threshold (e.g. 0.95), it returns the cached response immediately, bypassing the expensive LLM API call entirely.

### Q16. Explain Token-Level Logprobs and how they evaluate model confidence.

**Answer:** Requesting 'logprobs=True' returns the log-probabilities of generated tokens and top alternative candidates. A low log-probability indicates the model was uncertain about that specific word, allowing automated flagging of potential hallucinations or low-confidence assertions.

### Q17. How do you build a Production Fallback Strategy across multiple LLM providers?

**Answer:** Use proxy frameworks (LiteLLM, Portkey) that implement automated fallback cascades: if primary model (e.g. GPT-4o) returns 429 (Rate Limit) or 500 (Error), the request automatically routes to a secondary provider (e.g. Claude 3.5 Sonnet, Mistral Large, or Azure OpenAI) within milliseconds.

### Q18. What is Prompt Compression (LLMLingua)?

**Answer:** Prompt compression uses a compact, fast language model to evaluate token perplexity across long context documents, dynamically pruning low-information tokens and filler words while preserving semantic meaning, reducing context length by up to 50% with minimal loss in QA accuracy.

### Q19. Explain Guardrails AI and NeMo Guardrails for conversational safety.

**Answer:** These frameworks enforce programmable constraints on LLM inputs and outputs: validating data types, verifying factual grounding against context, masking PII, blocking toxic speech, and enforcing deterministic dialog flows via Colang rules.

### Q20. What is Structured Extraction from Unstructured Documents?

**Answer:** Injecting unstructured text (invoices, legal contracts, medical reports) into an LLM and enforcing a strict JSON Schema/Pydantic model to extract key fields (invoice_number, total_amount, line_items) with validated types and fallback defaults.

### Q21. How do you handle Token Limits when summarizing a 200-page PDF document?

**Answer:** Techniques: (1) Map-Reduce (chunk document, summarize each chunk in parallel, then summarize the concatenated summaries), (2) Refine (iteratively update a running summary chunk-by-chunk), and (3) Hierarchical Tree Summarization.

### Q22. What is the difference between In-Memory Chat History and Database-Backed History (Redis, Postgres)?

**Answer:** In-memory stores history in server RAM, which is lost when the server restarts and fails across distributed load-balanced multi-instance backends. Database-backed history (e.g. Redis with TTL expiration or PostgreSQL) persists sessions across user logins and scales horizontally.

### Q23. Explain Few-Shot Dynamic Demonstration Selection with Vector DBs.

**Answer:** Instead of hardcoding the same 5 few-shot examples into every prompt, store hundreds of diverse exemplars in a vector database. At query time, embed the user query and retrieve the top-3 most semantically similar exemplars, dynamically constructing the most relevant few-shot prompt.

### Q24. What are Tokenizers (tiktoken) and why is token count different from word or character count?

**Answer:** 'tiktoken' implements BPE (Byte Pair Encoding) used by OpenAI. On average, 1 token ≈ 4 characters or 0.75 words in English. Common words are single tokens; rare words, non-English scripts, and code indentation split into multiple tokens.

### Q25. How do you implement Content Filtering and PII Masking before sending data to external APIs?

**Answer:** Run local regex and named entity recognition (spaCy, Microsoft Presidio) to detect and redact sensitive PII (Social Security Numbers, credit cards, emails, names) with anonymized placeholders ('<PERSON_1>') before transmitting prompts over the wire.

### Q26. What is Chain-of-Thought Scratchpad in multi-agent tool execution?

**Answer:** A designated section in the prompt where an agent logs intermediate thoughts, plan revisions, tool input formulations, and observation analyses before presenting the final response to the user.

### Q27. Explain how to evaluate conversational AI applications using LLM-as-a-Judge.

**Answer:** Use a powerful judge model (e.g. GPT-4o) with a detailed scoring rubric to evaluate chatbot responses across dimensions: Relevancy, Factual Consistency, Tone, Helpfulness, and Safety, computing automated Likert-scale benchmark scores.

### Q28. What is Dynamic Few-Shot Prompting via KNN Exemplars?

**Answer:** K-Nearest Neighbors is applied in embedding space to select exemplars that are structurally closest to the input query, guiding the model on rare edge cases without filling the prompt with irrelevant examples.

### Q29. How does the OpenAI Moderation endpoint integrate into user-facing web applications?

**Answer:** Before dispatching a prompt to the generative model, pass it to 'POST /v1/moderations'. If 'flagged: true', abort execution immediately and return a predefined polite rejection, preventing abuse and conserving API budget.

### Q30. What is Function Calling Hallucination and how do you guard against it?

**Answer:** Function calling hallucination occurs when the model invents parameter keys not present in the schema or passes invalid arguments. Guard via strict mode ('strict: true' in OpenAI API), which enforces JSON Schema adherence, and validate incoming arguments with Pydantic.
