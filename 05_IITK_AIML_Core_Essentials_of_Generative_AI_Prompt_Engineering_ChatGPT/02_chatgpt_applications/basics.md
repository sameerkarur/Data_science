# Chapter 2: Enterprise LLM Applications & Structured Tool Use
**Comprehensive Textbook Guide — Generative AI & Large Language Models**

---

## 1. Executive Overview & Mental Models

Enterprise generative AI systems integrate LLMs with external software infrastructure. Through **Function Calling (Tool Use)**, the model acts as an intelligent router and parameter extractor, while deterministic backend code handles business logic and database mutations.

```
                 TOOL USE / FUNCTION CALLING CYCLE
    User Query ──► LLM Orchestrator + Available Tools JSON Schema
                            │
               Model generates Tool Call:
               {"name": "fetch_balance", "args": {"account_id": 104}}
                            │
               Host Application Executes Local API Function
                            │
               API Result fed back to LLM Context
                            │
               Model synthesizes natural response to User
```

---

## 2. Deep Theoretical Foundations

### 1. Generation Sampling Mathematics
An LLM outputs raw unnormalized logit vectors $z \in \mathbb{R}^V$ across vocabulary $V$. Probabilities are computed via Temperature-scaled Softmax:
$$P(w_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^V \exp(z_j / T)}$$
- **Temperature ($T$):**
  - $T \to 0$: Argmax sampling (greedy deterministic decoding).
  - $T = 1.0$: Standard categorical sampling.
  - $T > 1.0$: Flattens probability mass, increasing linguistic diversity.
- **Nucleus (Top-P) Sampling:** Truncates candidate vocabulary to the smallest subset $V^{(p)} \subset V$ such that:
  $$\sum_{w \in V^{(p)}} P(w) \ge p$$
  Tokens outside $V^{(p)}$ have their probabilities set to 0.

### 2. Tokenization & Byte-Pair Encoding (BPE)
LLMs process tokens, not words. BPE iteratively merges the most frequent byte pairs in the training corpus into vocabulary units. Understanding token boundaries is essential for optimizing cost, latency, and context window budget allocation.

---

## 3. Production Implementation: Structured Tool Definition & Dispatch

```python
import json
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Executes read-only SQL queries on the analytics warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Safe SQL SELECT query"}
            },
            "required": ["query"]
        }
    }
}

def dispatch_tool_call(tool_name: str, arguments_json: str) -> str:
    """Safely dispatches tool calls with JSON validation and error trapping."""
    try:
        args = json.loads(arguments_json)
        if tool_name == "query_database":
            # Simulate secure query execution
            return json.dumps({"rows": [{"id": 1, "status": "active"}], "count": 1})
        return json.dumps({"error": f"Unknown function: {tool_name}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
```
