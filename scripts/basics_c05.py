"""
Textbook-Scale Architectural & Conceptual Guides for Course 5:
IITK AIML Core: Essentials of Generative AI, Prompt Engineering & ChatGPT
Modules:
  01_prompt_engineering
  02_chatgpt_applications
  03_genai_optimization
"""

C05_BASICS = {}

# =====================================================================
# 1. Prompt Engineering & Reasoning Frameworks
# =====================================================================
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering"] = r'''# Chapter 1: Prompt Engineering & Systematic Reasoning Frameworks
**Comprehensive Textbook Guide — Generative AI & Large Language Models**

---

## 1. Executive Overview & Mental Models

Autoregressive Large Language Models (LLMs) model conditional probability distributions over sequence tokens:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$
Prompt engineering is the systematic art and science of structuring input context to guide the model's conditional generation toward desired outputs without altering the underlying model weights.

```
                 REASONING SYSTEMATIC ARCHITECTURES
    ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
    │ CHAIN-OF-THOUGHT (CoT):         │   │ REACT AGENT FRAMEWORK:          │
    │ Input Prompt                    │   │ 1. Thought: Reason about task   │
    │      ▼                          │   │ 2. Action: Call external tool   │
    │ Intermediate Reasoning Steps    │   │ 3. Observation: Read tool result│
    │      ▼                          │   │ 4. Repeat until Final Answer!   │
    │ Final Deductive Answer          │   └─────────────────────────────────┘
    └─────────────────────────────────┘
```

---

## 2. Deep Theoretical Foundations

### 1. In-Context Learning Taxonomy
- **Zero-Shot Prompting:** Presenting instructions without demonstration examples. Relies strictly on pre-trained parametric knowledge.
- **Few-Shot In-Context Demonstrations:** Providing $k$ input-output exemplars. Activates task-specific induction heads in transformer attention layers without weight updates.
- **Chain-of-Thought (CoT - Wei et al.):** By compelling the model to generate explicit intermediate computation tokens before emitting the final answer, CoT expands the computational budget (number of transformer forward passes) dedicated to reasoning.
- **Tree of Thoughts (ToT - Yao et al.):** Explores multiple reasoning trajectories simultaneously, evaluating intermediate thoughts via heuristic search (BFS / DFS / A*).

### 2. The ReAct (Reason + Act) Agent Framework
Combines reasoning traces with task-specific actions:
$$\text{Trajectory} = (\text{Thought}_1, \text{Action}_1, \text{Observation}_1, \dots, \text{Thought}_n, \text{Action}_n, \text{Observation}_n, \text{Answer})$$
This synergy enables LLMs to interact with external environments, databases, and APIs to ground reasoning in factual real-time data.

### 3. Prompt Injection & Semantic Guardrails
Adversarial prompt injection attacks attempt to break system constraints via delimiter collisions or instruction hijacking. Production defenses utilize strict structural encapsulation:
- **XML Tagged Enclosures:** Isolating user payload inside `<user_query>` tags.
- **Dual-LLM Validator:** Passing candidate outputs through an independent evaluator model with strict classification criteria before returning to client applications.

---

## 3. Production Implementation: Minimal ReAct Agent Framework in Python

```python
import re
from typing import Callable

class SimpleReActAgent:
    """Production implementation of the ReAct (Reason + Act) runtime loop."""
    def __init__(self, tools: dict[str, Callable[[str], str]]):
        self.tools = tools
        self.action_pattern = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\[(.*?)\]")

    def run_step(self, trajectory: str, llm_generate_fn: Callable[[str], str]) -> tuple[str, bool]:
        """Executes a single reasoning-action cycle."""
        response = llm_generate_fn(trajectory)
        match = self.action_pattern.search(response)
        
        if match:
            tool_name, tool_arg = match.group(1), match.group(2)
            if tool_name in self.tools:
                obs = self.tools[tool_name](tool_arg)
                updated_trajectory = f"{trajectory}{response}\nObservation: {obs}\n"
                return updated_trajectory, False
            else:
                return f"{trajectory}{response}\nObservation: Error tool '{tool_name}' not found.\n", False
        else:
            # Reached Final Answer
            return f"{trajectory}{response}", True
```
'''

# =====================================================================
# 2. Enterprise ChatGPT Applications & Function Calling
# =====================================================================
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications"] = r'''# Chapter 2: Enterprise LLM Applications & Structured Tool Use
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
'''

# =====================================================================
# 3. Parameter-Efficient Fine-Tuning (PEFT) & LoRA Architecture
# =====================================================================
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization"] = r'''# Chapter 3: Parameter-Efficient Fine-Tuning (PEFT) & LoRA Architecture
**Comprehensive Textbook Guide — Generative AI & Large Language Models**

---

## 1. Executive Overview & Mental Models

Full fine-tuning of 7B–70B parameter models requires massive GPU VRAM to store optimizer states (Adam requires 16 bytes per parameter: 2 for weights, 2 for gradients, 4 for master weights, 4 for first momentum, 4 for second momentum). **Low-Rank Adaptation (LoRA)** freezes the pre-trained weights and injects trainable rank decomposition matrices, reducing memory overhead by over 90%.

```
                   LORA (LOW-RANK ADAPTATION) MATRIX DECOMPOSITION
    Input Activation (x)
          │
          ├──► Frozen Original Weights W₀ (d × k) ───────┐
          │    (Requires ZERO Backprop Gradients!)       │
          │                                              ▼
          └──► Low-Rank Adapter Matrices:             Sum (+) ──► Output (h)
               Down-Projection A (r × k, Gaussian)       ▲
               Up-Projection B (d × r, Zeros)            │
               h_adapter = (B · A) · x · (α / r) ────────┘
```

---

## 2. Deep Theoretical Foundations

### 1. Mathematical Formulation of LoRA (Hu et al.)
For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constrains the update $\Delta W$ by representing it as a low-rank factorization:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
Where:
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$ (typically $r \in \{8, 16, 32\}$).
- $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, and $B$ is initialized to 0, ensuring $\Delta W = 0$ at the start of training.
- $\alpha$ is a constant scaling hyperparameter.

### 2. QLoRA: 4-Bit Quantized LoRA (Dettmers et al.)
QLoRA enables fine-tuning a 65B model on a single 48GB GPU by introducing three core innovations:
1. **NF4 (NormalFloat 4):** An information-theoretically optimal quantile quantization data type for normally distributed weights.
2. **Double Quantization (DQ):** Quantizes the quantization constants themselves, saving 0.37 bits per parameter.
3. **Paged Optimizers:** Uses CUDA unified memory to page optimizer states between GPU VRAM and CPU RAM during gradient checkpointing spikes.

---

## 3. Production Implementation: PyTorch LoRA Linear Layer from Scratch

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear implementing exact low-rank adaptation."""
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r
        
        # Frozen base weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear transformation
        base_out = nn.functional.linear(x, self.weight, self.bias)
        # Low-rank adapted transformation
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out
```
'''

print(f"Loaded {len(C05_BASICS)} textbook chapters for Course 5.")
