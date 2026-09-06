"""
Comprehensive, high-depth Basics & Architecture Guides for Course 5:
Generative AI, Prompt Engineering & Optimization (3 modules)
"""

C05_BASICS = {}

# 1. Prompt Engineering
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering"] = """# Prompt Engineering & Reasoning Frameworks
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. In-Context Learning Taxonomy
- **Zero-Shot Prompting:** Providing task description and input without explicit exemplars.
- **Few-Shot Prompting:** Guiding formatting and task mapping with 2–5 structured input-output demonstrations.
- **Chain-of-Thought (CoT):** Encouraging step-by-step reasoning tokens before generating the final answer, dramatically improving performance on math and symbolic logic.
- **Tree of Thoughts (ToT):** Tree-search exploration evaluating multiple diverse reasoning paths with backtrack search algorithms.

### 2. Prompt Injection & Jailbreak Defenses
Malicious users attempt to override system instructions via direct or indirect prompt injection (e.g., hidden instructions in ingested PDFs). Defenses include strict XML tag encapsulation (`<system_instructions>`, `<untrusted_user_input>`), dual-LLM guardrail evaluators, and deterministic parameter schema validation.
"""

# 2. ChatGPT Applications
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications"] = """# Enterprise LLM Applications & Function Calling
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Sampling Mathematics: Temperature vs Top-P
LLMs output a vector of unnormalized logits $z_i$. Softmax with temperature $T$ scales probabilities:
$$P(w_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
- **Low Temperature ($T \to 0$):** Deterministic argmax selection (code, math).
- **High Temperature ($T > 1.0$):** Flattens distribution, increasing vocabulary diversity (creative writing).
- **Nucleus Sampling (Top-P):** Truncates candidate pool to the smallest set of tokens whose cumulative probability exceeds $p$.
"""

# 3. GenAI Optimization
C05_BASICS["05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization"] = """# Parameter-Efficient Fine-Tuning (PEFT) & LoRA Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

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

## 🧭 Deep Theoretical Foundations

### 1. Low-Rank Adaptation (LoRA) Mechanics
Full fine-tuning updates massive parameter matrices $\Delta W \in \mathbb{R}^{d \times k}$, requiring hundreds of gigabytes of optimizer memory. LoRA factorizes weight updates into two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A, \quad \text{where } B \in \mathbb{R}^{d \times r}, \, A \in \mathbb{R}^{r \times k}, \, r \ll \min(d, k)$$
This reduces trainable parameters by **99.9%** while enabling multiple task-specific LoRA adapters to be swapped dynamically on a single frozen base model.
"""

print(f"Loaded {len(C05_BASICS)} comprehensive guides for Course 5.")
