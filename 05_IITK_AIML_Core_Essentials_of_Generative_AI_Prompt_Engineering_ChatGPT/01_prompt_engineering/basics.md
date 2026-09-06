# Prompt Engineering & Reasoning Frameworks
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
