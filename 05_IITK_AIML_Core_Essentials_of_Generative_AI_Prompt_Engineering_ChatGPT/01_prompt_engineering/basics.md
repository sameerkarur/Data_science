# Chapter 1: Prompt Engineering & Systematic Reasoning Frameworks
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
