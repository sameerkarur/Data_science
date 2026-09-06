# Generative AI Optimization, Hyperparameter Tuning & PEFT (LoRA)
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Generation Sampling Parameters (Temperature, Top-p, Top-k)](#1-generation-sampling-parameters)
2. [Frequency & Presence Penalties (Repetition Mitigation)](#2-frequency--presence-penalties)
3. [Full Fine-Tuning vs Parameter-Efficient Fine-Tuning (PEFT)](#3-full-fine-tuning-vs-peft)
4. [LoRA: Low-Rank Adaptation Architecture & Mathematics](#4-lora-low-rank-adaptation-architecture)
5. [QLoRA: 4-Bit NormalFloat Quantization & Paged Optimizers](#5-qlora-quantization)
6. [Simulating LoRA Weight Injection in Python](#6-simulating-lora-weight-injection-in-python)
7. [Try It Yourself! (Hands-On Practice Exercises)](#7-try-it-yourself-hands-on-practice-exercises)
8. [Quick Reference Cheat Sheet](#8-quick-reference-cheat-sheet)

---

## 1. Generation Sampling: Temperature, Top-p & Top-k

When generating text, the model converts logit activations into probability distributions over its vocabulary via Softmax:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **Temperature ($T$):**
  - $T \to 0$: Deterministic greedy argmax sampling (Best for code, JSON, SQL).
  - $T = 0.7 - 1.0$: Creative, balanced sampling (Best for copywriting, brainstorming).
- **Top-p (Nucleus Sampling):** Retains the smallest cumulative probability set exceeding $p$ (e.g. $p = 0.9$).
- **Top-k:** Filters logits to strictly the top $k$ highest-probability tokens.

```
                    TEMPERATURE SAMPLING PROBABILITY SHIFT
     Probability P(w)
          ▲
          │    T = 0.2 (Sharp, greedy, near-deterministic peak)
          │      ╭┴╮
          │     ╭╯ │ ╰╮
          │    ╭╯  │  ╰╮
          │    │   │   │
          │  ──┴───┴───┴─────── T = 1.0 (Flatter, diverse, creative distribution)
          └────────────────────────────────────────────────────────► Vocabulary Tokens
```

---

## 2. LoRA: Low-Rank Adaptation Architecture

Full fine-tuning updates all billions of parameters in a pretrained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$, requiring hundreds of gigabytes of VRAM.
**LoRA** freezes $\mathbf{W}_0$ and decomposes the weight update $\Delta \mathbf{W}$ into two low-rank matrices:

$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} (\mathbf{B} \cdot \mathbf{A})$$

$$\text{Where } \mathbf{B} \in \mathbb{R}^{d \times r}, \quad \mathbf{A} \in \mathbb{R}^{r \times k}, \quad \text{with rank } r \ll \min(d, k) \text{ (typically } r = 4, 8, 16\text{)}$$

```
                      LoRA FORWARD PASS ARCHITECTURE
                         Input Feature x ∈ ℝᵈ
                                   │
                      ┌────────────┴────────────┐
                      │                         │
                      ▼                         ▼
             Pretrained Weight W₀         Down-Projection A
             (FROZEN in 16-bit / 4-bit)    (ℝᵈˣʳ, initialized Gaussian)
                      │                         │
                      │                         ▼ r-dimensional bottleneck
                      │                    Up-Projection B
                      │                    (ℝʳˣᵏ, initialized to 0)
                      │                         │
                      ▼                         ▼ × (α / r)
                      └────────────┬────────────┘
                                   │
                                   ▼ Add()
                         Output Feature h ∈ ℝᵏ
```

---

## 3. Simulating LoRA Parameter Reduction in Python

```python
import numpy as np

# Model dimensions (e.g. Llama-3 hidden dimension)
d_model = 4096
rank = 8

# Full fine-tuning parameter count
full_params = d_model * d_model

# LoRA parameter count: Matrix A (d x r) + Matrix B (r x d)
lora_params = (d_model * rank) + (rank * d_model)
reduction_pct = (1 - (lora_params / full_params)) * 100

print(f"Full Layer Weight Parameters: {full_params:,}")
print(f"LoRA Adapter Parameters (r={rank}): {lora_params:,}")
print(f"🚀 VRAM / Trainable Parameter Reduction: {reduction_pct:.2f}% fewer parameters!")
```

#### Output:
```text
Full Layer Weight Parameters: 16,777,216
LoRA Adapter Parameters (r=8): 65,536
🚀 VRAM / Trainable Parameter Reduction: 99.61% fewer parameters!
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: LoRA Forward Pass Verification
**Task:** Code a NumPy function `lora_linear_forward(x, W0, A, B, alpha=16, r=8)` demonstrating that at initialization (where matrix $\mathbf{B}$ is zeros), the LoRA output is identically equal to the original pretrained layer output:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np

def lora_linear_forward(x, W0, A, B, alpha=16, r=8):
    scaling = alpha / r
    h_pretrained = x @ W0
    h_lora = (x @ A @ B) * scaling
    return h_pretrained + h_lora

np.random.seed(42)
x = np.random.randn(1, 16)
W0 = np.random.randn(16, 16)
A = np.random.randn(16, 4)
B = np.zeros((4, 16))  # Initialized to zero as per LoRA paper

original_out = x @ W0
lora_out = lora_linear_forward(x, W0, A, B, alpha=8, r=4)

print("Original Output == LoRA Output at Init?", np.allclose(original_out, lora_out))
```
#### Output:
```text
Original Output == LoRA Output at Init? True
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Parameter / Technique | Target Setting | Description |
|---|---|---|
| **Temperature** | `0.0` for code, `0.7` for prose | Controls output randomness / entropy |
| **Top-p** | `0.9` | Nucleus sampling threshold |
| **Frequency Penalty** | `0.1 - 0.5` | Penalizes tokens proportional to frequency |
| **LoRA Rank ($r$)** | `8` or `16` | Dimension of low-rank adapter bottleneck |
| **LoRA Alpha ($\alpha$)**| Typically $2 \times r$ | Scaling factor for adapter activations |
| **QLoRA** | NF4 4-bit quantization | Reduces 70B parameter model VRAM to 48 GB |
