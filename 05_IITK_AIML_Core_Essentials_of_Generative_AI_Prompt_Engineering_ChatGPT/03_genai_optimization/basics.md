# Chapter 3: Parameter-Efficient Fine-Tuning (PEFT) & LoRA Architecture
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
