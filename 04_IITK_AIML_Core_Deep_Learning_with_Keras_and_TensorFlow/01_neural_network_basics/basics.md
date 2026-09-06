# Deep Neural Network Foundations & Backpropagation
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Feedforward artificial neural networks propagate tensors through non-linear functional compositions, updating weights via the multivariate calculus **Chain Rule**.

```
                   FORWARD & BACKPROPAGATION TENSOR FLOW
    Input (x) ──► Linear: z = Wx + b ──► Non-Linear: a = σ(z) ──► Loss (L)
                                                                    │
    Update: W ← W - η·(∂L/∂W) ◄── Backprop: ∂L/∂W = (∂L/∂a)·σ'(z)·x ┘
```

---

## 🧭 Deep Theoretical Foundations

### 1. The Backpropagation Chain Rule
For layer $l$ with activations $a^{[l]} = \sigma(z^{[l]})$ and pre-activations $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$:
$$\delta^{[l]} = rac{\partial \mathcal{L}}{\partial z^{[l]}} = \left( (W^{[l+1]})^T \delta^{[l+1]} ight) \odot \sigma'(z^{[l]})$$
$$rac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad rac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

### 2. Vanishing & Exploding Gradients
In deep networks using sigmoid or tanh activations, saturating tails have derivatives $\sigma'(z) 	o 0$. Repeated chain-rule multiplications cause gradients to diminish exponentially:
- Mitigated by **ReLU / LeakyReLU / GELU** activations with non-saturating gradients.
- Stabilized by **He (Kaiming) & Xavier (Glorot) Initialization** preserving variance:
  $$	ext{He Variance: } 	ext{Var}(W) = rac{2}{n_{	ext{in}}}, \quad 	ext{Xavier Variance: } 	ext{Var}(W) = rac{2}{n_{	ext{in}} + n_{	ext{out}}}$$
