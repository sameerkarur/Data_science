# Chapter 1: Deep Neural Network Foundations & Backpropagation
**Comprehensive Textbook Guide — Advanced Deep Learning**

---

## 1. Executive Overview & Mental Models

Deep neural networks are parameterized compositions of non-linear tensor transformations. Optimization proceeds by computing analytical gradients of a scalar objective with respect to every weight tensor via reverse-mode automatic differentiation (the **Backpropagation Calculus**), subsequently updating parameters using stochastic gradient descent variants.

```
                   FORWARD & BACKPROPAGATION TENSOR FLOW
    Input (x) ──► Linear: z = Wx + b ──► Non-Linear: a = σ(z) ──► Loss (L)
                                                                    │
    Update: W ← W - η·(∂L/∂W) ◄── Backprop: ∂L/∂W = (∂L/∂a)·σ'(z)·x ┘
```

---

## 2. Deep Theoretical Foundations

### 1. The Backpropagation Calculus (Reverse-Mode AD)
Consider layer $l$ in an $L$-layer network. Let:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$
Defining the error sensitivity vector $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}$, the recurrence relation propagates backward:
$$\delta^{[l]} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot g'^{[l]}(z^{[l]})$$
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$
This matrix-vector formulation executes in $O(E)$ time where $E$ is the number of edges in the computational graph.

### 2. Vanishing Gradients & Activation Physics
For standard Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, the derivative is:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 0.25$$
Multiplying $L$ such derivatives across deep layers causes gradients to decay exponentially as $(0.25)^L \to 0$.
- **Modern Solution (GELU - Gaussian Error Linear Unit):**
  $$\text{GELU}(x) = x \cdot P(X \le x) = x \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$
  Used in BERT, GPT-4, and modern Transformers to provide smooth, non-saturating gradients.

### 3. Modern Optimizers: AdamW vs Adam
Adam calculates exponentially decaying moving averages of past gradients ($m_t$) and squared gradients ($v_t$):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
- **L2 Regularization in Adam:** Adds $\lambda \theta$ to gradient $g_t$, which interacts pathologically with the denominator $\sqrt{\hat{v}_t}$.
- **Decoupled Weight Decay (AdamW - Loshchilov & Hutter):** Decouples weight decay from the gradient step, updating weights directly:
  $$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

---

## 3. Production Implementation: NumPy Neural Network from First Principles

```python
import numpy as np

class DenseLayer:
    """Fully-connected layer with He initialization and analytical backpropagation."""
    def __init__(self, in_features: int, out_features: int):
        # He (Kaiming) normal initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.x: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        return self.z

    def backward(self, delta: np.ndarray) -> np.ndarray:
        # Compute parameter gradients
        self.dW = np.dot(self.x.T, delta) / len(self.x)
        self.db = np.sum(delta, axis=0, keepdims=True) / len(self.x)
        # Propagate error backward to preceding layer
        dx = np.dot(delta, self.W.T)
        return dx
```
