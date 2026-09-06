# Artificial Neural Networks, Backpropagation & Gradient Descent
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [From Biological Neurons to the Artificial Perceptron](#1-from-biological-neurons-to-the-artificial-perceptron)
2. [Multi-Layer Perceptron (MLP) Architecture](#2-multi-layer-perceptron-mlp-architecture)
3. [Activation Functions (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Softmax)](#3-activation-functions)
4. [Forward Propagation (The Mathematical Matrix Product)](#4-forward-propagation)
5. [Loss Functions (Binary Cross-Entropy, Categorical Cross-Entropy, MSE)](#5-loss-functions)
6. [Backpropagation & the Computational Chain Rule](#6-backpropagation--the-computational-chain-rule)
7. [Gradient Descent Optimizers (SGD, Momentum, RMSprop, AdamW)](#7-gradient-descent-optimizers)
8. [Building a Complete Neural Network from Scratch in Pure NumPy](#8-building-a-complete-neural-network-from-scratch)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. The Artificial Perceptron

The Perceptron (Frank Rosenblatt, 1957) takes a vector of inputs $\mathbf{x}$, multiplies them by learnable weights $\mathbf{w}$, adds a bias scalar $b$, and passes the pre-activation $z$ through a non-linear activation function $\sigma(z)$:

```
                         THE ARTIFICIAL NEURON
       Inputs (x)       Weights (w)        Summation & Activation
        ┌──────┐          w₁
        │  x₁  │ ───────────┐
        └──────┘            │
        ┌──────┐          w₂ │           ┌──────────────────────┐
        │  x₂  │ ───────────┼──────────► │ z = Σ(wᵢ·xᵢ) + b     │ ──► y = σ(z) (Output)
        └──────┘            │            │ Pre-Activation Sum   │
        ┌──────┐          w₃ │           └──────────────────────┘
        │  x₃  │ ───────────┘                      ▲
        └──────┘                                   │
                                            ┌──────┴─────┐
                                            │ Bias (b)   │
                                            └────────────┘
```

---

## 2. Multi-Layer Perceptron (MLP) Architecture

An MLP stacks layers of neurons to learn hierarchical, non-linear representations of input data:

```
    INPUT LAYER                HIDDEN LAYER                  OUTPUT LAYER
      (x ∈ ℝ³)                   (h ∈ ℝ⁴)                      (ŷ ∈ ℝ²)
        ( x₁ ) ───────────────► ( h₁ ) ─────────────────────► ( ŷ₁ )
               \             /          \                  /
        ( x₂ ) ───────────────► ( h₂ ) ─────────────────────► ( ŷ₂ )
               /             \          /                  \
        ( x₃ ) ───────────────► ( h₃ ) ─────────────────────► Loss L(y, ŷ)
                             \          /
                               ( h₄ ) ─┘
```

---

## 3. Activation Functions Comparison

```python
import numpy as np

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def gelu(z): return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / exp_z.sum(axis=-1, keepdims=True)

test_inputs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

print("Inputs:        ", test_inputs)
print("ReLU:          ", np.round(relu(test_inputs), 3))
print("Sigmoid:       ", np.round(sigmoid(test_inputs), 3))
print("GELU (Modern): ", np.round(gelu(test_inputs), 3))
print("Softmax (Norm):", np.round(softmax(test_inputs), 3))
```

#### Output:
```text
Inputs:         [-2.  -0.5  0.   0.5  2. ]
ReLU:           [0.  0.  0.  0.5 2. ]
Sigmoid:        [0.119 0.378 0.5   0.622 0.881]
GELU (Modern):  [-0.045 -0.154  0.     0.346  1.955]
Softmax (Norm): [0.012 0.053 0.088 0.145 0.702]
```

---

## 4. Backpropagation & The Computational Chain Rule

Backpropagation calculates the analytical gradient of the loss function with respect to every weight in the network by systematically applying the calculus chain rule backwards from output to input:

$$\frac{\partial L}{\partial \mathbf{W}^{[1]}} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}^{[2]}} \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{a}^{[1]}} \cdot \frac{\partial \mathbf{a}^{[1]}}{\partial \mathbf{z}^{[1]}} \cdot \frac{\partial \mathbf{z}^{[1]}}{\partial \mathbf{W}^{[1]}}$$

```
               FORWARD PASS (Computation of Activations & Predictions)
    Input X ───────► Hidden Layer 1 ───────► Hidden Layer 2 ───────► Output ŷ ──► Loss L
                       ▲                       ▲                       ▲
                       │                       │                       │
               BACKWARD PASS (Gradient Propagation via Calculus Chain Rule)
    ∂L/∂X   ◄─────── ∂L/∂W¹         ◄─────── ∂L/∂W²         ◄─────── ∂L/∂ŷ
```

---

## 5. Complete 2-Layer Neural Network from Scratch in Pure NumPy

```python
import numpy as np

class PureNumPyMLP:
    """A complete 2-layer neural network trained on XOR logic."""
    def __init__(self, in_dim=2, hidden_dim=4, out_dim=1, lr=0.1):
        np.random.seed(42)
        self.lr = lr
        # He / Xavier weight initialization
        self.W1 = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, out_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = 1 / (1 + np.exp(-np.clip(self.z2, -20, 20)))  # Sigmoid
        return self.a2

    def train_step(self, X, y):
        m = X.shape[0]
        # 1. Forward Pass
        y_pred = self.forward(X)
        loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

        # 2. Backward Pass (Chain rule gradients)
        dz2 = (y_pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 3. Gradient Descent Parameter Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss

# XOR Problem (Non-linear dataset)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

mlp = PureNumPyMLP(lr=1.0)
for epoch in range(1000):
    loss = mlp.train_step(X, y)

print("Trained XOR Predictions:\n", mlp.forward(X).round(3))
```

#### Output:
```text
Trained XOR Predictions:
 [[0.015]
 [0.982]
 [0.981]
 [0.019]]
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Derivative of Sigmoid Function
**Task:** Prove and code the derivative of the Sigmoid function $\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

z_test = np.array([-2.0, 0.0, 2.0])
print("Sigmoid values:     ", np.round(sigmoid(z_test), 3))
print("Sigmoid derivatives:", np.round(sigmoid_derivative(z_test), 3))
```
#### Output:
```text
Sigmoid values:      [0.119 0.5   0.881]
Sigmoid derivatives: [0.105 0.25  0.105]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Activation | Formula | Gradient Range | Primary Location |
|---|---|---|---|
| **ReLU** | $\max(0, z)$ | $\{0, 1\}$ | Hidden layers in MLPs & CNNs |
| **GELU** | $z \cdot \Phi(z)$ | Smooth non-zero | Modern LLMs & Vision Transformers |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $(0, 0.25]$ | Binary classification output layer |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | Non-linear | Multi-class categorical output layer |
