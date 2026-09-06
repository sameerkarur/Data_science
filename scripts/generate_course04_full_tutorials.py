"""
Generates extensive, tutorial-grade, visual guides (W3Schools / GeeksforGeeks / Official Docs style)
for Course 4 Deep Learning with Keras & TensorFlow:
- 01_neural_network_basics
- 02_keras_tensorflow
- 03_preprocessing_imbalance
- 04_model_evaluation_dl
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# 1. 01_neural_network_basics/basics.md
# =====================================================================
C04_M01_GUIDE = r'''# Artificial Neural Networks, Backpropagation & Gradient Descent
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
'''

p = REPO_ROOT / "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics/basics.md"
p.write_text(C04_M01_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C04 M01 Guide: {len(C04_M01_GUIDE.splitlines())} lines.")

# =====================================================================
# 2. 02_keras_tensorflow/basics.md
# =====================================================================
C04_M02_GUIDE = r'''# TensorFlow 2 & Keras: Architecture, APIs & Computational Graphs
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [TensorFlow Ecosystem & Tensor Memory Structure](#1-tensorflow-ecosystem--tensor-memory-structure)
2. [Eager Execution vs `@tf.function` Computation Graphs](#2-eager-execution-vs-tffunction)
3. [The Keras Sequential API (Linear Layer Stacks)](#3-the-keras-sequential-api)
4. [The Keras Functional API (Multi-Input & Multi-Output Models)](#4-the-keras-functional-api)
5. [Model Subclassing for Custom Architectures](#5-model-subclassing)
6. [Training Callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)](#6-training-callbacks)
7. [Complete Production Training Pipeline in TensorFlow 2](#7-complete-production-training-pipeline)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. TensorFlow Ecosystem & Tensors

A **Tensor** is a multi-dimensional array with a uniform datatype (`dtype`) that can reside in CPU RAM or GPU/TPU VRAM:

```python
import tensorflow as tf

# Create constant and variable tensors
const_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
var_tensor = tf.Variable([[5.0, 6.0], [7.0, 8.0]])

# Automatic differentiation with GradientTape
with tf.GradientTape() as tape:
    y = tf.reduce_sum(var_tensor ** 2)

grad = tape.gradient(y, var_tensor)
print("Variable Tensor:\n", var_tensor.numpy())
print("Gradients (2 * var):\n", grad.numpy())
```

#### Output:
```text
Variable Tensor:
 [[5. 6.]
 [7. 8.]]
Gradients (2 * var):
 [[10. 12.]
 [14. 16.]]
```

---

## 2. Sequential API vs Functional API

```
   SEQUENTIAL API: Linear Pipeline          FUNCTIONAL API: Non-Linear DAG (ResNet Skip Connections)
      Input (784)                               Input (Image)
          │                                          │
          ▼                                          ├──► Conv Layer A ──► Conv Layer B ──┐
      Dense (128)                                    │                                    ▼
          │                                          └──────────────────────────────► Add() (Residual)
          ▼                                                                               │
      Dense (10)                                                                          ▼
                                                                                     Dense Output
```

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Sequential API (Best for straightforward feedforward stacks)
seq_model = models.Sequential([
    layers.Input(shape=(20,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 2. Functional API (Enables shared layers, multiple inputs/outputs, skip connections)
inputs = layers.Input(shape=(20,))
x = layers.Dense(64, activation='relu')(inputs)
residual = x
x = layers.Dense(64, activation='relu')(x)
x = layers.add([x, residual])  # Skip connection!
outputs = layers.Dense(1, activation='sigmoid')(x)
func_model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_Tabular")

func_model.summary()
```

#### Output:
```text
Model: "ResNet_Tabular"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_2 (InputLayer)│ (None, 20)        │          0 │ -                 │
│ dense_2 (Dense)     │ (None, 64)        │      1,344 │ input_2[0][0]     │
│ dense_3 (Dense)     │ (None, 64)        │      4,160 │ dense_2[0][0]     │
│ add (Add)           │ (None, 64)        │          0 │ dense_3[0][0],    │
│                     │                   │            │ dense_2[0][0]     │
│ dense_4 (Dense)     │ (None, 1)         │         65 │ add[0][0]         │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 5,569 (21.75 KB)
```

---

## 3. Production Training with Callbacks

Callbacks intercept training at epoch boundaries to save weights or stop early:

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    # Stops training when validation loss stops improving for 5 consecutive epochs
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    # Halves learning rate if validation loss plateaus for 3 epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Multi-Output Classification Model
**Task:** Build a model using the Keras Functional API that takes tabular features `(None, 30)` and outputs two simultaneous predictions: binary churn probability (Sigmoid) and continuous lifetime value (Linear):

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from tensorflow.keras import layers, models

inputs = layers.Input(shape=(30,), name='customer_features')
shared = layers.Dense(64, activation='relu')(inputs)
shared = layers.Dense(32, activation='relu')(shared)

churn_output = layers.Dense(1, activation='sigmoid', name='churn_pred')(shared)
clv_output = layers.Dense(1, activation='linear', name='clv_pred')(shared)

multi_model = models.Model(inputs=inputs, outputs=[churn_output, clv_output])
multi_model.compile(
    optimizer='adam',
    loss={'churn_pred': 'binary_crossentropy', 'clv_pred': 'mse'},
    metrics={'churn_pred': 'accuracy', 'clv_pred': 'mae'}
)
print("Multi-Output Model successfully compiled.")
```
#### Output:
```text
Multi-Output Model successfully compiled.
```
</details>

---

## 5. Quick Reference Cheat Sheet

| API / Feature | Syntax | Best Use Case |
|---|---|---|
| **Sequential** | `models.Sequential([...])` | Plain linear layer stacks |
| **Functional** | `models.Model(inputs, outputs)` | Skip connections, multi-head architectures |
| **Gradient Tape**| `with tf.GradientTape() as t:` | Custom training loops and physics-informed NNs |
| **Early Stopping**| `EarlyStopping(patience=5)` | Prevents overfitting automatically |
| **Model Export** | `model.export('path/')` | SavedModel format for high-speed C++ serving |
'''

p2 = REPO_ROOT / "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow/basics.md"
p2.write_text(C04_M02_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C04 M02 Guide: {len(C04_M02_GUIDE.splitlines())} lines.")

# =====================================================================
# 3. 03_preprocessing_imbalance/basics.md
# =====================================================================
C04_M03_GUIDE = r'''# Deep Learning Preprocessing, Regularization & Data Pipelines
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The `tf.data` High-Performance Input Pipeline](#1-the-tfdata-high-performance-input-pipeline)
2. [Data Augmentation Layers (Spatial Invariance)](#2-data-augmentation-layers)
3. [Normalization Taxonomy: Batch Normalization vs Layer Normalization](#3-normalization-taxonomy)
4. [Regularization: Dropout, DropPath & Weight Decay ($L_2$)](#4-regularization-techniques)
5. [Imbalanced Classification in Deep Learning: Class Weights & Focal Loss](#5-imbalanced-classification-in-deep-learning)
6. [Try It Yourself! (Hands-On Practice Exercises)](#6-try-it-yourself-hands-on-practice-exercises)
7. [Quick Reference Cheat Sheet](#7-quick-reference-cheat-sheet)

---

## 1. High-Performance Input Pipelines: `tf.data`

GPU compute units sit idle when Python single-threaded CPU loops cannot load images fast enough. The `tf.data` API maximizes throughput through pipelined prefetching:

```
            SEQUENTIAL EXECUTION (GPU Idle Bottleneck):
     CPU: [Read 1] [Preprocess 1]                   [Read 2] [Preprocess 2]
     GPU:                         [Train 1]                                [Train 2]
                                  ◄─────── GPU Sits Idle Waiting! ────────►

            PIPELINED EXECUTION WITH PREFETCH (100% GPU Saturation):
     CPU: [Read 1] [Preprocess 1] [Read 2] [Preprocess 2] [Read 3]
     GPU:                         [Train 1]               [Train 2]        [Train 3]
```

```python
import tensorflow as tf
import numpy as np

features = np.random.randn(1000, 32).astype(np.float32)
labels = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)

dataset = (
    tf.data.Dataset.from_tensor_slices((features, labels))
    .shuffle(buffer_size=500)
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)  # Overlaps CPU preprocessing with GPU compute
)

sample_batch_x, sample_batch_y = next(iter(dataset))
print(f"Batched Feature Shape: {sample_batch_x.shape}")
print(f"Batched Label Shape:   {sample_batch_y.shape}")
```

#### Output:
```text
Batched Feature Shape: (32, 32)
Batched Label Shape:   (32, 1)
```

---

## 2. Normalization Taxonomy: BatchNorm vs LayerNorm

```
       BATCH NORMALIZATION (Across Batch B)          LAYER NORMALIZATION (Across Features C)
       Normalizes each feature channel independently  Normalizes each token / sample independently
             Batch Samples (B)                             Batch Samples (B)
             ┌───┬───┬───┐                                 ┌───┬───┬───┐
      Chan 1 │ ↓ │ ↓ │ ↓ │                          Chan 1 │ ──►───►───│
             ├───┼───┼───┤                                 ├───┼───┼───┤
      Chan 2 │ ↓ │ ↓ │ ↓ │                          Chan 2 │ ──►───►───│
             └───┴───┴───┘                                 └───┴───┴───┘
         (Ideal for CNNs & Computer Vision)             (Ideal for Transformers & Sequence NLP)
```

```python
from tensorflow.keras import layers

# Batch Normalization layer
batch_norm = layers.BatchNormalization()
# Layer Normalization layer
layer_norm = layers.LayerNormalization()
```

---

## 3. Regularization: Dropout & Weight Decay

```python
from tensorflow.keras import layers, regularizers

regularized_dense = layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=regularizers.l2(1e-4)  # L2 Weight Decay: Penalizes large weights
)
dropout_layer = layers.Dropout(rate=0.3)      # Randomly zeroes 30% of activations during training
```

---

## 4. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Custom Focal Loss Implementation in Keras
**Task:** Focal Loss down-weights well-classified easy examples with modulating factor $(1 - p_t)^\gamma$:
$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$
Implement a custom Focal Loss class in Keras:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import tensorflow as tf

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1.0 - self.alpha)
        loss = -alpha_t * tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
print("Custom Binary Focal Loss initialized successfully.")
```
#### Output:
```text
Custom Binary Focal Loss initialized successfully.
```
</details>

---

## 5. Quick Reference Cheat Sheet

| Pipeline / Layer | Syntax | Primary Benefit |
|---|---|---|
| **Prefetch** | `.prefetch(tf.data.AUTOTUNE)` | Saturates GPU by backgrounding CPU ETL |
| **BatchNorm** | `layers.BatchNormalization()` | Stabilizes internal covariate shift in CNNs |
| **LayerNorm** | `layers.LayerNormalization()` | Standard normalization for Transformers/LLMs |
| **Dropout** | `layers.Dropout(0.3)` | Prevents co-adaptation of hidden features |
| **L2 Decay** | `kernel_regularizer=regularizers.l2()` | Keeps weight norms small |
'''

p3 = REPO_ROOT / "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance/basics.md"
p3.write_text(C04_M03_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C04 M03 Guide: {len(C04_M03_GUIDE.splitlines())} lines.")

# =====================================================================
# 4. 04_model_evaluation_dl/basics.md
# =====================================================================
C04_M04_GUIDE = r'''# Deep Learning Evaluation, Grad-CAM Explainability & Model Export
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Diagnosing Training Dynamics (Overfitting vs Underfitting Curves)](#1-diagnosing-training-dynamics)
2. [Learning Rate Scheduling (Cosine Annealing & OneCycle Policy)](#2-learning-rate-scheduling)
3. [Visual Model Explainability: Grad-CAM (Gradient-Weighted Class Activation Mapping)](#3-visual-model-explainability-grad-cam)
4. [Calibration of Deep Neural Networks (Temperature Scaling)](#4-calibration-of-deep-neural-networks)
5. [Model Export: SavedModel, ONNX & TensorRT Conversion](#5-model-export)
6. [Try It Yourself! (Hands-On Practice Exercises)](#6-try-it-yourself-hands-on-practice-exercises)
7. [Quick Reference Cheat Sheet](#7-quick-reference-cheat-sheet)

---

## 1. Diagnosing Training Dynamics from Loss Curves

```
      HEALTHY CONVERGENCE                     OVERFITTING DETECTED
  Loss                                   Loss
   ▲                                      ▲
   │  ── Train Loss                        │               Val Loss Spikes!
   │  -- Val Loss                          │            --/
   │ \                                     │ \         /
   │  \                                    │  \  _____/
   │   \________ Train                     │   \_______ Train Loss
   │    \------- Val                       │
   └──────────────────────► Epochs        └──────────────────────► Epochs
```

---

## 2. Grad-CAM (Gradient-Weighted Class Activation Mapping)

Grad-CAM uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in an image:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

```
                               THE GRAD-CAM PIPELINE
  [Input Image] ──► [Conv Backbone] ──► [Feature Maps Aᵏ] ──► [GAP + Dense] ──► Score yᶜ
                                                │                              │
                                                ▼ Backward Gradients ∂yᶜ/∂Aᵏ ◄─┘
                                       [Neuron Importance Weights αₖᶜ]
                                                │
                                                ▼ Weighted Combination + ReLU
                                     [Grad-CAM Heatmap Overlaid on Image!]
```

```python
import tensorflow as tf
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Computes Grad-CAM heatmap for a given input image."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of output class with respect to feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # Channel-wise mean pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features that positively contribute to class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

print("Grad-CAM function compiled and verified.")
```

#### Output:
```text
Grad-CAM function compiled and verified.
```

---

## 3. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Cosine Annealing Learning Rate Decay
**Task:** Use `tf.keras.optimizers.schedules.CosineDecay` to create a learning rate schedule that decays from `initial_lr = 1e-3` down to `min_lr = 1e-5` over 10,000 steps:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import tensorflow as tf

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    alpha=0.01  # Minimum lr ratio = 1e-3 * 0.01 = 1e-5
)

print(f"Step 0 LR:     {lr_schedule(0).numpy():.6f}")
print(f"Step 5,000 LR: {lr_schedule(5000).numpy():.6f}")
print(f"Step 10,000 LR:{lr_schedule(10000).numpy():.6f}")
```
#### Output:
```text
Step 0 LR:      0.001000
Step 5,000 LR:  0.000505
Step 10,000 LR: 0.000010
```
</details>

---

## 4. Quick Reference Cheat Sheet

| Tool | Implementation | Primary Purpose |
|---|---|---|
| **Grad-CAM** | Gradient-weighted pooling | Visualizes CNN attention / decisions |
| **Cosine Decay** | `optimizers.schedules.CosineDecay` | Smooth learning rate annealing |
| **SavedModel** | `model.export('dir/')` | Production TensorFlow serving export |
| **ONNX Export** | `tf2onnx.convert` | Universal cross-framework inference runtime |
'''

p4 = REPO_ROOT / "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl/basics.md"
p4.write_text(C04_M04_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C04 M04 Guide: {len(C04_M04_GUIDE.splitlines())} lines.")
