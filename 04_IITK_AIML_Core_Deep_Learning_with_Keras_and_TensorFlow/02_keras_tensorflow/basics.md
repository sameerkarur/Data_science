# TensorFlow 2 & Keras: Architecture, APIs & Computational Graphs
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
