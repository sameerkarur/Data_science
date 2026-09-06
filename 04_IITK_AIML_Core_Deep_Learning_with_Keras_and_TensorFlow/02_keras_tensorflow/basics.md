# TensorFlow 2.x & Keras 3 Architecture: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official TensorFlow / Keras Style)**

---

## 📑 Table of Contents (On this page)
1. [TensorFlow 2.x Architecture: Eager Execution vs `@tf.function` Graphs](#1-tensorflow-architecture)
2. [Tensors, Variables & Device Placement (`/GPU:0`, `/TPU:0`)](#2-tensors-variables--device-placement)
3. [Automatic Differentiation with `tf.GradientTape`](#3-automatic-differentiation-gradienttape)
4. [The 3 Keras Model Authoring Paradigms](#4-the-3-keras-model-paradigms)
5. [Custom Training Loops vs `model.compile()` & `model.fit()`](#5-custom-training-loops)
6. [Keras Callbacks: Checkpointing, Early Stopping & TensorBoard](#6-keras-callbacks)
7. [Common Pitfalls: In-Graph Tensor Mutations & Retracing Overhead](#7-common-pitfalls)
8. [Production Case Study: Multi-Task Learning Architecture with Residual Skips](#8-production-case-study-multitask-learning)
9. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet & Best Website Citations](#10-quick-reference-cheat-sheet--citations)

---

## 1. TensorFlow 2.x Architecture: Eager Execution vs `@tf.function`

TF2 defaults to **Eager Execution** (imperative Python debugging), but compiles compute graphs into optimized C++ binaries using `@tf.function` and AutoGraph:

```python
import tensorflow as tf

@tf.function
def fast_matrix_power(A, power=3):
    result = A
    for _ in tf.range(power - 1):
        result = tf.matmul(result, A)
    return result

mat = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print("Compiled Graph Output:\n", fast_matrix_power(mat).numpy())
```

#### Output:
```text
Compiled Graph Output:
 [[ 37.  54.]
 [ 81. 118.]]
```

---

## 2. Automatic Differentiation with `tf.GradientTape`

```python
# Computing first and second derivatives of y = x^3 at x = 3.0
x = tf.Variable(3.0)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = x ** 3
    dy_dx = tape1.gradient(y, x)  # dy/dx = 3 * x^2 = 27.0
d2y_dx2 = tape2.gradient(dy_dx, x)  # d^2y/dx^2 = 6 * x = 18.0

print(f"y = x^3 at x=3.0 -> dy/dx = {dy_dx.numpy():.1f} | d^2y/dx^2 = {d2y_dx2.numpy():.1f}")
```

#### Output:
```text
y = x^3 at x=3.0 -> dy/dx = 27.0 | d^2y/dx^2 = 18.0
```

---

## 3. The 3 Keras Model Authoring Paradigms

```
                       THE THREE KERAS AUTHORING PATTERNS
    1. SEQUENTIAL API          2. FUNCTIONAL API           3. MODEL SUBCLASSING
    model = Sequential([       inputs = Input(shape=(10,)) class ResBlock(Model):
      Dense(64), Dense(1)      x = Dense(64)(inputs)         def call(self, x):
    ])                         out = Dense(1)(x)               return x + self.dense(x)
    Simple linear pipeline     Residual skips, multi-head  Dynamic branching / custom loops
```

---

## 4. Production Case Study: Functional Multi-Output Residual Architecture

```python
from tensorflow.keras import layers, Model

def build_multitask_network(input_dim=16):
    inputs = layers.Input(shape=(input_dim,), name="features_input")

    # Dense Backbone with Residual Connection
    x = layers.Dense(64, activation="relu")(inputs)
    residual = x
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Add()([x, residual])  # Skip connection!

    # Head 1: Binary Classification (Churn)
    churn_head = layers.Dense(1, activation="sigmoid", name="churn_output")(x)

    # Head 2: Regression (Revenue)
    revenue_head = layers.Dense(1, activation="linear", name="revenue_output")(x)

    model = Model(inputs=inputs, outputs=[churn_head, revenue_head], name="multi_task_enterprise_net")
    return model

multitask_model = build_multitask_network()
multitask_model.summary(line_length=80)
```

#### Output:
```text
Model: "multi_task_enterprise_net"
________________________________________________________________________________
 Layer (type)                       Output Shape                    Param #     
================================================================================
 features_input (InputLayer)        [(None, 16)]                    0           
 dense (Dense)                      (None, 64)                      1088        
 dense_1 (Dense)                    (None, 64)                      4160        
 add (Add)                          (None, 64)                      0           
 churn_output (Dense)               (None, 1)                       65          
 revenue_output (Dense)             (None, 1)                       65          
================================================================================
Total params: 5,378 (21.01 KB)
Trainable params: 5,378 (21.01 KB)
Non-trainable params: 0 (0.00 Byte)
________________________________________________________________________________
```

---

## 5. Quick Reference Cheat Sheet & Best Website Citations

| API Paradigm | Flexibility | Ease of Use | Serialization Safety |
|---|---|---|---|
| **Sequential** | Low (Single In/Out) | High | Perfect (`.keras`) |
| **Functional** | High (DAGs, Skips) | High | Perfect (`.keras`) |
| **Subclassing** | Maximum (Dynamic Python) | Medium | Requires custom `get_config` |

### 🌐 Official References & Recommended Reading:
- [Keras Official Documentation](https://keras.io/)
- [TensorFlow Guide: `tf.GradientTape`](https://www.tensorflow.org/guide/autodiff)
- [W3Schools TensorFlow Tutorial](https://www.w3schools.com/python/python_ml_getting_started.asp)
