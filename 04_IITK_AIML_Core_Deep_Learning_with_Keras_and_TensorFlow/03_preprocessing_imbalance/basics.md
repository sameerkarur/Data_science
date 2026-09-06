# Deep Learning Preprocessing, Regularization & Data Pipelines
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
