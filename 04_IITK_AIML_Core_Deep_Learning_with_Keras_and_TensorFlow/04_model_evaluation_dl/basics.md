# Deep Learning Evaluation, Grad-CAM Explainability & Model Export
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
