# Chapter 4: Deep Learning Evaluation & Grad-CAM Interpretability
**Comprehensive Textbook Guide — Advanced Deep Learning**

---

## 1. Executive Overview & Mental Models

Evaluating deep neural networks requires diagnosing loss trajectory dynamics, learning rate decay schedules, and computing visual saliency attributions to verify that predictions are based on causally meaningful features rather than dataset artifacts.

```
                  GRAD-CAM (GRADIENT-WEIGHTED CLASS ACTIVATION MAP)
    Input Image ──► Conv Backbone ──► Feature Maps Aᵏ ──► Dense Layer ──► Score yᶜ
                                            ▲
    Gradients ∂yᶜ/∂Aᵏ ──► Global Avg Pool ──┘
                               │
    Linear Combination: L_GradCAM = ReLU(Σ αₖ Aᵏ) ──► Heatmap Overlay!
```

---

## 2. Deep Theoretical Foundations

### 1. Grad-CAM Mathematical Formulation
Grad-CAM computes the gradient of class score $y^c$ (prior to softmax) with respect to feature map activations $A^k$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i, j}^k}$$
Where $Z = U \times V$ is the spatial dimensions of the feature map.
The class-discriminative localization map $L_{\text{Grad-CAM}}^c$ is computed by a weighted linear combination passed through a ReLU activation:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
The ReLU operation retains features that have a positive influence on the target class of interest while discarding negative evidence.

### 2. Cosine Annealing Learning Rate Schedule with Warmup
To avoid early divergence and escape saddle points:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$
Combined with a linear warm-up phase for the first $T_{\text{warmup}}$ epochs.

---

## 3. Production Implementation: Complete Grad-CAM Heatmap Extractor

```python
import tensorflow as tf
import numpy as np

def generate_gradcam_heatmap(model: tf.keras.Model, 
                              img_array: np.ndarray, 
                              last_conv_layer_name: str, 
                              pred_index: int | None = None) -> np.ndarray:
    """Computes Grad-CAM 2D heatmap attribution for a given image array."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of top predicted class wrt last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only positive influences
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()
```
