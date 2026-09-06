# Chapter 3: Deep Learning Preprocessing & Focal Loss
**Comprehensive Textbook Guide — Advanced Deep Learning**

---

## 1. Executive Overview & Mental Models

Deep neural networks require specialized input conditioning. In computer vision and tabular deep learning, standard cross-entropy fails when millions of easy negative background instances drown out rare positive signals during gradient aggregation.

```
                    FOCAL LOSS DYNAMIC DOWNEIGHTING
    Standard Cross-Entropy Loss: CE = -log(p_t)
    Focal Loss: FL = -(1 - p_t)^γ · log(p_t)
    
    [Easy Negative: p_t = 0.99] ──► Modulating Factor (1 - 0.99)² = 0.0001 (Loss suppressed!)
    [Hard Minority: p_t = 0.10] ──► Modulating Factor (1 - 0.10)² = 0.81   (Loss prioritized!)
```

---

## 2. Deep Theoretical Foundations

### 1. Focal Loss Formulation (Lin et al.)
Focal loss dynamically scales the standard cross-entropy loss by a modulating factor $(1 - p_t)^\gamma$:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where:
$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise} \end{cases}$$
- When an example is misclassified ($p_t$ is small), the modulating factor $(1 - p_t)^\gamma \approx 1$, leaving the loss unaffected.
- As $p_t \to 1$ (high-confidence correct prediction), the modulating factor approaches 0, suppressing the gradient contribution of well-classified easy samples.

### 2. Modern Data Augmentations: Mixup & CutMix
- **Mixup (Zhang et al.):** Convex combinations of pairs of examples and labels:
  $$\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$
- **CutMix (Yun et al.):** Replaces a rectangular patch of image $A$ with a patch from image $B$, scaling labels proportionally to the area of the patch.

---

## 3. Production Implementation: Custom Focal Loss in TensorFlow/Keras

```python
import tensorflow as tf

class BinaryFocalLoss(tf.keras.losses.Loss):
    """Numerically stable Binary Focal Loss with alpha balancing and gamma focusing."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, name: str = "binary_focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        # Calculate alpha_t
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        # Calculate focal loss
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        loss = -focal_weight * tf.math.log(p_t)
        return tf.reduce_mean(loss)
```
