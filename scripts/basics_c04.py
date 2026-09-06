"""
Textbook-Scale Architectural & Conceptual Guides for Course 4:
IITK AIML Core: Deep Learning with Keras and TensorFlow
Modules:
  01_neural_network_basics
  02_keras_tensorflow
  03_preprocessing_imbalance
  04_model_evaluation_dl
"""

C04_BASICS = {}

# =====================================================================
# 1. Deep Neural Network Foundations & Backpropagation
# =====================================================================
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics"] = r'''# Chapter 1: Deep Neural Network Foundations & Backpropagation
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
'''

# =====================================================================
# 2. Keras 3 & TensorFlow Computation Graph Architecture
# =====================================================================
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow"] = r'''# Chapter 2: Keras 3 & TensorFlow Computational Architecture
**Comprehensive Textbook Guide — Advanced Deep Learning**

---

## 1. Executive Overview & Mental Models

Modern deep learning frameworks bridge dynamic imperative debugging with high-performance static compilation. TensorFlow 2 / Keras 3 allows eager pythonic execution while compiling critical functions into static XLA (Accelerated Linear Algebra) computation graphs via `@tf.function`.

```
                 TF.DATA HIGH-THROUGHPUT PIPELINE
    Disk (Files) ──► Interleave ──► Prefetch ──► Map (Augment) ──► Batch ──► GPU Memory
                           │
                 Runs in parallel C++ threads!
                 GPU never starves for data!
```

---

## 2. Deep Theoretical Foundations

### 1. High-Throughput `tf.data` ETL Architecture
To achieve near 100% GPU compute utilization, data ingestion must overlap with backward pass computation:
- **`.interleave(cycle_length=AUTOTUNE)`:** Concurrently reads from multiple disk shards.
- **`.map(num_parallel_calls=AUTOTUNE)`:** Distributes CPU-intensive image decoding and augmentations across all available host CPU cores.
- **`.prefetch(buffer_size=AUTOTUNE)`:** Utilizes double-buffering to pre-stage the $(N+1)$-th mini-batch in GPU VRAM while the GPU CUDA cores are actively executing the $N$-th forward-backward training step.

### 2. Mixed Precision Training (FP16 / BF16)
Mixed precision uses 16-bit floating point representations (IEEE FP16 or Brain Floating Point BF16) for activations and weights while maintaining 32-bit (FP32) master weights:
- Doubles tensor throughput via GPU Tensor Cores.
- Reduces activation memory in half.
- Utilizes dynamic **Loss Scaling** to prevent gradient underflow in small FP16 exponent ranges.

---

## 3. Production Implementation: Optimized Keras Model with High-Throughput Pipeline

```python
import tensorflow as tf

def build_production_vision_backbone(input_shape: tuple = (224, 224, 3), num_classes: int = 10) -> tf.keras.Model:
    """Builds a high-throughput convolutional network with residual connections."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stem convolution
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    
    # Residual Block
    residual = tf.keras.layers.Conv2D(64, kernel_size=1, strides=2, padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.add([x, residual])
    x = tf.keras.layers.Activation('swish')(x)
    
    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet_Stem")
    return model
```
'''

# =====================================================================
# 3. Deep Learning Preprocessing & Focal Loss
# =====================================================================
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance"] = r'''# Chapter 3: Deep Learning Preprocessing & Focal Loss
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
'''

# =====================================================================
# 4. Model Evaluation, Interpretability & Grad-CAM
# =====================================================================
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl"] = r'''# Chapter 4: Deep Learning Evaluation & Grad-CAM Interpretability
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
'''

print(f"Loaded {len(C04_BASICS)} textbook chapters for Course 4.")
