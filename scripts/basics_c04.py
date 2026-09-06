"""
Comprehensive, high-depth Basics & Architecture Guides for Course 4:
Deep Learning with Keras & TensorFlow (4 modules)
"""

C04_BASICS = {}

# 1. Neural Network Basics
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics"] = """# Deep Neural Network Foundations & Backpropagation
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Feedforward artificial neural networks propagate tensors through non-linear functional compositions, updating weights via the multivariate calculus **Chain Rule**.

```
                   FORWARD & BACKPROPAGATION TENSOR FLOW
    Input (x) ──► Linear: z = Wx + b ──► Non-Linear: a = σ(z) ──► Loss (L)
                                                                    │
    Update: W ← W - η·(∂L/∂W) ◄── Backprop: ∂L/∂W = (∂L/∂a)·σ'(z)·x ┘
```

---

## 🧭 Deep Theoretical Foundations

### 1. The Backpropagation Chain Rule
For layer $l$ with activations $a^{[l]} = \sigma(z^{[l]})$ and pre-activations $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$:
$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot \sigma'(z^{[l]})$$
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T, \quad \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

### 2. Vanishing & Exploding Gradients
In deep networks using sigmoid or tanh activations, saturating tails have derivatives $\sigma'(z) \to 0$. Repeated chain-rule multiplications cause gradients to diminish exponentially:
- Mitigated by **ReLU / LeakyReLU / GELU** activations with non-saturating gradients.
- Stabilized by **He (Kaiming) & Xavier (Glorot) Initialization** preserving variance:
  $$\text{He Variance: } \text{Var}(W) = \frac{2}{n_{\text{in}}}, \quad \text{Xavier Variance: } \text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$
"""

# 2. Keras & TensorFlow
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow"] = """# Keras 3 & TensorFlow Computation Graph Architecture
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 TF.DATA HIGH-THROUGHPUT PIPELINE
    Disk (Files) ──► Interleave ──► Prefetch ──► Map (Augment) ──► Batch ──► GPU Memory
                           │
                 Runs in parallel C++ threads!
                 GPU never starves for data!
```

---

## 🧭 Deep Theoretical Foundations

### 1. TensorFlow API Tiers
- **Sequential API:** Linear stack of layers for simple feedforward pipelines.
- **Functional API:** Directed Acyclic Graph (DAG) supporting multi-input, multi-output, and residual skip connections.
- **Model Subclassing:** Imperative dynamic execution via `call(inputs, training=False)` allowing dynamic loops and custom attention.

### 2. High-Performance `tf.data` Pipeline
To achieve 100% GPU saturation, `tf.data` uses:
- `.cache()`: Keeps decoded records in RAM.
- `.map(num_parallel_calls=tf.data.AUTOTUNE)`: Parallel CPU preprocessing.
- `.prefetch(buffer_size=tf.data.AUTOTUNE)`: Pre-stages next training batch in GPU VRAM during backward pass execution.
"""

# 3. Preprocessing & Imbalance
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance"] = """# Deep Learning Preprocessing & Focal Loss
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                    FOCAL LOSS DYNAMIC DOWNEIGHTING
    Standard Cross-Entropy Loss: CE = -log(p_t)
    Focal Loss: FL = -(1 - p_t)^γ · log(p_t)
    
    [Easy Negative: p_t = 0.99] ──► Modulating Factor (1 - 0.99)² = 0.0001 (Loss suppressed!)
    [Hard Minority: p_t = 0.10] ──► Modulating Factor (1 - 0.10)² = 0.81   (Loss prioritized!)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Focal Loss Formulation (Lin et al.)
Addresses severe class imbalance (e.g. 1000:1 background vs foreground) by focusing learning on hard false negatives:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
The focusing parameter $\gamma$ dynamically scales down the loss contribution from easy examples, preventing the gradient from being swamped by negative background instances.
"""

# 4. Model Evaluation DL
C04_BASICS["04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl"] = """# Deep Learning Evaluation, Interpretability & Grad-CAM
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                  GRAD-CAM (GRADIENT-WEIGHTED CLASS ACTIVATION MAP)
    Input Image ──► Conv Backbone ──► Feature Maps Aᵏ ──► Dense Layer ──► Score yᶜ
                                            ▲
    Gradients ∂yᶜ/∂Aᵏ ──► Global Avg Pool ──┘
                               │
    Linear Combination: L_GradCAM = ReLU(Σ αₖ Aᵏ) ──► Heatmap Overlay!
```

---

## 🧭 Deep Theoretical Foundations

### 1. Grad-CAM Mathematical Formulation
Grad-CAM computes the gradient of the target class score $y^c$ with respect to feature map activations $A^k$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
The ReLU operation retains features that have a positive influence on the target class of interest while discarding negative evidence.
"""

print(f"Loaded {len(C04_BASICS)} comprehensive guides for Course 4.")
