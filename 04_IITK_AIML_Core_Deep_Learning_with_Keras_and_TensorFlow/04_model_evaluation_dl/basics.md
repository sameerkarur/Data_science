# Deep Learning Evaluation, Interpretability & Grad-CAM
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
$$lpha_k^c = rac{1}{Z} \sum_{i} \sum_{j} rac{\partial y^c}{\partial A_{i, j}^k}$$
$$L_{	ext{Grad-CAM}}^c = 	ext{ReLU}\left( \sum_k lpha_k^c A^k ight)$$
The ReLU operation retains features that have a positive influence on the target class of interest while discarding negative evidence.
