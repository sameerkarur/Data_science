# Chapter 3: Multimodal Generative AI, CLIP & Diffusion Models
**Comprehensive Textbook Guide — Advanced Generative AI**

---

## 1. Executive Overview & Mental Models

Multimodal generative AI models bridge different perceptual modalities (vision, text, audio). Contrastive models (CLIP) align images and text into a shared latent manifold, while Latent Diffusion Models (LDMs) synthesize high-fidelity images by iteratively inverting a continuous Gaussian noise process.

```
                   LATENT DIFFUSION MODEL (STABLE DIFFUSION)
    Text Prompt ──► Text Encoder (CLIP) ──► Conditioning Vector (c)
                                                   │
                                                   ▼
    Gaussian Noise z_T ──► [U-Net Denoising Iterations with Cross-Attention] ──► Latent z_0
                                                                                   │
                                                         Decoder (VAE) ◄──────────┘
                                                               │
                                                               ▼
                                                      Output 1024x1024 Image
```

---

## 2. Deep Theoretical Foundations

### 1. Contrastive Language-Image Pretraining (CLIP - Radford et al.)
CLIP trains an image encoder $f(I)$ and text encoder $g(T)$ jointly across a mini-batch of $N$ image-text pairs:
$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{I \to T} + \mathcal{L}_{T \to I} \right)$$
Where the image-to-text contrastive cross-entropy loss is:
$$\mathcal{L}_{I \to T} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(f(I_i), g(T_i)) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(f(I_i), g(T_j)) / \tau)}$$
Maximizing the cosine similarity of true $(I_i, T_i)$ pairs while penalizing all $N^2 - N$ mismatched pairs in the batch.

### 2. Classifier-Free Guidance (CFG - Ho & Salimans)
During diffusion denoising, CFG balances prompt adherence against generation sample diversity. At each denoising step $t$, the predicted noise vector $\hat{\epsilon}_\theta$ combines unconditional and conditional predictions:
$$\hat{\epsilon}_\theta(z_t, c) = \epsilon_\theta(z_t, \emptyset) + s \cdot \left( \epsilon_\theta(z_t, c) - \epsilon_\theta(z_t, \emptyset) \right)$$
Where:
- $c$ is the text prompt conditioning embedding.
- $\emptyset$ is the null/empty prompt embedding.
- $s \ge 1$ is the guidance scale. Setting $s > 1$ amplifies the direction pointing toward the conditioned text prompt, sharpening visual fidelity.

---

## 3. Production Implementation: Zero-Shot Multimodal Classification via CLIP Embeddings

```python
import numpy as np

def zero_shot_clip_predict(image_embedding: np.ndarray, 
                           text_candidate_embeddings: np.ndarray, 
                           class_labels: list[str], 
                           temperature: float = 0.01) -> dict[str, float]:
    """Computes zero-shot classification probabilities from normalized CLIP embeddings."""
    # Ensure L2 unit norms
    img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-10)
    txt_norm = text_candidate_embeddings / (np.linalg.norm(text_candidate_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarities
    logits = np.dot(txt_norm, img_norm) / temperature
    # Stable Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    return {label: float(prob) for label, prob in zip(class_labels, probs)}
```
