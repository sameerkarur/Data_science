# Elective 01: Advanced Deep Learning & Computer Vision
**E&ICT Academy, IIT Kanpur — Specialization Syllabus & Engineering Guide**

---

## 📌 Domain Overview

Advanced Computer Vision represents the confluence of deep representation learning and spatial-temporal pattern analysis. This elective delves into the mathematical mechanics, model architectures, and production deployment paradigms required to process, classify, segment, and synthesize visual media.

---

## 🧭 Specialization Architecture & Curriculum Roadmap

### 1. Modern Deep Convolutional & Hybrid Architectures
- **Residual & Dense Feature Propagation:** Mathematical analysis of residual connections ($y = \mathcal{F}(x, \{W_i\}) + x$) mitigating vanishing gradient phenomena across 100+ layer topologies (ResNet, DenseNet, ResNeXt).
- **Compound Scaling Principles:** EfficientNet scaling depth ($d$), width ($w$), and resolution ($r$) concurrently via the composite coefficient $\phi$ subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.
- **Modern ConvNets:** ConvNeXt architectures adopting 7x7 depthwise separable convolutions, inverted bottlenecks, and LayerNorm to compete with Vision Transformers.

### 2. Vision Transformers (ViT) & Multi-Scale Attention
- **Patch Projection Mechanics:** Linearly flattening non-overlapping 2D image patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$ into 1D token embeddings with learnable 1D position representations.
- **Swin Transformers:** Shifted window multi-head self-attention operating with linear computational complexity $\mathcal{O}(M \times N)$ relative to image dimensions.
- **Self-Supervised Visual Representation:** Masked Autoencoders (MAE) and DINOv2 self-distillation for robust visual features without human annotation.

### 3. Object Detection & Spatial Localization
- **Two-Stage Detectors:** Faster R-CNN with Region Proposal Networks (RPN), Anchor Boxes, and Region of Interest (RoI) Align.
- **Single-Stage Real-Time Detectors:** YOLOv8/YOLOv9 architectures utilizing anchor-free decoupled heads and Task-Aligned Assigner.
- **Transformer Detectors (DETR):** Bipartite matching via the Hungarian algorithm for direct set-prediction without Non-Maximum Suppression (NMS).

### 4. Semantic & Instance Segmentation
- **Fully Convolutional Networks (FCN) & U-Net:** Encoder-decoder skip connections for precision biomedical and spatial pixel classification.
- **Instance Segmentation:** Mask R-CNN parallel branch generating binary segmentation masks atop bounding box regressions.
- **Foundation Segmentation:** Segment Anything Model (SAM) promptable zero-shot segmentation engine.

---

## 📐 Mathematical Cheatsheet & Key Formulations

### Spatial Convolutions & Receptive Fields
$$\text{Output Feature Dimension: } O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
$$\text{Receptive Field Growth: } RF_{l} = RF_{l-1} + (K_l - 1) \cdot \prod_{i=1}^{l-1} S_i$$

### Intersection over Union (IoU) & Generalized IoU (GIoU)
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}, \quad \text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$
*(where $C$ is the smallest convex hull enclosing both bounding boxes $A$ and $B$)*

### Combined Cross-Entropy & Soft Dice Loss
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

---

## 💻 Recommended Applied Projects & Research Benchmarks

1. **Multi-Camera Visual Perception Pipeline:** Real-time multi-class tracking with DeepSORT and YOLOv8 on urban driving datasets.
2. **Medical Scan Micro-Lesion Segmentation:** Dual-encoder U-Net leveraging Swin Transformer backbones for CT/MRI tumor delineation.
3. **Diffusion-Based Visual Inpainting:** Guided latent diffusion models with ControlNet conditioning for structural image restoration.
