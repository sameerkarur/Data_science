# Elective 05: Academic Masterclass by IIT Kanpur Faculty
**E&ICT Academy, IIT Kanpur — Advanced Theoretical Insights & Frontiers of AI**

---

## 📌 Domain Overview

The Academic Masterclass series, curated and delivered by distinguished faculty and researchers from the **Department of Computer Science & Engineering (CSE) and Electrical Engineering at IIT Kanpur**, provides deep theoretical foundations, mathematical proofs, and insights into the mathematical frontiers of modern artificial intelligence and machine learning.

---

## 🧭 Specialization Architecture & Curriculum Roadmap

### 1. Statistical Learning Theory & Generalization Bounds
- **Probably Approximately Correct (PAC) Learning:** Formal mathematical definition of sample complexity $m \ge \frac{1}{\epsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)$ ensuring bounded error $\epsilon$ with confidence $1 - \delta$.
- **Vapnik-Chervonenkis (VC) Dimension:** Combinatorial measure of hypothesis class capacity, shattering points in $\mathbb{R}^d$, and the Fundamental Theorem of Statistical Learning.
- **Rademacher Complexity:** Data-dependent capacity measures and uniform convergence bounds for neural networks:
  $$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma \left[ \sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h(x_i) \right]$$

### 2. Advanced Optimization in High-Dimensional Landscapes
- **Loss Surfaces of Deep Networks:** Non-convex optimization dynamics, saddle points vs. local minima, and the geometry of over-parameterized models.
- **Implicit Regularization in Gradient Descent:** Why first-order gradient descent biased towards minimum norm solutions generalizes even when models possess capacity to memorize arbitrary noise.
- **Stochastic Optimization Frontiers:** Stochastic Polyak momentum, Sharpness-Aware Minimization (SAM), and adaptive preconditioners.

### 3. Representation Learning & Manifold Geometry
- **The Manifold Hypothesis:** Mathematical justification for high-dimensional real-world data lying on low-dimensional Riemannian sub-manifolds.
- **Geometric Deep Learning:** Invariance and equivariance under symmetry groups ($SE(2), SE(3)$), graph neural networks (GNNs), and gauge equivariant convolutions.
- **Information Bottleneck Principle:** Minimizing mutual information $I(X; T)$ while maximizing $I(T; Y)$ for optimal latent representations $T$.

### 4. Frontiers of Generative Modeling & Alignment Theory
- **Score-Based Diffusion & SDEs:** Unifying diffusion models under continuous-time Stochastic Differential Equations (Song et al.):
  $$dx = f(x, t)dt + g(t)dw$$
- **Optimal Transport & Wasserstein Distances:** Earth Mover's Distance, Kantorovich-Rubinstein duality, and continuous normalizing flows.
- **Theoretical Alignment & Game Theory:** Nash equilibrium in multi-agent generative systems, RLHF mathematical limits, and reward gaming bounds.

---

## 📚 Seminal Academic References & Key Publications

1. **Vapnik, V. N. (1998):** *Statistical Learning Theory.* Wiley-Interscience.
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016):** *Deep Learning.* MIT Press.
3. **Song, Y., et al. (2020):** *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR Outstanding Paper.
4. **Bronstein, M. M., et al. (2021):** *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* arXiv:2104.13478.

---

## 💻 Recommended Academic Explorations & Implementations

1. **Empirical Loss Surface Visualization:** Filter-normalized loss landscape projection for ResNets vs. VGG architectures.
2. **Sharpness-Aware Minimization (SAM) Implementation:** Custom PyTorch training step computing perturbation $\hat{\epsilon}(w) = \rho \frac{\nabla_w L(w)}{\|\nabla_w L(w)\|_2}$.
3. **Continuous-Time Score Matching Toy Engine:** 2D Swiss Roll distribution modeling via denoising score matching and Euler-Maruyama sampling.
