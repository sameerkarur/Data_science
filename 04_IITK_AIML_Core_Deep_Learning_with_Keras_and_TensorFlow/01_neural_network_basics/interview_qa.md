# Interview Q&A — Neural Network Foundations & Backpropagation

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the mathematical model of a single Perceptron and its limitations.

**Answer:** A Perceptron computes a linear combination of inputs: z = sum(w_i * x_i) + b = w^T x + b, followed by a step threshold activation function f(z) = 1 if z >= 0 else 0. As proven by Minsky & Papert (1969), a single-layer perceptron can only learn linearly separable functions and fails on non-linear logic gates like XOR.

### Q2. What is a Multi-Layer Perceptron (MLP) and how does it overcome Perceptron limitations?

**Answer:** An MLP consists of an input layer, one or more hidden layers with non-linear activation functions, and an output layer. Non-linear activations introduce non-linear decision boundaries, enabling MLPs to solve non-linear problems like XOR and act as Universal Function Approximators.

### Q3. State the Universal Approximation Theorem and its practical implications.

**Answer:** The theorem (Cybenko, 1989; Hornik, 1991) states that a feedforward network with a single hidden layer containing a finite number of neurons and non-linear activation functions can approximate any continuous function on compact subsets of R^n to arbitrary precision. In practice, deeper networks with fewer total neurons learn hierarchical representations much more efficiently than a single excessively wide layer.

### Q4. Explain the Backpropagation algorithm step-by-step.

**Answer:** 1. Forward Pass: compute activations layer-by-layer up to output predictions ŷ, then calculate loss L(y, ŷ). 2. Backward Pass: compute error gradient at output layer ∂L/∂ŷ, then apply the multivariable Chain Rule of calculus backward through each layer to compute gradients of the loss with respect to weights ∂L/∂W and biases ∂L/∂b. 3. Optimization Step: update parameters using an optimizer (e.g. W = W - η * ∂L/∂W).

### Q5. Derive the gradient update using the Chain Rule for a 2-layer network.

**Answer:** For loss L, activation a^[2] = σ(z^[2]) where z^[2] = W^[2] a^[1] + b^[2]: ∂L/∂W^[2] = (∂L/∂a^[2]) * (∂a^[2]/∂z^[2]) * (∂z^[2]/∂W^[2]) = δ^[2] * (a^[1])^T, where δ^[2] = (∂L/∂z^[2]). The error propagates backward: δ^[1] = (W^[2])^T δ^[2] * σ'(z^[1]), and ∂L/∂W^[1] = δ^[1] * (x)^T.

### Q6. What is the Vanishing Gradient problem and what causes it?

**Answer:** During backpropagation in deep networks, gradients are multiplied across layers via the chain rule. Saturating activation functions like Sigmoid and Tanh have maximum derivatives of 0.25 and 1.0 respectively. In deep architectures, repeatedly multiplying fractions << 1 causes gradients in earlier layers to shrink exponentially toward zero, preventing early layers from updating their weights.

### Q7. What is the Exploding Gradient problem and how is it mitigated?

**Answer:** Exploding gradients occur when large weights or derivatives compound across deep networks, causing gradient norms to grow exponentially, resulting in numerical overflow (NaNs) and unstable weight updates. Mitigated via: (1) Gradient Clipping (clipping gradient norm to a maximum threshold), (2) proper weight initialization (He/Glorot), and (3) Batch Normalization.

### Q8. Compare Sigmoid, Tanh, ReLU, Leaky ReLU, and GELU activation functions.

**Answer:** Sigmoid: σ(z) = 1/(1+e^-z); outputs [0, 1], non-zero-centered, suffers from severe vanishing gradients. Tanh: zero-centered [-1, 1], but still saturates at tails. ReLU: max(0, z); computationally cheap, does not saturate for z > 0, but suffers from 'Dying ReLU'. Leaky ReLU: max(αz, z) with α ≈ 0.01; allows small gradient when z < 0. GELU: x * Φ(x); smooth, probabilistic gating standard in modern Transformers (BERT, GPT).

### Q9. What is the 'Dying ReLU' problem and how can it be prevented?

**Answer:** Dying ReLU occurs when a large gradient pushes neuron weights such that the neuron outputs z <= 0 for all training examples. Because ReLU's derivative is 0 for negative inputs, gradient flow halts completely, rendering the neuron permanently inactive ('dead'). Prevented by using Leaky ReLU, Parametric ReLU (PReLU), ELU, or reducing the learning rate.

### Q10. Explain Weight Initialization: Xavier/Glorot vs He/Kaiming initialization.

**Answer:** Initializing weights to zeros makes all neurons compute identical gradients (symmetry problem). Xavier/Glorot draws from variance Var(W) = 2 / (fan_in + fan_out), designed to preserve activation and gradient variance for Sigmoid/Tanh. He/Kaiming draws from Var(W) = 2 / fan_in, specifically derived to account for ReLU zeroing out half of the activations.

### Q11. What is the loss surface of a neural network and what are Saddle Points?

**Answer:** The loss surface is a high-dimensional landscape mapping network weights to loss values. In high dimensions, local minima are rare; most critical points (where gradient = 0) are Saddle Points (minima along some dimensions, maxima along others). First-order gradient descent can slow down severely near saddle points, but stochastic noise and momentum help escape them.

### Q12. Explain the difference between Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch SGD.

**Answer:** Batch GD computes gradients across the ENTIRE dataset before updating weights (exact gradient, but slow and memory-prohibitive). Pure SGD updates weights after EVERY single sample (fast, but noisy and cannot leverage GPU vectorization). Mini-Batch SGD computes gradients over small batches (e.g. 32 to 256 samples), balancing gradient stability with GPU SIMD parallelism.

### Q13. What does Momentum do in gradient descent?

**Answer:** Momentum adds a fraction γ of the previous update vector to the current gradient update: v_t = γ v_{t-1} + η ∇L(W). It accelerates gradient descent along persistent directional trajectories while dampening wild oscillations perpendicular to ravines, speeding up convergence.

### Q14. Explain Nesterov Accelerated Gradient (NAG).

**Answer:** Standard momentum computes the gradient at the current position, then takes a momentum step. Nesterov 'looks ahead' by computing the gradient at the anticipated future position: ∇L(W - γ v_{t-1}), applying a proactive course correction that prevents overshooting sharp turns.

### Q15. How does the learning rate hyperparameter affect optimization?

**Answer:** If learning rate η is too small, training converges at an agonizingly slow pace and gets trapped in flat plateaus. If η is too large, optimization oscillates wildly, overshoots minima, and can diverge toward infinite loss (NaNs). Learning rate schedules (warmup, cosine decay) adapt η dynamically.

### Q16. What is Learning Rate Warmup and why is it standard in deep learning?

**Answer:** Warmup starts training with a very small learning rate for the first few epochs/steps, gradually ramping up to the base learning rate. Early in training, network weights are randomized and gradients are huge and unstable; warmup prevents catastrophic early updates that destabilize representations.

### Q17. Explain Cosine Annealing learning rate schedule.

**Answer:** Cosine annealing decays the learning rate following a half-cosine curve: η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(t * π / T_max)). It decays gradually initially, drops rapidly through intermediate epochs, and smoothly flattens near zero, enabling fine convergence into deep minima.

### Q18. What is an Epoch vs an Iteration vs a Batch Size?

**Answer:** Batch Size: number of training samples processed in one forward/backward pass. Iteration (Step): one parameter update pass over a single mini-batch. Epoch: one complete pass through the entire training dataset (Iterations per epoch = Total Samples / Batch Size).

### Q19. Why is Shuffle=True critical during training mini-batches?

**Answer:** If data is not shuffled between epochs, mini-batches contain identical samples in identical order, introducing cyclical bias into gradient paths and potentially clustering identical labels together, which causes optimization instability. Shuffling breaks correlations and ensures stochastic gradient variance.

### Q20. Explain the Softmax function and how temperature scaling works.

**Answer:** Softmax maps raw logits z into a probability distribution: P(y=k) = e^(z_k / T) / sum_j e^(z_j / T). Temperature T modulates output entropy: T = 1 is standard softmax; T > 1 smooths probabilities toward uniform distribution (higher exploration); T < 1 sharpens probabilities toward argmax (higher confidence).

### Q21. What is Categorical Cross-Entropy loss mathematically?

**Answer:** L = - sum_{k=1}^K y_k ln(p_k), where y is a one-hot vector and p is the predicted probability distribution. For the true class c, the loss simplifies to -ln(p_c). Minimizing cross-entropy is mathematically equivalent to minimizing the Kullback-Leibler (KL) divergence between empirical and predicted distributions.

### Q22. What is Sparse Categorical Cross-Entropy and when is it preferred over Categorical Cross-Entropy?

**Answer:** Categorical Cross-Entropy expects one-hot encoded targets of shape (N, K), consuming O(N*K) memory. Sparse Categorical Cross-Entropy accepts integer class labels (0 to K-1) of shape (N, 1) and computes cross-entropy directly without one-hot expansion, saving significant RAM on large vocabularies (e.g. 50,000 classes).

### Q23. Explain computational graphs and automatic differentiation (Autograd).

**Answer:** A computational graph represents mathematical operations as directed acyclic graphs (DAGs) where nodes are operations and edges are tensors. Automatic differentiation traverses the graph using the chain rule, decomposing complex matrix equations into primitive operations with known analytic derivatives, tracking intermediate variables dynamically during the forward pass.

### Q24. What is the difference between Reverse-Mode and Forward-Mode automatic differentiation?

**Answer:** Forward-mode computes derivatives alongside the forward pass; efficient when outputs >> inputs. Reverse-mode (backpropagation) performs a forward pass to store intermediate activations, then sweeps backward from outputs to inputs. Reverse-mode is vastly more efficient for neural networks where inputs (millions of weights) >> outputs (single scalar loss).

### Q25. What are residual skip connections (ResNet) and why do they solve degradation?

**Answer:** Skip connections add the input of a layer directly to its output: y = F(x, {W}) + x. Instead of forcing layers to fit an underlying mapping H(x), they fit the residual mapping F(x) = H(x) - x. During backpropagation, the identity term (+x) creates a direct gradient highway where gradients propagate backward unaltered (∂y/∂x = ∂F/∂x + 1), completely bypassing vanishing gradients.

### Q26. Explain the Forward Pass computational complexity of a Fully Connected layer.

**Answer:** For an input vector of dimension d_in and output dimension d_out, the matrix multiplication W * x requires d_in * d_out multiplications and additions, yielding O(d_in * d_out) FLOPs per sample, and storing W requires d_in * d_out parameters.

### Q27. What is Gradient Checking (Grad Check) and how does it verify backprop implementations?

**Answer:** Grad check compares analytic gradients ∂L/∂W computed by backprop against numerical two-sided finite difference approximations: (L(W + ε) - L(W - ε)) / (2ε). A relative error <= 1e-7 verifies that backpropagation is mathematically bug-free.

### Q28. What is inductive bias in neural network architectures?

**Answer:** Inductive bias is the set of architectural assumptions a model relies on to predict outputs on unseen data. MLPs have weak inductive bias (assuming only smoothness). CNNs have strong inductive bias (spatial locality and translation equivariance). RNNs assume sequential temporal causality. Strong inductive bias requires less training data to learn specific domain patterns.

### Q29. What is catastrophic forgetting in neural networks?

**Answer:** Catastrophic forgetting occurs when a neural network trained on Task A is subsequently trained on Task B: gradient updates overwrite the weight representations established for Task A, causing performance on Task A to abruptly collapse.

### Q30. Explain the difference between Parameter and Hyperparameter in deep learning.

**Answer:** Parameters are internal configuration variables learned automatically from data during training (weights W, biases b). Hyperparameters are external configuration choices set by the practitioner prior to training that govern the learning process (learning rate, batch size, number of layers, activation functions, dropout rate).
