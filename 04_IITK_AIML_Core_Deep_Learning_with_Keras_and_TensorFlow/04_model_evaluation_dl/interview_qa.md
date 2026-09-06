# Interview Q&A — Deep Learning Model Evaluation & Interpretability

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. How do you identify Underfitting, Optimal Fitting, and Overfitting from Learning Curves?

**Answer:** Underfitting: Training loss and validation loss both remain high with negligible improvement (high bias). Optimal Fitting: Both training and validation loss decrease steadily and plateau close together with minimal generalization gap. Overfitting: Training loss continues to plummet toward zero while validation loss reverses direction and climbs upward (high variance).

### Q2. Explain the Double Descent phenomenon in modern deep learning.

**Answer:** Traditional statistical theory states that model test error decreases, hits a minimum, and then increases as model capacity grows (U-shaped curve). Deep learning exhibits Double Descent: as model parameters exceed the number of training samples (interpolation threshold), test error initially spikes, but then drops a second time and continues decreasing as over-parameterization increases.

### Q3. Why is Accuracy insufficient for evaluating deep tabular and medical models?

**Answer:** Accuracy treats all errors equally and masks minority failure under class imbalance. In cancer detection or fraud prevention, false negatives carry catastrophic financial or life-safety costs. Sensitivity, Specificity, PR-AUC, and Expected Cost are mandatory.

### Q4. Explain Area Under the Precision-Recall Curve (PR-AUC / Average Precision) in deep classification.

**Answer:** PR-AUC calculates the area under the curve plotting Precision vs Recall across all thresholds: AP = sum (R_n - R_{n-1}) * P_n. It evaluates ranking quality on positive classes without being inflated by overwhelming true negative counts.

### Q5. What is Top-1 vs Top-5 Error Rate in Image Classification?

**Answer:** Top-1 Error: percentage of test images where the model's highest-probability prediction is NOT the ground truth class. Top-5 Error: percentage of images where the true class is not among the model's top 5 highest-probability predictions (standard benchmark for 1000-class ImageNet).

### Q6. Explain Expected Calibration Error (ECE) for assessing neural network calibration.

**Answer:** ECE groups model predictions into M equal-width confidence bins, computing the absolute difference between average confidence and true empirical accuracy per bin weighted by sample count: ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|. Lower ECE indicates reliable probabilistic predictions.

### Q7. What is Temperature Scaling and why is it effective for post-processing calibration?

**Answer:** Temperature scaling divides raw model logits z by a learned scalar T > 0 before applying Softmax: p̂ = softmax(z / T). Because T is optimized on validation data using negative log-likelihood without changing the argmax, it preserves classification accuracy while adjusting confidence to match empirical reality.

### Q8. Explain Grad-CAM (Gradient-weighted Class Activation Mapping) for deep model interpretability.

**Answer:** Grad-CAM computes the gradient of the score for target class c with respect to feature activation maps A^k of the last convolutional layer. Gradients are globally pooled to calculate importance weights α_k^c. A weighted combination of feature maps followed by ReLU yields a coarse 2D heatmap highlighting regions supporting the decision.

### Q9. What is Integrated Gradients (Axiomatic Attribution)?

**Answer:** Integrated Gradients calculates feature attributions by integrating the gradients of the model prediction along a straight line path from a neutral baseline x' (e.g. black image or zero embedding) to the input x: Attribution_i = (x_i - x'_i) * int_0^1 (∂F(x' + α(x - x')) / ∂x_i) dα, satisfying Completeness and Implementation Invariance axioms.

### Q10. Explain SHAP (SHapley Additive exPlanations) DeepExplainer for neural networks.

**Answer:** DeepExplainer leverages the connection between DeepLIFT and Shapley values from cooperative game theory, recursively passing analytic attribution scores backward through neural network layers based on reference background samples, explaining non-linear feature attributions in real time.

### Q11. What is Mean Average Precision (mAP) in Object Detection?

**Answer:** mAP evaluates object detection across all classes. An object detection is a True Positive if Intersection over Union (IoU) with ground truth >= threshold (e.g. 0.5 for mAP@0.5). Average Precision (AP) is the area under the PR curve for a class. mAP is the mean AP averaged across all classes and IoU thresholds (0.50 to 0.95 in COCO).

### Q12. Explain Intersection over Union (IoU / Jaccard Index) for bounding boxes.

**Answer:** IoU = Area of Overlap / Area of Union between predicted bounding box B_p and ground truth box B_gt. An IoU >= 0.5 is standardly accepted as a valid localization.

### Q13. What is Dice Coefficient (F1-score) vs IoU in Semantic Image Segmentation?

**Answer:** Dice = 2 * |A ∩ B| / (|A| + |B|) = 2 * TP / (2*TP + FP + FN). IoU = |A ∩ B| / |A ∪ B| = TP / (TP + FP + FN). Mathematically: Dice = 2 * IoU / (1 + IoU). Dice penalizes false positives and false negatives smoothly and is directly differentiable as a loss function (Dice Loss).

### Q14. Explain BLEU score and ROUGE score for generative sequence evaluation.

**Answer:** BLEU (Bilingual Evaluation Understudy) measures n-gram precision between generated text and reference text with a brevity penalty, standard in translation. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram recall, particularly ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (Longest Common Subsequence), standard in text summarization.

### Q15. What is Perplexity (PPL) in language model evaluation?

**Answer:** Perplexity is the exponentiated average negative log-likelihood per token: PPL = exp(- (1/T) sum_{t=1}^T ln P(w_t | w_{<t})). It measures how surprised the model is by test text; lower perplexity corresponds to a better predictive language model.

### Q16. How do you evaluate Inception Score (IS) and Fréchet Inception Distance (FID) for GANs and Diffusion models?

**Answer:** IS evaluates generated images using an Inception-v3 classifier: computes exp(KL(p(y|x) || p(y))), penalizing lack of diversity or realism. FID compares feature activations of real and generated images in Inception-v3 feature space: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2)). Lower FID indicates generated images match real distribution statistics.

### Q17. What is Adversarial Robustness and how do you test for Fast Gradient Sign Method (FGSM) attacks?

**Answer:** Adversarial examples add imperceptible noise to inputs to induce misclassification: x_adv = x + ε * sign(∇_x L(θ, x, y)). Evaluate robustness by measuring accuracy drop as perturbation magnitude ε increases, and harden models using adversarial training.

### Q18. Explain Out-Of-Distribution (OOD) Detection in production neural networks.

**Answer:** OOD detection identifies inputs originating from distributions entirely outside the training domain (e.g. showing a picture of a car to a dermatology classifier). Techniques: Maximum Softmax Probability (MSP), Energy-Based Out-of-Distribution scoring, and Mahalanobis distance in feature space.

### Q19. What is Inference Latency vs Throughput in deep learning production?

**Answer:** Latency (P50, P95, P99) is the elapsed time in milliseconds to compute a single prediction for one request. Throughput is the total volume of requests or samples processed per second (QPS) across all concurrent threads and GPUs.

### Q20. How do FLOPs and Parameter Counts determine deep learning computational efficiency?

**Answer:** Parameter count dictates memory footprint (RAM/VRAM storage). FLOPs (Floating-Point Operations) measures the mathematical operations required for one forward pass, governing compute runtime. However, real-world latency is also constrained by memory bandwidth (Arithmetic Intensity = FLOPs / Memory Access Bytes).

### Q21. What is Model Pruning (Structured vs Unstructured) and how is it evaluated?

**Answer:** Pruning removes unneeded weights to compress models. Unstructured pruning zeroes individual weights based on magnitude, creating sparse matrices (requires specialized hardware to accelerate). Structured pruning removes entire channels, attention heads, or layers, producing smaller dense matrices that accelerate natively on standard hardware.

### Q22. Explain Knowledge Distillation (Teacher-Student network).

**Answer:** A large, high-capacity Teacher network trains a compact Student network. The student minimizes a combined loss: standard cross-entropy with hard labels plus KL divergence with the Teacher's softened probabilities (computed with temperature T > 1), capturing 'dark knowledge' (inter-class relationships).

### Q23. What is Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)?

**Answer:** PTQ quantizes weights and activations of a pre-trained model to INT8 after training, which is fast but can cause accuracy drops. QAT models quantization error during the training forward pass using fake-quantization nodes while computing gradients in FP32, allowing the network to adapt and retain full accuracy.

### Q24. How does A/B Testing evaluate deep learning models against live production baselines?

**Answer:** Route a random fraction of live traffic (e.g. 50%) to Model A (baseline) and 50% to Model B (new deep model). Measure statistical lift on business KPIs (conversion, retention, CTR) over 2-4 weeks, verifying with hypothesis tests before full rollout.

### Q25. What is Canary Deployment in machine learning?

**Answer:** A Canary deployment rolls out a new model to a tiny subset of users (e.g. 1% to 5%) while routing the remaining 95% to the stable model. It monitors telemetry (latency, error codes, crash rates) for an initial burn-in period before incrementally ramping up traffic to 100%.

### Q26. Explain Shadow Mode (Dark Launch) deployment.

**Answer:** Shadow mode sends live production requests in parallel to both the active production model and the new candidate model. The candidate model computes predictions for logging and evaluation without serving responses to users, validating production latency, throughput, and stability risk-free.

### Q27. What is Feature Drift vs Concept Drift in computer vision models?

**Answer:** Feature drift: shift in visual image properties (e.g. night-time driving, snowy weather, new camera sensor blur). Concept drift: shift in the semantic meaning or relationship to classes (e.g. fashion trends redefining what constitutes 'business casual').

### Q28. Explain Population Stability Index (PSI) applied to deep feature embeddings.

**Answer:** Compute average cosine distance or maximum mean discrepancy (MMD) between production embedding clusters and baseline training embedding clusters. Drastic drops in similarity indicate feature distribution drift.

### Q29. What are Golden Evaluation Sets in continuous integration (CI) for deep learning?

**Answer:** A fixed, hand-curated suite of mission-critical regression test examples and difficult edge cases that every newly trained model checkpoint must satisfy before being promoted to staging or production.

### Q30. Explain Algorithmic Bias and Disparate Impact in deep learning models.

**Answer:** Disparate impact occurs when a model produces substantially different positive prediction rates across protected demographic groups (Four-Fifths Rule: selection rate for minority group < 80% of majority group). Evaluate using Equalized Odds, Demographic Parity, and Equality of Opportunity metrics.
