# Interview Q&A — Supervised Classification & Ensembles

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain Logistic Regression and derive the Sigmoid function.

**Answer:** Logistic Regression models the probability of a binary outcome y in {0, 1} given features x: P(y=1|x) = σ(w^T x). The Sigmoid function σ(z) = 1 / (1 + e^(-z)) maps unbounded linear real values z in (-inf, +inf) strictly into probabilities in [0, 1]. Its derivative has the elegant form σ'(z) = σ(z)(1 - σ(z)).

### Q2. What is Binary Cross-Entropy (Log Loss) and why is Mean Squared Error (MSE) not used for logistic regression?

**Answer:** Binary Cross-Entropy Loss: L(w) = - [y ln(ŷ) + (1 - y) ln(1 - ŷ)]. When combined with the Sigmoid activation, cross-entropy produces a smooth convex optimization surface with non-vanishing gradients. Using MSE with Sigmoid produces a highly non-convex loss surface with numerous poor local minima and flat saddle regions where gradients vanish.

### Q3. How do Decision Trees choose the best split point (Gini Impurity vs Information Gain)?

**Answer:** At each candidate split, the tree evaluates reduction in impurity. Gini Impurity: G = 1 - sum(p_i²). Information Gain = Entropy(parent) - sum((N_child/N_parent) * Entropy(child)), where Entropy = - sum(p_i log2(p_i)). The split that maximizes impurity reduction across children is selected.

### Q4. What is the difference between Bagging and Boosting?

**Answer:** Bagging (Bootstrap Aggregating): trains independent, deep, high-variance base estimators in parallel on random bootstrap subsets of data and averages predictions, reducing variance. Boosting: trains shallow, high-bias base estimators sequentially, where each successive estimator is trained to correct the residual errors made by preceding estimators, primarily reducing bias.

### Q5. Explain Random Forest and how Feature Bagging decorrelates trees.

**Answer:** Random Forest combines bootstrap sampling (data bagging) with feature bagging: at each candidate split in every individual tree, only a random subset of features (typically sqrt(p) for classification) is considered. This prevents dominant features from dictating every tree split, decorrelating tree predictions and drastically reducing ensemble variance.

### Q6. What is Out-Of-Bag (OOB) error in Random Forest?

**Answer:** Because bootstrap sampling samples with replacement, ~36.8% of training samples (1 - 1/N)^N ≈ 1/e are excluded from each individual tree's training set. The OOB error evaluates each sample using only trees that did not include that sample during training, providing an unbiased validation estimate without separate cross-validation.

### Q7. Explain AdaBoost (Adaptive Boosting) and how sample weights are updated.

**Answer:** AdaBoost trains sequential decision stumps. Misclassified training instances have their sample weights increased, forcing subsequent stumps to concentrate on difficult edge cases. The final model is a weighted majority vote where tree weights depend on their individual accuracy: α_m = 0.5 * ln((1 - err_m) / err_m).

### Q8. How does Gradient Boosting differ from AdaBoost?

**Answer:** AdaBoost minimizes exponential loss by adjusting sample instance weights. Gradient Boosting minimizes any differentiable loss function (e.g. cross-entropy, Huber loss) by fitting subsequent trees to the negative gradients (pseudo-residuals) of the loss function evaluated at the current ensemble predictions.

### Q9. What makes XGBoost faster and more regularized than standard Gradient Boosting?

**Answer:** XGBoost incorporates second-order Taylor expansion (computing both first-order gradients g_i and second-order Hessians h_i), adds explicit L1 (α) and L2 (λ) leaf weight regularization to the objective function, uses histogram-based split finding, handles missing values natively, and parallelizes tree construction across CPU cores.

### Q10. Explain LightGBM's Leaf-Wise tree growth and GOSS (Gradient-based One-Side Sampling).

**Answer:** LightGBM uses leaf-wise (best-first) tree growth, splitting the leaf with maximum loss reduction regardless of depth (unlike level-wise growth in traditional trees), achieving lower loss at the risk of overfitting on small data. GOSS retains all instances with large gradients (under-trained) and randomly samples instances with small gradients, accelerating training by up to 10x.

### Q11. How does CatBoost handle categorical features natively?

**Answer:** CatBoost computes target encodings on-the-fly using 'Ordered Target Statistics' calculated strictly on preceding rows in random permutations of the data, completely eliminating target leakage and avoiding expensive one-hot encoding.

### Q12. What is the Support Vector Machine (SVM) objective and what is the Maximum Margin Hyperplane?

**Answer:** SVM finds a hyperplane w^T x + b = 0 that separates binary classes while maximizing the geometric margin 2 / ||w|| between the hyperplane and the closest training samples (Support Vectors), minimizing 0.5 * ||w||² subject to y_i(w^T x_i + b) >= 1 - ξ_i.

### Q13. What are Slack Variables (ξ_i) and the C parameter in Soft-Margin SVM?

**Answer:** Slack variables allow margin violations and misclassifications for non-linearly separable data. The hyperparameter C controls the regularization trade-off: large C penalizes violations heavily, producing a narrow margin with lower bias but higher risk of overfitting; small C allows more violations, producing a wider margin with higher bias and better generalization.

### Q14. Explain the Kernel Trick in SVM and name common kernel functions.

**Answer:** The kernel trick computes inner products in a high-dimensional feature space without explicitly computing coordinates in that space: K(x, z) = <φ(x), φ(z)>. Common kernels: Linear, Polynomial K(x, z) = (γ x^T z + r)^d, and Radial Basis Function (RBF / Gaussian) K(x, z) = exp(-γ ||x - z||²).

### Q15. What does the γ (gamma) hyperparameter control in an RBF SVM?

**Answer:** Gamma controls the radius of influence of individual support vectors: γ = 1 / (2σ²). High gamma means support vectors have a tight radius of influence, creating complex, winding decision boundaries that overfit. Low gamma creates broad, smooth, linear-like boundaries that underfit.

### Q16. Explain K-Nearest Neighbors (KNN) classification and why it is a 'Lazy Learner'.

**Answer:** KNN is a lazy, instance-based non-parametric classifier: it has no explicit training phase (it stores training data in memory). At prediction time, it computes distances to all training points, identifies the k-nearest neighbors, and performs majority voting. Prediction is computationally heavy (O(N * D)), and vulnerable to the curse of dimensionality.

### Q17. What is Naive Bayes classification and why is it called 'Naive'?

**Answer:** Naive Bayes applies Bayes' Theorem: P(y|x) ∝ P(y) prod P(x_i|y). It is 'naive' because it assumes that all predictor features are strictly conditionally independent given the class label: P(x1, x2|y) = P(x1|y) * P(x2|y). Despite this unrealistic assumption, it performs surprisingly well for high-dimensional text classification (spam detection).

### Q18. Explain Laplace Smoothing in Naive Bayes.

**Answer:** If a word or category never appeared in class y in the training data, its conditional probability P(x|y) = 0, which zeroes out the entire multiplicative posterior probability. Laplace smoothing adds a pseudo-count α (typically 1): P(x|y) = (count + α) / (total_count + α * vocabulary_size), preventing zero-probability collapse.

### Q19. How do you extend Binary Classifiers to Multi-Class Classification (One-vs-Rest vs One-vs-One)?

**Answer:** One-vs-Rest (OvR): trains K separate binary models, where model k predicts class k vs all other K-1 classes combined; prediction chooses the class with highest confidence. One-vs-One (OvO): trains K*(K-1)/2 binary models for every unique pair of classes; prediction chooses the class winning the most pairwise majority votes.

### Q20. What is Softmax Regression (Multinomial Logistic Regression)?

**Answer:** Softmax generalizes logistic regression to K > 2 mutually exclusive classes. It maps linear scores z_k into a normalized probability distribution using the softmax function: P(y = k | x) = e^(z_k) / sum_{j=1}^K e^(z_j). It is trained using Categorical Cross-Entropy loss.

### Q21. What are ensemble Stacking (Stacked Generalization) and Blending?

**Answer:** Stacking trains diverse base models (e.g. Random Forest, XGBoost, SVM) on training folds, collects their out-of-fold probability predictions into a new feature matrix, and trains a meta-model (e.g. Logistic Regression) to make final predictions. Blending is similar but uses a simple holdout validation split rather than out-of-fold K-fold cross-validation.

### Q22. What is Cost-Sensitive Learning in classification?

**Answer:** Cost-sensitive learning modifies the loss function by assigning asymmetric penalty costs C(actual, predicted) to different misclassification errors (e.g. classifying a fraudulent transaction as legitimate costs $10,000, while flagging a normal transaction costs $5), forcing the model to minimize financial loss rather than 0-1 error.

### Q23. How does Voting Classifier work (Hard Voting vs Soft Voting)?

**Answer:** Hard Voting aggregates discrete class predictions from individual models and picks the majority class. Soft Voting averages predicted class probabilities across all models and selects the class with the highest average probability. Soft voting gives higher influence to models that are confident in their predictions, usually yielding superior performance.

### Q24. Explain decision tree pruning (Pre-pruning vs Post-pruning / Cost-Complexity Pruning).

**Answer:** Pre-pruning halts tree growth early during training using stopping criteria (max_depth, min_samples_split, min_impurity_decrease). Post-pruning (Cost-Complexity Pruning / Minimal Cost-Complexity Pruning) grows a full unconstrained tree, then prunes back subtrees that minimize cost R_α(T) = R(T) + α |T|, preventing overfitting without heuristic limits.

### Q25. What is Feature Importance in tree models: MDI (Mean Decrease Impurity) vs Permutation Importance?

**Answer:** MDI (Gini importance) sums the impurity reduction achieved by splits on a feature across all trees; it is fast, but severely biased toward high-cardinality numerical features. Permutation Importance shuffles a feature's values in the validation set and measures the drop in model metric (e.g. accuracy/AUC); it is computationally heavier but unbiased and model-agnostic.

### Q26. How does Linear Discriminant Analysis (LDA) differ from Logistic Regression for classification?

**Answer:** Logistic Regression is discriminative: it models P(y|x) directly without assuming feature distributions. LDA is generative: it models P(x|y) assuming features follow multivariate Gaussian distributions with a shared covariance matrix, applying Bayes' rule to derive linear decision boundaries. LDA is optimal when normality assumptions hold.

### Q27. Explain Probability Calibration (Platt Scaling vs Isotonic Regression).

**Answer:** Many models (SVMs, Naive Bayes, boosted trees) produce uncalibrated probabilities (confidence does not reflect true empirical frequency). Platt Scaling fits a logistic regression model on raw model scores. Isotonic Regression fits a non-parametric piecewise constant isotonic function. Calibration ensures that a predicted probability of 0.8 represents an 80% real-world event rate.

### Q28. What is the Brier Score and how does it evaluate probability calibration?

**Answer:** The Brier Score is the mean squared difference between predicted probabilities and true binary labels: BS = (1/N) sum (p_i - y_i)². Lower values indicate better calibrated predictions (0 is perfect; 0.25 is random coin flip). It decomposes into Reliability, Resolution, and Uncertainty.

### Q29. How do decision boundaries differ between Linear Models, Decision Trees, and RBF SVMs?

**Answer:** Linear models create flat hyperplanes. Decision trees create axis-aligned orthogonal step-like rectangular decision regions. RBF SVMs create smooth, non-linear, flexible closed curved contours capable of isolating complex islands of data.

### Q30. What is the difference between Transductive and Inductive learning in classification?

**Answer:** Inductive learning learns a generalized mapping function f(x) from training data that can predict on unseen future instances (e.g. standard supervised ML). Transductive learning predicts labels strictly for a specific, known set of unlabelled test points provided during training (e.g. graph node classification, semi-supervised transductive SVMs) without learning a general function.
