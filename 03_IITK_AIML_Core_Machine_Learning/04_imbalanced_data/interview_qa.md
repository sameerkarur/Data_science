# Interview Q&A — Handling Imbalanced Datasets & Anomaly Detection

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Why is Accuracy a dangerously misleading metric on highly imbalanced datasets?

**Answer:** In a dataset with 99% negative (class 0) and 1% positive (class 1) instances, a trivial dummy model that predicts class 0 for every instance achieves 99% accuracy while having a Recall of 0% on the minority class of interest, failing to detect fraud, disease, or system failures.

### Q2. Explain the SMOTE (Synthetic Minority Over-sampling Technique) algorithm.

**Answer:** For each minority class sample x, SMOTE computes its k-nearest neighbors among other minority samples, randomly chooses one neighbor x_nn, and creates a synthetic instance along the line segment connecting them: x_syn = x + λ * (x_nn - x), where λ ~ Uniform(0, 1). This expands the minority decision region without exact duplication.

### Q3. What are the primary risks and limitations of standard SMOTE?

**Answer:** SMOTE blindly interpolates between minority samples: if a minority sample is an outlier inside the majority class distribution, SMOTE generates synthetic points directly inside majority clusters, aggravating class overlap and increasing false positive rates.

### Q4. Explain Borderline-SMOTE and ADASYN.

**Answer:** Borderline-SMOTE applies oversampling ONLY to minority instances near the decision boundary (whose k-nearest neighbors are heavily majority class), ignoring interior points. ADASYN (Adaptive Synthetic) generates synthetic points proportional to the difficulty level: minority samples with more majority neighbors receive higher synthetic point generation weights.

### Q5. How does Random Under-Sampling differ from Tomek Links?

**Answer:** Random Under-Sampling randomly discards majority class instances, which speeds up training but discards potentially valuable informative data. Tomek Links identifies pairs of minimally distanced nearest neighbors of opposite classes (x_min, x_maj); removing the majority member of the pair cleans the decision boundary and sharpens separation.

### Q6. What is Edited Nearest Neighbors (ENN) and SMOTEENN?

**Answer:** ENN removes majority samples whose class label differs from the majority of its k-nearest neighbors. SMOTEENN combines SMOTE (oversampling minority points) followed by ENN (pruning noisy and overlapping samples from both classes), producing clean, well-defined decision boundaries.

### Q7. Explain Cost-Sensitive Learning and Class Weights in 'class_weight="balanced"'.

**Answer:** Cost-sensitive learning modifies the loss function by scaling the penalty for minority class misclassifications: w_j = N / (n_classes * N_j). In binary cross-entropy, misclassifying a minority instance incurs a proportionally larger gradient update, forcing the model to shift its decision boundary toward the majority class.

### Q8. What is Focal Loss and how does it address extreme class imbalance?

**Answer:** Focal Loss FL(p_t) = - α_t (1 - p_t)^γ ln(p_t). The modulating factor (1 - p_t)^γ down-weights the loss contribution of easy, well-classified examples (where p_t is large), focusing model gradients on hard, ambiguous, and rare instances (popularized in RetinaNet).

### Q9. Why is the Precision-Recall (PR) Curve preferred over the ROC Curve for severe imbalance?

**Answer:** The False Positive Rate in ROC includes True Negatives in its denominator: FPR = FP / (TN + FP). When the majority class (TN) is vast (e.g. millions), huge increases in false positives (FP) barely change FPR, making the ROC curve look deceptively outstanding. The PR curve plots Precision (TP / (TP + FP)) vs Recall without TNs, exposing drops in precision immediately.

### Q10. Explain Threshold Tuning (Threshold Moving) for imbalanced classification.

**Answer:** By default, classifiers predict class 1 if P(y=1|x) >= 0.5. On imbalanced data, 0.5 is arbitrary. Threshold tuning sweeps decision thresholds from 0 to 1, evaluating business metrics (F1-score, F_beta, or total financial cost) on a validation set to select the optimal operating threshold (e.g. threshold = 0.15).

### Q11. What is the F-beta score and when should you choose β = 2 vs β = 0.5?

**Answer:** F_beta = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall). β controls the relative weight of Recall vs Precision. When β = 2, Recall is weighted twice as heavily as Precision (ideal for cancer screening or fraud detection where false negatives are disastrous). When β = 0.5, Precision is weighted twice as heavily (ideal for spam filtering).

### Q12. What is Balanced Random Forest (EasyEnsemble)?

**Answer:** Balanced Random Forest draws a bootstrap sample from the minority class and randomly under-samples the majority class to match the minority size for each individual tree in the ensemble, ensuring every tree is trained on a balanced 50:50 distribution without discarding majority information globally.

### Q13. Explain the dangers of applying SMOTE before Train-Test splitting (Data Leakage).

**Answer:** If SMOTE is applied to the entire dataset before splitting, synthetic instances in the test set will be linear interpolations of training points, and training instances will be interpolations of test points. This causes catastrophic data leakage, producing wildly over-optimistic test evaluation metrics.

### Q14. What is Stratified K-Fold Cross-Validation and why is it mandatory for imbalanced data?

**Answer:** Standard K-Fold randomly splits data, which can result in some folds receiving zero minority instances. Stratified K-Fold partitions data such that each fold contains the exact same percentage of each class as the complete dataset, ensuring stable evaluation across all splits.

### Q15. How does One-Class Classification (e.g. One-Class SVM) handle extreme 1:1,000,000 imbalance?

**Answer:** Instead of learning a boundary between two classes, One-Class SVM fits a tight hypersphere or support vector envelope around the abundant majority class (normal data). Any future instance that falls outside this learned boundary is classified as an anomaly/minority instance.

### Q16. What is the Geometric Mean (G-Mean) metric in imbalanced evaluation?

**Answer:** G-Mean = sqrt(Sensitivity * Specificity). It measures the balance between classification performances on both majority and minority classes simultaneously; a low score on either class penalizes the overall G-Mean.

### Q17. Explain Balanced Accuracy.

**Answer:** Balanced Accuracy is the unweighted average of Recall obtained on each class: Balanced Accuracy = 0.5 * (TPR + TNR). For a dummy classifier predicting all negatives on a 99:1 dataset, standard accuracy is 99%, but balanced accuracy correctly reports 50% (random guess).

### Q18. How do Ensemble Methods with Downsampling (RUSBoost) work?

**Answer:** RUSBoost integrates Random Under-Sampling (RUS) into AdaBoost: at each boosting iteration, it randomly downsamples the majority class to match minority class weights before fitting the weak learner, combining boosting performance with the speed of undersampling.

### Q19. What is the difference between Over-sampling with replacement vs Synthetic Over-sampling?

**Answer:** Over-sampling with replacement simply duplicates existing minority instances, which increases the density of specific points and causes tree models to learn overly specific rules (overfitting). Synthetic over-sampling (SMOTE) generates novel points along feature vectors, expanding the decision space.

### Q20. What is Class Overlap and how does it exacerbate class imbalance?

**Answer:** Class overlap occurs when feature distributions of minority and majority classes overlap significantly in feature space. In imbalanced scenarios, majority samples vastly outnumber minority samples in the overlapping region, causing models to predict the majority class for all points in that region.

### Q21. How does cost matrix formulation optimize commercial fraud detection?

**Answer:** Define cost matrix: Cost(FP) = customer friction ($5); Cost(FN) = transaction value lost ($2,500); Cost(TP) = investigation fee ($10); Cost(TN) = $0. Total Expected Cost is minimized by picking classification threshold θ* = Cost(FP) / (Cost(FP) + Cost(FN)).

### Q22. Explain how Autoencoders are used for extreme class imbalance.

**Answer:** Train an Autoencoder neural network exclusively on normal majority class samples to minimize reconstruction error (MSE). When an anomalous minority instance is fed to the network, the autoencoder fails to reconstruct it accurately, resulting in a high reconstruction error that triggers anomaly detection.

### Q23. What is NearMiss undersampling and what are its three variants?

**Answer:** NearMiss-1: selects majority instances with smallest average distance to 3 closest minority instances. NearMiss-2: selects majority instances with smallest average distance to 3 furthest minority instances. NearMiss-3: selects a fixed number of closest majority instances for each minority instance.

### Q24. How does temperature scaling or label smoothing interact with imbalanced datasets?

**Answer:** Label smoothing replaces hard 0/1 targets with soft targets (e.g. ε and 1 - ε), preventing overconfidence. On imbalanced data, uniform label smoothing can disproportionately penalize the minority class; class-aware label smoothing adjusts smoothing proportional to class frequencies.

### Q25. Explain the role of probability calibration when using under-sampling.

**Answer:** Under-sampling shifts the empirical class distribution (e.g. from 1% positive to 50% positive). As a result, model output probabilities are severely shifted toward 0.5. To recover true population probabilities, apply Bayes' recalibration: p_true = (p_pred * r) / (p_pred * r + (1 - p_pred)) where r is the sampling ratio.

### Q26. What is Two-Phase Learning for imbalanced neural networks?

**Answer:** Phase 1: train the neural network on an artificially balanced dataset (via oversampling/undersampling) to learn strong, generalizable feature representations. Phase 2: fine-tune the final classification layer on the true imbalanced dataset to properly calibrate decision thresholds.

### Q27. How do you evaluate multi-class imbalanced classification?

**Answer:** Use Macro-Averaged F1-score (unweighted average of F1 across all classes, treating rare classes equally) or Balanced Multi-Class Log-Loss. Avoid Micro-Averaged metrics, which are dominated by performance on the majority class.

### Q28. What is Neighborhood Cleaning Rule (NCR)?

**Answer:** NCR is an undersampling technique that combines Edited Nearest Neighbors (ENN) to clean ambiguous majority instances with a secondary rule: if a minority instance is misclassified by its 3 nearest neighbors, the majority neighbors are removed.

### Q29. How does Metric Learning (Triplet Loss / Contrastive Loss) assist imbalanced classification?

**Answer:** Metric learning trains neural networks to project samples into an embedding space where instances of the same class are pulled close together and instances of different classes are pushed apart by a margin, creating clustered representations that separate minority points from majority points.

### Q30. What are the key production monitoring alerts for models operating on imbalanced data?

**Answer:** Monitor: (1) Prediction Distribution Drift (alert if predicted positive rate shifts from historical baseline), (2) Feature Drift (KS-test / PSI on input features), (3) Operational False Positive Rate, and (4) Delayed True Positive Feedback Loops.
