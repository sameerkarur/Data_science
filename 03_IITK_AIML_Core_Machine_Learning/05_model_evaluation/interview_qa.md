# Interview Q&A — Model Evaluation & Cross-Validation Strategies

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the Confusion Matrix and define TP, FP, TN, and FN.

**Answer:** A Confusion Matrix cross-tabulates actual vs predicted classes: True Positive (TP): correctly predicted positive. False Positive (FP / Type I error): negative incorrectly predicted as positive. True Negative (TN): correctly predicted negative. False Negative (FN / Type II error): positive incorrectly predicted as negative.

### Q2. Define Precision, Recall, Specificity, and Fall-out mathematically.

**Answer:** Precision = TP / (TP + FP) (purity of positive predictions). Recall (Sensitivity / TPR) = TP / (TP + FN) (proportion of actual positives identified). Specificity (TNR) = TN / (TN + FP) (proportion of actual negatives identified). Fall-out (FPR) = FP / (TN + FP) = 1 - Specificity.

### Q3. What is the mathematical formulation of F1-Score and why does it use Harmonic Mean?

**Answer:** F1 = 2 * (Precision * Recall) / (Precision + Recall). It uses the Harmonic Mean rather than the Arithmetic Mean because the harmonic mean approaches the smaller of the two numbers. If either Precision or Recall is near zero, F1 drops close to zero, preventing models with 100% recall but 1% precision from scoring well.

### Q4. Differentiate between Micro-average, Macro-average, and Weighted-average in multi-class metrics.

**Answer:** Macro-average: calculates metric independently for each class and computes unweighted mean; treats all classes equally (ideal for spotting issues in rare classes). Micro-average: pools all TPs, FPs, and FNs globally across all classes; dominated by majority classes. Weighted-average: averages class metrics weighted by the support (frequency) of each class.

### Q5. Explain the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC).

**Answer:** The ROC curve plots TPR (Recall) on the y-axis against FPR (1 - Specificity) on the x-axis across all decision thresholds from 1.0 to 0.0. The diagonal line represents random guessing (AUC = 0.5). AUC measures the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance (AUC = 1.0 is perfect ranking).

### Q6. What are the mathematical properties of ROC-AUC and why is it threshold-independent?

**Answer:** ROC-AUC evaluates ranking quality across ALL possible decision thresholds simultaneously. It is scale-invariant (evaluates rank order rather than absolute probability values) and classification-threshold-invariant.

### Q7. When does ROC-AUC give an overly optimistic view compared to PR-AUC?

**Answer:** On datasets with extreme class imbalance (e.g. 99.9% negative), the vast number of True Negatives suppresses the False Positive Rate (FPR = FP / (TN + FP)), keeping FPR tiny even when thousands of false positives occur. PR-AUC excludes TNs, revealing drops in precision immediately.

### Q8. What is Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE)?

**Answer:** MSE = (1/N) sum (y_i - ŷ_i)²; penalizes large errors heavily due to squaring. RMSE = sqrt(MSE); in the same units as target variable. MAE = (1/N) sum |y_i - ŷ_i|; robust to outliers and represents the median regression error.

### Q9. When should you choose Mean Absolute Percentage Error (MAPE) and when does it fail?

**Answer:** MAPE = (100%/N) sum |(y_i - ŷ_i) / y_i| expresses error as a percentage of actual values, making it easy to communicate to business executives. It fails completely when actual values y_i = 0 (division by zero) and penalizes positive errors differently from negative errors.

### Q10. Explain Mean Squared Logarithmic Error (MSLE) and why it is used in pricing models.

**Answer:** MSLE = (1/N) sum (ln(1 + y_i) - ln(1 + ŷ_i))². It evaluates relative percentage differences rather than absolute values and penalizes under-estimates more heavily than over-estimates, making it ideal for real-estate or product price predictions.

### Q11. What is R-squared (Coefficient of Determination) and can it be negative?

**Answer:** R² = 1 - (SS_res / SS_tot) = 1 - sum(y_i - ŷ_i)² / sum(y_i - ȳ)². It measures the proportion of variance in y explained by the model compared to a naive mean baseline. Yes, R² can be negative if the model's predictions are worse than simply predicting the mean of the target variable on out-of-sample test data.

### Q12. Explain Adjusted R-squared and its degrees of freedom penalty.

**Answer:** Standard R² increases mechanically with every additional feature added to the model. Adjusted R² = 1 - [(1 - R²)(N - 1) / (N - p - 1)], penalizing the addition of extraneous features p unless they reduce residual error sufficiently to justify the lost degree of freedom.

### Q13. What is K-Fold Cross-Validation and what is the optimal choice of K?

**Answer:** K-Fold partitions data into K equal folds, training on K-1 folds and validating on the remaining fold, rotating K times. Standard choice is K = 5 or K = 10, balancing empirical bias (lower at K=10) with variance and computational expense.

### Q14. Explain Leave-One-Out Cross-Validation (LOOCV) and its trade-offs.

**Answer:** LOOCV is K-Fold where K = N (trains on N-1 samples, tests on 1, repeated N times). It provides an almost unbiased estimate of true performance, but has high computational cost (O(N) model trainings) and high variance because the N training sets are nearly identical.

### Q15. What is TimeSeriesSplit and why must regular K-Fold NEVER be used on temporal data?

**Answer:** Regular K-Fold randomly shuffles data, training on future observations to predict past observations, causing catastrophic temporal look-ahead leakage. TimeSeriesSplit uses expanding rolling windows (walk-forward validation): trains strictly on past data [1 to t] and tests on future window [t+1 to t+k], preserving temporal causality.

### Q16. Explain GroupKFold and when it is required.

**Answer:** GroupKFold ensures that all samples originating from the same group (e.g. patient ID, user ID, household ID) are assigned entirely to either the training set or the validation set, never split across both. It prevents data leakage when multiple correlated records belong to the same entity.

### Q17. What is Stratified K-Fold and why is it standard for classification?

**Answer:** Stratified K-Fold preserves the percentage of samples for each target class within every individual fold to match the complete dataset distribution, preventing folds from suffering from class starvation.

### Q18. Explain the Bias-Variance Tradeoff mathematically.

**Answer:** Expected Prediction Error = Bias(ŷ)² + Var(ŷ) + σ² (Irreducible Noise). Bias measures error from erroneous assumptions in the learning algorithm (underfitting). Variance measures sensitivity to small fluctuations in the training set (overfitting). As model complexity increases, bias decreases while variance increases.

### Q19. How do Learning Curves diagnose High Bias (Underfitting) vs High Variance (Overfitting)?

**Answer:** High Bias: training score and validation score converge quickly to a low, unacceptable plateau; adding more training data does not improve performance. High Variance: large gap between high training score and substantially lower validation score; adding more training data helps close the generalization gap.

### Q20. What is a Calibration Curve (Reliability Diagram)?

**Answer:** A calibration curve bins predicted probabilities (e.g. [0-0.1, 0.1-0.2...]) on the x-axis and plots the true fraction of positive outcomes on the y-axis. A perfectly calibrated model follows the 45-degree diagonal line y = x. S-shaped curves indicate under-confidence; inverted S-curves indicate over-confidence.

### Q21. What is the Cumulative Gains Chart and Lift Chart?

**Answer:** Cumulative Gains plots the percentage of total positive targets identified on the y-axis against the percentage of population contacted on the x-axis (sorted by predicted probability). The Lift Chart plots the ratio of model gain to random baseline gain (Lift = Gain / Random). High initial lift (e.g. 5x lift in top decile) proves model efficiency in marketing campaigns.

### Q22. Explain Normalized Discounted Cumulative Gain (NDCG) in ranking and recommendation systems.

**Answer:** DCG = sum_{i=1}^p (2^(rel_i) - 1) / log2(i + 1). NDCG = DCG / IDCG, where IDCG is the Ideal DCG achieved by perfect ranking. It rewards relevant items appearing at top ranks and heavily penalizes relevant items pushed down the ranked list, scaled between 0 and 1.

### Q23. What is Mean Reciprocal Rank (MRR)?

**Answer:** MRR = (1/|Q|) sum_{i=1}^{|Q|} (1 / rank_i), where rank_i is the position of the FIRST relevant item found for query i. It evaluates systems where only the top successful retrieval matters (e.g. search engines, QA systems).

### Q24. Explain Top-K Accuracy (Hit Rate@K).

**Answer:** Top-K Accuracy measures the proportion of test instances where the true target label is included within the model's top K highest-confidence predicted classes (common in ImageNet top-5 evaluation and recommender candidate generation).

### Q25. What is Cohen's Kappa statistic?

**Answer:** Kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is expected agreement by chance. It evaluates inter-rater agreement or classification performance adjusted for chance agreement, bounded in [-1, 1].

### Q26. Explain Matthew's Correlation Coefficient (MCC).

**Answer:** MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). It evaluates binary classification using all four confusion matrix quadrants, producing values in [-1, 1]. It remains reliable even on heavily imbalanced datasets where F1 and accuracy fail.

### Q27. What is Population Stability Index (PSI) and how does it detect production model drift?

**Answer:** PSI = sum ((Actual% - Expected%) * ln(Actual% / Expected%)). It measures shift in feature or prediction distributions between training (expected) and production (actual). PSI < 0.1 indicates stability; 0.1 <= PSI <= 0.2 indicates moderate shift; PSI > 0.2 signals significant distribution drift requiring retraining.

### Q28. Explain Cross-Entropy (Log-Loss) as an evaluation metric.

**Answer:** Log-Loss = - (1/N) sum [y_i ln(p_i) + (1 - y_i) ln(1 - p_i)]. Unlike accuracy or F1 which evaluate hard binary decisions, Log-Loss penalizes overconfident wrong predictions exponentially (e.g. predicting p = 0.99 when true y = 0 incurs huge loss), measuring true probabilistic confidence.

### Q29. What is the difference between Data Drift and Concept Drift in evaluation pipelines?

**Answer:** Data drift (covariate shift) occurs when the input distribution P(X) changes over time while P(Y|X) remains unchanged. Concept drift occurs when the conditional relationship P(Y|X) changes (the underlying definition of the target changes), causing a model to degrade even if input distributions appear stable.

### Q30. What are Golden Evaluation Sets and Shadow Deployments in enterprise production ML?

**Answer:** A Golden Set is a curated, human-verified, static benchmark test dataset representing critical edge cases and core business logic that every candidate model must pass before release. A Shadow Deployment runs the new model in production alongside the active legacy model on live traffic without serving predictions to users, comparing latency, throughput, and predictions in real time.
