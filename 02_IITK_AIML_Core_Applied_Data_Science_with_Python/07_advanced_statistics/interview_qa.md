# Interview Q&A — Advanced Hypothesis Testing & Statistical Inference

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the Null Hypothesis (H0) and Alternative Hypothesis (H1).

**Answer:** The Null Hypothesis (H0) assumes no effect, no difference, or no relationship exists (default status quo). The Alternative Hypothesis (H1 or Ha) contradicts H0, asserting that an observed difference is real, statistically significant, or driven by an underlying effect.

### Q2. Differentiate between Type I Error (α) and Type II Error (β).

**Answer:** Type I error (α) is a False Positive: rejecting a true null hypothesis (e.g. concluding a drug is effective when it is not). Type II error (β) is a False Negative: failing to reject a false null hypothesis (e.g. concluding a drug is ineffective when it actually works). Power = 1 - β.

### Q3. Define the p-value rigorously and refute common misinterpretations.

**Answer:** The p-value is the probability, assuming the null hypothesis is true, of observing a test statistic as extreme as, or more extreme than, the observed data. It is NOT the probability that the null hypothesis is true, nor is it the probability that results were produced by random chance alone.

### Q4. What is significance level (alpha) and how is it chosen?

**Answer:** Alpha (α) is the maximum acceptable threshold for committing a Type I error (commonly set to 0.05 or 0.01). If p-value <= α, we reject H0. In high-consequence applications (clinical trials, fraud detection), α is set lower (e.g. 0.001) to protect against false positives.

### Q5. Explain the difference between a One-Tailed test and a Two-Tailed test.

**Answer:** A two-tailed test evaluates for any directional difference (H1: μ != μ0), dividing α equally across both tails (α/2 each). A one-tailed test tests for an effect in a single specific direction (H1: μ > μ0 or μ < μ0), concentrating the entire α in that tail and providing higher statistical power.

### Q6. When should you use a Student's t-test vs a Z-test?

**Answer:** Use a Z-test when the population standard deviation σ is known and sample size N is large (N >= 30). Use a Student's t-test when the population standard deviation σ is unknown and must be estimated from the sample standard deviation s, especially for small samples (N < 30).

### Q7. Differentiate between One-Sample, Two-Sample Independent, and Paired t-tests.

**Answer:** One-Sample: compares sample mean to a known benchmark value. Two-Sample Independent: compares means of two unrelated groups (e.g. control vs treatment group in A/B test). Paired t-test: compares means from the same subjects measured twice (e.g. blood pressure before and after medication), controlling for subject-specific baseline variability.

### Q8. What are the core assumptions of an independent two-sample t-test?

**Answer:** Assumptions: (1) Independence of observations, (2) Continuous outcome variable, (3) Normal distribution of data within each group (or N >= 30 by CLT), and (4) Homogeneity of variance (homoscedasticity across groups; if violated, use Welch's t-test).

### Q9. Explain Welch's t-test and why it is preferred over Student's t-test in production.

**Answer:** Welch's t-test does not assume equal population variances between the two groups. It adjusts degrees of freedom using the Welch-Satterthwaite equation, maintaining robust Type I error rates even when sample sizes and variances differ greatly.

### Q10. What is ANOVA (Analysis of Variance) and why shouldn't you run multiple t-tests instead?

**Answer:** ANOVA tests whether the means of three or more groups are statistically different by comparing between-group variance to within-group variance. Running multiple pairwise t-tests causes Family-Wise Error Rate inflation: for 5 groups (10 pairwise tests at α = 0.05), overall false positive risk jumps to 1 - (1 - 0.05)^10 ≈ 40%.

### Q11. How does One-Way ANOVA calculate the F-statistic?

**Answer:** F = MS_between / MS_within, where MS_between = SS_between / (k - 1) and MS_within = SS_within / (N - k). If group means differ significantly, MS_between exceeds random within-group variance MS_within, producing a large F-ratio (F >> 1) and small p-value.

### Q12. What is a Post-Hoc test (e.g. Tukey's HSD) following ANOVA?

**Answer:** If ANOVA rejects H0, it proves that at least two group means differ, but does not identify which specific pairs differ. Post-hoc tests like Tukey's Honestly Significant Difference (HSD) test all pairwise comparisons while controlling the Family-Wise Error Rate.

### Q13. Explain Two-Way ANOVA and interaction effects.

**Answer:** Two-Way ANOVA evaluates the impact of two independent categorical factors on a continuous outcome simultaneously. An interaction effect occurs when the effect of Factor A depends on the level of Factor B (e.g. medicine dosage effectiveness depends on patient age group).

### Q14. What is the Chi-Square (χ²) Test of Independence?

**Answer:** The Chi-Square test evaluates whether two categorical variables are statistically independent. It computes the test statistic χ² = sum((O - E)² / E), where O is observed cell frequency and E is expected frequency under the assumption of independence. df = (rows - 1) * (cols - 1).

### Q15. Explain the Chi-Square Goodness-of-Fit test.

**Answer:** Goodness-of-fit evaluates whether an observed categorical sample distribution matches a hypothesized theoretical distribution (e.g. testing whether dice rolls follow a uniform distribution or gene frequencies match Mendelian ratios).

### Q16. What is Fisher's Exact Test and when is it preferred over Chi-Square?

**Answer:** Fisher's Exact Test calculates exact hypergeometric probabilities for 2x2 contingency tables. It is preferred when sample sizes are small or expected frequencies in any cell drop below 5, where Chi-Square large-sample approximations break down.

### Q17. What are non-parametric alternatives to the t-test and ANOVA?

**Answer:** For independent two-sample t-test: Mann-Whitney U test (Wilcoxon rank-sum). For paired t-test: Wilcoxon signed-rank test. For One-Way ANOVA: Kruskal-Wallis test. These operate on ranked data and do not require normality.

### Q18. Explain multiple testing corrections: Bonferroni vs False Discovery Rate (Benjamini-Hochberg).

**Answer:** Bonferroni controls Family-Wise Error Rate (FWER) by setting α_new = α / m (where m is the number of hypotheses), but it is overly conservative and inflates Type II errors. Benjamini-Hochberg controls the False Discovery Rate (FDR = expected proportion of false positives among rejected hypotheses), providing significantly higher statistical power for high-throughput testing (e.g. thousands of A/B test metrics or genomic loci).

### Q19. What is Cohen's d and why is Effect Size important alongside p-values?

**Answer:** Cohen's d = (μ1 - μ2) / s_pooled measures the standardized magnitude of the difference between groups. With massive datasets (N = 1,000,000), trivial, meaningless differences achieve tiny p-values (p < 0.0001); effect size quantifies whether the difference has practical, real-world business significance.

### Q20. Explain statistical Power Analysis and how to determine required sample size for an A/B test.

**Answer:** Power analysis relates four variables: Alpha (α), Power (1 - β), Effect Size (minimum detectable effect MDE), and Sample Size (N). Given α = 0.05, power = 0.80, and target MDE, power formulas determine the exact minimum sample size required before running an experiment to avoid underpowered tests.

### Q21. What is Sample Ratio Mismatch (SRM) in A/B testing and why is it dangerous?

**Answer:** SRM occurs when the observed ratio of visitors in Treatment vs Control deviates significantly from the intended design (e.g. expecting 50:50, but observing 52:48 with χ² p < 0.001). SRM indicates severe underlying execution bugs (broken redirect bots, user filtering, browser caching) that invalidate all experimental conclusions.

### Q22. What is the difference between Fixed Effects and Random Effects models?

**Answer:** Fixed effects models assume observed levels of a factor are fixed and of direct interest, estimating specific parameters for each group. Random effects models treat levels as random draws from a broader population distribution, modeling variance components (common in hierarchical/multilevel modeling).

### Q23. Explain the difference between R-squared and Adjusted R-squared in regression.

**Answer:** R² measures the proportion of variance in the dependent variable explained by independent features. However, R² mechanically increases with every added feature, even random noise. Adjusted R² penalizes the addition of non-informative predictors: Adj R² = 1 - [(1 - R²)(N - 1) / (N - p - 1)].

### Q24. What is the Durbin-Watson statistic and what does it detect?

**Answer:** The Durbin-Watson statistic tests for first-order autocorrelation in regression residuals (common in time-series data). Values range from 0 to 4: a value around 2 indicates zero autocorrelation, values < 2 indicate positive autocorrelation, and values > 2 indicate negative autocorrelation.

### Q25. Explain Cook's Distance and Leverage in regression diagnostics.

**Answer:** Leverage measures how far an observation's feature values are from the mean of all predictors. Cook's Distance measures the overall influence of an observation by calculating how much all fitted values change when that observation is omitted. Points with Cook's distance > 1 or 4/N exert outsized leverage and skew model coefficients.

### Q26. What is Logistic Regression and why is the Logit (log-odds) link function used?

**Answer:** Logistic regression models binary outcomes P(y = 1|x) by mapping linear combinations of features w^T x to probabilities using the sigmoid function σ(z) = 1 / (1 + e^(-z)). The logit link function log(p / (1 - p)) = w^T x transforms probabilities bounded in [0, 1] to unbounded log-odds in (-inf, +inf), making linear modeling mathematically valid.

### Q27. How do you interpret coefficients in Logistic Regression?

**Answer:** For a continuous feature x, a one-unit increase multiplies the odds of the positive outcome by e^(w_i) (the Odds Ratio). If w = 0.693, e^(0.693) ≈ 2.0, meaning odds of the outcome double for every unit increase in x.

### Q28. What is the AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)?

**Answer:** AIC = 2k - 2ln(L) and BIC = k ln(N) - 2ln(L) evaluate model quality by balancing goodness of fit (log-likelihood L) with model complexity (number of parameters k). Lower values indicate better models. BIC penalizes extra parameters more heavily than AIC as sample size N grows.

### Q29. Explain the Kolmogorov-Smirnov (K-S) test.

**Answer:** The K-S test evaluates whether a sample distribution matches a reference distribution (one-sample) or whether two independent sample distributions differ (two-sample). The test statistic D is the maximum vertical distance between the empirical cumulative distribution functions (ECDFs).

### Q30. What is Survival Analysis and how does the Kaplan-Meier estimator handle censored data?

**Answer:** Survival analysis models the expected duration of time until an event occurs (e.g. customer churn, machine failure). Right-censored data occurs when subjects leave the study without experiencing the event. The Kaplan-Meier estimator calculates step-wise survival probabilities at each event time, correctly factoring in censored subjects up to their last observation.
