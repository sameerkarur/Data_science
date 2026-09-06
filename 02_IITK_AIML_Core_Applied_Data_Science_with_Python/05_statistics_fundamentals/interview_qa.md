# Interview Q&A — Statistics Fundamentals & EDA

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the difference between Mean, Median, and Mode in skewed distributions.

**Answer:** The Mean is the arithmetic average, sensitive to extreme outliers. The Median is the middle value of sorted data, robust to outliers. The Mode is the most frequent value. In a right-skewed (positively skewed) distribution: Mode < Median < Mean. In a left-skewed (negatively skewed) distribution: Mean < Median < Mode.

### Q2. Why is sample variance divided by (N - 1) instead of N (Bessel's Correction)?

**Answer:** Dividing by N systematically underestimates the true population variance because sample deviations are measured around the sample mean x̄ rather than the true population mean μ. Bessel's correction (dividing by N - 1) provides an unbiased estimator: E[s²] = σ².

### Q3. State the Central Limit Theorem (CLT) and its practical significance in machine learning.

**Answer:** The CLT states that the sampling distribution of the sample mean approaches a normal distribution as the sample size N grows large (typically N >= 30), regardless of the shape of the underlying population distribution, provided the data has finite variance. This justifies using normal-theory confidence intervals and hypothesis tests on non-normal real-world data.

### Q4. What is Chebyshev's Inequality and when is it applied?

**Answer:** Chebyshev's inequality guarantees that for any probability distribution with finite variance, at least 1 - (1/k²) of values fall within k standard deviations of the mean (e.g. at least 75% within 2σ, at least 88.9% within 3σ). It applies universally without requiring normality assumptions.

### Q5. Explain Skewness and Kurtosis.

**Answer:** Skewness measures asymmetry around the mean (0 for symmetric, >0 for right-skewed, <0 for left-skewed). Kurtosis measures tail heaviness relative to a normal distribution: Mesokurtic (kurtosis ≈ 3, normal), Leptokurtic (>3, heavy tails, high outlier risk), Platykurtic (<3, light tails).

### Q6. Differentiate between Pearson Correlation and Spearman Rank Correlation.

**Answer:** Pearson correlation evaluates linear relationships between continuous, normally distributed variables: r = Cov(X, Y) / (σ_X σ_Y). Spearman rank correlation evaluates monotonic relationships by computing Pearson on ranked values; it does not assume normality and is robust to extreme outliers.

### Q7. What is the difference between Covariance and Correlation?

**Answer:** Covariance measures the joint directional variability of two variables: Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)], but its magnitude depends on the units of measurement. Correlation is covariance standardized by the product of standard deviations: r = Cov(X,Y)/(σ_X σ_Y), producing a dimensionless metric strictly bounded in [-1, 1].

### Q8. What is the 68-95-99.7 Rule (Empirical Rule) for Normal distributions?

**Answer:** For data that is normally distributed: ~68.27% of observations fall within μ ± 1σ, ~95.45% fall within μ ± 2σ, and ~99.73% fall within μ ± 3σ. Any point beyond 3σ is typically flagged as a potential statistical outlier.

### Q9. What is a Z-score and how is it calculated?

**Answer:** A Z-score standardizes an individual observation by measuring how many standard deviations it lies from the mean: Z = (x - μ) / σ. It transforms the distribution to have a mean of 0 and standard deviation of 1.

### Q10. Explain the Interquartile Range (IQR) and the 1.5 * IQR outlier rule.

**Answer:** IQR = Q3 (75th percentile) - Q1 (25th percentile). Tukey's fence defines outliers as points falling below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR. Because quartiles are rank-based, IQR is far more robust to extreme anomalies than mean/std.

### Q11. What is the difference between standard error (SE) and standard deviation (SD)?

**Answer:** Standard deviation (SD) measures the dispersion of individual data points around the mean in a single dataset. Standard error (SE = SD / sqrt(N)) measures the precision and dispersion of the sample mean estimate across repeated random samples from the population.

### Q12. What is a Confidence Interval and how do you interpret a 95% Confidence Interval?

**Answer:** A 95% Confidence Interval [L, U] means that if we repeated the experiment and sampling procedure infinitely many times, 95% of the calculated intervals would contain the true, fixed population parameter. It does NOT mean there is a 95% probability the parameter lies within that specific realized interval.

### Q13. Explain the concept of Degrees of Freedom in statistics.

**Answer:** Degrees of freedom (df) is the number of independent values that can vary in an analysis without violating statistical constraints. In estimating variance of N samples, calculating the sample mean imposes 1 constraint, leaving df = N - 1.

### Q14. What is a Box Plot (Box-and-Whisker Plot) and what five statistics does it display?

**Answer:** A Box Plot summarizes distributions via the 'Five-Number Summary': Minimum (Q1 - 1.5*IQR), First Quartile (Q1), Median (Q2), Third Quartile (Q3), and Maximum (Q3 + 1.5*IQR). Individual points beyond the whiskers are plotted as individual outlier points.

### Q15. Explain the Law of Large Numbers (LLN).

**Answer:** The Weak and Strong Laws of Large Numbers state that as the number of identically distributed, randomly generated trials N increases, the sample average x̄ converges deterministically to the expected value μ: lim_{N->inf} P(|x̄ - μ| > ε) = 0.

### Q16. What is Quantile-Quantile (Q-Q) Plot and how is it interpreted?

**Answer:** A Q-Q plot plots the quantiles of sample data against theoretical quantiles of a target distribution (typically standard normal). If the data follows the distribution, points form a straight diagonal line. S-shaped curves indicate heavy or light tails; convex/concave curves indicate skewness.

### Q17. What is the difference between point estimation and interval estimation?

**Answer:** A point estimate provides a single best numerical value for a population parameter (e.g. sample mean x̄ = 42.5). An interval estimate specifies a range of plausible values accompanied by a specified degree of confidence (e.g. [40.2, 44.8] with 95% confidence).

### Q18. What is an Unbiased Estimator?

**Answer:** An estimator θ̂ is unbiased if its mathematical expected value equals the true population parameter: E[θ̂] = θ across all possible random samples.

### Q19. Explain the Trimmed Mean and Winsorized Mean.

**Answer:** A Trimmed Mean discards a fixed percentage of extreme values from both tails before computing the mean. A Winsorized Mean replaces extreme values with the nearest retained boundary value (e.g. setting all values above the 95th percentile to the 95th percentile value) before computing the mean.

### Q20. What is Variance Inflation Factor (VIF) and how does it detect multicollinearity?

**Answer:** VIF measures how much the variance of an estimated regression coefficient is inflated due to collinearity with other predictors: VIF_j = 1 / (1 - R_j²), where R_j² is the R² from regressing feature j on all other features. VIF > 5 or 10 indicates severe multicollinearity.

### Q21. Explain the coefficient of variation (CV) and when to use it.

**Answer:** The coefficient of variation CV = (σ / μ) * 100% is a normalized measure of relative dispersion. It enables comparing variability between datasets with completely different units (e.g. comparing volatility of stock prices in USD vs sales volume in units).

### Q22. What is a Bimodal Distribution and what does it typically signify in business data?

**Answer:** A bimodal distribution has two distinct peaks, typically signifying that the dataset is an unsegregated mixture of two distinct sub-populations (e.g. customer heights mixing adult males and females, or purchase amounts mixing retail and wholesale buyers).

### Q23. What is the Geometric Mean and when is it preferred over Arithmetic Mean?

**Answer:** The Geometric Mean (nth root of product of n numbers) is preferred when analyzing multiplicative growth rates, investment returns, compounding ratios, and data spanning multiple orders of magnitude.

### Q24. What is the Harmonic Mean and why is it used for F1-Score?

**Answer:** The Harmonic Mean H = 2 / (1/x + 1/y) gives higher weight to smaller values. In classification, F1-Score uses the harmonic mean of Precision and Recall so that a model cannot achieve a high score if either Precision or Recall is terribly low.

### Q25. Explain the difference between parametric and non-parametric statistical methods.

**Answer:** Parametric methods assume the data follows a specific theoretical distribution (like Normal) with fixed parameters (mean, variance). Non-parametric methods make no assumptions about probability distributions, operating on ranks or empirical counts (e.g. Mann-Whitney U, Kruskal-Wallis).

### Q26. What is the difference between pooled variance and unpooled variance?

**Answer:** Pooled variance combines the variances of two samples under the assumption that both populations have equal true variances (homoscedasticity). Unpooled variance (used in Welch's t-test) calculates separate sample variances when population variances may differ.

### Q27. How does missing data introduce statistical bias if missing not at random (MNAR)?

**Answer:** If data is MNAR, the probability of missingness depends on the unobserved value itself (e.g. high earners refusing to report income). Simple imputation or deletion shifts the sample mean and distorts variance, introducing non-correctable bias unless modeled explicitly.

### Q28. What is heteroscedasticity and how does it affect linear models?

**Answer:** Heteroscedasticity occurs when the variance of the residuals is non-constant across levels of an independent variable (e.g. funnel-shaped residual plots). It does not bias regression coefficients, but it invalidates standard error calculations, making t-tests and p-values unreliable.

### Q29. Explain Bootstrapping in non-parametric statistics.

**Answer:** Bootstrapping estimates the sampling distribution of almost any statistic by repeatedly resampling the observed dataset with replacement B times (e.g. B = 10,000) and calculating the statistic on each bootstrap sample, generating robust confidence intervals without theoretical distribution assumptions.

### Q30. What is the Jackknife resampling method and how does it differ from Bootstrapping?

**Answer:** The Jackknife systematically recalculates the statistic by leaving out one observation at a time (N total resamples of size N-1). Unlike Bootstrapping, which is stochastic and samples with replacement, Jackknife is deterministic.
