# Interview Q&A — Data Cleaning, Imputation & Wrangling

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. What are Hadley Wickham's three principles of Tidy Data?

**Answer:** Tidy Data principles state: (1) Each variable must have its own column. (2) Each observation must have its own row. (3) Each type of observational unit forms a table. Tidy structure standardizes data manipulation, simplifies filtering, and aligns seamlessly with vectorization.

### Q2. Differentiate between Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR).

**Answer:** MCAR: Missingness is entirely independent of observed and unobserved data (e.g. random sensor glitch); safe for complete-case deletion. MAR: Missingness depends systematically on other observed variables but not the missing value itself (e.g. men are less likely to report depression score, but within gender it is random); handled via multiple imputation. MNAR: Missingness depends on the unobserved value itself (e.g. wealthy individuals refusing to disclose high income); introduces non-ignorable bias.

### Q3. What are the dangers of Mean Imputation for missing values?

**Answer:** Mean imputation artificially reduces sample variance, distorts probability distributions, attenuates covariance and correlation between features, and under-estimates standard errors of regression coefficients, leading to artificially narrow confidence intervals and inflated Type I error rates.

### Q4. Explain KNN Imputation and Iterative (MICE) Imputation.

**Answer:** KNN Imputation finds the k-nearest neighbors using Euclidean distance on non-missing features and imputes the missing value as the average of neighbors. MICE (Multivariate Imputation by Chained Equations) models each missing feature as a function of all other features in an iterative round-robin cycle, capturing complex multivariate dependencies.

### Q5. What is Winsorization and how does it handle extreme outliers?

**Answer:** Winsorization caps extreme values at predefined percentiles (e.g. 1st and 99th percentiles) rather than discarding rows. Values below the 1st percentile are replaced by the 1st percentile value; values above the 99th percentile are replaced by the 99th percentile value, preserving sample size while taming extreme variance.

### Q6. Explain the difference between Min-Max Normalization and Z-score Standardization.

**Answer:** Min-Max Normalization: x_norm = (x - x_min) / (x_max - x_min), scaling values strictly to [0, 1]; highly sensitive to extreme outliers. Z-score Standardization: z = (x - μ) / σ, scaling data to zero mean and unit variance; handles unbounded distributions and assumes roughly Gaussian properties.

### Q7. What is RobustScaler and when is it preferred over StandardScaler?

**Answer:** RobustScaler centers data by subtracting the Median and scales by dividing by the Interquartile Range: x_scaled = (x - Q2) / (Q3 - Q1). Because both median and IQR are non-parametric order statistics, RobustScaler is immune to distortion by extreme outliers.

### Q8. What is the Box-Cox transformation and when does it fail?

**Answer:** The Box-Cox transformation stabilizes variance and transforms non-normal features into approximate Gaussian distributions via power parameter λ: y = (x^λ - 1) / λ for λ != 0, and ln(x) for λ = 0. It fails on negative numbers or zeros (requires x > 0). The Yeo-Johnson transformation extends Box-Cox to negative and zero values.

### Q9. Explain One-Hot Encoding vs Target (Mean) Encoding for high-cardinality categorical variables.

**Answer:** One-Hot Encoding creates a binary column per category, causing dimensionality explosion and severe sparsity on high-cardinality features (e.g. 50,000 ZIP codes). Target Encoding replaces each category with the average target value for that category, keeping dimensionality to 1 column. However, Target Encoding risks severe target leakage and overfitting unless smoothed using m-estimate or K-fold out-of-fold target encoding.

### Q10. What is Target Leakage in feature preprocessing?

**Answer:** Target leakage occurs when target information leaks into predictor features during data wrangling (e.g. calculating global target means across the entire dataset before train/test splitting, or including features like 'account_closed_date' when predicting customer churn).

### Q11. Explain string distance metrics: Levenshtein distance vs Jaro-Winkler distance.

**Answer:** Levenshtein distance counts the minimum single-character edits (insertions, deletions, substitutions) to transform string A into string B. Jaro-Winkler measures character matching and transpositions, giving higher similarity weights to strings sharing common initial prefixes, making it ideal for deduplicating person names and addresses.

### Q12. How do you perform fuzzy entity resolution and deduplication across messy datasets?

**Answer:** Use blocking (grouping by soundex, zip code, or initial characters to reduce O(N²) comparisons) followed by approximate string matching (Levenshtein, Token Sort Ratio in rapidfuzz/fuzzywuzzy) and graph connected-components clustering to merge duplicate entities.

### Q13. What is data reshaping using 'melt' vs 'pivot' in Pandas?

**Answer:** 'melt' converts wide-format data (measurements spread across multiple columns) into long tidy format (single column for variable names, single column for values). 'pivot' does the exact reverse, spreading unique values of an index into new columns.

### Q14. How do you handle date/time inconsistencies across different data sources?

**Answer:** Parse using 'pd.to_datetime(col, errors='coerce', utc=True)'. Standardize all timestamps to UTC immediately to eliminate timezone ambiguities and daylight saving time jumps. Extract cyclical features (sine/cosine of hour or day of year) for machine learning models.

### Q15. What are cyclical features and why shouldn't hour-of-day be fed as raw integers (0-23)?

**Answer:** Hours 23 (11 PM) and 0 (midnight) are contiguous in real time (distance = 1 hour), but as raw integers, their numerical distance is 23. Encoding hour as cyclical features: x_sin = sin(2π * hour / 24) and x_cos = cos(2π * hour / 24) maps time onto a continuous 2D unit circle, preserving physical temporal continuity.

### Q16. What is data sanitization and whitespace trimming?

**Answer:** Data sanitization strips invisible control characters, non-breaking spaces ('\u00a0'), leading/trailing whitespaces ('.str.strip()'), and normalizes Unicode strings via 'unicodedata.normalize("NFKD", text)' to eliminate homoglyph discrepancies.

### Q17. Explain how to detect and resolve duplicate records with contradictory fields.

**Answer:** Identify duplicates using primary candidate keys. For contradictory non-key fields: (1) apply business conflict-resolution rules (e.g. retain record with latest timestamp), (2) mark as conflicting and route to manual review, or (3) compute consensus/median values across matching entries.

### Q18. What is an Isolation Forest and how does it detect anomalies during data wrangling?

**Answer:** Isolation Forest builds ensembles of random trees that isolate individual anomalies by randomly selecting features and split values. Because anomalies are few and structurally different, they are isolated in fewer tree splits (shallow tree depth) compared to normal dense points, providing fast, non-parametric multi-dimensional outlier detection.

### Q19. Explain the difference between complete-case analysis and available-case analysis.

**Answer:** Complete-case analysis (listwise deletion) drops any row with even a single missing value, discarding valuable data and reducing sample size. Available-case analysis (pairwise deletion) calculates statistics using all rows that have data present for the specific pair of variables being analyzed, but can result in mathematically invalid non-positive-definite covariance matrices.

### Q20. How does frequency encoding work for categorical features?

**Answer:** Frequency encoding replaces each categorical value with its relative frequency (count / total_rows) in the training dataset. It is useful for tree-based models where the rarity of a category carries predictive signal.

### Q21. What is the dummy variable trap and how do you avoid it?

**Answer:** The dummy variable trap occurs when all K categories of a feature are converted into K binary columns, creating perfect linear dependence (collinearity) because the sum of all columns equals the intercept column (1). Avoid by dropping one column ('drop_first=True' in pd.get_dummies), retaining K - 1 dummy variables.

### Q22. What is the difference between ordinal encoding and nominal one-hot encoding?

**Answer:** Ordinal encoding maps ordered categories to integers that reflect natural ranking (e.g. Low=1, Medium=2, High=3). One-hot encoding creates independent binary indicator columns for unordered nominal categories (e.g. Red, Blue, Green) to prevent models from learning false numerical rank relationships.

### Q23. How do you handle outliers in skewed target variables in regression models?

**Answer:** Apply a monotonic non-linear transformation such as natural log (log1p(y) = ln(1 + y)) to compress long right tails, stabilize residual variance, and normalize error distributions. After prediction, invert using expm1(ŷ).

### Q24. Explain data binning (discretization) and its trade-offs.

**Answer:** Binning groups continuous values into discrete intervals (e.g. age groups: 18-25, 26-35). Trade-offs: it handles non-linear relationships and tames outliers, but destroys within-bin variance, reduces statistical power, and introduces arbitrary boundary discontinuities.

### Q25. What is an entity-relationship diagram (ERD) in relational data wrangling?

**Answer:** An ERD visually models entities (tables), their attributes (columns), and relationships (one-to-one, one-to-many, many-to-many) governed by Primary Keys and Foreign Keys, ensuring relational join integrity during pipeline development.

### Q26. How do you detect structural data anomalies (like trailing spaces or casing differences)?

**Answer:** Group by lowercase/stripped strings and compare unique counts: 'df['category'].str.lower().str.strip().nunique() vs df['category'].nunique()'. If lower/stripped unique count is smaller, casing or whitespace inconsistencies exist.

### Q27. Explain data validation frameworks like Great Expectations or Pydantic.

**Answer:** These frameworks enforce automated declarative data quality test suites ('expectations') at pipeline boundaries (e.g. asserting column values are non-null, within specific ranges, or matching regex patterns), halting execution before bad data enters machine learning models.

### Q28. What is the difference between forward-fill and linear interpolation for missing sensor data?

**Answer:** Forward-fill assumes step-function dynamics (value stays flat until new reading arrives). Linear interpolation assumes continuous linear transition between known endpoints, which is more physically realistic for temperature, speed, or pressure sensors.

### Q29. How do you safely split data into Train, Validation, and Test sets without data snooping?

**Answer:** Perform the random or temporal train/test split as the FIRST step before ANY imputation, scaling, outlier removal, or feature engineering. Compute all transformation parameters (mean, std, min, max, target encodings) strictly on the Training split, and apply those frozen parameters to Validation and Test sets.

### Q30. What is data lineage and why is it crucial in enterprise AI systems?

**Answer:** Data lineage tracks the complete lifecycle of data: origin source, transformations applied, intermediate staging tables, and downstream ML models. It provides auditability, regulatory compliance (GDPR right to explanation), faster root cause debugging, and impact analysis.
