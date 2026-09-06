# Interview Q&A — EDA & Feature Engineering

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the difference between Filter, Wrapper, and Embedded feature selection methods.

**Answer:** Filter methods evaluate individual feature relevance using statistical scores independent of model algorithms (e.g. Pearson correlation, Chi-Square, ANOVA F-test, Mutual Information); very fast, but ignore feature interactions. Wrapper methods search feature subsets by training and evaluating an ML model repeatedly (e.g. Recursive Feature Elimination RFE, Forward/Backward Stepwise); captures interactions, but computationally expensive and risks overfitting. Embedded methods perform feature selection during model training as part of the objective function (e.g. L1 Lasso penalty, Random Forest feature importance).

### Q2. Why does L1 Regularization (Lasso) produce sparse feature weights while L2 (Ridge) does not?

**Answer:** The mathematical constraint region for L1 is an L1-ball (a diamond with sharp vertices situated on coordinate axes), whereas L2 is an L2-ball (a smooth hypersphere). When the elliptical contours of the Ordinary Least Squares loss expand, they are geometrically most likely to intersect the L1 diamond at a corner vertex where one or more feature weights are exactly zero.

### Q3. Explain Target (Mean) Encoding and how to prevent target leakage.

**Answer:** Target encoding replaces each categorical level with the mean of the continuous or binary target variable for that category. Target leakage occurs if calculated globally across all training samples, leading to extreme overfitting. Mitigate using Out-Of-Fold (K-fold) target encoding, adding smoothing parameters (m-estimate blending category mean with global prior mean), or injecting Gaussian noise.

### Q4. How does Mutual Information differ from Pearson Correlation for feature selection?

**Answer:** Pearson correlation measures only linear relationships (r = 0 for Y = X² centered at 0). Mutual Information is an information-theoretic metric based on entropy: I(X; Y) = H(X) + H(Y) - H(X, Y). It quantifies any relationship—linear, quadratic, sinusoidal, or non-linear—and does not assume normality.

### Q5. What is the Box-Cox transformation and when does Yeo-Johnson replace it?

**Answer:** Box-Cox transforms skewed continuous variables into Gaussian shapes via power parameter λ: (x^λ - 1)/λ. It requires strictly positive values (x > 0). The Yeo-Johnson transformation extends the Box-Cox power formulation to accommodate zero and negative values.

### Q6. Explain Polynomial Features and Interaction Terms. What are the risks of high-degree expansions?

**Answer:** Interaction terms (x1 * x2) model joint multiplicative effects. Polynomial expansions (x1², x2²) capture non-linear curvature in linear models. Expanding to degree d on p features causes combinatorial dimensionality explosion (O(p^d)), leading to severe collinearity, memory bottlenecks, and extreme overfitting unless constrained by strong L1/L2 penalties.

### Q7. How do you handle cyclic features like hour of day (0-23) or month of year (1-12)?

**Answer:** Transform each cyclic feature into two continuous orthogonal trigonometric coordinates: x_sin = sin(2π * val / period) and x_cos = cos(2π * val / period). This projects time onto a continuous 2D unit circle where hour 23 and hour 0 are mathematically adjacent (distance = 1 hour).

### Q8. What is the difference between Min-Max Scaler and Standard Scaler? When does StandardScaler fail?

**Answer:** MinMaxScaler scales data to [0, 1] using min and max values, preserving exact zero values in sparse matrices but severely compressing normal ranges if extreme outliers exist. StandardScaler scales to zero mean and unit variance. StandardScaler fails when features exhibit extreme outliers or multi-modal heavy tails, where RobustScaler (median and IQR) is preferred.

### Q9. What is High Cardinality in categorical features and what are three robust strategies to handle it?

**Answer:** High cardinality occurs when a categorical feature has hundreds or thousands of unique levels (e.g. user IDs, postal codes). Strategies: (1) Target Encoding with regularization, (2) Frequency/Count Encoding, (3) Grouping rare categories into an 'Other' bucket based on a threshold (e.g. < 1% frequency), and (4) Entity Embeddings learned via neural networks.

### Q10. Explain Principal Component Analysis (PCA) as an unsupervised feature extraction method.

**Answer:** PCA projects high-dimensional centered data onto orthogonal linear axes (principal components) that sequentially maximize retained variance. The first principal component aligns with the eigenvector of the covariance matrix having the largest eigenvalue. PCA eliminates collinearity and compresses dimensions, but sacrifices raw feature interpretability.

### Q11. What is Data Leakage during feature engineering and how do you prevent it?

**Answer:** Data leakage occurs when information from outside the training dataset (such as validation/test sets or future data) contaminates the feature transformation pipeline (e.g. scaling before train/test splitting, computing global target statistics, using future timestamps). Prevent by fitting transformers strictly on training folds and using Scikit-learn Pipeline objects.

### Q12. How do you encode ordinal categorical variables correctly?

**Answer:** Map ordinal categories to integers that reflect their natural domain hierarchy (e.g. {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}). Never use nominal One-Hot Encoding for ordinal data if the intrinsic ranking provides predictive ordering signal.

### Q13. What is Recursive Feature Elimination (RFE)?

**Answer:** RFE is a greedy wrapper feature selection algorithm that trains a model, ranks features by their coefficients or feature importances, removes the weakest feature(s), and retrains the model iteratively until a target number of features remains. Cross-validated RFE (RFECV) automatically finds the optimal feature count.

### Q14. Explain Variance Thresholding for feature selection.

**Answer:** Variance thresholding is a baseline filter method that removes all features whose sample variance fails to exceed a specified threshold. For binary features, thresholding removes constant or near-constant features (e.g. p * (1 - p) < 0.01 where >99% of values are 0).

### Q15. What is Weight of Evidence (WoE) and Information Value (IV) in credit risk modeling?

**Answer:** In binary classification, WoE = ln(% Non-Events / % Events) within a binned feature bucket. Information Value IV = sum((% Non-Events - % Events) * WoE). IV evaluates overall predictive strength: IV < 0.02 is unpredictive, 0.1-0.3 is medium, and >0.5 warrants caution for suspicious leakage.

### Q16. How do you extract features from raw text data without deep learning?

**Answer:** Classical techniques: (1) Bag-of-Words (CountVectorizer) counting token frequencies, (2) TF-IDF (Term Frequency-Inverse Document Frequency) penalizing common corpus words, and (3) N-grams (bi-grams, tri-grams) to capture local word order and phrase idioms.

### Q17. What is TF-IDF mathematically?

**Answer:** TF(t, d) = count of term t in document d. IDF(t) = ln((1 + N) / (1 + DF(t))) + 1, where N is total documents and DF(t) is documents containing term t. TF-IDF(t, d) = TF(t, d) * IDF(t). High TF-IDF indicates words that are frequent in a specific document but rare across the broader corpus.

### Q18. How do you extract features from time-series timestamps?

**Answer:** Extract calendar components: year, month, day of week, day of month, hour, is_weekend, is_holiday, quarter. Compute lag features (y_{t-1}, y_{t-7}), rolling window statistics (7-day rolling mean, rolling standard deviation), and exponential moving averages.

### Q19. What are Lag Features and Rolling Statistics in time-series feature engineering?

**Answer:** Lag features shift historical values backward in time: lag_k = y_{t-k}, providing immediate temporal context. Rolling statistics compute aggregate metrics (mean, min, max, volatility) over sliding past windows [t-w, t-1], smoothing short-term noise and capturing trend dynamics.

### Q20. Explain the difference between Label Encoding and One-Hot Encoding.

**Answer:** Label Encoding assigns arbitrary integers (0, 1, 2...) to categories; if applied to nominal features in linear or distance-based models, the model falsely assumes numerical ranking (2 > 1). One-Hot Encoding creates K binary indicator columns, preserving nominal independence at the cost of expanding dimensionality.

### Q21. What is the Dummy Variable Trap?

**Answer:** When K categories are encoded into K one-hot columns, the sum of all columns equals 1, creating perfect multicollinearity with the regression intercept. Avoid by dropping one category ('drop_first=True'), leaving K - 1 dummy variables.

### Q22. How do tree-based models handle non-scaled features compared to distance-based algorithms?

**Answer:** Decision trees and ensemble methods (Random Forest, XGBoost) are invariant to monotonic feature scaling because split points depend strictly on feature ordering, not magnitude. Distance-based algorithms (KNN, SVM, K-Means) and gradient-based models (Neural Networks, Logistic Regression) are heavily distorted unless all features are standardized to identical scales.

### Q23. What is Quantile Transformer and when should it be applied?

**Answer:** QuantileTransformer maps feature empirical distributions to a uniform or normal distribution using cumulative distribution functions. It smooths out multi-modal distributions and dampens extreme outliers, but distorts linear correlations between features.

### Q24. How do you extract domain-specific features from geospatial coordinates (latitude, longitude)?

**Answer:** Compute Haversine distance to key landmarks (city center, airport, coastline), segment into spatial grid cells (Uber H3, geohashes), or calculate local density of amenities using spatial KD-Trees.

### Q25. Explain feature binarization and thresholding.

**Answer:** Feature binarization maps continuous numerical features to boolean values based on a domain threshold (e.g. converting income to is_high_income = income > 100k). It simplifies models and captures sharp cliff-edge behavioral shifts.

### Q26. What is the curse of dimensionality in feature engineering?

**Answer:** As the number of features D grows, the volume of feature space grows exponentially, causing sample data points to become extremely sparse. Distances between all points converge to equal values (Euclidean distance metric breakdown), increasing model variance and risk of overfitting.

### Q27. How do you detect and remove collinear features automatically in a pipeline?

**Answer:** Compute correlation matrix or Variance Inflation Factor (VIF). Iteratively remove features with VIF > 10, or use hierarchical clustering on correlation distance (1 - |r|) to group collinear features and select one representative feature per cluster.

### Q28. What is Feature Cross in tabular modeling?

**Answer:** A feature cross combines two or more categorical features into a single composite feature (e.g. crossing 'Device_Type=Mobile' with 'Country=IN' to form 'Mobile_IN'). It allows linear models to learn non-linear interactions without polynomial expansion.

### Q29. Explain Discretization (Binning) and why it helps linear models learn non-linear relationships.

**Answer:** Discretization converts continuous variables into discrete categorical bins (e.g. age groups). One-hot encoding the resulting bins allows linear models to assign independent weights to each interval, fitting step-function approximations to non-linear relationships.

### Q30. What are automated feature engineering frameworks like Featuretools (Deep Feature Synthesis)?

**Answer:** Frameworks like Featuretools automatically build complex feature representations by applying relational aggregations (sum, mean, max) and transformations across multiple connected relational database tables along Foreign Key relationships.
