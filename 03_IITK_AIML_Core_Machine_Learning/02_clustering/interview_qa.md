# Interview Q&A — Unsupervised Clustering Algorithms

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the step-by-step algorithm of K-Means clustering.

**Answer:** 1. Initialize K cluster centroids (randomly or via K-Means++). 2. Assignment Step: assign each data point to its closest centroid based on squared Euclidean distance. 3. Update Step: recompute each centroid as the arithmetic mean of all data points assigned to that cluster. 4. Repeat Steps 2 and 3 until centroid positions converge (shift < tolerance) or max iterations is reached.

### Q2. What is K-Means++ initialization and why is it superior to random initialization?

**Answer:** Standard random initialization often converges to poor local minima with high inertia. K-Means++ picks the first centroid uniformly at random, then samples subsequent centroids with probability proportional to the squared distance D(x)² from the closest already chosen centroid. This disperses initial centroids across data space, providing O(log K) competitive approximation to the optimal clustering.

### Q3. Explain the difference between Inertia (WCSS) and Silhouette Score.

**Answer:** Inertia (Within-Cluster Sum of Squares) measures cluster compactness: sum of squared distances from points to their assigned centroid; it monotonically decreases toward 0 as K approaches N. Silhouette Score s = (b - a) / max(a, b) balances compactness a (mean intra-cluster distance) with separation b (mean distance to nearest neighboring cluster), bounded in [-1, 1].

### Q4. How does the Elbow Method determine the optimal number of clusters K?

**Answer:** Plot Inertia (WCSS) on the y-axis against K on the x-axis. As K increases, inertia drops rapidly at first, then flattens. The 'elbow' point represents the optimal trade-off where adding further clusters yields diminishing marginal gains in compactness.

### Q5. What are the primary assumptions and limitations of K-Means clustering?

**Answer:** Assumptions: spherical clusters of roughly equal variance and size, linearly separable boundaries. Limitations: sensitive to initial centroid placement, fails on non-convex arbitrary geometries (interlocking rings, crescent moons), heavily corrupted by outliers, and requires pre-specifying K.

### Q6. Explain DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

**Answer:** DBSCAN groups points based on local density without specifying K. It requires two parameters: epsilon (ε, neighborhood radius) and min_samples. A point is a Core point if at least min_samples lie within radius ε. Core points within ε connect into dense clusters. Border points lie within ε of a Core point. Outliers/Noise points have < min_samples within ε and are left unassigned (-1).

### Q7. What are the advantages of DBSCAN over K-Means?

**Answer:** DBSCAN does not require pre-specifying cluster count K, detects clusters of arbitrary non-convex shapes, automatically isolates and flags outliers/noise points, and is robust to initialization.

### Q8. When does DBSCAN fail?

**Answer:** DBSCAN struggles with datasets containing clusters of varying densities (a single fixed ε cannot capture both dense and sparse clusters simultaneously) and high-dimensional data where the curse of dimensionality makes distances uniform.

### Q9. Explain HDBSCAN (Hierarchical Density-Based Spatial Clustering).

**Answer:** HDBSCAN extends DBSCAN by building a cluster hierarchy across varying epsilon thresholds and extracting the most stable persistent clusters using mutual reachability distance and minimum spanning trees, eliminating the sensitive global ε parameter.

### Q10. What is Hierarchical Agglomerative Clustering (HAC)?

**Answer:** HAC is a bottom-up hierarchical method: starts with each sample in its own individual cluster, then iteratively merges the two closest clusters according to a linkage criterion until all points belong to a single root cluster, visualized as a Dendrogram.

### Q11. Explain Linkage Criteria in Hierarchical Clustering: Single, Complete, Average, and Ward.

**Answer:** Single Linkage: minimum distance between any point in cluster A and any point in B (prone to chaining artifacts). Complete Linkage: maximum distance between points in A and B (produces compact, equal-diameter clusters). Average Linkage: average distance between all pairs. Ward's Linkage: minimizes the increase in total within-cluster variance upon merging, producing well-separated spherical clusters.

### Q12. What is a Dendrogram and how do you determine cluster cutoffs?

**Answer:** A Dendrogram is a tree diagram displaying the sequence of cluster merges and their linkage distances. To extract clusters, draw a horizontal line across the dendrogram at a chosen height; the number of vertical lines intersected equals the number of extracted clusters. Cutting across the largest vertical gap without horizontal cross-branches identifies natural clustering.

### Q13. Explain Gaussian Mixture Models (GMM) and Soft Clustering.

**Answer:** GMM assumes data is generated from a mixture of K Gaussian distributions with parameters (mean μ_k, covariance Σ_k, mixing weight π_k). Unlike K-Means hard assignments, GMM assigns soft posterior probabilities P(cluster k | x) to each point, capturing elliptical clusters with varying orientations and variances.

### Q14. How does the Expectation-Maximization (EM) algorithm train a GMM?

**Answer:** E-step: computes the posterior probability (responsibility) that each Gaussian component generated each data point given current parameters. M-step: updates means, covariance matrices, and mixing weights by maximizing expected log-likelihood weighted by responsibilities. Repeats until log-likelihood converges.

### Q15. What is the difference between K-Means and GMM?

**Answer:** K-Means is a special case of GMM where all covariance matrices are constrained to be isotropic and equal (Σ_k = σ² I) as σ -> 0, producing hard assignments and spherical boundaries. GMM allows full covariance matrices, soft probabilistic assignments, and anisotropic elliptical shapes.

### Q16. What is Davies-Bouldin Index for evaluating clustering?

**Answer:** Davies-Bouldin Index measures the average similarity between each cluster and its most similar cluster, where similarity is the ratio of within-cluster dispersion to between-cluster separation: R_ij = (s_i + s_j) / d(c_i, c_j). Lower values indicate better clustering (compact, well-separated clusters).

### Q17. What is Calinski-Harabasz Index (Variance Ratio Criterion)?

**Answer:** Calinski-Harabasz is the ratio of between-cluster dispersion to within-cluster dispersion: CH = (SS_B / (K - 1)) / (SS_W / (N - K)). Higher scores indicate distinct, well-separated, compact clusters.

### Q18. How do you cluster high-dimensional data effectively?

**Answer:** Apply dimensionality reduction first (PCA, UMAP, t-SNE) to mitigate distance metric collapse caused by the curse of dimensionality, or use cosine distance/spectral clustering.

### Q19. Explain Spectral Clustering and when it is used.

**Answer:** Spectral clustering constructs an affinity graph between data points, computes the Graph Laplacian matrix L = D - W, performs eigendecomposition to project data into the low-dimensional subspace spanned by the bottom eigenvectors, and runs K-Means in that subspace. It excels at identifying non-convex, interconnected manifolds.

### Q20. What is Mean Shift clustering?

**Answer:** Mean Shift is a centroid-based non-parametric algorithm that places kernel density windows over data and iteratively shifts centroids toward regions of maximum density gradient (modes) until convergence. Number of clusters is determined automatically by bandwidth parameter.

### Q21. What is Affinity Propagation?

**Answer:** Affinity Propagation clusters data by passing messages ('responsibility' and 'availability') between pairs of data points until an optimal set of representative exemplars emerges. It does not require specifying K but scales as O(N²), limiting it to small datasets.

### Q22. How does feature scaling impact distance-based clustering algorithms?

**Answer:** Because Euclidean distance is sum of squared coordinate differences, unscaled features with large numerical magnitudes (e.g. Income: 100,000) completely dominate features with small numerical ranges (e.g. Age: 30), rendering the smaller feature irrelevant. Scaling is mandatory.

### Q23. What is the difference between Extrinsic and Intrinsic clustering evaluation metrics?

**Answer:** Intrinsic metrics evaluate cluster quality using the geometry of data points without ground truth labels (Silhouette Score, Inertia, Davies-Bouldin). Extrinsic metrics compare discovered clusters against known ground truth class labels (Adjusted Rand Index ARI, Normalized Mutual Information NMI, V-Measure).

### Q24. Explain Adjusted Rand Index (ARI).

**Answer:** ARI measures similarity between predicted clusters and true class assignments by counting pairwise agreements and disagreements, adjusted for chance: ARI = (Index - Expected Index) / (Max Index - Expected Index). 1 indicates identical clustering; 0 indicates random labeling.

### Q25. What is Biclustering (Co-clustering)?

**Answer:** Biclustering simultaneously clusters both rows (samples) and columns (features) of a data matrix, discovering sub-matrices of correlated behaviors (widely used in gene expression analysis and collaborative filtering recommendations).

### Q26. Explain OPTICS (Ordering Points To Identify the Clustering Structure).

**Answer:** OPTICS addresses DBSCAN's inability to detect clusters of varying density by ordering points based on core distance and reachability distance, producing a reachability plot that reveals clusters of arbitrary densities at multiple scales.

### Q27. How do you handle categorical variables in clustering (K-Modes and K-Prototypes)?

**Answer:** Standard K-Means cannot compute mathematical means on categories. K-Modes uses simple matching dissimilarity for categories and updates cluster modes (most frequent values). K-Prototypes integrates Euclidean distance for numeric features and matching dissimilarity for categorical features.

### Q28. What is the difference between hard clustering and soft (fuzzy) clustering?

**Answer:** Hard clustering (K-Means, DBSCAN) assigns each point exclusively to exactly one cluster. Soft clustering (GMM, Fuzzy C-Means) assigns membership probabilities or degrees of belonging across all clusters, allowing points on boundaries to belong partially to multiple clusters.

### Q29. How do you cluster streaming real-time data?

**Answer:** Use streaming algorithms like Mini-Batch K-Means (updates centroids incrementally using online mini-batch gradient descent) or CluStream (maintains micro-clusters in real-time, performing macro-clustering on demand).

### Q30. What is Cluster Profiling and how do you interpret cluster results for business stakeholders?

**Answer:** Cluster profiling calculates summary statistics (mean, median, mode) of original unscaled features per cluster, contrasts cluster averages against population baselines, and builds persona archetypes (e.g. 'High-Frequency Low-Basket Shoppers') to drive targeted business strategy.
