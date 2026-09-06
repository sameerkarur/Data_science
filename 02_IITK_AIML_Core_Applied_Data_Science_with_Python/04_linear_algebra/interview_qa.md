# Interview Q&A — Linear Algebra for Machine Learning

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. What is the geometric meaning of the dot product between two vectors?

**Answer:** The dot product u · v = ||u|| ||v|| cos(θ). Geometrically, it measures the length of the projection of vector u onto vector v scaled by the length of v. If the vectors are orthogonal (θ = 90°), the dot product is zero; if collinear, it is maximized.

### Q2. Explain the difference between Euclidean Distance and Cosine Similarity.

**Answer:** Euclidean distance measures the straight-line spatial distance between points in coordinate space, sensitive to vector magnitude. Cosine similarity measures the cosine of the angle between vectors, normalizing for magnitude: cos(θ) = (u · v) / (||u|| ||v||). In text embeddings and RAG, cosine similarity is preferred because document length should not bias semantic relevance.

### Q3. What is linear independence of a set of vectors?

**Answer:** A set of vectors {v1, ..., vn} is linearly independent if the only linear combination that sums to the zero vector is the trivial solution where all coefficients are zero (c1*v1 + ... + cn*vn = 0 implies c1 = ... = cn = 0). If one vector can be written as a linear combination of others, the set is linearly dependent.

### Q4. What is the Rank of a matrix and why does it matter in machine learning?

**Answer:** The rank of a matrix is the maximum number of linearly independent row or column vectors. In feature matrices (N x D), if rank < D, features exhibit multicollinearity (redundant dimensions), making (X^T X) non-invertible in Ordinary Least Squares.

### Q5. Explain the geometric meaning of a matrix determinant.

**Answer:** The determinant det(A) represents the factor by which the linear transformation A scales area (in 2D) or volume (in higher dimensions). If det(A) = 0, the transformation collapses dimensions onto a lower subspace, meaning the matrix is singular and cannot be inverted.

### Q6. What are Eigenvalues and Eigenvectors?

**Answer:** For a square matrix A, an eigenvector v is a non-zero vector whose direction is unchanged by the transformation A, only scaled by a scalar factor λ (the eigenvalue): A v = λ v. In PCA, eigenvectors of the covariance matrix represent principal axes of maximum variance.

### Q7. Explain Principal Component Analysis (PCA) from a linear algebra perspective.

**Answer:** PCA computes the covariance matrix of centered data X^T X / (N-1), finds its orthogonal eigenvectors and eigenvalues via eigendecomposition, and projects the data onto the eigenvectors associated with the largest eigenvalues, maximizing preserved variance while minimizing reconstruction error.

### Q8. What is Singular Value Decomposition (SVD)?

**Answer:** SVD factors any real M x N matrix A into three matrices: A = U Σ V^T, where U (M x M) and V (N x N) are orthogonal matrices whose columns are left- and right-singular vectors, and Σ (M x N) is a diagonal matrix containing non-negative singular values ordered by magnitude.

### Q9. How does Truncated SVD perform dimensionality reduction and latent semantic analysis (LSA)?

**Answer:** Truncated SVD approximates matrix A by retaining only the top-k singular values and vectors: A_k = U_k Σ_k V_k^T. According to the Eckart-Young-Mirsky theorem, A_k is the optimal rank-k approximation that minimizes the Frobenius norm error ||A - A_k||_F.

### Q10. What is the Moore-Penrose Pseudoinverse and when is it used?

**Answer:** The pseudoinverse A^+ generalizes matrix inversion to non-square or singular matrices: A^+ = (A^T A)^(-1) A^T. In linear regression, it yields the minimum-norm least-squares solution: w = A^+ y.

### Q11. Explain the Gram-Schmidt process and QR Decomposition.

**Answer:** Gram-Schmidt converts a set of linearly independent vectors into an orthonormal basis. QR decomposition factors a matrix A into an orthogonal matrix Q (Q^T Q = I) and an upper triangular matrix R (A = Q R), providing high numerical stability when solving least-squares problems.

### Q12. What is a positive semi-definite (PSD) matrix and why are covariance matrices PSD?

**Answer:** A symmetric matrix A is positive semi-definite if x^T A x >= 0 for all non-zero vectors x. Covariance matrices are always PSD because x^T (X^T X) x = ||X x||^2 >= 0, guaranteeing that variance is always non-negative and all eigenvalues are >= 0.

### Q13. Explain the difference between L1 (Manhattan) norm and L2 (Euclidean) norm mathematically.

**Answer:** The L1 norm ||x||_1 = sum(|x_i|) sums absolute values, producing diamond-shaped unit contours. The L2 norm ||x||_2 = sqrt(sum(x_i^2)) computes straight-line distance, producing spherical unit contours.

### Q14. Why does L1 regularization (Lasso) induce feature sparsity while L2 (Ridge) does not?

**Answer:** The L1 diamond constraint boundary has sharp corners located exactly on coordinate axes. When the elliptical loss contour of regression touches the L1 boundary, it is most likely to hit a corner, forcing coefficients to exactly zero. The smooth L2 sphere touches contours at non-zero points, shrinking weights without zeroing them.

### Q15. What is an Orthogonal Matrix and what are its computational benefits?

**Answer:** A square matrix Q is orthogonal if Q^T Q = Q Q^T = I, meaning its transpose equals its inverse (Q^(-1) = Q^T). Orthogonal transformations preserve vector lengths, angles, and distances (isometries), preventing numerical instability and gradient explosion.

### Q16. Explain the Trace of a matrix and its invariant properties.

**Answer:** The trace tr(A) is the sum of diagonal elements. It equals the sum of eigenvalues: tr(A) = sum(λ_i). It is invariant under cyclic permutations: tr(ABC) = tr(BCA) = tr(CAB) and invariant under basis transformations.

### Q17. What is the spectral theorem for symmetric matrices?

**Answer:** The spectral theorem states that any real symmetric matrix A can be diagonalized by an orthogonal matrix Q of its eigenvectors: A = Q Λ Q^T, where Λ is a diagonal matrix of real eigenvalues. This guarantees that real symmetric matrices always have real eigenvalues and orthogonal eigenvectors.

### Q18. How does matrix projection onto a subspace work in linear regression?

**Answer:** Given feature matrix X, the projection matrix (hat matrix) H = X (X^T X)^(-1) X^T projects the target vector y orthogonally onto the column space of X, producing predicted values ŷ = H y such that the residual vector e = y - ŷ is orthogonal to the column space.

### Q19. What is the Kernel Trick in Support Vector Machines (SVM)?

**Answer:** The kernel trick computes inner products in a high-dimensional feature space without explicitly calculating coordinates in that space: K(x, z) = <φ(x), φ(z)>. Mercer's theorem guarantees that any positive semi-definite kernel function corresponds to an inner product in some Hilbert space.

### Q20. What is the condition number of a matrix and how does it affect gradient descent?

**Answer:** The condition number κ(A) = λ_max / λ_min of the Hessian matrix measures curvature anisotropy. A high condition number creates an elongated, ravine-like loss surface where gradient descent oscillates wildly perpendicular to the valley, requiring momentum or second-order methods.

### Q21. Explain Cholesky Decomposition and its role in Gaussian processes.

**Answer:** Cholesky decomposition factors a symmetric positive-definite matrix A into A = L L^T, where L is a lower triangular matrix. It requires half the FLOPs of LU decomposition and is widely used for sampling multivariate normal distributions and Gaussian process regression.

### Q22. What is a basis of a vector space?

**Answer:** A basis is a linearly independent subset of vectors that spans the entire vector space, meaning every vector in the space can be uniquely represented as a linear combination of basis vectors. The number of vectors in a basis equals the dimension of the space.

### Q23. Explain the Null Space (Kernel) of a matrix.

**Answer:** The null space Null(A) is the set of all vectors x such that A x = 0. According to the Rank-Nullity theorem: Rank(A) + Nullity(A) = Number of Columns of A.

### Q24. What is the Frobenius norm of a matrix?

**Answer:** The Frobenius norm ||A||_F = sqrt(sum_i sum_j |a_ij|^2) is the Euclidean norm applied to all matrix elements flattened. It equals the square root of the sum of squared singular values: sqrt(sum(σ_i^2)).

### Q25. What is matrix factorization and how does it power recommendation systems (Collaborative Filtering)?

**Answer:** Matrix factorization decomposes a sparse user-item interaction matrix R (M x N) into low-rank latent feature matrices P (M x K) and Q (N x K) such that R ≈ P Q^T. The latent factors capture hidden user tastes and item attributes.

### Q26. Explain cosine distance vs cosine similarity.

**Answer:** Cosine similarity ranges from -1 to 1 (or 0 to 1 for non-negative vectors). Cosine distance is defined as: Distance = 1 - Cosine Similarity. Note: Cosine distance is not a strict mathematical metric because it does not satisfy the triangle inequality.

### Q27. What is an affine transformation?

**Answer:** An affine transformation is a linear mapping followed by a translation: f(x) = A x + b. It preserves points, straight lines, and planes (e.g. rotation, scaling, shearing, translation), represented in homogeneous coordinates as a single matrix multiplication.

### Q28. How does batch matrix multiplication work in attention layers of Transformers?

**Answer:** In multi-head attention with batch size B, heads H, sequence length S, and head dimension D, queries Q and keys K have shape (B, H, S, D). Batched matrix multiplication 'Q @ K.swapaxes(-1, -2)' computes attention score matrices of shape (B, H, S, S) in parallel.

### Q29. What is the Cayley-Hamilton theorem?

**Answer:** The Cayley-Hamilton theorem states that every square matrix satisfies its own characteristic equation: p(A) = det(A - λ I) = 0. It allows high powers of matrices to be expressed as polynomials of degree less than N.

### Q30. Why is numerical precision critical when computing matrix inversions in data science?

**Answer:** Floating-point rounding errors in floating-point operations can accumulate rapidly in ill-conditioned matrices, producing inverted matrices with garbage coefficients. Always use stable factorizations (QR, SVD, Cholesky) or regularized inverses ((X^T X + λ I)^(-1)).
