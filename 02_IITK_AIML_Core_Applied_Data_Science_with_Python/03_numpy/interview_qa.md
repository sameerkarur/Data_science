# Interview Q&A — NumPy & High-Performance Array Computing

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain NumPy array broadcasting rules with mathematical conditions.

**Answer:** Two dimensions are compatible for broadcasting when: (1) they are equal, or (2) one of them is 1. Starting from the trailing (rightmost) dimensions, NumPy compares dimensions. A dimension of size 1 is virtually stretched to match the other array's shape without allocating extra memory.

### Q2. What are NumPy strides and how do they determine memory indexing?

**Answer:** Strides are tuples of bytes to step in each dimension when moving to the next element. For a 2D float64 array of shape (M, N), row stride is N*8 bytes, and column stride is 8 bytes. Transposing an array simply swaps the strides without copying underlying memory buffer.

### Q3. Explain the difference between C-contiguous and Fortran-contiguous array memory layouts.

**Answer:** C-contiguous (row-major) stores row elements consecutively in memory; advancing row index requires a large jump in memory address. Fortran-contiguous (column-major) stores column elements consecutively. C-contiguous aligns with C/Python indexing; Fortran with BLAS/Fortran libraries.

### Q4. What is the difference between a view and a copy in NumPy?

**Answer:** A view shares the exact same data buffer as the original array (checked via arr.base is not None); modifying a view mutates the original. A copy allocates an entirely new independent memory buffer.

### Q5. How does basic slicing differ from advanced (fancy) indexing in NumPy?

**Answer:** Basic slicing (e.g. arr[1:5, ::2]) ALWAYS returns a view. Advanced indexing (indexing with integer arrays, lists, or boolean masks like arr[[0, 2]]) ALWAYS returns a copy.

### Q6. Explain the difference between 'np.dot()', 'np.matmul()', and the '@' operator.

**Answer:** 'np.dot()' performs matrix multiplication for 2D arrays but calculates inner products for higher-dimensional tensors. 'np.matmul()' and '@' treat higher dimensions as batches of 2D matrices, which is essential for deep learning mini-batch tensor multiplications.

### Q7. What are universal functions (ufuncs) in NumPy?

**Answer:** Ufuncs are vectorized functions operating element-wise in compiled C, supporting broadcasting, type casting, and reduction methods like np.add.reduce() and np.add.accumulate().

### Q8. Explain how boolean masking works in NumPy and how to combine masks.

**Answer:** Boolean masking evaluates element-wise conditions, returning a boolean array of identical shape. Combine multiple conditions using bitwise operators (& for AND, | for OR, ~ for NOT), wrapping each condition in parentheses: '(arr > 0) & (arr < 10)'.

### Q9. What is the difference between 'np.concatenate()', 'np.stack()', 'np.vstack()', and 'np.hstack()'?

**Answer:** 'np.concatenate()' joins existing arrays along an existing axis. 'np.stack()' joins arrays along a NEW axis. 'np.vstack()' stacks along vertical row axis (axis 0). 'np.hstack()' stacks along horizontal column axis (axis 1).

### Q10. What does 'np.vectorize' do and is it truly as fast as C-level vectorization?

**Answer:** 'np.vectorize' is a convenience wrapper providing a vectorized interface over a Python function. It does NOT compile Python code into C or accelerate execution; it simply runs a Python for-loop internally.

### Q11. How does NumPy handle floating-point special values: 'np.nan', 'np.inf', and '-np.inf'?

**Answer:** They adhere to IEEE 754 standards. 'np.nan' represents Not-a-Number (np.nan == np.nan evaluates to False; use np.isnan()). Arithmetic with nan results in nan. 'np.inf' represents positive infinity from division by zero.

### Q12. What is 'np.where()' and how does its three-argument form work?

**Answer:** 'np.where(condition, x, y)' is a vectorized ternary operator: where condition is True, yield x; where False, yield y. When called with only condition, it returns the indices where condition is True.

### Q13. Explain the concept of memory alignment and SIMD operations in NumPy.

**Answer:** Modern CPUs use SIMD (Single Instruction, Multiple Data) instructions (AVX2, AVX-512) to execute mathematical operations on 4 to 8 floating-point numbers simultaneously in parallel hardware registers. NumPy contiguous memory buffers align with cache lines to maximize SIMD throughput.

### Q14. What is 'np.einsum()' and why is it powerful in linear algebra and machine learning?

**Answer:** Einstein summation ('np.einsum') computes complex tensor contractions, matrix multiplications, traces, and batched dot products using index notation (e.g. 'ik,kj->ij' for matrix multiplication) without allocating large intermediate memory arrays.

### Q15. How do you find unique elements and their counts in a NumPy array?

**Answer:** Use 'np.unique(arr, return_counts=True)'. It sorts the array and returns unique values and their frequencies in O(N log N) time.

### Q16. What is the difference between 'arr.reshape()' and 'arr.resize()'?

**Answer:** 'arr.reshape()' returns a new view with the requested shape if compatible with total elements (or raises ValueError). 'arr.resize()' modifies the array in-place, truncating or padding with zeros if size changes.

### Q17. How does 'np.pad()' work in computer vision and signal processing?

**Answer:** 'np.pad()' pads array boundaries with constants, reflection, or edge replication, widely used in convolutional layers to maintain spatial dimensions across image borders.

### Q18. What is structured array in NumPy?

**Answer:** A structured array allows defining compound datatypes (like C structs) with named fields of differing types (e.g. ('age', 'i4'), ('salary', 'f8')) stored contiguously in memory.

### Q19. Explain 'np.argsort()' and how it is used in ranking algorithms.

**Answer:** 'np.argsort()' returns the indices that would sort an array along a given axis, essential for ranking top-k predictions, nearest neighbors, and sorting parallel arrays simultaneously.

### Q20. How do you compute pairwise Euclidean distance matrix using NumPy broadcasting without for-loops?

**Answer:** Given matrix X of shape (N, D), expand dimensions: 'diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]' (shape N, N, D). The distance matrix is 'np.sqrt(np.sum(diff**2, axis=-1))'.

### Q21. What is 'np.clip()' and when is it used in deep learning?

**Answer:** 'np.clip(arr, a_min, a_max)' constrains values within an interval. In deep learning, gradient clipping prevents exploding gradients, and probability clipping (e.g. clip to [1e-15, 1-1e-15]) prevents log(0) in cross-entropy loss.

### Q22. How does 'np.tile()' differ from 'np.repeat()'?

**Answer:** 'np.repeat()' repeats individual elements consecutively ('[1, 2]' -> '[1, 1, 2, 2]'). 'np.tile()' replicates the entire array structure ('[1, 2]' -> '[1, 2, 1, 2]').

### Q23. Explain 'np.argmax()' along an axis in classification models.

**Answer:** 'np.argmax(arr, axis=1)' returns the column index containing the maximum predicted probability for each sample row, converting probability distributions into discrete class label predictions.

### Q24. What is memory leakage in NumPy arrays due to small views of huge arrays?

**Answer:** If you create a tiny slice from a massive 10 GB array ('small = huge[:5]'), the small view holds a reference to the entire 10 GB buffer ('small.base is huge'), preventing the 10 GB array from being garbage-collected. Fixed by forcing a copy: 'small = huge[:5].copy()'.

### Q25. How do you perform element-wise random sampling in NumPy?

**Answer:** Use the modern 'np.random.default_rng()' generator: '.integers()' for discrete uniform, '.random()' for continuous [0, 1), and '.normal(mean, std, size)' for Gaussian sampling.

### Q26. What is 'np.bincount()' and why is it preferred over np.unique for counting non-negative integers?

**Answer:** 'np.bincount(arr)' counts occurrences of each integer value from 0 up to max(arr) directly using an index-based bucket array, running in O(N) linear time compared to O(N log N) for np.unique.

### Q27. How does 'np.memmap' handle datasets larger than physical RAM?

**Answer:** 'np.memmap' maps large binary files on disk directly into an ndarray view; the operating system page cache transparently loads slices into RAM on-demand, enabling manipulation of terabyte-scale tensors without Out-Of-Memory errors.

### Q28. What is the difference between 'np.percentile()' and 'np.quantile()'?

**Answer:** 'np.percentile()' takes percentages in the range [0, 100]. 'np.quantile()' takes probabilities in the range [0, 1]. Both compute sample percentiles using linear interpolation.

### Q29. How do you invert a square matrix in NumPy and when should you avoid it?

**Answer:** Invert using 'np.linalg.inv(A)'. In production, never invert matrices to solve Ax = b; solving 'np.linalg.solve(A, b)' using LU or Cholesky decomposition is twice as fast and mathematically much more numerically stable.

### Q30. What is condition number of a matrix ('np.linalg.cond')?

**Answer:** The condition number measures how sensitive linear systems are to small errors in input data. A high condition number (ill-conditioned matrix) indicates that tiny perturbations in features produce huge fluctuations in linear regression weights.
