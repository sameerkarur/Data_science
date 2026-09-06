# NumPy - Numerical Python

## What is NumPy?
NumPy is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on them efficiently.

**Why use NumPy?**
- Much faster than Python lists (written in C under the hood)
- Uses less memory
- Supports vectorized operations (no need for loops)
- Foundation for Pandas, Matplotlib, Scikit-learn, etc.

## Installation
```bash
pip install numpy
```

## Importing
```python
import numpy as np
```

---

## 1. Creating Arrays

### From Python Lists
```python
# 1D Array
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(type(a))    # <class 'numpy.ndarray'>

# 2D Array (Matrix)
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 3D Array
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

### Using Built-in Functions
```python
# Array of zeros
np.zeros(5)              # [0. 0. 0. 0. 0.]
np.zeros((3, 4))         # 3x4 matrix of zeros

# Array of ones
np.ones(5)               # [1. 1. 1. 1. 1.]
np.ones((2, 3))          # 2x3 matrix of ones

# Array filled with a specific value
np.full((3, 3), 7)       # 3x3 matrix filled with 7

# Identity matrix
np.eye(4)                # 4x4 identity matrix

# Range of values
np.arange(0, 10, 2)      # [0 2 4 6 8]  (start, stop, step)

# Evenly spaced values
np.linspace(0, 1, 5)     # [0.   0.25 0.5  0.75 1.  ]  (start, stop, num_points)

# Empty array (uninitialized, random values)
np.empty((2, 3))
```

### Random Arrays
```python
# Random floats between 0 and 1
np.random.rand(3, 3)           # 3x3 matrix

# Random integers
np.random.randint(1, 100, 5)   # 5 random ints between 1-99

# Random from normal distribution
np.random.randn(3, 3)          # 3x3 matrix, mean=0, std=1

# Set seed for reproducibility
np.random.seed(42)

# Random choice from an array
np.random.choice([10, 20, 30, 40], size=3)
```

---

## 2. Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

arr.shape      # (2, 3)  - rows, columns
arr.ndim       # 2       - number of dimensions
arr.size       # 6       - total number of elements
arr.dtype      # int64   - data type of elements
arr.itemsize   # 8       - size of each element in bytes
arr.nbytes     # 48      - total bytes consumed
```

---

## 3. Array Indexing & Slicing

### 1D Indexing
```python
a = np.array([10, 20, 30, 40, 50])

a[0]       # 10       (first element)
a[-1]      # 50       (last element)
a[1:4]     # [20 30 40]  (slice: start to stop-1)
a[:3]      # [10 20 30]  (first 3 elements)
a[2:]      # [30 40 50]  (from index 2 to end)
a[::2]     # [10 30 50]  (every 2nd element)
```

### 2D Indexing
```python
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b[0, 0]      # 1         (row 0, col 0)
b[1, 2]      # 6         (row 1, col 2)
b[0]         # [1 2 3]   (entire row 0)
b[:, 1]      # [2 5 8]   (entire column 1)
b[0:2, 1:]   # [[2 3]    (rows 0-1, cols 1 onwards)
              #  [5 6]]
```

### Boolean Indexing (Filtering)
```python
a = np.array([10, 20, 30, 40, 50])

a[a > 25]          # [30 40 50]
a[a % 20 == 0]     # [20 40]
a[(a > 15) & (a < 45)]  # [20 30 40]
```

### Fancy Indexing
```python
a = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
a[indices]    # [10 30 50]
```

---

## 4. Reshaping Arrays

```python
a = np.arange(1, 13)     # [ 1  2  3  4  5  6  7  8  9 10 11 12]

# Reshape to 3x4
a.reshape(3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# Reshape with -1 (auto-calculate)
a.reshape(2, -1)          # 2x6 (NumPy figures out 6)
a.reshape(-1, 3)          # 4x3

# Flatten (multi-dim to 1D)
b = np.array([[1, 2], [3, 4]])
b.flatten()               # [1 2 3 4]
b.ravel()                 # [1 2 3 4]  (returns a view, not a copy)

# Transpose
b.T                       # [[1 3]
                          #  [2 4]]

# Add a new axis
a = np.array([1, 2, 3])
a[:, np.newaxis]          # Column vector: [[1], [2], [3]]
a[np.newaxis, :]          # Row vector: [[1, 2, 3]]
```

---

## 5. Array Operations

### Arithmetic (Element-wise)
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b       # [5 7 9]
a - b       # [-3 -3 -3]
a * b       # [4 10 18]
a / b       # [0.25 0.4  0.5]
a ** 2      # [1 4 9]
a % 2       # [1 0 1]

# Same as:
np.add(a, b)
np.subtract(a, b)
np.multiply(a, b)
np.divide(a, b)
np.power(a, 2)
```

### Scalar Operations (Broadcasting)
```python
a = np.array([1, 2, 3])

a + 10      # [11 12 13]
a * 3       # [3 6 9]
a / 2       # [0.5 1.  1.5]
```

### Comparison
```python
a = np.array([1, 2, 3, 4, 5])

a > 3       # [False False False  True  True]
a == 3      # [False False  True False False]
a != 2      # [ True False  True  True  True]
```

---

## 6. Mathematical Functions

### Aggregation
```python
a = np.array([1, 2, 3, 4, 5])

np.sum(a)       # 15
np.mean(a)      # 3.0
np.median(a)    # 3.0
np.std(a)       # 1.4142...
np.var(a)       # 2.0
np.min(a)       # 1
np.max(a)       # 5
np.argmin(a)    # 0   (index of min)
np.argmax(a)    # 4   (index of max)
np.cumsum(a)    # [ 1  3  6 10 15]  (cumulative sum)
np.cumprod(a)   # [  1   2   6  24 120]  (cumulative product)
```

### Axis-wise Operations (for 2D arrays)
```python
b = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(b, axis=0)    # [5 7 9]     (sum along columns)
np.sum(b, axis=1)    # [6 15]      (sum along rows)
np.mean(b, axis=0)   # [2.5 3.5 4.5]
np.mean(b, axis=1)   # [2. 5.]
```

### Universal Functions (ufuncs)
```python
a = np.array([1, 4, 9, 16])

np.sqrt(a)       # [1. 2. 3. 4.]
np.exp(a)        # exponential (e^x)
np.log(a)        # natural log
np.log10(a)      # log base 10
np.abs(a)        # absolute value
np.sin(a)        # sine
np.cos(a)        # cosine
np.round(np.array([1.23, 4.56]), 1)  # [1.2 4.6]
```

---

## 7. Linear Algebra

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Dot product
np.dot(a, b)            # or a @ b
# [[19 22]
#  [43 50]]

# Determinant
np.linalg.det(a)        # -2.0

# Inverse
np.linalg.inv(a)

# Eigenvalues & Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(a)

# Matrix rank
np.linalg.matrix_rank(a)

# Solve linear equations (Ax = b)
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 10])
x = np.linalg.solve(A, b)   # x = [1. 3.]
```

---

## 8. Stacking & Splitting

### Stacking (Joining)
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.hstack((a, b))     # [1 2 3 4 5 6]       (horizontal)
np.vstack((a, b))     # [[1 2 3] [4 5 6]]   (vertical)
np.concatenate((a, b))  # [1 2 3 4 5 6]

# For 2D
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
np.hstack((c, d))     # [[1 2 5 6] [3 4 7 8]]
np.vstack((c, d))     # [[1 2] [3 4] [5 6] [7 8]]
```

### Splitting
```python
a = np.array([1, 2, 3, 4, 5, 6])

np.split(a, 3)          # [array([1, 2]), array([3, 4]), array([5, 6])]
np.hsplit(a, 3)         # same for 1D

b = np.arange(1, 13).reshape(2, 6)
np.hsplit(b, 3)         # split into 3 equal parts along columns
np.vsplit(b, 2)         # split into 2 parts along rows
```

---

## 9. Copying Arrays

```python
a = np.array([1, 2, 3])

# View (shallow copy) - changes affect original
b = a.view()
b[0] = 99
print(a)    # [99  2  3]  ← original changed!

# Copy (deep copy) - independent
c = a.copy()
c[0] = 100
print(a)    # [99  2  3]  ← original NOT changed
```

---

## 10. Sorting

```python
a = np.array([3, 1, 4, 1, 5, 9, 2])

np.sort(a)              # [1 1 2 3 4 5 9]  (returns sorted copy)
np.argsort(a)           # [1 3 6 0 2 4 5]  (indices that would sort)

# Sort 2D array
b = np.array([[3, 1], [2, 4]])
np.sort(b, axis=0)      # sort along columns
np.sort(b, axis=1)      # sort along rows
```

---

## 11. Useful Tricks

### Where (Conditional)
```python
a = np.array([1, 2, 3, 4, 5])
np.where(a > 3, 'yes', 'no')   # ['no' 'no' 'no' 'yes' 'yes']
np.where(a > 3)                 # (array([3, 4]),)  indices where True
```

### Unique Values
```python
a = np.array([1, 2, 2, 3, 3, 3])
np.unique(a)                          # [1 2 3]
np.unique(a, return_counts=True)      # (array([1, 2, 3]), array([1, 2, 3]))
```

### Clip Values
```python
a = np.array([1, 5, 10, 15, 20])
np.clip(a, 5, 15)    # [ 5  5 10 15 15]
```

### NaN Handling
```python
a = np.array([1, np.nan, 3, np.nan, 5])
np.isnan(a)          # [False  True False  True False]
np.nanmean(a)        # 3.0  (ignores NaN)
np.nansum(a)         # 9.0
np.nanmax(a)         # 5.0
```

---

## Quick Reference Table

| Operation | Syntax |
|---|---|
| Create array | `np.array([1, 2, 3])` |
| Zeros | `np.zeros((rows, cols))` |
| Ones | `np.ones((rows, cols))` |
| Range | `np.arange(start, stop, step)` |
| Linspace | `np.linspace(start, stop, num)` |
| Random | `np.random.rand(rows, cols)` |
| Shape | `arr.shape` |
| Reshape | `arr.reshape(rows, cols)` |
| Flatten | `arr.flatten()` |
| Transpose | `arr.T` |
| Sum | `np.sum(arr, axis=)` |
| Mean | `np.mean(arr, axis=)` |
| Std Dev | `np.std(arr)` |
| Min/Max | `np.min(arr)` / `np.max(arr)` |
| Dot product | `np.dot(a, b)` or `a @ b` |
| Sort | `np.sort(arr)` |
| Filter | `arr[arr > value]` |
| Where | `np.where(condition, x, y)` |
