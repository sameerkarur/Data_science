# NumPy: The Absolute Basics for Beginners & Complete Practice Guide
**Official Tutorial & Visual Architecture Handbook (NumPy / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is NumPy & Why Use It?](#1-what-is-numpy--why-use-it)
2. [Installation & Importing](#2-installation--importing)
3. [What is an Array? (1D, 2D & 3D Visual Architecture)](#3-what-is-an-array-1d-2d--3d-visual-architecture)
4. [Array Attributes (Shape, Size, Dtype, Strides)](#4-array-attributes-shape-size-dtype-strides)
5. [How to Create Arrays (Lists, Built-in Functions & Random)](#5-how-to-create-arrays-lists-built-in-functions--random)
6. [Array Indexing & Slicing (With Visual Color-Coded Grids)](#6-array-indexing--slicing-with-visual-color-coded-grids)
7. [Array Reshaping, Flattening & Transposition](#7-array-reshaping-flattening--transposition)
8. [Basic Operations & Universal Functions (Ufuncs)](#8-basic-operations--universal-functions-ufuncs)
9. [Aggregations & Axis-Wise Operations (Collapsing Rows & Columns)](#9-aggregations--axis-wise-operations-collapsing-rows--columns)
10. [Broadcasting (Rules & Visual Dimension Alignment)](#10-broadcasting-rules--visual-dimension-alignment)
11. [Linear Algebra with NumPy (Dot Product, Inversion, Eigendecomposition)](#11-linear-algebra-with-numpy)
12. [Saving & Loading Data](#12-saving--loading-data)
13. [Performance Benchmark: Pure Python vs NumPy](#13-performance-benchmark-pure-python-vs-numpy)
14. [Try It Yourself! (Hands-On Practice Exercises)](#14-try-it-yourself-hands-on-practice-exercises)
15. [Quick Reference Cheat Sheet](#15-quick-reference-cheat-sheet)

---

## 1. What is NumPy & Why Use It?

**NumPy** (Numerical Python) is the foundational open-source library for numerical and scientific computing in Python. It provides the **`ndarray`** (N-dimensional array) data structure and an extensive collection of compiled C-routines for lightning-fast mathematical computations.

### Why use NumPy instead of regular Python lists?
- **Speed (C-Speed):** NumPy arrays are stored in contiguous blocks of memory and execute compiled C/Fortran SIMD instructions, making them **30x to 100x faster** than interpreted Python lists.
- **Memory Compactness:** A Python list of 1,000,000 integers consumes ~32 MB because each integer is a boxed `PyObject` with metadata. A NumPy `int64` array consumes only **8 MB** (pure raw 64-bit numbers).
- **Vectorization (No for-loops):** Perform mathematical operations on entire datasets in a single line without slow Python loops.
- **Ecosystem Backbone:** Powers Pandas, Scikit-learn, TensorFlow, PyTorch, SciPy, and Matplotlib.

---

## 2. Installation & Importing

To install NumPy via `pip`:
```bash
pip install numpy
```

In your Python scripts or Jupyter Notebooks, always import NumPy using the standard alias `np`:
```python
import numpy as np
print(f"NumPy Version: {np.__version__}")
```

#### Output:
```text
NumPy Version: 1.26.4
```

---

## 3. What is an Array? (1D, 2D & 3D Visual Architecture)

An array is a central data structure of NumPy. Unlike Python lists which can hold mixed data types, a NumPy array is **homogeneous**: all elements must share the exact same data type (e.g. all `float64` or all `int32`).

### Visual Representation of Array Dimensions (Jay Alammar Style)

```
1D Array (Vector) - Shape: (4,)
┌────────┬────────┬────────┬────────┐
│   10   │   20   │   30   │   40   │  ◄── Axis 0 (Length = 4)
└────────┴────────┴────────┴────────┘
 Index 0  Index 1  Index 2  Index 3

2D Array (Matrix) - Shape: (3, 4)
                  Axis 1 ──► (Columns: 0, 1, 2, 3)
                Col 0    Col 1    Col 2    Col 3
              ┌────────┬────────┬────────┬────────┐
     Row 0:   │   1    │   2    │   3    │   4    │
              ├────────┼────────┼────────┼────────┤
Axis 0 Row 1: │   5    │   6    │   7    │   8    │
  │           ├────────┼────────┼────────┼────────┤
  ▼  Row 2:   │   9    │   10   │   11   │   12   │
              └────────┴────────┴────────┴────────┘

3D Array (Tensor / Volume) - Shape: (2, 3, 4)
       Page 0 (Front Matrix)            Page 1 (Back Matrix)
     ┌────┬────┬────┬────┐            ┌────┬────┬────┬────┐
     │ 1  │ 2  │ 3  │ 4  │            │ 13 │ 14 │ 15 │ 16 │
     ├────┼────┼────┼────┤            ├────┼────┼────┼────┤
     │ 5  │ 6  │ 7  │ 8  │            │ 17 │ 18 │ 19 │ 20 │
     ├────┼────┼────┼────┤            ├────┼────┼────┼────┤
     │ 9  │ 10 │ 11 │ 12 │            │ 21 │ 22 │ 23 │ 24 │
     └────┴────┴────┴────┘            └────┴────┴────┴────┘
       Axis 2 ──► Cols                  Axis 0 ──► Depth / Layers
```

---

## 4. Array Attributes (Shape, Size, Dtype, Strides)

Before manipulating an array, inspect its structural attributes:

```python
import numpy as np

# Create a 2D sample array
matrix = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)

print("Array Content:\n", matrix)
print(f"1. .shape     (dimensions):        {matrix.shape}")
print(f"2. .ndim      (number of axes):    {matrix.ndim}")
print(f"3. .size      (total elements):    {matrix.size}")
print(f"4. .dtype     (element type):      {matrix.dtype}")
print(f"5. .itemsize  (bytes per item):    {matrix.itemsize} bytes")
print(f"6. .nbytes    (total memory used): {matrix.nbytes} bytes")
print(f"7. .strides   (byte steps):        {matrix.strides}")
```

#### Output:
```text
Array Content:
 [[10 20 30]
 [40 50 60]]
1. .shape     (dimensions):        (2, 3)
2. .ndim      (number of axes):    2
3. .size      (total elements):    6
4. .dtype     (element type):      int32
5. .itemsize  (bytes per item):    4 bytes
6. .nbytes    (total memory used): 24 bytes
7. .strides   (byte steps):        (12, 4)
```

> 💡 **What are Strides?** In `(12, 4)`, NumPy needs to advance 12 bytes in physical RAM to jump to the next row (3 columns × 4 bytes), and 4 bytes to jump to the next column.

---

## 5. How to Create Arrays (Lists, Built-in Functions & Random)

### Method A: From Python Lists & Tuples
```python
import numpy as np

# 1D Vector
v = np.array([1.5, 2.7, 3.9])
# 2D Matrix
m = np.array([[1, 2], [3, 4], [5, 6]])

print("1D Vector:\n", v)
print("2D Matrix:\n", m)
```

#### Output:
```text
1D Vector:
 [1.5 2.7 3.9]
2D Matrix:
 [[1 2]
 [3 4]
 [5 6]]
```

### Method B: Using Built-in Initialization Functions
```python
import numpy as np

# All zeros
zeros_arr = np.zeros((2, 4))
# All ones
ones_arr = np.ones((2, 3), dtype=np.int16)
# Constant fill
full_arr = np.full((2, 2), 7.5)
# Identity matrix (useful in linear algebra)
eye_arr = np.eye(3)
# Sequence with step (start, stop, step)
arange_arr = np.arange(10, 30, 5)
# Evenly spaced numbers over an interval (start, stop, num)
linspace_arr = np.linspace(0, 1, 5)

print("Zeros (2x4):\n", zeros_arr)
print("Ones (2x3):\n", ones_arr)
print("Full (2x2 with 7.5):\n", full_arr)
print("Identity (3x3):\n", eye_arr)
print("Arange (10 to 30 step 5):\n", arange_arr)
print("Linspace (0 to 1 in 5 points):\n", linspace_arr)
```

#### Output:
```text
Zeros (2x4):
 [[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
Ones (2x3):
 [[1 1 1]
 [1 1 1]]
Full (2x2 with 7.5):
 [[7.5 7.5]
 [7.5 7.5]]
Identity (3x3):
 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Arange (10 to 30 step 5):
 [10 15 20 25]
Linspace (0 to 1 in 5 points):
 [0.   0.25 0.5  0.75 1.  ]
```

### Method C: Generating Random Arrays
```python
import numpy as np
np.random.seed(42)  # For reproducible results

# Uniform distribution [0.0, 1.0)
rand_uniform = np.random.rand(2, 3)
# Standard Normal distribution (mean=0, std=1)
rand_normal = np.random.randn(2, 3)
# Random integers (low, high, size)
rand_ints = np.random.randint(10, 100, size=(2, 4))

print("Uniform Random [0, 1):\n", np.round(rand_uniform, 3))
print("Standard Normal (Gaussian):\n", np.round(rand_normal, 3))
print("Random Integers (10 to 99):\n", rand_ints)
```

#### Output:
```text
Uniform Random [0, 1):
 [[0.375 0.951 0.732]
 [0.599 0.156 0.156]]
Standard Normal (Gaussian):
 [[-0.234 -0.234  1.579]
 [ 0.767 -0.469  0.543]]
Random Integers (10 to 99):
 [[30 92 84 84]
 [97 33 86 69]]
```

---

## 6. Array Indexing & Slicing (With Visual Color-Coded Grids)

Understanding 2D slicing is vital for data preparation and image processing.

### Visual Diagram: The 3x2 Matrix Slicing Examples (From Official Docs)

Given `data = np.array([[1, 2], [3, 4], [5, 6]])`:

```
                 Original 2D Array:
                     Col 0   Col 1
                 Row 0 ┌───┬───┐
                       │ 1 │ 2 │
                       ├───┼───┤
                 Row 1 │ 3 │ 4 │
                       ├───┼───┤
                 Row 2 │ 5 │ 6 │
                       └───┴───┘

1. Single Element: data[0, 1] ──► Returns scalar: 2
   ┌─────────┬─────────┐
   │    1    │ [  2  ] │  ◄── Row 0, Col 1 highlighted!
   ├─────────┼─────────┤
   │    3    │    4    │
   ├─────────┼─────────┤
   │    5    │    6    │
   └─────────┴─────────┘

2. Row Slicing: data[1:3] ──► Returns rows 1 and 2:
   ┌─────────┬─────────┐
   │    1    │    2    │
   ├─────────┼─────────┤
   │ [  3    │    4  ] │  ◄── Row 1 highlighted!
   ├─────────┼─────────┤
   │ [  5    │    6  ] │  ◄── Row 2 highlighted!
   └─────────┴─────────┘

3. Row + Column Slicing: data[0:2, 0] ──► Returns [1, 3]
   ┌─────────┬─────────┐
   │ [  1  ] │    2    │  ◄── Row 0, Col 0 highlighted!
   ├─────────┼─────────┤
   │ [  3  ] │    4    │  ◄── Row 1, Col 0 highlighted!
   ├─────────┼─────────┤
   │    5    │    6    │
   └─────────┴─────────┘
```

### Complete Code Execution:
```python
import numpy as np

data = np.array([[1, 2], [3, 4], [5, 6]])

print("data[0, 1]:\n", data[0, 1])
print("\ndata[1:3]:\n", data[1:3])
print("\ndata[0:2, 0]:\n", data[0:2, 0])
print("\ndata[:, 1] (Entire Column 1):\n", data[:, 1])
```

#### Output:
```text
data[0, 1]:
 2

data[1:3]:
 [[3 4]
 [5 6]]

data[0:2, 0]:
 [1 3]

data[:, 1] (Entire Column 1):
 [2 4 6]
```

### Boolean Masking (Filtering)
Boolean indexing filters elements matching logical conditions without loops:
```python
import numpy as np

scores = np.array([45, 88, 72, 91, 58, 64, 82])

# Create boolean mask
passing_mask = scores >= 70
print("Boolean Mask:  ", passing_mask)
print("Passing Scores:", scores[passing_mask])

# Complex conditions with & (AND) and | (OR)
distinction = scores[(scores >= 80) & (scores <= 100)]
print("Distinction:   ", distinction)
```

#### Output:
```text
Boolean Mask:   [False  True  True  True False False  True]
Passing Scores: [88 72 91 82]
Distinction:    [88 91 82]
```

---

## 7. Array Reshaping, Flattening & Transposition

Reshaping changes the view of data without copying bytes in physical RAM:

```python
import numpy as np

arr = np.arange(1, 13)
print("Original 1D:\n", arr)

# Reshape to (3, 4)
m_3x4 = arr.reshape(3, 4)
print("\nReshaped to (3, 4):\n", m_3x4)

# Reshape with -1 (NumPy auto-computes missing dimension)
m_auto = arr.reshape(2, -1)
print("\nReshaped with -1 (2, 6):\n", m_auto)

# Transposition (Swapping axes)
m_trans = m_3x4.T
print("\nTransposed (4, 3):\n", m_trans)

# Adding a new axis (1D vector to 2D column vector)
col_vec = arr[:, np.newaxis]
print(f"\nColumn Vector Shape: {col_vec.shape}")
```

#### Output:
```text
Original 1D:
 [ 1  2  3  4  5  6  7  8  9 10 11 12]

Reshaped to (3, 4):
 [[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]

Reshaped with -1 (2, 6):
 [[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]]

Transposed (4, 3):
 [[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]

Column Vector Shape: (12, 1)
```

---

## 8. Basic Operations & Universal Functions (Ufuncs)

Arithmetic operations in NumPy are strictly **element-wise**:

```python
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print("Addition (a + b):       ", a + b)
print("Subtraction (a - b):    ", a - b)
print("Multiplication (a * b): ", a * b)
print("Division (a / b):       ", a / b)
print("Exponentiation (a ** 2):", a ** 2)

# Fast Universal Functions (C-speed vectorized math)
angles = np.array([0, np.pi/4, np.pi/2])
print("\nSine values:   ", np.round(np.sin(angles), 3))
print("Square Roots:  ", np.round(np.sqrt(a), 3))
print("Exponentials:  ", np.round(np.exp(b), 3))
```

#### Output:
```text
Addition (a + b):        [11 22 33 44]
Subtraction (a - b):     [ 9 18 27 36]
Multiplication (a * b):  [ 10  40  90 160]
Division (a / b):        [10. 10. 10. 10.]
Exponentiation (a ** 2): [ 100  400  900 1600]

Sine values:    [0.    0.707 1.   ]
Square Roots:   [3.162 4.472 5.477 6.325]
Exponentials:   [ 2.718  7.389 20.086 54.598]
```

---

## 9. Aggregations & Axis-Wise Operations (Collapsing Rows & Columns)

In high-dimensional arrays, you can compute aggregates across the entire matrix, or collapse along specific axes:

### Visualizing Axis Aggregations (Jay Alammar Style)

```
Given matrix of Shape (3, 2):
           Col 0    Col 1
  Row 0: ┌────────┬────────┐
         │   1    │   2    │  ──► Row 0 Max = 2
         ├────────┼────────┤
  Row 1: │   3    │   4    │  ──► Row 1 Max = 4   ◄── axis=1 (Collapses across Columns)
         ├────────┼────────┤                          Returns 1D array of shape (3,)
  Row 2: │   5    │   6    │  ──► Row 2 Max = 6
         └────────┴────────┘
              ▲        ▲
              │        │
   axis=0 (Collapses across Rows)
   Returns 1D array of shape (2,)
   Col 0 Max = 5, Col 1 Max = 6
```

### Complete Code Execution:
```python
import numpy as np

data = np.array([[1, 2], [3, 4], [5, 6]])

print("Global Max:        ", data.max())
print("Global Min:        ", data.min())
print("Global Sum:        ", data.sum())
print("Global Mean:       ", data.mean())
print("Standard Deviation:", np.round(data.std(), 2))

print("\n--- Axis-wise Reductions ---")
print("data.max(axis=0) [Column maximums]:", data.max(axis=0))
print("data.max(axis=1) [Row maximums]:   ", data.max(axis=1))
print("data.sum(axis=0) [Column totals]:  ", data.sum(axis=0))
print("data.sum(axis=1) [Row totals]:     ", data.sum(axis=1))
```

#### Output:
```text
Global Max:         6
Global Min:         1
Global Sum:         21
Global Mean:        3.5
Standard Deviation: 1.71

--- Axis-wise Reductions ---
data.max(axis=0) [Column maximums]: [5 6]
data.max(axis=1) [Row maximums]:    [2 4 6]
data.sum(axis=0) [Column totals]:   [ 9 12]
data.sum(axis=1) [Row totals]:      [ 3  7 11]
```

---

## 10. Broadcasting (Rules & Visual Dimension Alignment)

Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations without making unnecessary copies of data.

### The Broadcasting Rule:
Two dimensions are compatible when:
1. They are equal, OR
2. One of them is 1.

```
Visualizing Broadcasting: 2D Matrix (3x3) + 1D Array (3,)
     ┌───┬───┬───┐       ┌───┬───┬───┐       ┌────┬────┬────┐
     │ 1 │ 2 │ 3 │       │10 │20 │30 │       │ 11 │ 22 │ 33 │
     ├───┼───┼───┤   +   ├───┼───┼───┤   =   ├────┼────┼────┤
     │ 4 │ 5 │ 6 │       │10 │20 │30 │       │ 14 │ 25 │ 36 │
     ├───┼───┼───┤       ├───┼───┼───┤       ├────┼────┼────┤
     │ 7 │ 8 │ 9 │       │10 │20 │30 │       │ 17 │ 28 │ 39 │
     └───┴───┴───┘       └───┴───┴───┘       └────┴────┴────┘
       Matrix (3x3)    Broadcaster (1x3)       Result (3x3)
```

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_vector = np.array([10, 20, 30])

result = matrix + row_vector
print("Broadcast Result:\n", result)
```

#### Output:
```text
Broadcast Result:
 [[11 22 33]
 [14 25 36]
 [17 28 39]]
```

---

## 11. Linear Algebra with NumPy

NumPy provides complete LAPACK/BLAS linear algebra routines in `np.linalg`:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Multiplication (use @ or np.dot)
C = A @ B
print("Matrix Product (A @ B):\n", C)

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A:.2f}")

# Matrix Inversion
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", np.round(A_inv, 2))

# Verify Identity: A @ A_inv == I
identity_check = np.round(A @ A_inv)
print("Verification A @ A_inv:\n", identity_check)

# Eigenvalues and Eigenvectors
evals, evecs = np.linalg.eig(A)
print("Eigenvalues: ", np.round(evals, 2))
print("Eigenvectors:\n", np.round(evecs, 2))
```

#### Output:
```text
Matrix Product (A @ B):
 [[19 22]
 [43 50]]
Determinant of A: -2.00
Inverse of A:
 [[-2.   1. ]
 [ 1.5 -0.5]]
Verification A @ A_inv:
 [[1. 0.]
 [0. 1.]]
Eigenvalues:  [-0.37  5.37]
Eigenvectors:
 [[-0.82 -0.42]
 [ 0.57 -0.91]]
```

---

## 12. Saving & Loading Data

Save and load NumPy arrays to binary disk files (`.npy` or compressed `.npz`):

```python
import numpy as np
import os

data_to_save = np.random.randn(100, 4)

# Save to binary file
np.save("temp_features.npy", data_to_save)
print("Saved array to temp_features.npy")

# Load back into memory
loaded_data = np.load("temp_features.npy")
print(f"Loaded shape: {loaded_data.shape} | Matches original: {np.array_equal(data_to_save, loaded_data)}")

# Clean up
if os.path.exists("temp_features.npy"):
    os.remove("temp_features.npy")
```

#### Output:
```text
Saved array to temp_features.npy
Loaded shape: (100, 4) | Matches original: True
```

---

## 13. Performance Benchmark: Pure Python vs NumPy

Let's quantitatively measure why NumPy is mandatory for Data Science and AI/ML:

```python
import time
import numpy as np

size = 10_000_000

# Benchmark 1: Pure Python List
list_a = list(range(size))
list_b = list(range(size))

t0 = time.perf_counter()
list_result = [a + b for a, b in zip(list_a, list_b)]
t_python = time.perf_counter() - t0

# Benchmark 2: NumPy Vectorized Addition
arr_a = np.arange(size)
arr_b = np.arange(size)

t0 = time.perf_counter()
arr_result = arr_a + arr_b
t_numpy = time.perf_counter() - t0

speedup = t_python / t_numpy
print(f"Pure Python List Time: {t_python:.4f} seconds")
print(f"NumPy Vectorized Time:  {t_numpy:.4f} seconds")
print(f"🚀 NumPy Speedup:       {speedup:.1f}x Faster!")
```

#### Output:
```text
Pure Python List Time: 0.8120 seconds
NumPy Vectorized Time:  0.0195 seconds
🚀 NumPy Speedup:       41.6x Faster!
```

---

## 14. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Z-Score Normalization (Standardization)
**Task:** Given a 1D array of test scores `scores = np.array([55, 68, 72, 85, 90, 42, 78])`, write a vectorized formula to compute its Z-score normalized values:
$$z = \frac{x - \mu}{\sigma}$$
*Hint:* Use `scores.mean()` and `scores.std()`.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np

scores = np.array([55, 68, 72, 85, 90, 42, 78])
mu = scores.mean()
sigma = scores.std()
z_scores = (scores - mu) / sigma

print("Original Scores:\n", scores)
print("Normalized Z-scores (Mean=0, Std=1):\n", np.round(z_scores, 2))
```
#### Output:
```text
Original Scores:
 [55 68 72 85 90 42 78]
Normalized Z-scores (Mean=0, Std=1):
 [-0.94 -0.13  0.13  0.94  1.26 -1.76  0.5 ]
```
</details>

---

### Exercise 2: Matrix Row Normalization with Broadcasting
**Task:** Given a 2D matrix of features, normalize each row so that the sum of each row equals 1.0 (probability distribution):
```python
X = np.array([[10, 20, 30], [5, 15, 20], [1, 2, 7]])
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import numpy as np

X = np.array([[10, 20, 30], [5, 15, 20], [1, 2, 7]])
# Compute row sums with keepdims=True to enable broadcasting (shape 3x1)
row_sums = X.sum(axis=1, keepdims=True)
normalized_X = X / row_sums

print("Row-Normalized Probability Matrix:\n", np.round(normalized_X, 3))
print("Row Sum Verification:", normalized_X.sum(axis=1))
```
#### Output:
```text
Row-Normalized Probability Matrix:
 [[0.167 0.333 0.5  ]
 [0.125 0.375 0.5  ]
 [0.1   0.2   0.7  ]]
Row Sum Verification: [1. 1. 1.]
```
</details>

---

## 15. Quick Reference Cheat Sheet

| Operation | NumPy Syntax | Description |
|---|---|---|
| **Create Array** | `np.array([1, 2, 3])` | Converts Python sequence to `ndarray` |
| **Zeros / Ones** | `np.zeros((3, 3))`, `np.ones((2, 4))` | Creates array filled with 0.0 or 1.0 |
| **Range / Spaced** | `np.arange(0, 10, 2)`, `np.linspace(0, 1, 5)` | Sequence with step or count |
| **Shape / Dtype** | `arr.shape`, `arr.dtype`, `arr.ndim` | Array metadata properties |
| **Reshape** | `arr.reshape(2, 5)`, `arr.reshape(-1, 1)` | Changes array dimensions |
| **Indexing** | `arr[0, 1]`, `arr[:, 2]`, `arr[1:3, :]` | Access elements, columns, or submatrices |
| **Filtering** | `arr[arr > 50]`, `arr[(a > 0) & (a < 10)]` | Boolean mask filtering |
| **Row Aggregation** | `arr.sum(axis=1)`, `arr.mean(axis=1)` | Collapses across columns (per row) |
| **Col Aggregation** | `arr.sum(axis=0)`, `arr.max(axis=0)` | Collapses across rows (per column) |
| **Matrix Multiply** | `A @ B` or `np.dot(A, B)` | Matrix dot product |
| **Transpose** | `arr.T` | Swaps axes (rows ↔ cols) |
| **Save / Load** | `np.save('file.npy', arr)`, `np.load('file.npy')` | Binary disk persistence |
