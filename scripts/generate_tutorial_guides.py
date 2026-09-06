"""
Generates extensive, tutorial-grade, visual guides (W3Schools / GeeksforGeeks / NumPy Official Docs style)
with Table of Contents, Jay Alammar-style visual ASCII diagrams, complete code blocks, explicit Output blocks,
and "Try It Yourself" practice exercises across all curriculum modules.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# =====================================================================
# 1. NumPy Master Guide
# =====================================================================
NUMPY_MASTER_GUIDE = r'''# NumPy: The Absolute Basics for Beginners & Complete Practice Guide
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
'''

# =====================================================================
# 2. Pandas Master Guide
# =====================================================================
PANDAS_MASTER_GUIDE = r'''# Pandas: Complete Step-by-Step Tutorial & Data Wrangling Handbook
**Official Tutorial & Practical Analytics Guide (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Pandas & Why Use It?](#1-what-is-pandas--why-use-it)
2. [Installation & Importing](#2-installation--importing)
3. [Pandas Data Structures: Series (1D) & DataFrame (2D)](#3-pandas-data-structures-series-1d--dataframe-2d)
4. [Creating DataFrames from Dictionaries, Lists & CSV](#4-creating-dataframes-from-dictionaries-lists--csv)
5. [Viewing & Inspecting Data (Head, Tail, Info, Describe)](#5-viewing--inspecting-data-head-tail-info-describe)
6. [Selection & Slicing (loc, iloc & Boolean Filtering)](#6-selection--slicing-loc-iloc--boolean-filtering)
7. [Data Cleaning (Missing Values, Duplicates & Types)](#7-data-cleaning-missing-values-duplicates--types)
8. [Data Transformation & Feature Engineering](#8-data-transformation--feature-engineering)
9. [GroupBy & Aggregations (Split-Apply-Combine)](#9-groupby--aggregations-split-apply-combine)
10. [Merging, Joining & Concatenating (Inner, Outer, Left, Right)](#10-merging-joining--concatenating)
11. [Pivot Tables & Cross-Tabulations](#11-pivot-tables--cross-tabulations)
12. [Reading & Writing External Files (CSV, Excel, JSON)](#12-reading--writing-external-files)
13. [Try It Yourself! (Hands-On Practice Exercises)](#13-try-it-yourself-hands-on-practice-exercises)
14. [Quick Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What is Pandas & Why Use It?

**Pandas** is the premiere Python library for data manipulation and tabular data analysis. It provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data intuitive and natural.

### Why use Pandas?
- **Excel on Steroids:** Easily handle millions of rows with high performance.
- **Missing Data Handling:** Detect, drop, or impute missing values seamlessly (`NaN` / `None`).
- **Flexible Reshaping:** Pivot, melt, stack, and aggregate multi-dimensional tables.
- **SQL-like Joins:** Execute lightning-fast inner, outer, left, and cross joins between datasets.
- **Time Series Ready:** Specialized frequency conversion, date shifting, and rolling statistics.

---

## 2. Installation & Importing

Install Pandas via `pip`:
```bash
pip install pandas
```

Standard industry convention is to import Pandas as `pd`:
```python
import pandas as pd
print(f"Pandas Version: {pd.__version__}")
```

#### Output:
```text
Pandas Version: 2.2.2
```

---

## 3. Pandas Data Structures: Series (1D) & DataFrame (2D)

Pandas provides two foundational data structures:
1. **`Series`:** A one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.).
2. **`DataFrame`:** A two-dimensional tabular data structure with labeled axes (rows and columns). A DataFrame is essentially a collection of Series sharing a common index.

### Visual Representation of Series vs DataFrame:

```
        PANDAS SERIES (1D)                         PANDAS DATAFRAME (2D)
                                                 Columns ──► ['Name', 'Age', 'City']
     Index ──► Data Values                         Index      Col 0    Col 1    Col 2
    ┌───────┬─────────────┐                       ┌───────┬─────────┬──────┬─────────┐
    │   0   │    10.5     │                       │   0   │  Alice  │  25  │   NYC   │
    ├───────┼─────────────┤                       ├───────┼─────────┼──────┼─────────┤
    │   1   │    20.8     │                       │   1   │   Bob   │  30  │   LA    │
    ├───────┼─────────────┤                       ├───────┼─────────┼──────┼─────────┤
    │   2   │    35.2     │                       │   2   │ Charlie │  35  │ Chicago │
    └───────┴─────────────┘                       └───────┴─────────┴──────┴─────────┘
     dtype: float64                                Row 0 ──► Series: [Alice, 25, NYC]
                                                   Col 0 ──► Series: [Alice, Bob, Charlie]
```

---

## 4. Creating DataFrames from Dictionaries, Lists & CSV

### Example 1: Creating a Series (GeeksforGeeks Style)
```python
import pandas as pd
import numpy as np

# From a Python list
fruits = pd.Series(['Apple', 'Banana', 'Cherry'], index=['a', 'b', 'c'])
print("Pandas Series with Custom Index:\n", fruits)
```

#### Output:
```text
Pandas Series with Custom Index:
 a     Apple
 b    Banana
 c    Cherry
 dtype: object
```

### Example 2: Creating a DataFrame from a Dictionary
```python
import pandas as pd

employee_data = {
    'EmpID': [101, 102, 103, 104],
    'Name': ['Sarah', 'David', 'Elena', 'Michael'],
    'Department': ['Engineering', 'Marketing', 'Engineering', 'Finance'],
    'Salary': [85000, 62000, 92000, 78000],
    'Experience': [4, 2, 7, 5]
}

df = pd.DataFrame(employee_data)
print("Employee DataFrame:\n", df)
```

#### Output:
```text
Employee DataFrame:
    EmpID     Name   Department  Salary  Experience
0    101    Sarah  Engineering   85000           4
1    102    David    Marketing   62000           2
2    103    Elena  Engineering   92000           7
3    104  Michael      Finance   78000           5
```

---

## 5. Viewing & Inspecting Data (Head, Tail, Info, Describe)

When exploring a new dataset, always execute these diagnostic inspections:

```python
import pandas as pd

# 1. View first 2 rows
print("--- df.head(2) ---\n", df.head(2))

# 2. View shape and column names
print("\nShape (Rows, Columns):", df.shape)
print("Column Names:         ", df.columns.tolist())
print("Data Types:\n", df.dtypes)

# 3. Comprehensive Statistical Summary
print("\n--- df.describe() Numerical Summary ---\n", df.describe())
```

#### Output:
```text
--- df.head(2) ---
    EmpID   Name   Department  Salary  Experience
0    101  Sarah  Engineering   85000           4
1    102  David    Marketing   62000           2

Shape (Rows, Columns): (4, 5)
Column Names:          ['EmpID', 'Name', 'Department', 'Salary', 'Experience']
Data Types:
 EmpID          int64
Name          object
Department    object
Salary         int64
Experience     int64
dtype: object

--- df.describe() Numerical Summary ---
             EmpID        Salary  Experience
count     4.000000      4.000000    4.000000
mean    102.500000  79250.000000    4.500000
std       1.290994  12816.005618    2.081666
min     101.000000  62000.000000    2.000000
25%     101.750000  74000.000000    3.500000
50%     102.500000  81500.000000    4.500000
75%     103.250000  86750.000000    5.500000
max     104.000000  92000.000000    7.000000
```

---

## 6. Selection & Slicing (loc, iloc & Boolean Filtering)

Accessing subsets of data is the most common operation in Pandas.

### Visual Diagram: `.loc` vs `.iloc`

```
  df.loc[row_label, col_label]        vs        df.iloc[row_integer, col_integer]
  (Explicit Label / Name Based)                 (Pure 0-Indexed Position Based)
  
  df.loc[1:2, 'Name':'Salary']                  df.iloc[1:3, 1:4]
  (INCLUSIVE of endpoint 'Salary'!)             (EXCLUSIVE of endpoint index 3 & 4!)
```

```python
import pandas as pd

# 1. Select single column as Series
names = df['Name']

# 2. Select multiple columns as DataFrame
subset = df[['Name', 'Salary']]
print("Multiple Columns:\n", subset)

# 3. .iloc: Select rows 0 to 1, columns 1 to 3 by integer index
print("\n--- df.iloc[0:2, 1:4] ---")
print(df.iloc[0:2, 1:4])

# 4. .loc: Select by column names and condition
print("\n--- High Earners (Salary >= 80,000) ---")
high_earners = df.loc[df['Salary'] >= 80000, ['Name', 'Department', 'Salary']]
print(high_earners)
```

#### Output:
```text
Multiple Columns:
       Name  Salary
0    Sarah   85000
1    David   62000
2    Elena   92000
3  Michael   78000

--- df.iloc[0:2, 1:4] ---
    Name   Department  Salary
0  Sarah  Engineering   85000
1  David    Marketing   62000

--- High Earners (Salary >= 80,000) ---
    Name   Department  Salary
0  Sarah  Engineering   85000
2  Elena  Engineering   92000
```

---

## 7. Data Cleaning (Missing Values, Duplicates & Types)

In the real world, data is messy. Here is the canonical W3Schools cleaning workflow:

```python
import pandas as pd
import numpy as np

# Sample dataset with missing values and duplicates
raw_records = pd.DataFrame({
    'TransactionID': [1, 2, 3, 3, 4],
    'Customer': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David'],
    'Amount': [250.0, np.nan, 150.0, 150.0, 420.0],
    'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-03', 'InvalidDate']
})

print("Raw Dirty Data:\n", raw_records)

# 1. Identify missing values
print("\nMissing Values Count:\n", raw_records.isna().sum())

# 2. Impute missing numeric values with column median
median_amount = raw_records['Amount'].median()
raw_records['Amount'] = raw_records['Amount'].fillna(median_amount)

# 3. Remove duplicate rows
clean_df = raw_records.drop_duplicates()

# 4. Clean dates using errors='coerce' to turn bad dates into NaT
clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors='coerce')

print("\n--- Cleaned DataFrame ---\n", clean_df)
```

#### Output:
```text
Raw Dirty Data:
    TransactionID Customer  Amount         Date
0              1    Alice   250.0   2026-01-01
1              2      Bob     NaN   2026-01-02
2              3  Charlie   150.0   2026-01-03
3              3  Charlie   150.0   2026-01-03
4              4    David   420.0  InvalidDate

Missing Values Count:
 TransactionID    0
Customer         0
Amount           1
Date             0
dtype: int64

--- Cleaned DataFrame ---
    TransactionID Customer  Amount       Date
0              1    Alice   250.0 2026-01-01
1              2      Bob   200.0 2026-01-02
2              3  Charlie   150.0 2026-01-03
4              4    David   420.0        NaT
```

---

## 8. Data Transformation & Feature Engineering

Transforming raw columns into predictive features:

```python
import pandas as pd

df = pd.DataFrame({
    'Product': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard'],
    'UnitPrice': [1200, 35, 120],
    'Quantity': [2, 10, 4]
})

# 1. Vectorized Column Creation
df['TotalRevenue'] = df['UnitPrice'] * df['Quantity']

# 2. Custom Function Application with .apply()
def categorize_tier(price):
    if price > 500:
        return 'Premium'
    elif price > 50:
        return 'Mid-Range'
    return 'Budget'

df['Tier'] = df['UnitPrice'].apply(categorize_tier)

# 3. String Methods with .str accessor
df['Product_Upper'] = df['Product'].str.upper()

print("Engineered DataFrame:\n", df[['Product', 'TotalRevenue', 'Tier', 'Product_Upper']])
```

#### Output:
```text
Engineered DataFrame:
                Product  TotalRevenue       Tier        Product_Upper
0           Laptop Pro          2400    Premium           LAPTOP PRO
1       Wireless Mouse           350     Budget       WIRELESS MOUSE
2  Mechanical Keyboard           480  Mid-Range  MECHANICAL KEYBOARD
```

---

## 9. GroupBy & Aggregations (Split-Apply-Combine)

The **Split-Apply-Combine** strategy is the foundation of group aggregations:

```
                  SPLIT-APPLY-COMBINE PIPELINE
       Input Table ──► SPLIT by Department:
                          ├── Engineering Sub-table
                          ├── Marketing Sub-table
                          └── Finance Sub-table
                                    │
                       APPLY Aggregation: sum(Salary), mean(Experience)
                                    │
                       COMBINE into Summary Table:
                          Department     TotalSalary  AvgExp
                          Engineering       177,000     5.5
                          Marketing          62,000     2.0
                          Finance            78,000     5.0
```

```python
import pandas as pd

sales_data = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'South', 'North', 'West'],
    'Rep': ['Alex', 'Brian', 'Alex', 'David', 'Elena', 'Fiona'],
    'Units': [50, 40, 65, 30, 80, 45],
    'Revenue': [5000, 4200, 6800, 3100, 8400, 4700]
})

# Group by Region with multiple aggregations
region_summary = sales_data.groupby('Region').agg(
    TotalRevenue=('Revenue', 'sum'),
    AvgUnits=('Units', 'mean'),
    TotalTransactions=('Rep', 'count')
).reset_index()

print("Regional Performance Summary:\n", region_summary)
```

#### Output:
```text
Regional Performance Summary:
   Region  TotalRevenue   AvgUnits  TotalTransactions
0  North         20200  65.000000                  3
1  South          7300  35.000000                  2
2   West          4700  45.000000                  1
```

---

## 10. Merging, Joining & Concatenating

Combining distinct relational tables using primary keys:

```
                      VISUALIZING SQL-STYLE JOINS
      INNER JOIN                      LEFT JOIN                     OUTER JOIN
    ┌────┬─────────┐                ┌────┬─────────┐              ┌────┬─────────┐
    │ ID │ Shared  │                │ ID │ All Left│              │ ID │ All Rows│
    └────┴─────────┘                └────┴─────────┘              └────┴─────────┘
  (Keys in BOTH tables)         (All Left + Matching Right)   (Union of all keys)
```

```python
import pandas as pd

customers = pd.DataFrame({
    'CustID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'OrderID': [501, 502, 503],
    'CustID': [1, 2, 4],  # Customer 4 does not exist in customers table
    'Amount': [350, 120, 890]
})

# Inner Merge (Only matching keys)
inner_df = pd.merge(customers, orders, on='CustID', how='inner')
print("--- Inner Join ---\n", inner_df)

# Left Merge (Preserves all customers)
left_df = pd.merge(customers, orders, on='CustID', how='left')
print("\n--- Left Join ---\n", left_df)
```

#### Output:
```text
--- Inner Join ---
    CustID   Name  OrderID  Amount
0       1  Alice      501     350
1       2    Bob      502     120

--- Left Join ---
    CustID     Name  OrderID  Amount
0       1    Alice    501.0   350.0
1       2      Bob    502.0   120.0
2       3  Charlie      NaN     NaN
```

---

## 11. Pivot Tables & Cross-Tabulations

Pivot tables summarize complex multi-dimensional tables into presentation grids:

```python
import pandas as pd

orders_df = pd.DataFrame({
    'Year': [2025, 2025, 2026, 2026, 2026],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Electronics', 'Clothing'],
    'Sales': [1500, 400, 2200, 1800, 650]
})

pivot = pd.pivot_table(
    orders_df,
    values='Sales',
    index='Category',
    columns='Year',
    aggfunc='sum',
    fill_value=0
)

print("Sales Pivot Table:\n", pivot)
```

#### Output:
```text
Sales Pivot Table:
 Year          2025  2026
Category                 
Clothing       400   650
Electronics   1500  4000
```

---

## 12. Reading & Writing External Files

```python
import pandas as pd
import tempfile
import os

# Create sample DataFrame
df = pd.DataFrame({'Model': ['ResNet50', 'BERT', 'GPT-4'], 'Parameters_M': [25.6, 110, 175000]})

# Write to temporary CSV
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
    tmp_path = tmp.name

df.to_csv(tmp_path, index=False)
print(f"Written to CSV: {tmp_path}")

# Read CSV back into DataFrame
df_read = pd.read_csv(tmp_path)
print("Read DataFrame:\n", df_read)

# Clean up
if os.path.exists(tmp_path):
    os.remove(tmp_path)
```

#### Output:
```text
Written to CSV: /var/folders/.../temp.csv
Read DataFrame:
       Model  Parameters_M
0  ResNet50          25.6
1      BERT         110.0
2     GPT-4      175000.0
```

---

## 13. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding Top Customers by Expenditure
**Task:** Given a DataFrame of e-commerce orders, compute the total expenditure per customer and find the top 2 customers with the highest spending:
```python
orders = pd.DataFrame({
    'Customer': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
    'Spend': [120, 450, 80, 210, 310, 400]
})
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import pandas as pd

orders = pd.DataFrame({
    'Customer': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
    'Spend': [120, 450, 80, 210, 310, 400]
})

top_spenders = (orders.groupby('Customer')['Spend']
                .sum()
                .sort_values(ascending=False)
                .head(2)
                .reset_index())

print("Top 2 Customers by Total Spend:\n", top_spenders)
```
#### Output:
```text
Top 2 Customers by Total Spend:
   Customer  Spend
0      Bob    760
1    Alice    600
```
</details>

---

## 14. Quick Reference Cheat Sheet

| Task | Pandas Command | Description |
|---|---|---|
| **Read CSV** | `pd.read_csv('file.csv')` | Ingests CSV to DataFrame |
| **Inspect Data** | `df.head()`, `df.info()`, `df.describe()` | Examines structure & statistics |
| **Filter Rows** | `df[df['age'] > 30]`, `df.query('age > 30')` | Boolean conditional selection |
| **Select Columns**| `df[['name', 'salary']]` | Extracts column subset |
| **Label Slice** | `df.loc[0:5, ['name', 'age']]` | Label-based row and column slice |
| **Positional Slice**| `df.iloc[0:5, 0:2]` | 0-indexed integer slice |
| **Fill Missing** | `df['col'].fillna(df['col'].median())` | Imputes missing values |
| **Drop Missing** | `df.dropna(subset=['id', 'date'])` | Removes records with NaNs |
| **Drop Duplicates**| `df.drop_duplicates()` | Eliminates duplicate rows |
| **GroupBy** | `df.groupby('dept')['salary'].mean()` | Aggregates across categories |
| **Merge / Join** | `pd.merge(df1, df2, on='key', how='inner')` | SQL-style relational merge |
| **Pivot Table** | `pd.pivot_table(df, values='x', index='y', columns='z')`| 2D multi-index summary |
| **Export CSV** | `df.to_csv('output.csv', index=False)` | Writes DataFrame to disk |
'''

# Write to both locations so both basics.md and pandas_basics.md have the master visual tutorial
p_pd1 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas/pandas_basics.md"
p_pd2 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas/basics.md"
p_pd1.write_text(PANDAS_MASTER_GUIDE.strip() + "\n", encoding="utf-8")
p_pd2.write_text(PANDAS_MASTER_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated Pandas Master Guide: {len(PANDAS_MASTER_GUIDE.splitlines())} lines.")

p_np1 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy/numpy_basics.md"
p_np2 = REPO_ROOT / "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy/basics.md"
p_np1.write_text(NUMPY_MASTER_GUIDE.strip() + "\n", encoding="utf-8")
p_np2.write_text(NUMPY_MASTER_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated NumPy Master Guide: {len(NUMPY_MASTER_GUIDE.splitlines())} lines.")
