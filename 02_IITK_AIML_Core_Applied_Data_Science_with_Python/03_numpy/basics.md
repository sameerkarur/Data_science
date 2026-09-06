# Chapter 3: NumPy Numerical Computing & Array Memory Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

A NumPy `ndarray` separates **Array Metadata** (shape, strides, dtype, flags) from the contiguous **Data Buffer**. This decoupling allows NumPy to perform operations like transposition, slicing, and reshaping in $O(1)$ constant time without duplicating array memory.

```
                NUMPY NDARRAY MEMORY ARCHITECTURE
    ┌────────────────────────────────────────────────────────┐
    │ ndarray Metadata Header:                               │
    │   • dtype: float64 (8 bytes per item)                  │
    │   • shape: (2, 3) ──► 2 rows, 3 columns                │
    │   • strides: (24, 8) ──► Bytes to advance per axis!   │
    └───────────────────────────┬────────────────────────────┘
                                │ Points to Raw C-Buffer
                                ▼
    ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
    │ Byte 0-7 │ Byte 8-15│Byte 16-23│Byte 24-31│Byte 32-39│Byte 40-47│
    │  [0, 0]  │  [0, 1]  │  [0, 2]  │  [1, 0]  │  [1, 1]  │  [1, 2]  │
    └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. Architectural Flowchart: Broadcasting Rules Engine

NumPy broadcasts arrays of mismatched shapes across element-wise operations using a deterministic alignment protocol:

```
                     BROADCASTING COMPATIBILITY RESOLUTION
                     
       Array A: Shape (5, 1, 32)
       Array B: Shape    (4, 32)
                            │
                            ▼
       Step 1: Right-Align Dimensions
               Array A:  5  x  1  x  32
               Array B:  1  x  4  x  32  (Prepend 1 to shorter array)
                            │
                            ▼
       Step 2: Compare Dimensions from Right to Left
               Dim 3: 32 vs 32  ──► Equal? YES! Match.
               Dim 2:  1 vs  4  ──► One is 1? YES! Broadcast 1 ➔ 4.
               Dim 1:  5 vs  1  ──► One is 1? YES! Broadcast 1 ➔ 5.
                            │
                            ▼
       Resulting Broadcast Output Shape: (5, 4, 32)
       (Achieved with ZERO data replication via stride 0 manipulation!)
```

---

## 3. Deep Theoretical Foundations

### 1. Memory Strides & Zero-Copy Views
The `strides` attribute is a tuple specifying the number of bytes to step in memory to advance by one index along each dimension.
For a 2D array of shape $(M, N)$ and data type size $S$ bytes:
- **C-Contiguous (Row-Major):** Elements in a row are adjacent in memory.
  $$\text{Strides} = (N \times S, S)$$
- **Fortran-Contiguous (Column-Major):** Elements in a column are adjacent.
  $$\text{Strides} = (S, M \times S)$$
- **Transposition (`arr.T`):** Swapping axes simply swaps the stride tuple $(N \times S, S) \to (S, N \times S)$. No bytes are moved in memory; it executes instantaneously in $O(1)$ time.

### 2. The Stride 0 Trick
When broadcasting a dimension of size 1 across size $K$, NumPy sets that dimension's stride to **0 bytes**. Every index access along that dimension references the exact same physical memory address, consuming zero additional RAM.

---

## 4. Production Implementation: Memory-Mapped Arrays for Massive Datasets

```python
import numpy as np
from pathlib import Path

def process_huge_matrix(filepath: Path | str, rows: int = 100_000, cols: int = 256) -> np.ndarray:
    """Uses memmap to process multi-gigabyte matrices with minimal RAM footprint."""
    # 1. Create a binary memory-mapped array on disk (100,000 x 256 x 4 bytes ≈ 102 MB)
    mmap_arr = np.memmap(filepath, dtype='float32', mode='w+', shape=(rows, cols))
    
    # 2. Populate chunks incrementally without memory bloat
    chunk_size = 10_000
    for i in range(0, rows, chunk_size):
        mmap_arr[i : i + chunk_size] = np.random.randn(chunk_size, cols).astype('float32')
        
    mmap_arr.flush()  # Commit to storage
    
    # 3. Read specific submatrix with zero-copy slice
    read_view = np.memmap(filepath, dtype='float32', mode='r', shape=(rows, cols))
    top_embeddings = read_view[:5, :]  # Instantaneous slice
    return np.array(top_embeddings)
```

---

## 5. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Modifying a View Expecting an Isolated Copy
```python
original = np.zeros((3, 3))
view_slice = original[:2, :2]
view_slice[0, 0] = 999  # MODIFIES 'original[0, 0]' AS WELL!

# PRODUCTION FIX: Force copy if isolation is required:
isolated_copy = original[:2, :2].copy()
```

### Pitfall 2: Inadvertent Temporary Array Allocation
```python
# ALLOCATES THREE TEMPORARY ARRAYS IN MEMORY:
# result = 2 * A + 3 * B - C

# PRODUCTION IN-PLACE FIX:
A *= 2
A += (3 * B)
A -= C
```
