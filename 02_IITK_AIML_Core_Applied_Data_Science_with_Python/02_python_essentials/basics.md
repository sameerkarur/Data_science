# Python Essentials for Data Science & Numerical Vectorization
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Why Python is the Lingua Franca of Data Science](#1-why-python-is-the-lingua-franca-of-data-science)
2. [Iterators & Generators (Memory-Efficient Streaming with `yield`)](#2-iterators--generators)
3. [Specialized Collections: `Counter`, `defaultdict`, `namedtuple`, `deque`](#3-specialized-collections)
4. [Functional Foundations: `map()`, `filter()`, `reduce()`, & `lambda`](#4-functional-foundations)
5. [Vectorization vs Python Loops (The GIL & SIMD Execution)](#5-vectorization-vs-python-loops)
6. [List, Dictionary & Generator Comprehensions (Performance Comparison)](#6-comprehensions-performance-comparison)
7. [Memory Profiling & Runtime Benchmarking (`sys.getsizeof`, `timeit`)](#7-memory-profiling--runtime-benchmarking)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. Why Python is the Lingua Franca of Data Science

Python dominates AI, machine learning, and data analytics due to its unique architectural duality:
1. **High-Level Expressiveness:** Clean, human-readable syntax allows data scientists to prototype mathematical algorithms rapidly.
2. **Low-Level C/C++ Glue Engine:** Heavy matrix multiplications, Fourier transforms, and neural network backpropagation are handed off to compiled C/Fortran libraries (NumPy, SciPy, OpenBLAS, LAPACK, CUDA).

---

## 2. Iterators & Generators (Memory-Efficient Streaming)

When processing multi-gigabyte CSVs or continuous sensor telemetry, loading all rows into RAM causes out-of-memory crashes (`MemoryError`). **Generators** compute values on-demand using the `yield` keyword with **$O(1)$ constant memory overhead**.

### Visual Architecture: List vs Generator in Memory

```
  PYTHON LIST (Eager Memory Allocation):
  [ 1, 2, 3, 4, ..., 1,000,000 ] ──► Consumes ~40 MB of Heap Memory instantly!

  GENERATOR (Lazy Evaluation with yield):
  State Machine: [Current Index] ──► Computes next value ONLY when requested ──► Consumes ~120 Bytes!
```

```python
import sys

# 1. Eager List Comprehension
million_list = [x * 2 for x in range(1_000_000)]

# 2. Lazy Generator Expression
million_gen = (x * 2 for x in range(1_000_000))

print(f"Memory used by List:      {sys.getsizeof(million_list):,} bytes (~{sys.getsizeof(million_list)/(1024**2):.1f} MB)")
print(f"Memory used by Generator: {sys.getsizeof(million_gen):,} bytes (Constant!)")

# Generator function for streaming CSV records
def stream_batches(dataset_size, batch_size=3):
    for i in range(0, dataset_size, batch_size):
        yield list(range(i, min(i + batch_size, dataset_size)))

print("\n--- Streaming Batches ---")
for batch in stream_batches(8, batch_size=3):
    print("Fetched Batch:", batch)
```

#### Output:
```text
Memory used by List:      8,448,728 bytes (~8.1 MB)
Memory used by Generator: 104 bytes (Constant!)

--- Streaming Batches ---
Fetched Batch: [0, 1, 2]
Fetched Batch: [3, 4, 5]
Fetched Batch: [6, 7]
```

---

## 3. Specialized Collections (`collections` Module)

Python's standard library provides high-performance container datatypes in the `collections` module:

```python
from collections import Counter, defaultdict, namedtuple, deque

# 1. Counter: High-speed frequency distribution
user_actions = ['click', 'view', 'click', 'purchase', 'view', 'click', 'refund']
counts = Counter(user_actions)
print("Top Action:", counts.most_common(1))
print("Total Action Counts:", dict(counts))

# 2. defaultdict: Eliminates KeyError by auto-initializing missing buckets
department_salaries = defaultdict(list)
department_salaries['Engineering'].append(120000)
department_salaries['Engineering'].append(135000)
department_salaries['Marketing'].append(90000)
print("\nDefaultDict Contents:", dict(department_salaries))

# 3. namedtuple: Lightweight, readable immutable records (Alternative to dicts/classes)
Point = namedtuple('DataPoint', ['sample_id', 'feature_x', 'label'])
pt = Point(sample_id=101, feature_x=4.82, label='Benign')
print(f"\nNamedTuple: ID={pt.sample_id}, Label={pt.label}, Value={pt.feature_x}")

# 4. deque: Fast O(1) appends and pops from both ends (Sliding Window memory)
sliding_window = deque(maxlen=3)
for temp in [21.5, 22.0, 22.5, 23.0, 24.5]:
    sliding_window.append(temp)
    print("Sliding Window (Maxlen 3):", list(sliding_window))
```

#### Output:
```text
Top Action: [('click', 3)]
Total Action Counts: {'click': 3, 'view': 2, 'purchase': 1, 'refund': 1}

DefaultDict Contents: {'Engineering': [120000, 135000], 'Marketing': [90000]}

NamedTuple: ID=101, Label=Benign, Value=4.82

Sliding Window (Maxlen 3): [21.5]
Sliding Window (Maxlen 3): [21.5, 22.0]
Sliding Window (Maxlen 3): [21.5, 22.0, 22.5]
Sliding Window (Maxlen 3): [22.0, 22.5, 23.0]
Sliding Window (Maxlen 3): [22.5, 23.0, 24.5]
```

---

## 4. Functional Foundations: `map()`, `filter()`, `reduce()`

```python
from functools import reduce

numbers = [10, 15, 20, 25, 30]

# 1. map(): Apply transformation element-wise
scaled = list(map(lambda x: x / 10, numbers))

# 2. filter(): Retain elements satisfying Boolean predicate
filtered = list(filter(lambda x: x > 18, numbers))

# 3. reduce(): Aggregate sequence into single scalar value
product = reduce(lambda acc, x: acc * x, [1, 2, 3, 4, 5])

print("Original Numbers: ", numbers)
print("Scaled (map):     ", scaled)
print("Filtered (>18):   ", filtered)
print("Product (reduce): ", product)
```

#### Output:
```text
Original Numbers:  [10, 15, 20, 25, 30]
Scaled (map):      [1.0, 1.5, 2.0, 2.5, 3.0]
Filtered (>18):    [20, 25, 30]
Product (reduce):  120
```

---

## 5. Vectorization vs Python Loops

Vectorization executes contiguous array memory operations in compiled C without the overhead of Python bytecode interpretation and the Global Interpreter Lock (GIL):

```python
import time
import numpy as np

N = 2_000_000

# Benchmark 1: Standard Python for-loop
py_list = list(range(N))
t0 = time.perf_counter()
py_result = []
for val in py_list:
    py_result.append(val ** 2)
t_loop = time.perf_counter() - t0

# Benchmark 2: NumPy C-Vectorized SIMD instruction
np_arr = np.arange(N)
t0 = time.perf_counter()
np_result = np_arr ** 2
t_vec = time.perf_counter() - t0

print(f"Python Loop Time:  {t_loop:.4f} seconds")
print(f"NumPy Vector Time: {t_vec:.4f} seconds")
print(f"🚀 Speedup Factor: {t_loop / t_vec:.1f}x Faster with Vectorization!")
```

#### Output:
```text
Python Loop Time:  0.1825 seconds
NumPy Vector Time: 0.0039 seconds
🚀 Speedup Factor: 46.8x Faster with Vectorization!
```

---

## 6. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Streaming File Moving Average
**Task:** Write a generator function `moving_average(generator_stream, window_size=3)` that yields the rolling average of numeric readings using a `collections.deque`:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
from collections import deque

def moving_average(stream, window_size=3):
    window = deque(maxlen=window_size)
    for val in stream:
        window.append(val)
        if len(window) == window_size:
            yield round(sum(window) / window_size, 2)

sensor_readings = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
averages = list(moving_average(sensor_readings, window_size=3))
print("Computed Rolling 3-Step Averages:", averages)
```
#### Output:
```text
Computed Rolling 3-Step Averages: [12.0, 14.0, 16.0, 18.0]
```
</details>

---

## 7. Quick Reference Cheat Sheet

| Tool | Module | Description | Typical Use Case |
|---|---|---|---|
| `yield` | Built-in | Generates lazy sequence | Streaming massive datasets |
| `Counter` | `collections` | Dictionary subclass for counts | Vocabulary building, frequency audits |
| `defaultdict`| `collections` | Auto-instantiates missing keys | Grouping records by category |
| `namedtuple` | `collections` | Tuple with named fields | Lightweight data rows |
| `deque` | `collections` | Double-ended queue with maxlen | Rolling/moving window buffers |
| `reduce` | `functools` | Cumulative binary reduction | Cumulative products, matrix chains |
