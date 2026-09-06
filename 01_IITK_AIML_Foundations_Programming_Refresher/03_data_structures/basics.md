# Python Data Structures & Algorithmic Complexity
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Python provides core data structures optimized for different memory layouts and algorithmic access patterns.

```
       DYNAMIC LIST (PyListObject)              HASH TABLE (PyDictObject)
    ┌───────────────────────────────┐        ┌───────────────────────────────┐
    | Array of Pointers (Contiguous)|        | Hash Table with Compact Array |
    | ┌─────┬─────┬─────┬─────┬───┐ |        | Hash Index Table ──► Indices  |
    | | *P1 | *P2 | *P3 | *P4 |...| |        | Entries Table:                |
    | └─────┴─────┴─────┴─────┴───┘ |        | [hash, *key_ptr, *val_ptr]    |
    └───────────────────────────────┘        └───────────────────────────────┘
          Index Access: O(1)                        Key Lookup: O(1) avg
          Insert at Head: O(n)                      Hash Collision: Open Addressing
```

---

## 🧭 Deep Theoretical Foundations

### 1. Python `list` Over-Allocation Algorithm
CPython lists are variable-length arrays of object pointers (`PyObject**`). When appending items, CPython grows the underlying array using an over-allocation formula:
$$	ext{new\_allocated} = 	ext{new\_size} + (	ext{new\_size} \gg 3) + (	ext{new\_size} < 9 \,?\, 3 : 6)$$
This guarantees that while individual reallocations cost $O(n)$, the **amortized cost per append operation is $O(1)$**.

### 2. Modern Hash Table Architecture (`dict` & `set`)
Since Python 3.6+, dictionaries are compact and preserve insertion order:
- **Indices Array:** Dense hash bucket array mapping hash modulus to entry indices.
- **Entries Array:** Contiguous array of entries stored in exact chronological insertion order: `[me_hash, me_key, me_value]`.
- **Collision Resolution:** Open addressing with perturbation-driven pseudo-random probing sequence:
  $$j = (5j + 1 + 	ext{perturb}) \pmod{2^k}$$

### 3. Timsort: Python's Sorting Engine
Python's `list.sort()` and `sorted()` implement Timsort, a hybrid stable sorting algorithm combining **Insertion Sort** (for small chunks or "runs", $n \le 64$) and **Merge Sort** with galloping mode optimization. It runs in $O(n)$ time on already-sorted or nearly-sorted data.

---

## 💻 Production Implementation: High-Performance Queues & Heaps

```python
from collections import deque
import heapq

# 1. Double-Ended Queue (deque): O(1) pops and appends from both ends
# (Unlike lists which require O(n) memory shifts for pop(0))
stream_buffer = deque(maxlen=5)
for i in range(10):
    stream_buffer.append(i)
print(f"Rolling Window Buffer: {list(stream_buffer)}")  # [5, 6, 7, 8, 9]

# 2. Min-Heap Priority Queue: O(log k) top-k selection
data_stream = [54, 12, 89, 43, 76, 23, 99, 1]
# Find top 3 largest elements in O(n log k) instead of O(n log n) full sort
top_3_largest = heapq.nlargest(3, data_stream)
print(f"Top 3 Values: {top_3_largest}")  # [99, 89, 76]
```

---

## 📐 Computational Complexity Matrix

| Data Structure | Lookup / Access | Insertion (Head) | Insertion (Tail) | Deletion |
|---|---|---|---|---|
| **Python List** | $O(1)$ | $O(n)$ (shift memory) | $O(1)$ amortized | $O(n)$ |
| **Collections Deque** | $O(n)$ | $O(1)$ | $O(1)$ | $O(1)$ (at ends) |
| **Dictionary (`dict`)** | $O(1)$ avg / $O(n)$ worst | $O(1)$ | $O(1)$ | $O(1)$ |
| **Set (`set`)** | $O(1)$ avg / $O(n)$ worst | N/A | $O(1)$ | $O(1)$ |
| **Heap (`heapq`)** | $O(1)$ min element | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Using List as a FIFO Queue:** Calling `list.pop(0)` takes $O(n)$ time because every subsequent element in memory must be shifted left by one slot. Always use `collections.deque.popleft()` for $O(1)$ performance.
2. **Hashing Mutable Objects:** Dictionaries and sets require keys to be hashable (immutable with consistent `__hash__` and `__eq__`). Attempting to use a `list` as a dict key raises `TypeError: unhashable type: 'list'`.
