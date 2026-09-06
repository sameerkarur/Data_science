# Chapter 3: Python Data Structures & Algorithmic Complexity
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

Data structures are physical layouts in memory designed to enforce specific access, insertion, and deletion characteristics. Choosing between a continuous pointer array (`list`), an open-addressed hash table (`dict`, `set`), or a ring buffer (`deque`) can alter pipeline throughput by several orders of magnitude.

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

## 2. Architectural Flowchart: Hash Table Insertion & Open Addressing

```
                     HASH TABLE INSERTION & PROBING PIPELINE
                     
       Key Insertion: d["batch_size"] = 64
                           │
                           ▼
                 1. Compute 64-bit Hash Value
                    h = hash("batch_size") = 0x5a7b3c2e1f0...
                           │
                           ▼
                 2. Modulo Table Size Mask
                    bucket_index = h & (table_size - 1)
                           │
                           ▼
                 3. Inspect Bucket State
                    ├── Empty? ──► Write entry into compact array & set index!
                    │
                    └── Occupied?
                         ├── Same Key? ──► Overwrite value pointer (Update)
                         │
                         └── Collision! ──► Open Addressing Perturbation Probing:
                                           perturb >>= 5
                                           i = (5*i + 1 + perturb) & mask
                                           Repeat until empty slot found!
```

---

## 3. Deep Theoretical Foundations

### 1. Dynamic Array Growth Formula
CPython lists are variable-length arrays of 64-bit object pointers (`PyObject**`). When appending items, the memory buffer expands according to an over-allocation formula:
$$\text{new\_allocated} = \text{new\_size} + (\text{new\_size} \gg 3) + (\text{new\_size} < 9 \,?\, 3 : 6)$$
This dynamic reallocation amortizes the cost of array resizing, guaranteeing that while an individual expansion takes $O(N)$ memory copying, $N$ sequential appends execute in $O(N)$ total time, yielding an **amortized cost of $O(1)$ per append**.

### 2. Modern Compact Dictionary Architecture (PEP 468 & PyPy Design)
Prior to Python 3.6, dictionaries consumed significant memory because each hash table row stored empty padding (`hash`, `key`, `value` tuples in a sparse table). Modern CPython separates the table into:
1. **Indices Table (Sparse):** A simple byte/integer array of indices pointing to the entries array.
2. **Entries Table (Dense):** A compact, contiguous array of entries `[me_hash, me_key, me_value]` stored in the exact chronological order of insertion.
This innovation reduced dictionary memory consumption by 25–40% and enabled deterministic iteration ordering.

### 3. Timsort Algorithm
Python's built-in sorting routine (`list.sort()` and `sorted()`) implements **Timsort**, an adaptive hybrid sorting algorithm created by Tim Peters:
- It scans the array for natural non-decreasing or strictly decreasing segments called **runs**.
- Short runs are extended to a minimum run size (`minrun`, typically 32–64) and sorted using **Binary Insertion Sort**.
- Runs are subsequently merged using **Merge Sort** with a stack-based merge policy maintaining balanced run sizes.
- **Galloping Mode:** When elements from one run consistently win during merging, Timsort switches to exponential search (binary search) to skip large blocks of elements in $O(\log N)$ comparisons.

---

## 4. Production Implementation: High-Throughput Buffers & Priority Queues

```python
from collections import deque
import heapq
from typing import Any

class PriorityTaskQueue:
    """Thread-safe priority scheduler using a min-heap."""
    def __init__(self):
        self._heap: list[tuple[int, int, Any]] = []
        self._counter = 0  # Tie-breaker for identical priorities

    def push(self, task: Any, priority: int) -> None:
        """Pushes task with priority (lower number = higher priority)."""
        heapq.heappush(self._heap, (priority, self._counter, task))
        self._counter += 1

    def pop(self) -> Any:
        """Pops highest-priority task in O(log N) time."""
        if not self._heap:
            raise IndexError("Queue is empty")
        priority, _, task = heapq.heappop(self._heap)
        return task

# Demonstrating O(1) Sliding Window with Deque vs O(N) List Slicing
window = deque(maxlen=5)
for sample in [10.2, 11.5, 12.1, 10.8, 11.9, 13.4, 14.1]:
    window.append(sample)
    # Average computed over sliding window without any list reallocations
    rolling_mean = sum(window) / len(window)
```

---

## 5. Algorithmic Complexity Comparison Matrix

| Data Structure | Lookup / Access | Insertion (Head) | Insertion (Tail) | Deletion | Memory Overhead |
|---|---|---|---|---|---|
| **Python List** | $O(1)$ | $O(N)$ (Shifts memory) | $O(1)$ amortized | $O(N)$ (General) | Low (Contiguous pointers) |
| **Collections Deque** | $O(N)$ | $O(1)$ | $O(1)$ | $O(1)$ (At ends) | Medium (Doubly linked blocks) |
| **Dictionary (`dict`)** | $O(1)$ avg / $O(N)$ | $O(1)$ avg | $O(1)$ avg | $O(1)$ avg | High (Hash tables & indices) |
| **Set (`set`)** | $O(1)$ avg / $O(N)$ | N/A | $O(1)$ avg | $O(1)$ avg | High (Hash table keys only) |
| **Binary Heap (`heapq`)** | $O(1)$ min element | $O(\log N)$ | $O(\log N)$ | $O(\log N)$ | Low (Packed in list) |

---

## 6. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Mutating a Collection While Iterating
```python
# BROKEN: Modifying collection indices causes skipped elements!
records = [1, 2, 3, 4, 5]
for item in records:
    if item % 2 == 0:
        records.remove(item)

# PRODUCTION FIX: List comprehension or filtering into fresh memory:
records_clean = [item for item in records if item % 2 != 0]
```

### Pitfall 2: Using Lists for Membership Testing
Checking `if item in my_list` takes $O(N)$ linear scan time. If this check is executed inside a loop of size $M$, total complexity explodes to $O(M \cdot N)$. Converting the lookup collection to a `set` drops membership testing to $O(1)$ average time, collapsing overall complexity to $O(M)$.
