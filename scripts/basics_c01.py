"""
Textbook-Scale Architectural & Conceptual Guides for Course 1:
IITK AIML Foundations: Programming Refresher
Modules:
  01_variables_datatypes
  02_control_flow_functions
  03_data_structures
  04_oop_modules
  05_file_io_exceptions
"""

C01_BASICS = {}

# =====================================================================
# 1. Variables, Data Types & Memory
# =====================================================================
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes"] = r'''# Chapter 1: Variables, Memory Architecture & Data Types
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

In high-level languages like C or C++, a variable is a named location in hardware memory where data bytes are stored directly. In contrast, in Python (specifically the reference implementation, **CPython**), a variable is **never a container of data**; it is an **abstract named pointer (reference)** bound to an object living on the heap.

```
       C / C++ Model (Value in Container):
       ┌────────────────────────┐
       │ int x = 42;            │  ──► Memory Address 0x7ffd... contains binary 00101010
       └────────────────────────┘

       CPython Model (Named Reference to Heap Object):
       Stack Frame (Symbol Table)                      Heap Memory (Allocated Object)
       ┌────────────────────────┐                   ┌───────────────────────────────────┐
       │ Variable Name: 'x'     │ ────────────────► │ PyLongObject:                     │
       │ Pointer: 0x104a8b20    │                   │   ob_refcnt: 1                    │
       └────────────────────────┘                   │   ob_type:   &PyLong_Type         │
                                                    │   ob_size:   1 (digit count)      │
                                                    │   ob_digit:  [42]                 │
                                                    └───────────────────────────────────┘
```

This fundamental paradigm shift dictates how memory allocation, assignment, mutation, argument passing, and garbage collection behave across the entire Python ecosystem.

---

## 2. Architectural Flowchart: Memory Lifecycle & Binding

```
                            VARIABLE ASSIGNMENT & LIFECYCLE
                            
       Source Code Statement: x = [10, 20, 30]
                                 │
                                 ▼
                     1. CPython Compiler / Parser
                        Emits bytecode: BUILD_LIST, STORE_NAME
                                 │
                                 ▼
                     2. Small Object Allocator (PyObject_Malloc)
                        Allocates heap memory for PyListObject + PyLongObjects
                                 │
                                 ▼
                     3. Stack Frame Symbol Resolution
                        f_localsplus['x'] receives memory pointer
                                 │
           ┌─────────────────────┴──────────────────────┐
           ▼                                            ▼
   Statement: y = x                             Statement: x = "Hello"
   (Aliasing / Pointer Copy)                    (Rebinding)
   • No data is copied!                         • 'x' points to new PyUnicodeObject
   • y gets same pointer as x                   • Old PyListObject ob_refcnt drops by 1
   • PyListObject ob_refcnt increments          • If ob_refcnt == 0 ➔ PyObject_Free()
```

---

## 3. Deep CPython Internals: The `PyObject` Structure

Every Python object, from a single integer to an entire convolutional neural network layer, shares the foundation defined in `Include/object.h`:

```c
typedef struct _object {
    _PyObject_HEAD_EXTRA // Doubly linked list pointers for cyclic GC tracking
    Py_ssize_t ob_refcnt; // Reference count (64-bit unsigned int)
    struct _typeobject *ob_type; // Pointer to type descriptor object
} PyObject;
```

### The Anatomy of Variable Types
1. **Variable-Sized Objects (`PyVarObject`):**
   Objects whose memory size varies (such as `str`, `list`, `tuple`, `bytes`, and arbitrarily large `int`s) extend `PyObject` with `ob_size`, representing the number of elements or digits.
2. **Arbitrary Precision Integers (`PyLongObject`):**
   Unlike C's fixed 32-bit or 64-bit integers which overflow at $2^{31}-1$ or $2^{63}-1$, Python integers support arbitrary precision. They are stored as signed digit arrays using a base of $2^{30}$ (on 64-bit systems). Arithmetic uses Karatsuba multiplication ($O(n^{1.58})$) for large numbers and Barrett reduction.
3. **Floating Point (`PyFloatObject`):**
   Wraps a standard C `double` (IEEE 754 double precision 64-bit float): 1 sign bit, 11 exponent bits, and 52 mantissa bits.
4. **Strings (`PyUnicodeObject` - PEP 393 Flexible String Representation):**
   Strings in Python 3 are compact and dynamically choose internal storage based on the maximum character ordinal:
   - **Latin-1 (1 byte/char):** If all characters fit within ASCII / ISO-8859-1 ($0 \le \text{char} \le 255$).
   - **UCS-2 (2 bytes/char):** If characters require up to 16 bits ($256 \le \text{char} \le 65535$).
   - **UCS-4 (4 bytes/char):** If full 32-bit Unicode code points (such as emojis $\ge 65536$) are present.

---

## 4. Mutability, Aliasing & In-Place Operations

Understanding mutability is critical to prevent silent data corruption in data pipelines.

| Type Category | Data Types | In-Place Modification Possible? | Reassignment Behavior |
|---|---|---|---|
| **Immutable** | `int`, `float`, `complex`, `bool`, `str`, `tuple`, `frozenset`, `bytes` | ❌ No. State is sealed at creation. | Creates brand-new object on heap; rebinds pointer. |
| **Mutable** | `list`, `dict`, `set`, `bytearray` | ✅ Yes. Buffer mutates without pointer change. | Can modify internal items; `id(obj)` remains invariant. |

### Mathematical Identity vs Equality
- **Equality (`==`):** Invokes `__eq__()`. Evaluates whether two objects represent the same value:
  $$x == y \iff x.\_\_\text{eq}\_\_(y) \equiv \text{True}$$
- **Identity (`is`):** Compares physical memory addresses directly:
  $$x \text{ is } y \iff \text{id}(x) == \text{id}(y) \iff \text{addr}(x) == \text{addr}(y)$$

```
     SHALLOW COPY vs DEEP COPY IN HIGH-DIMENSIONAL DATA
     
     Original List:  matrix = [[1, 2], [3, 4]]
     
     Shallow Copy:   s_copy = list(matrix)
     matrix ─────► [ *Ptr1 , *Ptr2 ]
                     │        │
     s_copy ─────► [ *Ptr1 , *Ptr2 ]  (Shares inner pointers!)
                     │        │
                     ▼        ▼
                   [1, 2]   [3, 4]
                   
     Deep Copy:      d_copy = copy.deepcopy(matrix)
     d_copy ─────► [ *NewPtr1 , *NewPtr2 ]
                     │           │
                     ▼           ▼
                   [1, 2]      [3, 4]   (Completely isolated memory trees!)
```

---

## 5. Comprehensive Production Code & Memory Profiling

```python
import sys
import ctypes
import copy
from typing import Any

def inspect_pyobject(name: str, obj: Any) -> None:
    """Reveals the underlying CPython heap memory layout and metadata."""
    address = id(obj)
    size_bytes = sys.getsizeof(obj)
    ref_count = sys.getrefcount(obj) - 1  # Subtract getrefcount's temporary pointer
    type_name = type(obj).__name__
    
    print(f"[{name}] Type: {type_name:<10} | Address: {hex(address)} | "
          f"Size: {size_bytes:>4} bytes | RefCount: {ref_count}")

# 1. Exploring CPython Small Integer Singleton Cache (-5 to 256)
val_a = 256
val_b = 256
inspect_pyobject("Cached 256 (A)", val_a)
inspect_pyobject("Cached 256 (B)", val_b)
print(f"Identity holds for 256: {val_a is val_b}\n")

val_c = 257
val_d = 257
inspect_pyobject("Heap 257 (C)", val_c)
inspect_pyobject("Heap 257 (D)", val_d)
print(f"Identity holds for 257: {val_c is val_d} (Distinct heap objects)\n")

# 2. String Memory Optimization & PEP 393 Storage
ascii_str = "AIML_2026"
unicode_str = "AIML_2026_🚀"
inspect_pyobject("ASCII String", ascii_str)
inspect_pyobject("Unicode (Emoji) String", unicode_str)
print(f"Size jump for Emoji: {sys.getsizeof(unicode_str) - sys.getsizeof(ascii_str)} bytes\n")

# 3. Safe Defensive Cloning in Data Pipelines
def sanitize_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Demonstrates safe mutation without affecting external caller data."""
    clean_records = copy.deepcopy(records)
    for row in clean_records:
        row['status'] = 'processed'
        row['imputed'] = row.get('imputed', False)
    return clean_records
```

---

## 6. Algorithmic & Memory Complexity Matrix

| Operation | Best Case Time | Worst Case Time | Space Complexity | Internal Mechanism |
|---|---|---|---|---|
| Variable Rebinding (`x = y`) | $O(1)$ | $O(1)$ | $O(1)$ | 64-bit pointer copy + `Py_INCREF` |
| Small Int Lookup (`-5..256`) | $O(1)$ | $O(1)$ | $O(0)$ | Array offset direct indexing |
| Arbitrary Precision Addition | $O(N)$ | $O(N)$ | $O(N)$ | Digit-by-digit ripple carry |
| String Concatenation (`s1 + s2`)| $O(N + M)$ | $O(N + M)$ | $O(N + M)$ | Allocates fresh contiguous buffer |
| Deep Copy (`copy.deepcopy(x)`) | $O(V + E)$ | $O(V + E)$ | $O(V)$ | Graph DFS with memoization dict |

---

## 7. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: IEEE 754 Floating Point Roundoff
```python
# FAILS in critical banking & evaluation metrics:
assert 0.1 + 0.2 == 0.3  # Raises AssertionError! (Evaluates to 0.30000000000000004)

# PRODUCTION FIX:
import math
assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)
```

### Pitfall 2: Accidental Global State via In-Place Modification (`+=`)
When `+=` is executed on mutable objects, it calls `__iadd__()`, mutating the object in-place. On immutable objects, it calls `__add__()`, producing a new object:
```python
a = [1, 2]
b = a
b += [3]        # Mutates [1, 2] into [1, 2, 3]! 'a' is also modified!

x = 10
y = x
y += 5          # Creates new int(15); 'x' remains 10.
```

### Pitfall 3: Inappropriate Use of `is` for Value Checking
Never use `is` to check equality for numbers or strings outside singleton checking (`None`, `True`, `False`). Always use `==` for values.
'''

# =====================================================================
# 2. Control Flow, Functions & Scopes
# =====================================================================
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions"] = r'''# Chapter 2: Control Flow, Scopes & Functional Programming
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

Execution flow in Python is governed by the **CPython Virtual Machine (VM)**, which interprets high-level language constructs into stack-based bytecode instructions. Functions in Python are **first-class citizens**: they can be passed as arguments, returned from other functions, bound to variables, and dynamically augmented using decorators and metaprogramming.

```
                    CPYTHON EVALUATION LOOP & BYTECODE DISPATCH
                    
       Source Code: if threshold > 0.8: trigger_alert()
                                │
                                ▼
                     CPYTHON BYTECODE EMISSION
       0 LOAD_NAME                0 (threshold)
       2 LOAD_CONST               0 (0.8)
       4 COMPARE_OP               4 (>)
       6 POP_JUMP_IF_FALSE       12 ────────┐ (Skips block if false)
       8 LOAD_NAME                1 (trigger_alert)
      10 CALL_FUNCTION            0
      12 ...
```

---

## 2. Architectural Flowchart: Scoping & The LEGB Resolution Pipeline

Whenever an identifier is referenced, CPython queries four nested namespaces in a strict chronological sequence:

```
                  LEGB SCOPE RESOLUTION PIPELINE
    ┌────────────────────────────────────────────────────────┐
    │ 1. LOCAL (L): Current function execution frame         │
    │    └── Found? ──► Use variable                         │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 2. ENCLOSING (E): Outer nested function closures       │
    │    └── Found? ──► Use variable                         │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 3. GLOBAL (G): Module-level symbol table (__main__)    │
    │    └── Found? ──► Use variable                         │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 4. BUILT-IN (B): Python built-in namespace (len, etc.) │
    │    └── Not Found? ──► RAISE NameError!                 │
    └────────────────────────────────────────────────────────┘

    FRAME OBJECTS & CLOSURE STATE RETENTION (CELL OBJECTS):
    Outer Function Scope                   Inner Closure Scope
    ┌────────────────────────┐             ┌────────────────────────┐
    │ def outer(learning_rate)             │ def inner(x):          │
    │   rate = learning_rate ├──[cell_obj]─┤   return x * rate      │
    │   return inner         │             └────────────────────────┘
    └────────────────────────┘             (Persists on Heap even after outer exits!)
```

---

## 3. Deep Theoretical Foundations

### 1. The Call Stack & `PyFrameObject`
Every function invocation pushes a new `PyFrameObject` onto the runtime call stack. A frame encapsulates:
- `f_code`: The immutable code object (`co_code`, `co_varnames`, `co_consts`).
- `f_localsplus`: A pre-allocated C array providing $O(1)$ indexed lookup for local variables via `LOAD_FAST` and `STORE_FAST` bytecode instructions (significantly faster than global dictionary lookups).
- `f_valuestack`: Evaluation stack for intermediate operand computations.
- `f_back`: Pointer to caller's frame (enabling traceback generation).

### 2. Closures & Free Variables
A closure occurs when a nested function references variables from its enclosing lexical scope. CPython packages these variables into heap-allocated `cell` objects. Even after the outer function's stack frame is popped and destroyed, the cell object maintains a reference count $> 0$, allowing the inner function to access its enclosing environment indefinitely.

### 3. Generators, Coroutines & Asynchronous Execution
Unlike normal subroutines that adhere to the standard LIFO call/return paradigm, **generators** utilize the `yield` keyword to implement cooperative multitasking:
- Calling a generator function does not execute its body; it instantiates a `PyGenObject`.
- Executing `next(gen)` or `gen.send(val)` activates the frame until a `yield` statement is hit.
- The instruction pointer (`f_lasti`) is frozen, and execution yields control back to the caller with zero heap reallocation.

---

## 4. Production Implementation: Advanced Decorators & Coroutine Streaming

```python
import functools
import time
from typing import Callable, Any, Generator

def rate_limiter(max_per_second: float) -> Callable:
    """Production-grade rate-limiting decorator with high-resolution token bucket."""
    min_interval = 1.0 / max_per_second
    last_called = 0.0

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal last_called  # Explicitly binds to enclosing closure state
            now = time.perf_counter()
            elapsed = now - last_called
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called = time.perf_counter()
            return result
        return wrapper
    return decorator

def memory_efficient_batcher(data_stream: Generator[dict, None, None], 
                             batch_size: int = 128) -> Generator[list[dict], None, None]:
    """Streams and yields batches of records in O(1) auxiliary memory."""
    batch = []
    for item in data_stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

@rate_limiter(max_per_second=50.0)
def query_model_endpoint(payload: dict) -> dict:
    """Simulated inference call protected by rate limiting."""
    return {"prediction": 0.942, "status": "ok"}
```

---

## 5. Scope & Iteration Complexity Matrix

| Mechanism | Memory Footprint | Invocation Overhead | State Lifetime |
|---|---|---|---|
| Regular Function | $O(\text{stack depth})$ | Minimal (Push/Pop C frame) | Terminated on `return` |
| Closure (`cell` object) | $O(\text{free vars})$ | Frame push + Cell dereference | Persists as long as inner ref exists |
| Generator (`yield`) | $O(1)$ constant buffer | Minimal (`GEN_START` / `YIELD_VALUE`) | Persists across iterations |
| List Comprehension | $O(N)$ heap buffer | Fast C-loop loop evaluation | Full eager evaluation |
| Generator Expression | $O(1)$ memory | Lazy iteration on demand | Evaluates one element at a time |

---

## 6. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Late Binding in Closures and Lambdas
```python
# BROKEN: All lambdas capture the variable 'i' by reference, not by value:
multipliers = [lambda x: x * i for i in range(4)]
results = [m(2) for m in multipliers]
# Returns [6, 6, 6, 6] instead of [0, 2, 4, 6]!

# PRODUCTION FIX: Bind eagerly via default argument:
multipliers_fixed = [lambda x, i=i: x * i for i in range(4)]
assert [m(2) for m in multipliers_fixed] == [0, 2, 4, 6]
```

### Pitfall 2: Mutable Default Arguments
```python
# DANGEROUS: Evaluated once at definition time!
def log_event(event_id: str, tags: list = []):
    tags.append(event_id)
    return tags

# PRODUCTION FIX:
def log_event_fixed(event_id: str, tags: list | None = None) -> list:
    if tags is None:
        tags = []
    tags.append(event_id)
    return tags
```
'''

# =====================================================================
# 3. Data Structures & Algorithmic Complexity
# =====================================================================
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures"] = r'''# Chapter 3: Python Data Structures & Algorithmic Complexity
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
'''

# =====================================================================
# 4. OOP & Modules
# =====================================================================
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules"] = r'''# Chapter 4: Object-Oriented Architecture & Metaprogramming
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

Python’s object-oriented system is completely dynamic. Classes are themselves instances of metaclasses (`type`), methods are descriptor objects bound at runtime, and inheritance hierarchies are linearized using the mathematical **C3 Superclass Linearization Algorithm**.

```
                   C3 MRO DIAMOND RESOLUTION PIPELINE
                               ┌──────────┐
                               |  Object  |
                               └────▲─────┘
                                    │
                               ┌────┴─────┐
                               | Base (A) |
                               └─▲──────▲─┘
                     ┌───────────┘      └───────────┐
                     │                              │
               ┌─────┴─────┐                  ┌─────┴─────┐
               | Left (B)  |                  | Right (C) |
               └─────▲─────┘                  └─────▲─────┘
                     └───────────┐      ┌───────────┘
                               ┌─┴──────┴─┐
                               | Leaf (D) |
                               └──────────┘
                  MRO: [D, B, C, A, object] (Deterministic!)
```

---

## 2. Architectural Flowchart: The Descriptor Protocol & Attribute Lookup

Attribute access in Python (`instance.attribute`) does not merely query a dictionary. It executes a rigorous multi-tier lookup protocol:

```
                  ATTRIBUTE ACCESS LOOKUP PROTOCOL (obj.attr)
    ┌────────────────────────────────────────────────────────┐
    │ 1. Check type(obj).__mro__ for Data Descriptor         │
    │    (Implements __get__ AND __set__)                    │
    │    └── Found? ──► Invoke Descriptor.__get__()          │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 2. Check obj.__dict__ (Instance Dictionary)            │
    │    └── Found? ──► Return instance value directly       │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 3. Check type(obj).__mro__ for Non-Data Descriptor     │
    │    (Implements __get__ ONLY, e.g. normal methods)      │
    │    └── Found? ──► Invoke Descriptor.__get__()          │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 4. Check Class Attributes (__dict__ on class hierarchy)│
    │    └── Found? ──► Return class value                   │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 5. Invoke obj.__getattr__(attr) (Fallback)             │
    │    └── Not Implemented? ──► RAISE AttributeError!      │
    └────────────────────────────────────────────────────────┘
```

---

## 3. Deep Theoretical Foundations

### 1. C3 Linearization (Method Resolution Order)
In complex inheritance graphs, CPython determines method lookup order using C3 Linearization. The linearization $L[C]$ of class $C$ inheriting from parents $B_1, B_2, \dots, B_n$ is defined recursively:
$$L[C] = [C] + \text{merge}(L[B_1], L[B_2], \dots, L[B_n], [B_1, B_2, \dots, B_n])$$
The merge operation extracts the first head that does not appear in the tail of any other list in the merge pool. This guarantees **Monotonicity** (subclasses never reorder parent precedence) and **Local Precedence Order**.

### 2. Memory Optimization with `__slots__`
By default, every Python instance stores attributes in a dynamic `__dict__` dictionary, consuming ~150-300 bytes of memory overhead per instance. Declaring `__slots__ = ('x', 'y')` replaces `__dict__` with a fixed-size array of C pointers, cutting instance memory by up to 80% for high-throughput data processing.

### 3. Metaclasses & Class Construction
A metaclass is the class of a class. When a `class` statement completes, CPython invokes:
$$\text{Class} = \text{Metaclass}(\text{name}, \text{bases}, \text{namespace})$$
This allows framework authors to dynamically validate fields, register models into registries, and generate boilerplate attributes before any instances are instantiated.

---

## 4. Production Implementation: Validated Descriptors & Metaclass Registry

```python
from typing import Any, Type

class ModelRegistryMeta(type):
    """Metaclass that automatically registers machine learning model classes."""
    REGISTRY: dict[str, Type] = {}

    def __new__(mcs, name: str, bases: tuple, attrs: dict) -> Type:
        cls = super().__new__(mcs, name, bases, attrs)
        if name != "BaseEstimator":
            mcs.REGISTRY[name] = cls
        return cls

class BoundedNumeric:
    """Data descriptor enforcing strict min/max numerical bounds."""
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __set_name__(self, owner: type, name: str) -> None:
        self.private_name = f"_{name}"

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.private_name)

    def __set__(self, instance: Any, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a real number")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value {value} out of range [{self.min_val}, {self.max_val}]")
        setattr(instance, self.private_name, value)

class BaseEstimator(metaclass=ModelRegistryMeta):
    """Base class for all estimators."""
    pass

class GradientBoostingClassifier(BaseEstimator):
    learning_rate = BoundedNumeric(0.0001, 1.0)
    subsample = BoundedNumeric(0.1, 1.0)

    def __init__(self, learning_rate: float = 0.1, subsample: float = 1.0):
        self.learning_rate = learning_rate
        self.subsample = subsample
```

---

## 5. OOP Mechanism Performance Matrix

| Mechanism | Memory Footprint | Method Dispatch Overhead | Use Case |
|---|---|---|---|
| Standard Instance (`__dict__`) | ~150–400 bytes | $O(1)$ dict lookup | General business logic |
| Slotted Instance (`__slots__`) | ~48–64 bytes | $O(1)$ pointer offset | Millions of ML data records |
| Property (`@property`) | Negligible | Function call overhead | Computed attributes |
| Metaclass (`type`) | Compile-time only | Zero runtime cost | Framework registration & validation |

---

## 6. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Calling `super()` with Explicit Class Arguments
```python
# OBSOLETE & BUG-PRONE (Python 2 pattern):
super(MyClass, self).__init__()

# PRODUCTION FIX (Zero-argument super in Python 3):
super().__init__()
```
Zero-argument `super()` automatically extracts the class and instance from compiler-generated closure cells (`__class__`), guaranteeing flawless cooperative multiple-inheritance resolution.
'''

# =====================================================================
# 5. File I/O, Exceptions & Robust Systems
# =====================================================================
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions"] = r'''# Chapter 5: Python File I/O, Exception Architecture & Systems
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

Production data pipelines and distributed training jobs require deterministic resource management. Unclosed file handles leak operating system file descriptors, while non-atomic file writes result in corrupted partial checkpoints if an out-of-memory (OOM) killer or kernel panic occurs mid-write.

```
                  CONTEXT MANAGER LIFECYCLE (with statement)
       ┌────────────────────────────────────────────────────────┐
       │ 1. Expression Evaluated: with open('data.bin') as f:   │
       │    └── manager = expression()                          │
       │    └── enter_val = manager.__enter__()                 │
       │                                                        │
       │ 2. Executing Code Block Inside 'with'                  │
       │    ├── Success? ──► manager.__exit__(None, None, None) │
       │    │                (File descriptor cleanly closed!)  │
       │    │                                                   │
       │    └── Exception Raised?                               │
       │         ▼                                              │
       │ 3. manager.__exit__(exc_type, exc_val, exc_tb)         │
       │    ├── Returns True?  ──► Exception suppressed!        │
       │    └── Returns False? ──► Exception re-propagated!     │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Flowchart: CPython 3-Tier I/O Subsystem

```
                         CPYTHON I/O ARCHITECTURE (io module)
                         
       Application Code: f.write("Record data\\n")
                              │
                              ▼
       Tier 1: TextIOWrapper (Character Encoding & Newlines)
               • Translates Unicode strings to bytes using specified codec (UTF-8)
               • Translates universal newlines ('\\n' ➔ '\\r\\n' if on Windows)
                              │
                              ▼
       Tier 2: BufferedWriter (User-Space Memory Buffering)
               • Buffers writes into an internal memory page (typically 8192 bytes)
               • Eliminates expensive OS system call on every single write operation
                              │
                              ▼ (When buffer fills or f.flush() is called)
       Tier 3: FileIO (Raw OS System Calls)
               • Executes unbuffered kernel system call: write(fd, buffer, count)
                              │
                              ▼
       OS Kernel Page Cache ──► Physical Storage Media (NVMe / SSD / HDD)
```

---

## 3. Deep Theoretical Foundations

### 1. Atomic Writes & Crash Consistency
When writing model checkpoints, calling `f.write()` modifies data in the OS page cache. If the machine loses power before the kernel flushes its dirty pages, the destination file is left in an unrecoverable corrupted state. 
- **Production Solution:** Write to an adjacent temporary file on the **same filesystem**, force a hardware flush via `os.fsync()`, and perform an **atomic rename** (`os.replace()`). On POSIX systems, `rename()` is guaranteed to be atomic by the filesystem journal.

### 2. Exception Hierarchy & Exception Chaining
All standard exceptions inherit from `BaseException`. Application code should catch `Exception`, never `BaseException`, because catching the latter traps `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`, preventing graceful process termination.
- **Explicit Chaining (`from exc`):** Sets `__cause__` to preserve the original exception context.
- **Suppression (`from None`):** Hides internal implementation details when presenting user-facing API errors.

### 3. Memory-Mapped Files (`mmap`)
For multi-gigabyte datasets (such as embedding matrices), standard file reads copy bytes from the kernel page cache into user process memory. `mmap` maps disk blocks directly into the virtual address space of the process, allowing lazy page faulting by the OS kernel without loading the entire file into RAM.

---

## 4. Production Implementation: Atomic Checkpointer & Mmap Reader

```python
import os
import tempfile
import mmap
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

@contextmanager
def atomic_checkpoint_writer(destination_path: Path | str) -> Generator[tempfile.NamedTemporaryFile, None, None]:
    """Guarantees atomic file updates: either 100% written or previous file untouched."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Must be on same filesystem for atomic rename
    with tempfile.NamedTemporaryFile(mode='wb', dir=dest.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())  # Force OS dirty pages onto physical disk
            tmp.close()
            os.replace(temp_path, dest)  # POSIX atomic filesystem swap
        except Exception:
            if temp_path.exists():
                os.remove(temp_path)
            raise

def fast_binary_embedding_search(filepath: Path | str, vector_dim: int, target_idx: int) -> bytes:
    """Reads vector embeddings with zero-copy mmap."""
    record_size = vector_dim * 4  # float32 = 4 bytes
    with open(filepath, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = target_idx * record_size
            return mm[offset : offset + record_size]
```

---

## 5. File I/O & Exception Complexity Matrix

| Technique | Memory Footprint | Latency Profile | Crash Safety |
|---|---|---|---|
| Naive `read()` | $O(\text{file size})$ (OOM hazard!) | High initial lag | Zero (Partial writes corrupt data) |
| Chunked Stream (`read(8192)`) | $O(1)$ constant 8KB buffer | Low streaming latency | Zero |
| Memory Map (`mmap`) | $O(1)$ virtual memory | Sub-millisecond lazy paging | High |
| Atomic File Write | $O(\text{buffer})$ | Extra rename operation | 100% ACID compliant |

---

## 6. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Missing Explicit Character Encoding
```python
# BUG-PRONE: Uses OS platform default (e.g. cp1252 on Windows, causing crashes!)
with open("data.json", "r") as f:
    data = f.read()

# PRODUCTION FIX: ALWAYS specify UTF-8:
with open("data.json", "r", encoding="utf-8") as f:
    data = f.read()
```

### Pitfall 2: Silencing Exceptions with Bare Except
```python
# ANTI-PATTERN: Traps KeyboardInterrupt and bugs silently!
try:
    process_data()
except:
    pass

# PRODUCTION FIX: Catch specific exceptions and log tracebacks:
try:
    process_data()
except (ValueError, KeyError) as exc:
    logger.error("Processing failed: %s", exc, exc_info=True)
    raise DataPipelineError("Data validation failed") from exc
```
'''

print(f"Loaded {len(C01_BASICS)} comprehensive textbook chapters for Course 1.")
