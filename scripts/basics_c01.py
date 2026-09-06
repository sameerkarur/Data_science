"""
Comprehensive, high-depth Basics & Architecture Guides for Course 1:
01_variables_datatypes
02_control_flow_functions
03_data_structures
04_oop_modules
05_file_io_exceptions
"""

C01_BASICS = {}

# -------------------------------------------------------------
# 1. Variables, Data Types & Memory
# -------------------------------------------------------------
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes"] = """# Python Variables, Memory Architecture & Data Types
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

In CPython, variables are **named references (pointers)** bound to heap-allocated objects. Every Python object is represented as a `PyObject` structure containing metadata (reference count and type pointer) alongside the actual data payload.

```
       VARIABLE BINDING (STACK)                 HEAP MEMORY ALLOCATION
     ┌────────────────────────┐              ┌─────────────────────────────┐
     |  user_id = 42          | ───────────► | PyLongObject:               |
     |  (Symbol Table Entry)  |              |   ob_refcnt = 1             |
     └────────────────────────┘              |   ob_type   = &PyLong_Type  |
                                             |   ob_digit  = 42            |
                                             └─────────────────────────────┘
                                                            ▲
     ┌────────────────────────┐                             |
     |  admin_id = user_id    | ────────────────────────────┘ (Shared Reference)
     |  (Alias Pointer)       |
     └────────────────────────┘

     MUTABILITY & REASSIGNMENT PIPELINE:
     [Original Var: x = 100] ─── (x += 1) ───► Creates NEW PyLongObject(101)
                                                 Original 100 refcnt decrements!
```

---

## 🧭 Deep Theoretical Foundations

### 1. The CPython Object Model (`PyObject`)
Every Python entity is a heap-allocated object. At the C level, all objects share the `PyObject_HEAD` macro:
- `ob_refcnt`: 64-bit integer tracking active references. When this drops to 0, memory is immediately reclaimed by Python's small-object allocator (`PyObject_Free`).
- `ob_type`: Pointer to the object's type descriptor (`PyTypeObject`), determining behavior, method tables, and memory size.

### 2. Mutability vs Immutability Mechanics
- **Immutable Types (`int`, `float`, `bool`, `str`, `tuple`, `frozenset`, `bytes`):** Once allocated in memory, their state cannot be altered in-place. Any modification creates a new object in memory and updates the pointer.
- **Mutable Types (`list`, `dict`, `set`, `bytearray`):** The internal data buffer can be mutated without changing the object's memory address (`id(obj)` remains constant).

### 3. Small Integer Caching & String Interning
- **Small Integer Array:** CPython pre-allocates an array of all integer objects between `-5` and `256` at runtime initialization. Any variable assigned in this range points to the identical singleton memory address.
- **String Interning:** Strings matching identifier naming rules (ASCII alphanumeric + underscores) are interned in an internal dictionary to allow instantaneous pointer comparisons (`is`) rather than character-by-character $O(n)$ equality checks.

---

## 💻 Production Implementation & Memory Inspection

```python
import sys
import ctypes
import copy

# 1. Inspecting Object Identity & Memory Footprint
value_a = 256
value_b = 256
print(f"Singleton Identity (256): {value_a is value_b}")  # True (Cached)

large_a = 10000
large_b = 10000
print(f"Distinct Heap Allocations: {large_a is large_b}")  # False

# 2. Shallow vs Deep Copying in Complex Structures
original_matrix = [[1, 2, 3], [4, 5, 6]]
shallow_matrix = copy.copy(original_matrix)
deep_matrix = copy.deepcopy(original_matrix)

# Modifying inner child element
original_matrix[0][0] = 999
print(f"Shallow Copy Affected: {shallow_matrix[0][0] == 999}")  # True! Shares references
print(f"Deep Copy Isolated:     {deep_matrix[0][0] == 1}")     # True! Independent tree
```

---

## 📐 Computational & Memory Complexity Matrix

| Operation | Time Complexity | Space Complexity | CPython Internal Mechanism |
|---|---|---|---|
| Variable Assignment (`x = y`) | $O(1)$ | $O(1)$ | Pointer copy + `Py_INCREF` |
| Small Int Lookup (`-5` to `256`) | $O(1)$ | $O(0)$ | Direct array offset index |
| String Concatenation (`s1 + s2`) | $O(n + m)$ | $O(n + m)$ | New buffer allocation |
| Deep Copy (`copy.deepcopy(x)`) | $O(V + E)$ | $O(V)$ | Memoization dict graph traversal |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Floating Point Precision Trap:** Never compare floating point numbers with `==` due to IEEE 754 binary representation limits (`0.1 + 0.2 == 0.3` evaluates to `False`). Always use `math.isclose(a, b, rel_tol=1e-9)`.
2. **Mutable Default Argument Anti-Pattern:** Defining `def append_to(item, target_list=[])` creates a single list at function definition time shared across all calls. Use `target_list=None` and initialize inside.
"""

# -------------------------------------------------------------
# 2. Control Flow, Functions & Scopes
# -------------------------------------------------------------
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions"] = """# Python Control Flow, Scopes & Functional Programming
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Python function execution relies on the **Call Stack** and **Frame Objects** (`PyFrameObject`), resolving symbols through the **LEGB (Local ➔ Enclosing ➔ Global ➔ Built-in)** scoping hierarchy.

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
    │ 3. GLOBAL (G): Module-level symbol table               │
    │    └── Found? ──► Use variable                         │
    │    └── Not Found?                                      │
    │         ▼                                              │
    │ 4. BUILT-IN (B): Python built-in namespace (len, etc.) │
    │    └── Not Found? ──► RAISE NameError!                 │
    └────────────────────────────────────────────────────────┘

    GENERATOR EXECUTION & SUSPENSION (yield):
    Caller Function ──[next()]──► Generator Frame Activates (RESUME)
                                           │
                                    Executes Logic
                                           │
    Caller Receives Value ◄──────[yield value] (FREEZE FRAME STATE)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Frame Objects & Execution Lifecycles
When a function is invoked, CPython creates a `PyFrameObject` containing:
- `f_locals`: Fast array access (`FAST_LOCAL`) for local variables resolved at compile time.
- `f_globals`: Dictionary lookup for module-level variables.
- `f_builtins`: Dictionary lookup for Python builtins.
- Evaluation stack: Virtual machine stack for executing bytecode instructions (`LOAD_FAST`, `BINARY_ADD`).

### 2. Closures & Free Variables
A closure occurs when an inner function retains access to variables in its enclosing scope even after the outer function has completed execution and exited the call stack. The free variables are packaged inside the inner function's `__closure__` tuple as `cell` objects.

### 3. Generator Mechanics & Cooperative Multitasking
Unlike standard functions that execute from start to finish and destroy their frame on `return`, generators use the `yield` expression to suspend execution, preserving the exact instruction pointer (`f_lasti`) and local variable state. This allows processing massive data streams in $O(1)$ auxiliary memory.

---

## 💻 Production Implementation: Parametric Decorator with Wraps

```python
import functools
import time
from typing import Callable, Any

def retry(max_attempts: int = 3, delay_sec: float = 1.0) -> Callable:
    \"\"\"Production-grade retry decorator with exponential backoff and metadata preservation.\"\"\"
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # Preserves __name__, __doc__, and type signatures
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            current_delay = delay_sec
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts") from exc
                    time.sleep(current_delay)
                    current_delay *= 2.0  # Exponential backoff
        return wrapper
    return decorator

@retry(max_attempts=3, delay_sec=0.1)
def fetch_api_payload(endpoint: str) -> dict:
    \"\"\"Simulates external API call.\"\"\"
    return {"status": "success", "endpoint": endpoint}
```

---

## 📐 Scope & Iteration Complexity Matrix

| Construct | Space Overhead | Time Complexity | State Retention |
|---|---|---|---|
| Regular Function Call | $O(\text{stack depth})$ | $O(1)$ setup | Frame destroyed on return |
| Closure (`cell` object) | $O(\text{free vars})$ | $O(1)$ lookup | Variables stored on heap |
| Generator Function | $O(1)$ memory | $O(1)$ per `next()` | Frame preserved until exhausted |
| List Comprehension | $O(n)$ heap | $O(n)$ eager | Evaluated fully in memory |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Late Binding in Closures:** In loops like `funcs = [lambda: i for i in range(5)]`, all lambdas evaluate `i` at invocation time (all return 4). Fix using default arguments: `lambda i=i: i`.
2. **Modifying Enclosing Variables Without `nonlocal`:** Reassigning an outer variable inside a nested function creates a local variable instead of mutating the outer one unless explicitly declared `nonlocal`.
"""

# -------------------------------------------------------------
# 3. Data Structures & Algorithms
# -------------------------------------------------------------
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures"] = """# Python Data Structures & Algorithmic Complexity
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
$$\text{new\_allocated} = \text{new\_size} + (\text{new\_size} \gg 3) + (\text{new\_size} < 9 \,?\, 3 : 6)$$
This guarantees that while individual reallocations cost $O(n)$, the **amortized cost per append operation is $O(1)$**.

### 2. Modern Hash Table Architecture (`dict` & `set`)
Since Python 3.6+, dictionaries are compact and preserve insertion order:
- **Indices Array:** Dense hash bucket array mapping hash modulus to entry indices.
- **Entries Array:** Contiguous array of entries stored in exact chronological insertion order: `[me_hash, me_key, me_value]`.
- **Collision Resolution:** Open addressing with perturbation-driven pseudo-random probing sequence:
  $$j = (5j + 1 + \text{perturb}) \pmod{2^k}$$

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
"""

# -------------------------------------------------------------
# 4. OOP & Modules
# -------------------------------------------------------------
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules"] = """# Python Object-Oriented Architecture & Metaprogramming
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Python's object-oriented system is dynamic, driven by the **C3 Superclass Linearization Algorithm** (Method Resolution Order - MRO), the **Descriptor Protocol**, and **Metaclasses**.

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

## 🧭 Deep Theoretical Foundations

### 1. C3 Linearization (Method Resolution Order)
In complex multi-inheritance graphs, CPython determines method lookup order using C3 Linearization. It guarantees:
- **Local Precedence:** Subclasses precede their parents.
- **Monotonicity:** Parent ordering is preserved across all inheritance branches.
Check order dynamically via `Class.__mro__`.

### 2. The Descriptor Protocol
Descriptors power `@property`, `@classmethod`, `@staticmethod`, and ORM fields. Any object implementing at least one of these dunder methods is a descriptor:
- `__get__(self, instance, owner)`
- `__set__(self, instance, value)`
- `__delete__(self, instance)`
Data descriptors (`__set__` implemented) take precedence over instance dictionary (`instance.__dict__`) lookups.

### 3. Memory Optimization with `__slots__`
By default, every Python instance stores attributes in a dynamic `__dict__` dictionary, consuming ~150-300 bytes of memory overhead per instance. Declaring `__slots__ = ('x', 'y')` replaces `__dict__` with a fixed-size array of C pointers, cutting instance memory by up to 80% for high-throughput data processing.

---

## 💻 Production Implementation: Data Validation Descriptor

```python
class ValidatedAttribute:
    \"\"\"Reusable descriptor enforcing positive numeric types.\"\"\"
    def __set_name__(self, owner: type, name: str) -> None:
        self.private_name = f"_{name}"

    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)

    def __set__(self, instance: Any, value: Any) -> None:
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Value must be a non-negative number, got: {value}")
        setattr(instance, self.private_name, value)

class MLModelConfig:
    __slots__ = ('_learning_rate', '_batch_size')  # Memory optimized!
    learning_rate = ValidatedAttribute()
    batch_size = ValidatedAttribute()

    def __init__(self, lr: float, batch_size: int):
        self.learning_rate = lr
        self.batch_size = batch_size
```

---

## 📐 OOP Mechanism Complexity Matrix

| Mechanism | Memory Footprint | Method Dispatch Overhead | Use Case |
|---|---|---|---|
| Standard Instance (`__dict__`) | ~150–400 bytes | $O(1)$ dict lookup | General business logic |
| Slotted Instance (`__slots__`) | ~48–64 bytes | $O(1)$ pointer offset | Millions of ML data records |
| Property (`@property`) | Negligible | Function call overhead | Computed attributes |
| Metaclass (`type`) | Compile-time only | Zero runtime cost | Framework registration & validation |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Calling `super()` Incorrectly:** Calling `Base.__init__(self)` explicitly in a multiple-inheritance hierarchy breaks the cooperative MRO chain and can result in duplicate parent calls or skipped classes. Always use `super().__init__(*args, **kwargs)`.
2. **Circular Import Trap:** Importing module A inside B and B inside A at the top level causes `ImportError: cannot import name`. Resolve via local imports inside functions or dependency injection.
"""

# -------------------------------------------------------------
# 5. File I/O, Exceptions & Robust Systems
# -------------------------------------------------------------
C01_BASICS["01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions"] = """# Python File I/O, Exception Architecture & Systems
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

Production data pipelines require deterministic resource management and exception handling to prevent file descriptor leaks and data corruption.

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

## 🧭 Deep Theoretical Foundations

### 1. OS File Descriptors & Buffering
When opening a file, the OS kernel allocates a file descriptor (integer index in the process table). Python's I/O library implements a three-tier architecture:
- `RawIOBase`: Direct, unbuffered OS system calls (`read`, `write`).
- `BufferedIOBase`: In-memory ring buffer (typically 8KB chunks) reducing expensive kernel context switches.
- `TextIOWrapper`: Handles encoding/decoding (e.g. UTF-8) and newline translations (`\\r\\n` to `\\n`).

### 2. Memory-Mapped Files (`mmap`)
For massive binary datasets (e.g., embeddings or gigabyte-scale arrays), standard `file.read()` copies data from the kernel disk cache to process user memory. `mmap` maps file pages directly into the process's virtual address space, enabling lazy OS-level paging without RAM saturation.

### 3. Exception Chaining & `traceback`
Python 3 tracks causal relationships between exceptions using:
- Explicit Chaining: `raise CustomError("Failure") from original_exc` (sets `__cause__`).
- Implicit Chaining: If an exception occurs inside an `except` block, Python automatically sets `__context__`.

---

## 💻 Production Implementation: Atomic File Writer

```python
import os
import tempfile
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

@contextmanager
def atomic_write(filepath: Path | str, mode: str = 'w', encoding: str = 'utf-8') -> Generator:
    \"\"\"Guarantees that a file is either completely written or untouched on crash/failure.\"\"\"
    dest_path = Path(filepath)
    temp_dir = dest_path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary file in same filesystem to enable atomic rename
    with tempfile.NamedTemporaryFile(mode=mode, dir=temp_dir, delete=False, encoding=encoding) as tmp_file:
        temp_name = tmp_file.name
        try:
            yield tmp_file
            tmp_file.flush()
            os.fsync(tmp_file.fileno())  # Force OS write to disk platter/SSD
            # Atomic OS-level filesystem rename
            os.replace(temp_name, dest_path)
        except Exception:
            if os.path.exists(temp_name):
                os.remove(temp_name)
            raise
```

---

## 📐 File I/O & Exception Complexity Matrix

| Technique | Memory Footprint | Latency Profile | Fault Tolerance |
|---|---|---|---|
| Naive `read()` | $O(\text{file size})$ (Dangerous!) | High initial lag | Low (crashes on OOM) |
| Chunked Iteration (`read(8192)`) | $O(1)$ constant buffer | Low streaming latency | High |
| Memory Map (`mmap`) | $O(1)$ virtual memory | Near-zero (OS page cache) | Highest |
| Atomic File Write | $O(\text{buffer})$ | Extra file rename | 100% crash proof |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Bare `except:` Catch-All:** Catching bare `except:` or `except Exception:` blindly catches system-level interrupts (`KeyboardInterrupt`, `SystemExit`), making applications impossible to terminate cleanly.
2. **Missing `encoding='utf-8'`:** Opening files with `open('file.txt')` defaults to platform-dependent encoding (e.g., `cp1252` on Windows), resulting in fatal `UnicodeDecodeError` in production environments.
"""

print(f"Loaded {len(C01_BASICS)} comprehensive guides for Course 1.")
