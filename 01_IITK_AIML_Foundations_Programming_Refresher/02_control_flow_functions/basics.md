# Chapter 2: Control Flow, Scopes & Functional Programming
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
