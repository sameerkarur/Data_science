# Python Control Flow, Scopes & Functional Programming
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
    """Production-grade retry decorator with exponential backoff and metadata preservation."""
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
    """Simulates external API call."""
    return {"status": "success", "endpoint": endpoint}
```

---

## 📐 Scope & Iteration Complexity Matrix

| Construct | Space Overhead | Time Complexity | State Retention |
|---|---|---|---|
| Regular Function Call | $O(	ext{stack depth})$ | $O(1)$ setup | Frame destroyed on return |
| Closure (`cell` object) | $O(	ext{free vars})$ | $O(1)$ lookup | Variables stored on heap |
| Generator Function | $O(1)$ memory | $O(1)$ per `next()` | Frame preserved until exhausted |
| List Comprehension | $O(n)$ heap | $O(n)$ eager | Evaluated fully in memory |

---

## ⚠️ Common Pitfalls & Anti-Patterns

1. **Late Binding in Closures:** In loops like `funcs = [lambda: i for i in range(5)]`, all lambdas evaluate `i` at invocation time (all return 4). Fix using default arguments: `lambda i=i: i`.
2. **Modifying Enclosing Variables Without `nonlocal`:** Reassigning an outer variable inside a nested function creates a local variable instead of mutating the outer one unless explicitly declared `nonlocal`.
