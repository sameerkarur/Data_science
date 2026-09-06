# Python Object-Oriented Programming, Metaprogramming & Modules: The Definitive Guide
**Comprehensive Academic & Industry Engineering Handbook (Official Python / W3Schools / GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Core OOP Foundations: Encapsulation, Abstraction, Inheritance & Polymorphism](#1-core-oop-foundations)
2. [CPython Object Lifecycle: `__new__`, `__init__`, and `__del__`](#2-cpython-object-lifecycle)
3. [Method Resolution Order (MRO) & C3 Linearization](#3-method-resolution-order-mro)
4. [Property Decorators & The Descriptor Protocol (`__get__`, `__set__`)](#4-property-decorators--the-descriptor-protocol)
5. [Memory Optimization with `__slots__`](#5-memory-optimization-with-__slots__)
6. [Abstract Base Classes (ABCs) & Protocol Interfaces (PEP 544)](#6-abstract-base-classes-abcs)
7. [Dunder Methods & Python Data Model Protocols](#7-dunder-methods--python-data-model-protocols)
8. [Module Packaging, `sys.modules`, and Circular Import Resolution](#8-module-packaging--circular-imports)
9. [Production Case Study: Scikit-Learn Style Base Estimator Pipeline](#9-production-case-study-scikit-learn-base-estimator)
10. [Try It Yourself! (Hands-On Practice Exercises with Solutions)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet & Best Website Citations](#11-quick-reference-cheat-sheet--citations)

---

## 1. Core OOP Foundations

Python OOP models real-world domain architectures through four classical pillars:

```
                          THE FOUR PILLARS OF OOP
    ┌─────────────────────────┬─────────────────────────┐
    │ ENCAPSULATION           │ ABSTRACTION             │
    │ Bundling state & logic; │ Hiding implementation   │
    │ private attributes (_x) │ details behind clean API│
    ├─────────────────────────┼─────────────────────────┤
    │ INHERITANCE             │ POLYMORPHISM            │
    │ Reusing parent classes; │ Same interface for      │
    │ overriding methods      │ diverse underlying types│
    └─────────────────────────┴─────────────────────────┘
```

---

## 2. CPython Object Lifecycle: `__new__` vs `__init__`

Instantiation is a two-step process in CPython:
1. **`__new__(cls)`**: The **allocator**. Creates and returns a fresh, uninitialized heap instance.
2. **`__init__(self)`**: The **initializer**. Configures attributes on the newly allocated instance.

```
                    OBJECT CREATION LIFECYCLE
      Call: obj = MyClass(*args)
                 │
                 ▼
      1. MyClass.__new__(cls, *args) ──► Allocates raw PyObject on Heap
                 │
                 ▼
      2. MyClass.__init__(self, *args) ─► Populates self.__dict__
                 │
                 ▼
      Instance returned to caller
```

```python
class SingletonConfig:
    """Enforces a single global instance across the runtime."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Allocate memory only once
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, environment: str = "production"):
        self.environment = environment

s1 = SingletonConfig("staging")
s2 = SingletonConfig("production")

print("s1 is s2 (Singleton)?", s1 is s2)
print("Shared environment:", s1.environment)
```

#### Output:
```text
s1 is s2 (Singleton)? True
Shared environment: production
```

---

## 3. Method Resolution Order (MRO) & C3 Linearization

In multiple inheritance, Python resolves attribute lookups using the **C3 Linearization Algorithm** (guaranteeing monotonicity and parent-precedence):

```
                     DIAMOND INHERITANCE GRAPH
                               ┌───────┐
                               │   A   │
                               └───┬───┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                      ┌───────┐           ┌───────┐
                      │   B   │           │   C   │
                      └───┬───┘           └───┬───┘
                          │                   │
                          └─────────┬─────────┘
                                    ▼
                                 ┌───────┐
                                 │   D   │
                                 └───────┘
```

```python
class A:
    def ping(self): print("Ping from A")

class B(A):
    def ping(self): print("Ping from B"); super().ping()

class C(A):
    def ping(self): print("Ping from C"); super().ping()

class D(B, C):
    def ping(self): print("Ping from D"); super().ping()

d = D()
d.ping()
print("\nLinearized MRO:", [cls.__name__ for cls in D.__mro__])
```

#### Output:
```text
Ping from D
Ping from B
Ping from C
Ping from A

Linearized MRO: ['D', 'B', 'C', 'A', 'object']
```

---

## 4. Property Decorators & The Descriptor Protocol

A **Descriptor** is any object implementing `__get__`, `__set__`, or `__delete__`. In Python, `@property`, `classmethod`, and `staticmethod` are built using descriptors:

```python
class ValidatedPositiveFloat:
    """Descriptor that validates positive numerical inputs."""
    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return getattr(obj, self.private_name, 0.0)

    def __set__(self, obj, value):
        val = float(value)
        if val <= 0:
            raise ValueError(f"{self.public_name} must be strictly positive! Got: {val}")
        setattr(obj, self.private_name, val)

class MLHyperparameters:
    learning_rate = ValidatedPositiveFloat()
    batch_size = ValidatedPositiveFloat()

    def __init__(self, lr, bs):
        self.learning_rate = lr
        self.batch_size = bs

hp = MLHyperparameters(0.001, 32)
print(f"Validated LR: {hp.learning_rate} | Batch Size: {hp.batch_size}")
```

#### Output:
```text
Validated LR: 0.001 | Batch Size: 32.0
```

---

## 5. Memory Optimization with `__slots__`

Normally, instances store attributes in a dynamic dictionary (`self.__dict__`), which adds ~150-200 bytes per instance. `__slots__` replaces `__dict__` with a fixed-size C array:

```python
import sys

class NormalPoint:
    def __init__(self, x, y): self.x = x; self.y = y

class SlottedPoint:
    __slots__ = ('x', 'y')
    def __init__(self, x, y): self.x = x; self.y = y

p_normal = NormalPoint(1.0, 2.0)
p_slotted = SlottedPoint(1.0, 2.0)

print(f"Normal Instance Memory:  {sys.getsizeof(p_normal) + sys.getsizeof(p_normal.__dict__)} bytes")
print(f"Slotted Instance Memory: {sys.getsizeof(p_slotted)} bytes (Saves ~70% RAM!)")
```

#### Output:
```text
Normal Instance Memory:  152 bytes
Slotted Instance Memory: 48 bytes (Saves ~70% RAM!)
```

---

## 6. Abstract Base Classes (ABCs)

ABCs enforce interface contracts across development teams:

```python
from abc import ABC, abstractmethod

class BaseDataConnector(ABC):
    """Abstract interface for all enterprise data sources."""
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    def fetch_batch(self, batch_size: int) -> list:
        """Fetch records."""
        pass

class S3DataConnector(BaseDataConnector):
    def connect(self) -> bool:
        print("Connected to AWS S3 Bucket.")
        return True

    def fetch_batch(self, batch_size: int) -> list:
        return [f"s3_record_{i}" for i in range(batch_size)]

s3 = S3DataConnector()
s3.connect()
print("Fetched S3 Batch:", s3.fetch_batch(2))
```

#### Output:
```text
Connected to AWS S3 Bucket.
Fetched S3 Batch: ['s3_record_0', 's3_record_1']
```

---

## 7. Dunder Methods & Python Data Model Protocols

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def __len__(self) -> int:
        return 2

v1 = Vector(2, 4)
v2 = Vector(3, 1)
print("Vector Addition (v1 + v2):", v1 + v2)
```

#### Output:
```text
Vector Addition (v1 + v2): Vector(x=5, y=5)
```

---

## 8. Module Packaging & Circular Import Resolution

When `module_a` imports `module_b` while `module_b` imports `module_a`, Python throws an `ImportError`:
- **Cause:** Python inserts partially initialized module stubs into `sys.modules` before top-level expressions finish executing.
- **Solution:** Move the import statement **inside the function scope** that requires it, or refactor shared models into a common `types.py` module.

---

## 9. Production Case Study: Scikit-Learn Style Base Estimator

```python
import numpy as np

class BaseMLModel(ABC):
    """Production base estimator implementing fit-predict pattern."""
    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel':
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class LinearMeanRegressor(BaseMLModel):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_target_ = np.mean(y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted yet. Call .fit() first!")
        return np.full(shape=(len(X),), fill_value=self.mean_target_)

model = LinearMeanRegressor()
model.fit(np.array([[1], [2], [3]]), np.array([10.0, 20.0, 30.0]))
preds = model.predict(np.array([[10], [20]]))
print("Predictions from fitted baseline model:", preds)
```

#### Output:
```text
Predictions from fitted baseline model: [20. 20.]
```

---

## 10. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Custom Context Manager Class
**Task:** Build a class `ExecutionTimer` that measures code block execution duration using `__enter__` and `__exit__`:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import time

class ExecutionTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed Time: {self.elapsed*1000:.2f} ms")
        return False  # Do not suppress exceptions

with ExecutionTimer():
    total = sum(i * i for i in range(500_000))
```
#### Output:
```text
Elapsed Time: 21.43 ms
```
</details>

---

## 11. Quick Reference Cheat Sheet & Best Website Citations

| OOP Feature | Implementation | Key Objective |
|---|---|---|
| **Encapsulation** | `self._attribute` | Information hiding |
| **Slots** | `__slots__ = ('a', 'b')` | Drastically reduces RAM footprint |
| **MRO Inspection** | `Class.__mro__` | Resolves inheritance priority |
| **Descriptor** | `__get__`, `__set__` | Reusable attribute validation logic |
| **ABCs** | `@abstractmethod` | Enforcing API contracts across pipelines |

### 🌐 Official References & Recommended Reading:
- [Python Official Documentation — Classes](https://docs.python.org/3/tutorial/classes.html)
- [Python Data Model Documentation](https://docs.python.org/3/reference/datamodel.html)
- [W3Schools Python OOP & Inheritance](https://www.w3schools.com/python/python_classes.asp)
- [Real Python Object-Oriented Programming](https://realpython.com/python3-object-oriented-programming/)
