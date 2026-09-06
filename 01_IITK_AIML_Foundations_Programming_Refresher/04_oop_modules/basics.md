# Python Object-Oriented Architecture & Metaprogramming
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
    """Reusable descriptor enforcing positive numeric types."""
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
