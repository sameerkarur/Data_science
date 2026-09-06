# Python Object-Oriented Programming (OOP) & Modules Handbook
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Core Principles of OOP (Encapsulation, Inheritance, Polymorphism, Abstraction)](#1-core-principles-of-oop)
2. [Classes, Objects & the `__init__` Constructor](#2-classes-objects--the-init-constructor)
3. [Instance vs Class Variables & Methods](#3-instance-vs-class-variables--methods)
4. [Inheritance & the `super()` Method](#4-inheritance--the-super-method)
5. [Encapsulation & Private Attributes (`_` vs `__`)](#5-encapsulation--private-attributes)
6. [Polymorphism & Method Overriding](#6-polymorphism--method-overriding)
7. [Dunder / Magic Methods (`__str__`, `__repr__`, `__len__`, `__eq__`)](#7-dunder--magic-methods)
8. [Python Modules & Packages Architecture](#8-python-modules--packages-architecture)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Core Principles of OOP

Object-Oriented Programming (OOP) bundles state (data attributes) and behavior (methods) into reusable models:
- **Encapsulation:** Hiding internal state behind public access methods.
- **Inheritance:** Deriving specialized classes from general base classes to avoid duplicate code.
- **Polymorphism:** A unified interface handling different underlying object types.
- **Abstraction:** Exposing what an object does while hiding how it does it.

---

## 2. Classes, Objects & the `__init__` Constructor

```python
class MachineLearningModel:
    """Blueprint for training ML models."""
    def __init__(self, name: str, framework: str):
        self.name = name          # Instance attribute
        self.framework = framework
        self.is_trained = False

    def train(self, epochs: int):
        self.is_trained = True
        return f"Trained {self.name} using {self.framework} for {epochs} epochs."

# Instantiating objects
model1 = MachineLearningModel("ResNet-50", "PyTorch")
model2 = MachineLearningModel("XGBoost", "Scikit-Learn")

print(model1.train(10))
print(f"Model 2 trained? {model2.is_trained}")
```

#### Output:
```text
Trained ResNet-50 using PyTorch for 10 epochs.
Model 2 trained? False
```

---

## 3. Instance vs Class Variables & Methods

```python
class NeuralNetwork:
    device_target = "CUDA:0"  # Class variable shared across ALL instances

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim  # Instance variable unique to each instance

    @classmethod
    def set_global_device(cls, new_device: str):
        cls.device_target = new_device

    @staticmethod
    def calculate_param_count(in_dim: int, out_dim: int) -> int:
        """Pure static utility method with no self or cls binding."""
        return (in_dim * out_dim) + out_dim

# Inspect class method and static method
print("Default Device:       ", NeuralNetwork.device_target)
NeuralNetwork.set_global_device("MPS (Apple Silicon)")
print("Updated Global Device:", NeuralNetwork.device_target)
print("Parameter Count:      ", NeuralNetwork.calculate_param_count(784, 128))
```

#### Output:
```text
Default Device:        CUDA:0
Updated Global Device: MPS (Apple Silicon)
Parameter Count:       100480
```

---

## 4. Inheritance & the `super()` Method

```python
class BaseTransformer:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads

    def describe(self):
        return f"Transformer(d_model={self.d_model}, heads={self.n_heads})"

class BertForClassification(BaseTransformer):
    def __init__(self, d_model: int, n_heads: int, num_classes: int):
        super().__init__(d_model, n_heads)  # Invoke base class constructor
        self.num_classes = num_classes

    def describe(self):
        base_desc = super().describe()
        return f"{base_desc} -> Classifier Head({self.num_classes} classes)"

bert = BertForClassification(768, 12, 3)
print(bert.describe())
```

#### Output:
```text
Transformer(d_model=768, heads=12) -> Classifier Head(3 classes)
```

---

## 5. Encapsulation & Private Attributes

In Python, name-mangling protects private attributes with double underscores `__`:

```python
class BankAccount:
    def __init__(self, account_holder: str, initial_balance: float):
        self.account_holder = account_holder
        self.__balance = initial_balance  # Private attribute

    def deposit(self, amount: float):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    @property
    def balance(self) -> float:
        """Getter property for controlled read-only access."""
        return self.__balance

account = BankAccount("Elena Rostova", 5000.0)
account.deposit(1500.0)
print(f"Account Balance: ${account.balance:.2f}")

# Direct private access triggers AttributeError
try:
    print(account.__balance)
except AttributeError as e:
    print("Direct private access prevented:", type(e).__name__)
```

#### Output:
```text
Account Balance: $6500.00
Direct private access prevented: AttributeError
```

---

## 6. Dunder / Magic Methods (`__repr__`, `__len__`, `__eq__`)

```python
class DatasetBatch:
    def __init__(self, data_list):
        self.data = list(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"DatasetBatch(size={len(self.data)}, sample={self.data[:2]})"

batch = DatasetBatch([10.5, 20.3, 40.1, 88.9])
print(f"Batch Length (len()):    {len(batch)}")
print(f"Batch Subscript ([1]):   {batch[1]}")
print(f"String Representation:    {repr(batch)}")
```

#### Output:
```text
Batch Length (len()):    4
Batch Subscript ([1]):   20.3
String Representation:    DatasetBatch(size=4, sample=[10.5, 20.3])
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Vector Math Class
**Task:** Build a 2D `Vector(x, y)` class supporting vector addition (`v1 + v2`) and scalar multiplication (`v * scalar`) via `__add__` and `__mul__`.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float):
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(5, 7)
print("Vector Sum:    ", v1 + v2)
print("Vector Scaled: ", v1 * 3)
```
#### Output:
```text
Vector Sum:     Vector(7, 10)
Vector Scaled:  Vector(6, 9)
```
</details>

---

## 8. Quick Reference Cheat Sheet

| OOP Mechanism | Syntax | Description |
|---|---|---|
| **Constructor** | `def __init__(self, ...):` | Initializes new instance |
| **Inheritance** | `class SubClass(BaseClass):` | Derives child class |
| **Super Call** | `super().__init__(...)` | Invokes parent class method |
| **Property** | `@property def x(self):` | Getter method disguised as attribute |
| **Class Method** | `@classmethod def f(cls):` | Receives class instead of instance |
| **Length Magic** | `def __len__(self):` | Custom `len()` support |
