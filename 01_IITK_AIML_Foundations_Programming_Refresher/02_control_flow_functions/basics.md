# Python Control Flow, Scopes & Functions: Complete Beginner-to-Pro Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [Conditional Statements (`if`, `elif`, `else`)](#1-conditional-statements-if-elif-else)
2. [Loops: `for` Loop & `range()` Function](#2-loops-for-loop--range-function)
3. [Loops: `while` Loop, `break`, `continue` & `else`](#3-loops-while-loop-break-continue--else)
4. [Function Architecture: `def`, Parameters & Return](#4-function-architecture-def-parameters--return)
5. [Arbitrary Arguments: `*args` and `**kwargs`](#5-arbitrary-arguments-args-and-kwargs)
6. [Lambda Expressions (Anonymous Functions)](#6-lambda-expressions-anonymous-functions)
7. [Variable Scope: LEGB Rule (Local, Enclosing, Global, Built-in)](#7-variable-scope-legb-rule)
8. [Decorators & Higher-Order Functions](#8-decorators--higher-order-functions)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Conditional Statements (`if`, `elif`, `else`)

Control flow executes different blocks of code based on Boolean truth conditions:

```python
score = 85

if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
else:
    grade = 'F'

print(f"Student Score: {score} -> Grade Assigned: {grade}")

# Ternary Conditional Expression (One-line if-else)
status = "Passing" if score >= 70 else "Failing"
print(f"Status: {status}")
```

#### Output:
```text
Student Score: 85 -> Grade Assigned: B
Status: Passing
```

---

## 2. Loops: `for` Loop & `range()` Function

```python
# Iterating over range(start, stop, step)
print("Range Step Loop:")
for i in range(10, 35, 5):
    print(f"Current Value: {i}")

# Iterating over list with enumerate() for index & value
tech_stack = ["Python", "NumPy", "Pandas", "Scikit-Learn"]
print("\nEnumerate Loop:")
for idx, tool in enumerate(tech_stack, start=1):
    print(f"Step {idx}: Learn {tool}")
```

#### Output:
```text
Range Step Loop:
Current Value: 10
Current Value: 15
Current Value: 20
Current Value: 25
Current Value: 30

Enumerate Loop:
Step 1: Learn Python
Step 2: Learn NumPy
Step 3: Learn Pandas
Step 4: Learn Scikit-Learn
```

---

## 3. Loops: `while` Loop, `break`, `continue` & `else`

```python
attempts = 0
max_attempts = 5

while attempts < max_attempts:
    attempts += 1
    if attempts == 2:
        print(f"Attempt {attempts}: Transient timeout, skipping with continue...")
        continue
    if attempts == 4:
        print(f"Attempt {attempts}: Success! Exiting with break.")
        break
    print(f"Attempt {attempts}: Processing request...")
```

#### Output:
```text
Attempt 1: Processing request...
Attempt 2: Transient timeout, skipping with continue...
Attempt 3: Processing request...
Attempt 4: Success! Exiting with break.
```

---

## 4. Function Architecture: `def`, Parameters & Return

```python
def calculate_compound_interest(principal: float, rate: float = 0.05, years: int = 1) -> float:
    """Computes compound interest balance: A = P(1 + r)^t"""
    final_amount = principal * ((1 + rate) ** years)
    return round(final_amount, 2)

# Call with positional and keyword arguments
bal1 = calculate_compound_interest(1000)
bal2 = calculate_compound_interest(1000, rate=0.08, years=5)

print(f"1 Year @ Default 5%:  ${bal1}")
print(f"5 Years @ Custom 8%:  ${bal2}")
```

#### Output:
```text
1 Year @ Default 5%:  $1050.0
5 Years @ Custom 8%:  $1469.33
```

---

## 5. Arbitrary Arguments: `*args` and `**kwargs`

```
  *args   ──► Packs positional arguments into a Tuple: (arg1, arg2, ...)
  **kwargs ──► Packs keyword arguments into a Dictionary: {'key': value, ...}
```

```python
def build_ml_pipeline(model_name, *metrics, **hyperparameters):
    print(f"Configuring Model: {model_name}")
    print(f"Evaluation Metrics (*args tuple):   {metrics}")
    print(f"Hyperparameters (**kwargs dict):    {hyperparameters}")

build_ml_pipeline(
    "XGBoost Classifier",
    "Accuracy", "F1-Score", "ROC-AUC",
    learning_rate=0.05,
    n_estimators=300,
    max_depth=6
)
```

#### Output:
```text
Configuring Model: XGBoost Classifier
Evaluation Metrics (*args tuple):   ('Accuracy', 'F1-Score', 'ROC-AUC')
Hyperparameters (**kwargs dict):    {'learning_rate': 0.05, 'n_estimators': 300, 'max_depth': 6}
```

---

## 6. Lambda Expressions (Anonymous Functions)

Small one-line functions written without `def`:

```python
# Sorting a list of tuples by second element using lambda
students = [("Alice", 88), ("Bob", 95), ("Charlie", 72), ("David", 91)]

sorted_by_score = sorted(students, key=lambda student: student[1], reverse=True)
print("Ranked by Score:\n", sorted_by_score)
```

#### Output:
```text
Ranked by Score:
 [('Bob', 95), ('David', 91), ('Alice', 88), ('Charlie', 72)]
```

---

## 7. Variable Scope: LEGB Rule

Python resolves variable names using the **LEGB** hierarchy:
1. **L**ocal: Inside the current function.
2. **E**nclosing: Inside any enclosing outer function.
3. **G**lobal: At the top module level.
4. **B**uilt-in: Python built-in namespace (`print`, `len`, `range`).

```python
counter = 10  # Global scope

def outer():
    tag = "OuterEnclosure"  # Enclosing scope
    def inner():
        local_val = 42      # Local scope
        print(f"Inside: local={local_val}, tag={tag}, global_counter={counter}")
    inner()

outer()
```

#### Output:
```text
Inside: local=42, tag=OuterEnclosure, global_counter=10
```

---

## 8. Decorators & Higher-Order Functions

A decorator wraps a function to modify or measure its behavior:

```python
import time

def execution_timer(func):
    """Decorator measuring runtime execution in milliseconds."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"⚡ [{func.__name__}] completed in {elapsed_ms:.3f} ms")
        return result
    return wrapper

@execution_timer
def compute_sum_of_squares(n):
    return sum(i * i for i in range(n))

val = compute_sum_of_squares(500_000)
print(f"Sum of squares computed: {val}")
```

#### Output:
```text
⚡ [compute_sum_of_squares] completed in 22.450 ms
Sum of squares computed: 41666541666750000
```

---

## 9. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Custom Filter Function with Lambda
**Task:** Write a function `custom_filter(numbers, predicate)` that takes a list of integers and returns only the elements where `predicate(num)` returns `True`. Test it with a lambda that selects all even numbers greater than 10.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
def custom_filter(numbers, predicate):
    return [x for x in numbers if predicate(x)]

raw_nums = [4, 12, 7, 18, 22, 9, 30, 2, 14]
evens_above_10 = custom_filter(raw_nums, lambda n: n % 2 == 0 and n > 10)
print("Filtered Numbers:", evens_above_10)
```
#### Output:
```text
Filtered Numbers: [12, 18, 22, 30, 14]
```
</details>

---

## 10. Quick Reference Cheat Sheet

| Construct | Syntax | Key Feature |
|---|---|---|
| **Ternary Operator** | `x if condition else y` | One-line conditional |
| **Enumerate** | `for idx, val in enumerate(lst)` | Returns index and value |
| **Zip** | `for a, b in zip(list1, list2)` | Parallel iteration across lists |
| **Default Arg** | `def f(x=10):` | Evaluated once at definition time |
| **Args Pack** | `*args` | Arbitrary positional arguments as tuple |
| **Kwargs Pack** | `**kwargs` | Arbitrary keyword arguments as dict |
| **Lambda** | `lambda x: x * 2` | Anonymous single-expression function |
