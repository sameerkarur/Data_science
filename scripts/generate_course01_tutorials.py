"""
Generates W3Schools / GeeksforGeeks style tutorials with visual diagrams, code, output blocks,
and practice exercises for all 5 modules of Course 1 (Programming Refresher).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Variables & Data Types
C01_M01_GUIDE = r'''# Python Variables, Data Types & Operators: Complete Beginner-to-Pro Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is a Variable in Python?](#1-what-is-a-variable-in-python)
2. [Variable Naming Rules & Conventions](#2-variable-naming-rules--conventions)
3. [Memory Architecture: Variables as Heap Object Pointers](#3-memory-architecture-variables-as-heap-object-pointers)
4. [Python Core Data Types (Int, Float, Bool, Str, None)](#4-python-core-data-types)
5. [Type Checking & Dynamic Typing (`type()`, `isinstance()`)](#5-type-checking--dynamic-typing)
6. [Type Casting & Conversion (Implicit vs Explicit)](#6-type-casting--conversion)
7. [Working with Strings (Indexing, Slicing & Methods)](#7-working-with-strings)
8. [Python Operators (Arithmetic, Comparison, Logical, Identity)](#8-python-operators)
9. [Small Integer Caching & Object Mutability](#9-small-integer-caching--object-mutability)
10. [Try It Yourself! (Hands-On Practice Exercises)](#10-try-it-yourself-hands-on-practice-exercises)
11. [Quick Reference Cheat Sheet](#11-quick-reference-cheat-sheet)

---

## 1. What is a Variable in Python?

A variable is a named reference that points to a value stored in your computer's memory. In Python, you do not need to declare variable types explicitly — Python is **dynamically typed**, inferring the type at runtime.

```python
# Creating variables
student_name = "Alex Mercer"  # string (str)
student_age = 23              # integer (int)
gpa_score = 3.85              # floating-point (float)
is_enrolled = True            # boolean (bool)

print(f"Student: {student_name} | Age: {student_age} | GPA: {gpa_score} | Active: {is_enrolled}")
```

#### Output:
```text
Student: Alex Mercer | Age: 23 | GPA: 3.85 | Active: True
```

---

## 2. Variable Naming Rules & Conventions

In Python (PEP 8 standard):
- Must begin with a letter (`a-z`, `A-Z`) or underscore (`_`).
- Cannot start with a number.
- Can only contain alphanumeric characters and underscores (`A-z`, `0-9`, and `_`).
- Case-sensitive (`total`, `Total`, and `TOTAL` are 3 distinct variables).
- Cannot use Python reserved keywords (`for`, `while`, `def`, `class`, `import`, etc.).

### Common Naming Styles:
- **`snake_case`** (Python convention for variables & functions): `total_revenue_usd = 5000`
- **`PascalCase`** (Used for Class definitions): `CustomerAccount`
- **`camelCase`** (Common in JavaScript): `totalRevenueUsd`

---

## 3. Memory Architecture: Variables as Heap Object Pointers

In CPython, variables do not store raw numbers directly in stack slots. Variables are **pointer references** stored in a symbol table pointing to heap-allocated `PyObject` structures.

```
       VARIABLE NAMES (STACK)                       HEAP MEMORY
      ┌───────────────────────┐                  ┌───────────────────────────────┐
      │  score = 42           │ ───────────────► │ PyLongObject:                 │
      │  (Symbol Table Entry) │                  │   ob_refcnt = 2               │
      └───────────────────────┘                  │   ob_type   = <class 'int'>   │
                                                 │   ob_digit  = 42              │
      ┌───────────────────────┐                  └───────────────────────────────┘
      │  result = score       │ ─────────────────────────────────┘ (Shared Reference)
      │  (Alias Pointer)      │
      └───────────────────────┘
```

```python
x = 100
y = x
print(f"Memory Address of x: {id(x)}")
print(f"Memory Address of y: {id(y)}")
print(f"Do x and y share the same object? {x is y}")
```

#### Output:
```text
Memory Address of x: 4352194880
Memory Address of y: 4352194880
Do x and y share the same object? True
```

---

## 4. Python Core Data Types

| Data Type | Class | Description | Example |
|---|---|---|---|
| Integer | `int` | Whole numbers of arbitrary precision | `count = 100` |
| Floating point | `float` | 64-bit IEEE 754 floating point numbers | `rate = 0.05` |
| Boolean | `bool` | Logical truth values (`True` or `False`) | `is_valid = True` |
| String | `str` | Immutable Unicode text sequence | `msg = "Hello"` |
| NoneType | `NoneType` | Singleton representing absence of value | `result = None` |

---

## 5. Type Checking & Dynamic Typing

Use `type()` to inspect the runtime class, and `isinstance()` for production validation:

```python
data_payload = "42000"

print("1. Type of payload:         ", type(data_payload))
print("2. Is payload a string?     ", isinstance(data_payload, str))
print("3. Is payload an int or str?", isinstance(data_payload, (int, str)))
```

#### Output:
```text
1. Type of payload:          <class 'str'>
2. Is payload a string?      True
3. Is payload an int or str? True
```

---

## 6. Type Casting & Conversion

Converting between data types is critical when parsing API responses or CSV files:

```python
raw_price = "149.99"
quantity_str = "5"

# Explicit casting
unit_price = float(raw_price)
quantity = int(quantity_str)
total_cost = unit_price * quantity

print(f"Total Cost: ${total_cost:.2f} (Type: {type(total_cost).__name__})")
```

#### Output:
```text
Total Cost: $749.95 (Type: float)
```

---

## 7. Working with Strings (Indexing, Slicing & Methods)

Strings in Python are **immutable sequences** of Unicode characters.

### Visual Diagram: String Slicing & Indexing

```
  String Value:   'P'   'Y'   'T'   'H'   'O'   'N'
  Forward Index:   0     1     2     3     4     5
  Reverse Index:  -6    -5    -4    -3    -2    -1

  text[0:2] ──► 'PY'  (From index 0 up to 2, exclusive)
  text[2:]  ──► 'THON' (From index 2 to the end)
  text[::-1]──► 'NOHTYP' (Reverse string with step -1)
```

```python
course = "  data science & ai  "

# String methods
cleaned = course.strip()
capitalized = cleaned.title()
tokens = cleaned.split()

print("Original:   ", repr(course))
print("Cleaned:    ", repr(cleaned))
print("Title Case: ", capitalized)
print("Words List: ", tokens)
print("Reversed:   ", cleaned[::-1])
```

#### Output:
```text
Original:    '  data science & ai  '
Cleaned:     'data science & ai'
Title Case:  Data Science & Ai
Words List:  ['data', 'science', '&', 'ai']
Reversed:    ia & ecneics atad
```

---

## 8. Python Operators

### Arithmetic & Comparison Operators
```python
a, b = 17, 5

print(f"Addition (a + b):        {a + b}")
print(f"Integer Floor Div (a // b): {a // b} (Removes decimal part)")
print(f"Modulus Remainder (a % b): {a % b}")
print(f"Exponent Power (a ** b):  {a ** b}")
print(f"Comparison (a > b):       {a > b}")
print(f"Equality Check (a == b):  {a == b}")
```

#### Output:
```text
Addition (a + b):        22
Integer Floor Div (a // b): 3 (Removes decimal part)
Modulus Remainder (a % b): 2
Exponent Power (a ** b):  1419857
Comparison (a > b):       True
Equality Check (a == b):  False
```

---

## 9. Small Integer Caching & Object Mutability

In CPython, integers in the range `[-5, 256]` are pre-allocated singletons in memory:

```python
x = 250
y = 250
print("250 is 250? (Cached):    ", x is y)

a = 1000
b = 1000
print("1000 is 1000? (Not cached):", a is b)
```

#### Output:
```text
250 is 250? (Cached):     True
1000 is 1000? (Not cached): False
```

---

## 10. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Formatted Invoice Generator
**Task:** Given product details, format a clean invoice string using an f-string, aligning numbers to 2 decimal places:
```python
item = "Mechanical Keyboard"
qty = 3
price = 89.99
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
item = "Mechanical Keyboard"
qty = 3
price = 89.99
subtotal = qty * price
tax = subtotal * 0.08
total = subtotal + tax

receipt = f"""
================ RECEIPT ================
Item:     {item}
Quantity: {qty} @ ${price:.2f} each
Subtotal: ${subtotal:.2f}
Tax (8%): ${tax:.2f}
Total:    ${total:.2f}
=========================================
"""
print(receipt)
```
#### Output:
```text
================ RECEIPT ================
Item:     Mechanical Keyboard
Quantity: 3 @ $89.99 each
Subtotal: $269.97
Tax (8%): $21.60
Total:    $291.57
=========================================
```
</details>

---

## 11. Quick Reference Cheat Sheet

| Task | Syntax | Output |
|---|---|---|
| **Check Type** | `type(x)` | `<class 'int'>` |
| **Validate Type** | `isinstance(x, (int, float))` | `True` or `False` |
| **Cast to String** | `str(100)` | `'100'` |
| **String Slice** | `'PYTHON'[1:4]` | `'YTH'` |
| **Reverse String** | `'HELLO'[::-1]` | `'OLLEH'` |
| **f-string format** | `f"{price:.2f}"` | `'19.95'` |
'''

# 2. Control Flow & Functions
C01_M02_GUIDE = r'''# Python Control Flow, Scopes & Functions: Complete Beginner-to-Pro Guide
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
'''

# 3. Data Structures
C01_M03_GUIDE = r'''# Python Data Structures: Lists, Tuples, Sets & Dictionaries
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [The 4 Built-in Collection Data Structures](#1-the-4-built-in-collection-data-structures)
2. [Python Lists: Dynamic Arrays, Methods & Memory](#2-python-lists-dynamic-arrays-methods--memory)
3. [Python Tuples: Immutable Sequences & Packing/Unpacking](#3-python-tuples-immutable-sequences--packingunpacking)
4. [Python Dictionaries: Hash Tables, Key-Value Pairs & Views](#4-python-dictionaries-hash-tables-key-value-pairs--views)
5. [Python Sets: Unique Elements & Mathematical Set Theory](#5-python-sets-unique-elements--mathematical-set-theory)
6. [List, Dictionary & Set Comprehensions](#6-list-dictionary--set-comprehensions)
7. [Algorithmic Complexity Matrix (Big-O Comparison)](#7-algorithmic-complexity-matrix-big-o-comparison)
8. [Try It Yourself! (Hands-On Practice Exercises)](#8-try-it-yourself-hands-on-practice-exercises)
9. [Quick Reference Cheat Sheet](#9-quick-reference-cheat-sheet)

---

## 1. The 4 Built-in Collection Data Structures

| Collection | Ordered? | Mutable? | Allows Duplicates? | Syntax Example |
|---|---|---|---|---|
| **List** | Yes | Yes (Append, Remove) | Yes | `['apple', 'banana', 'apple']` |
| **Tuple** | Yes | No (Immutable) | Yes | `(10, 20, 30)` |
| **Set** | No (Unordered) | Yes | No (Unique only) | `{'red', 'green', 'blue'}` |
| **Dictionary**| Yes (Insertion order)| Yes | Keys Unique, Values Dup | `{'user': 'Alex', 'id': 101}` |

---

## 2. Python Lists: Dynamic Arrays, Methods & Memory

Python lists are contiguous arrays of pointers that over-allocate memory to achieve $O(1)$ amortized append operations:

```python
# List creation and operations
items = ["GPU", "RAM", "CPU"]

# 1. Appending & Inserting
items.append("SSD")           # Adds to end: O(1)
items.insert(1, "Motherboard")# Inserts at index 1: O(N)

# 2. Removing elements
popped_item = items.pop()     # Removes and returns last element: O(1)
items.remove("RAM")           # Removes first occurrence by value: O(N)

# 3. Sorting & Reversing
numbers = [42, 12, 88, 5, 23]
numbers.sort()                # In-place Timsort: O(N log N)

print("Modified Hardware List:\n", items)
print("Sorted Numbers:\n", numbers)
```

#### Output:
```text
Modified Hardware List:
 ['GPU', 'Motherboard', 'CPU']
Sorted Numbers:
 [5, 12, 23, 42, 88]
```

---

## 3. Python Tuples: Immutable Sequences

Because tuples are immutable, Python optimizes them for faster iteration and memory efficiency. They can also be used as dictionary keys:

```python
# Tuple packing and unpacking
coordinate = (37.7749, -122.4194)
lat, lon = coordinate  # Unpacking

# Return multiple values from function
def get_user_status():
    return "Alex", 25, "Active"

name, age, status = get_user_status()
print(f"Latitude: {lat}, Longitude: {lon}")
print(f"User: {name} (Age {age}) - Status: {status}")
```

#### Output:
```text
Latitude: 37.7749, Longitude: -122.4194
User: Alex (Age 25) - Status: Active
```

---

## 4. Python Dictionaries: Hash Tables & Key-Value Pairs

Dictionaries in Python are hash tables that guarantee average $O(1)$ lookup, insertion, and deletion:

```python
model_cfg = {
    "architecture": "Transformer",
    "layers": 12,
    "heads": 8,
    "hidden_dim": 768
}

# Safe lookup with .get(key, default)
dropout = model_cfg.get("dropout_rate", 0.1)

# Updating & Iterating
model_cfg["vocab_size"] = 50257

print("Config Keys:  ", list(model_cfg.keys()))
print("Config Values:", list(model_cfg.values()))
print(f"Dropout Rate:  {dropout}")
```

#### Output:
```text
Config Keys:   ['architecture', 'layers', 'heads', 'hidden_dim', 'vocab_size']
Config Values: ['Transformer', 12, 8, 768, 50257]
Dropout Rate:  0.1
```

---

## 5. Python Sets: Unique Elements & Mathematical Set Operations

Sets use hashing to maintain distinct elements and perform set algebra:

```python
python_devs = {"Alice", "Bob", "Charlie", "David"}
ml_engineers = {"Charlie", "David", "Elena", "Frank"}

# Set Operations
both = python_devs & ml_engineers        # Intersection
all_talent = python_devs | ml_engineers  # Union
only_python = python_devs - ml_engineers # Difference

print("Intersection (Both Roles):   ", both)
print("Union (All Unique Talent):   ", all_talent)
print("Difference (Only Python Devs):", only_python)
```

#### Output:
```text
Intersection (Both Roles):    {'Charlie', 'David'}
Union (All Unique Talent):    {'David', 'Elena', 'Bob', 'Alice', 'Charlie', 'Frank'}
Difference (Only Python Devs): {'Bob', 'Alice'}
```

---

## 6. List, Dictionary & Set Comprehensions

Comprehensions provide concise syntax for creating collections:

```python
# 1. List Comprehension with condition
squared_evens = [x**2 for x in range(10) if x % 2 == 0]

# 2. Dictionary Comprehension (Inverting key-value pairs)
id_map = {"Alice": 101, "Bob": 102, "Charlie": 103}
inv_map = {v: k for k, v in id_map.items()}

# 3. Set Comprehension
unique_lengths = {len(name) for name in ["apple", "banana", "kiwi", "orange", "pear"]}

print("Squared Evens:  ", squared_evens)
print("Inverted Dict:  ", inv_map)
print("Unique Lengths: ", sorted(unique_lengths))
```

#### Output:
```text
Squared Evens:   [0, 4, 16, 36, 64]
Inverted Dict:   {101: 'Alice', 102: 'Bob', 103: 'Charlie'}
Unique Lengths:  [4, 5, 6]
```

---

## 7. Algorithmic Complexity Matrix (Big-O Comparison)

| Operation | List | Tuple | Set | Dictionary |
|---|---|---|---|---|
| **Access by Index / Key** | $O(1)$ | $O(1)$ | N/A | $O(1)$ avg |
| **Search / Contains (`in`)** | $O(N)$ | $O(N)$ | $O(1)$ avg | $O(1)$ avg |
| **Insert / Append** | $O(1)$ amortized | N/A | $O(1)$ avg | $O(1)$ avg |
| **Delete** | $O(N)$ | N/A | $O(1)$ avg | $O(1)$ avg |

---

## 8. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Word Frequency Counter
**Task:** Given a sentence, use a dictionary comprehension or dictionary counting logic to return word frequencies:

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
text = "machine learning and deep learning are branches of artificial intelligence"
words = text.split()

frequency = {}
for word in words:
    frequency[word] = frequency.get(word, 0) + 1

print("Word Frequency Count:\n", frequency)
```
#### Output:
```text
Word Frequency Count:
 {'machine': 1, 'learning': 2, 'and': 1, 'deep': 1, 'are': 1, 'branches': 1, 'of': 1, 'artificial': 1, 'intelligence': 1}
```
</details>

---

## 9. Quick Reference Cheat Sheet

| Task | Syntax | Collection |
|---|---|---|
| **Add Item** | `lst.append(x)` | List |
| **Unpack Tuple** | `a, b = (10, 20)` | Tuple |
| **Safe Lookup** | `d.get('key', default)` | Dict |
| **Set Union** | `set1 | set2` | Set |
| **Set Intersect** | `set1 & set2` | Set |
| **List Comp** | `[f(x) for x in seq if cond]` | List |
'''

# 4. OOP & Modules
C01_M04_GUIDE = r'''# Python Object-Oriented Programming (OOP) & Modules Handbook
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
'''

# 5. File I/O & Exceptions
C01_M05_GUIDE = r'''# Python File I/O & Exception Handling Handbook
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [File Handling Basics: Modes (`'r'`, `'w'`, `'a'`, `'b'`)](#1-file-handling-basics-modes)
2. [Context Managers: The `with` Statement](#2-context-managers-the-with-statement)
3. [Reading Files (Line by Line, Whole File, Chunking)](#3-reading-files)
4. [Writing & Appending to Files](#4-writing--appending-to-files)
5. [Structured Data Persistence: JSON Serialization](#5-structured-data-persistence-json-serialization)
6. [Exception Handling: `try`, `except`, `else`, `finally`](#6-exception-handling)
7. [Catching Specific Exceptions vs Broad Exceptions](#7-catching-specific-exceptions-vs-broad-exceptions)
8. [Custom User-Defined Exceptions](#8-custom-user-defined-exceptions)
9. [Try It Yourself! (Hands-On Practice Exercises)](#9-try-it-yourself-hands-on-practice-exercises)
10. [Quick Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. File Handling Basics: Modes

| Mode | Meaning | Creates File if Missing? | Overwrites Existing? |
|---|---|---|---|
| `'r'` | Read only (Default) | No (Raises `FileNotFoundError`) | No |
| `'w'` | Write only | Yes | **Yes (Truncates to 0 bytes)** |
| `'a'` | Append to end | Yes | No (Appends to end) |
| `'r+'`| Read and Write | No | No |
| `'b'` | Binary mode (e.g. `'rb'`, `'wb'`) for images/pickles | Same as above | Same as above |

---

## 2. Context Managers: The `with` Statement

Always use the `with` statement when opening files. It automatically closes the file descriptor even if an unhandled exception occurs:

```python
import tempfile
import os

# Create temporary file for demonstration
temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
temp_path = temp_file.name
temp_file.close()

# Safe writing with context manager
with open(temp_path, 'w', encoding='utf-8') as f:
    f.write("Line 1: Model Hyperparameters\n")
    f.write("Line 2: Epochs = 50\n")
    f.write("Line 3: Learning Rate = 0.001\n")

print(f"File closed automatically? {f.closed}")
```

#### Output:
```text
File closed automatically? True
```

---

## 3. Reading Files

```python
# 1. Read entire file into string
with open(temp_path, 'r', encoding='utf-8') as f:
    full_content = f.read()

# 2. Read line by line in memory-efficient stream (ideal for multi-GB log files)
print("--- Streaming Line-by-Line ---")
with open(temp_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, start=1):
        print(f"[{line_num}] {line.strip()}")
```

#### Output:
```text
--- Streaming Line-by-Line ---
[1] Line 1: Model Hyperparameters
[2] Epochs = 50
[3] Learning Rate = 0.001
```

---

## 4. Structured Data Persistence: JSON Serialization

JSON is the lingua franca of machine learning APIs and web apps:

```python
import json

experiment_config = {
    "run_id": "run_9841",
    "dataset": "CIFAR-100",
    "batch_size": 64,
    "augmentations": ["RandomCrop", "HorizontalFlip"],
    "metrics": {"val_acc": 0.842, "val_loss": 0.38}
}

# Serialize dictionary to JSON string
json_str = json.dumps(experiment_config, indent=2)
print("Formatted JSON Payload:\n", json_str)

# Parse JSON string back to Python dictionary
parsed_dict = json.loads(json_str)
print("\nParsed Run ID:    ", parsed_dict["run_id"])
print("Validation Accuracy:", parsed_dict["metrics"]["val_acc"])
```

#### Output:
```text
Formatted JSON Payload:
 {
  "run_id": "run_9841",
  "dataset": "CIFAR-100",
  "batch_size": 64,
  "augmentations": [
    "RandomCrop",
    "HorizontalFlip"
  ],
  "metrics": {
    "val_acc": 0.842,
    "val_loss": 0.38
  }
}

Parsed Run ID:     run_9841
Validation Accuracy: 0.842
```

---

## 5. Exception Handling: `try`, `except`, `else`, `finally`

```
  ┌────────────┐
  │    TRY     │ ──► Execute risky code block
  └─────┬──────┘
        │
   Exception?
   ├── YES ──► EXCEPT: Handle specific error gracefully
   └── NO  ──► ELSE:   Runs ONLY if no exception occurred
        │
  ┌─────▼──────┐
  │  FINALLY   │ ──► ALWAYS executes (Clean up resources / sockets)
  └────────────┘
```

```python
def safe_divide(numerator: float, denominator: float) -> float:
    try:
        result = numerator / denominator
    except ZeroDivisionError as err:
        print(f"⚠️ Caught Mathematical Error: {err}")
        return 0.0
    except TypeError as err:
        print(f"⚠️ Caught Type Error: {err}")
        return 0.0
    else:
        print("✅ Division calculated successfully.")
        return result
    finally:
        print("🔒 [Finally] Cleanup executed.")

print("Test 1 (Valid):    ", safe_divide(100, 4))
print("\nTest 2 (Zero Div): ", safe_divide(100, 0))
```

#### Output:
```text
✅ Division calculated successfully.
🔒 [Finally] Cleanup executed.
Test 1 (Valid):     25.0

⚠️ Caught Mathematical Error: division by zero
🔒 [Finally] Cleanup executed.
Test 2 (Zero Div):  0.0
```

---

## 6. Custom User-Defined Exceptions

```python
class ModelConvergenceError(Exception):
    """Raised when gradient descent diverges into NaN/Inf values."""
    def __init__(self, loss_value, epoch):
        super().__init__(f"Loss exploded to {loss_value} at epoch {epoch}. Training aborted.")
        self.loss_value = loss_value
        self.epoch = epoch

def simulate_training_step(loss, epoch):
    if loss > 10_000 or str(loss) == 'nan':
        raise ModelConvergenceError(loss, epoch)
    return f"Epoch {epoch} loss: {loss:.4f}"

try:
    print(simulate_training_step(0.42, 1))
    print(simulate_training_step(999_999, 2))
except ModelConvergenceError as e:
    print("Caught Custom Exception:\n", e)
```

#### Output:
```text
Epoch 1 loss: 0.4200
Caught Custom Exception:
 Loss exploded to 999999 at epoch 2. Training aborted.
```

---

## 7. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Safe File Number Summer
**Task:** Write a function `sum_numbers_from_file(filepath)` that reads a file where each line is a number. If a line contains invalid non-numeric text, catch `ValueError`, print a warning, and continue summing the valid numbers.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
def sum_numbers(lines):
    total = 0.0
    for idx, line in enumerate(lines, start=1):
        try:
            total += float(line.strip())
        except ValueError:
            print(f"Warning: Line {idx} '{line.strip()}' is not a valid number. Skipped.")
    return total

sample_lines = ["10.5", "20", "invalid_entry", "40.2"]
print("Total Sum Calculated:", sum_numbers(sample_lines))
```
#### Output:
```text
Warning: Line 3 'invalid_entry' is not a valid number. Skipped.
Total Sum Calculated: 70.7
```
</details>

---

## 8. Quick Reference Cheat Sheet

| Task | Syntax | Key Benefit |
|---|---|---|
| **Safe Open** | `with open(p, 'r') as f:` | Auto-closes on exit |
| **Dump JSON** | `json.dump(obj, f, indent=2)` | Serializes directly to file |
| **Load JSON** | `obj = json.load(f)` | Deserializes directly from file |
| **Catch Error** | `except (ValueError, KeyError) as e:` | Catches multiple types |
| **Raise Error** | `raise ValueError("Invalid arg")` | Triggers custom exception |
'''

p4 = REPO_ROOT / "01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules/basics.md"
p4.write_text(C01_M04_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C01 M04 Guide: {len(C01_M04_GUIDE.splitlines())} lines.")

p5 = REPO_ROOT / "01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions/basics.md"
p5.write_text(C01_M05_GUIDE.strip() + "\n", encoding="utf-8")
print(f"✅ Generated C01 M05 Guide: {len(C01_M05_GUIDE.splitlines())} lines.")
