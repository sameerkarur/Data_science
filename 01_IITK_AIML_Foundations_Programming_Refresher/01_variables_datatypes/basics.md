# Python Variables, Data Types & Operators: Complete Beginner-to-Pro Guide
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
