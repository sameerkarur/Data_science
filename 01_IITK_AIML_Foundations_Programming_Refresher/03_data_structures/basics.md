# Python Data Structures: Lists, Tuples, Sets & Dictionaries
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
