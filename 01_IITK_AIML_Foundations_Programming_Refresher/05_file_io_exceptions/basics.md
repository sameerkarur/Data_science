# Python File I/O & Exception Handling Handbook
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
