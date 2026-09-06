# Chapter 5: Python File I/O, Exception Architecture & Systems
**Comprehensive Textbook Guide — Advanced Python & Scientific Computing**

---

## 1. Executive Overview & Mental Models

Production data pipelines and distributed training jobs require deterministic resource management. Unclosed file handles leak operating system file descriptors, while non-atomic file writes result in corrupted partial checkpoints if an out-of-memory (OOM) killer or kernel panic occurs mid-write.

```
                  CONTEXT MANAGER LIFECYCLE (with statement)
       ┌────────────────────────────────────────────────────────┐
       │ 1. Expression Evaluated: with open('data.bin') as f:   │
       │    └── manager = expression()                          │
       │    └── enter_val = manager.__enter__()                 │
       │                                                        │
       │ 2. Executing Code Block Inside 'with'                  │
       │    ├── Success? ──► manager.__exit__(None, None, None) │
       │    │                (File descriptor cleanly closed!)  │
       │    │                                                   │
       │    └── Exception Raised?                               │
       │         ▼                                              │
       │ 3. manager.__exit__(exc_type, exc_val, exc_tb)         │
       │    ├── Returns True?  ──► Exception suppressed!        │
       │    └── Returns False? ──► Exception re-propagated!     │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Architectural Flowchart: CPython 3-Tier I/O Subsystem

```
                         CPYTHON I/O ARCHITECTURE (io module)
                         
       Application Code: f.write("Record data\\n")
                              │
                              ▼
       Tier 1: TextIOWrapper (Character Encoding & Newlines)
               • Translates Unicode strings to bytes using specified codec (UTF-8)
               • Translates universal newlines ('\\n' ➔ '\\r\\n' if on Windows)
                              │
                              ▼
       Tier 2: BufferedWriter (User-Space Memory Buffering)
               • Buffers writes into an internal memory page (typically 8192 bytes)
               • Eliminates expensive OS system call on every single write operation
                              │
                              ▼ (When buffer fills or f.flush() is called)
       Tier 3: FileIO (Raw OS System Calls)
               • Executes unbuffered kernel system call: write(fd, buffer, count)
                              │
                              ▼
       OS Kernel Page Cache ──► Physical Storage Media (NVMe / SSD / HDD)
```

---

## 3. Deep Theoretical Foundations

### 1. Atomic Writes & Crash Consistency
When writing model checkpoints, calling `f.write()` modifies data in the OS page cache. If the machine loses power before the kernel flushes its dirty pages, the destination file is left in an unrecoverable corrupted state. 
- **Production Solution:** Write to an adjacent temporary file on the **same filesystem**, force a hardware flush via `os.fsync()`, and perform an **atomic rename** (`os.replace()`). On POSIX systems, `rename()` is guaranteed to be atomic by the filesystem journal.

### 2. Exception Hierarchy & Exception Chaining
All standard exceptions inherit from `BaseException`. Application code should catch `Exception`, never `BaseException`, because catching the latter traps `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`, preventing graceful process termination.
- **Explicit Chaining (`from exc`):** Sets `__cause__` to preserve the original exception context.
- **Suppression (`from None`):** Hides internal implementation details when presenting user-facing API errors.

### 3. Memory-Mapped Files (`mmap`)
For multi-gigabyte datasets (such as embedding matrices), standard file reads copy bytes from the kernel page cache into user process memory. `mmap` maps disk blocks directly into the virtual address space of the process, allowing lazy page faulting by the OS kernel without loading the entire file into RAM.

---

## 4. Production Implementation: Atomic Checkpointer & Mmap Reader

```python
import os
import tempfile
import mmap
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

@contextmanager
def atomic_checkpoint_writer(destination_path: Path | str) -> Generator[tempfile.NamedTemporaryFile, None, None]:
    """Guarantees atomic file updates: either 100% written or previous file untouched."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Must be on same filesystem for atomic rename
    with tempfile.NamedTemporaryFile(mode='wb', dir=dest.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())  # Force OS dirty pages onto physical disk
            tmp.close()
            os.replace(temp_path, dest)  # POSIX atomic filesystem swap
        except Exception:
            if temp_path.exists():
                os.remove(temp_path)
            raise

def fast_binary_embedding_search(filepath: Path | str, vector_dim: int, target_idx: int) -> bytes:
    """Reads vector embeddings with zero-copy mmap."""
    record_size = vector_dim * 4  # float32 = 4 bytes
    with open(filepath, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            offset = target_idx * record_size
            return mm[offset : offset + record_size]
```

---

## 5. File I/O & Exception Complexity Matrix

| Technique | Memory Footprint | Latency Profile | Crash Safety |
|---|---|---|---|
| Naive `read()` | $O(\text{file size})$ (OOM hazard!) | High initial lag | Zero (Partial writes corrupt data) |
| Chunked Stream (`read(8192)`) | $O(1)$ constant 8KB buffer | Low streaming latency | Zero |
| Memory Map (`mmap`) | $O(1)$ virtual memory | Sub-millisecond lazy paging | High |
| Atomic File Write | $O(\text{buffer})$ | Extra rename operation | 100% ACID compliant |

---

## 6. Subtle Pitfalls, Bugs & Production Best Practices

### Pitfall 1: Missing Explicit Character Encoding
```python
# BUG-PRONE: Uses OS platform default (e.g. cp1252 on Windows, causing crashes!)
with open("data.json", "r") as f:
    data = f.read()

# PRODUCTION FIX: ALWAYS specify UTF-8:
with open("data.json", "r", encoding="utf-8") as f:
    data = f.read()
```

### Pitfall 2: Silencing Exceptions with Bare Except
```python
# ANTI-PATTERN: Traps KeyboardInterrupt and bugs silently!
try:
    process_data()
except:
    pass

# PRODUCTION FIX: Catch specific exceptions and log tracebacks:
try:
    process_data()
except (ValueError, KeyError) as exc:
    logger.error("Processing failed: %s", exc, exc_info=True)
    raise DataPipelineError("Data validation failed") from exc
```
