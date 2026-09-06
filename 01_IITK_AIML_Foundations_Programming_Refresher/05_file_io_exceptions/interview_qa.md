# Interview Q&A — File I/O, Exceptions & Context Managers

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain how Python's 'with' statement works and what methods a context manager must implement.

**Answer:** The 'with' statement manages resource allocation and cleanup automatically. A context manager must implement '__enter__()' (executed upon entering the block; its return value binds to the 'as' target) and '__exit__(exc_type, exc_val, exc_tb)' (executed upon exit, guaranteeing cleanup even if exceptions occur).

### Q2. How can you suppress exceptions using a context manager?

**Answer:** In the '__exit__' method, returning True signals to Python that the exception has been handled and suppressed, preventing it from propagating. Returning False or None allows the exception to propagate normally. The standard library provides 'contextlib.suppress(*exceptions)' for this purpose.

### Q3. What is 'contextlib.contextmanager' and how does it turn a generator into a context manager?

**Answer:** '@contextlib.contextmanager' is a decorator that converts a generator function with a single 'yield' into a context manager. Code before the 'yield' executes as '__enter__'; the yielded value binds to 'as'; code in a 'finally' block after 'yield' executes as '__exit__'.

### Q4. Explain the difference between 'pathlib.Path' and 'os.path'.

**Answer:** 'os.path' treats paths as raw strings and requires clunky function chaining ('os.path.join(os.path.dirname(p), "f.txt")'). 'pathlib.Path' provides an object-oriented API where paths are first-class objects with operator overloading (e.g. 'Path("/root") / "sub" / "file.txt"'), cross-platform normalization, and built-in file operations (.read_text(), .exists()).

### Q5. What is the difference between opening a file in 'r', 'w', 'a', and 'x' modes?

**Answer:** 'r' opens for reading (raises FileNotFoundError if missing). 'w' opens for writing, truncating existing content or creating a new file. 'a' opens for appending at the end of the file. 'x' opens for exclusive creation, failing with FileExistsError if the file already exists.

### Q6. Why should binary files (images, pickled models, audio) always be opened in 'rb' or 'wb' mode?

**Answer:** In text mode ('r'/'w'), Python translates newline characters (e.g. converting CRLF '\r\n' to LF '\n' on Windows) and decodes bytes into Unicode strings using system encoding. Binary mode ('rb'/'wb') disables newline translation and encoding, reading and writing raw byte streams untouched.

### Q7. How does file buffering work in Python and how can you control it?

**Answer:** Python buffers file I/O to minimize expensive kernel syscalls. You can control buffering via the 'buffering' argument in open(): 0 disables buffering (binary only), 1 enables line buffering (text only), and >1 specifies buffer size in bytes. Use 'flush()' to force writing buffer contents to disk.

### Q8. Explain the difference between 'read()', 'readline()', and 'readlines()'.

**Answer:** 'read(n)' reads up to n bytes/characters (or the entire file if n is omitted, risking memory exhaustion on large files). 'readline()' reads a single line up to the newline. 'readlines()' reads all lines into a Python list in memory. To stream large files efficiently, iterate over the file object directly: 'for line in f:'.

### Q9. How do 'tell()' and 'seek()' work in file handling?

**Answer:** 'f.tell()' returns the current cursor position in bytes from the start of the file. 'f.seek(offset, whence)' moves the cursor: whence=0 (default) measures from file start, whence=1 measures from current position, whence=2 measures from file end.

### Q10. What is the difference between catching 'Exception' vs 'BaseException'?

**Answer:** 'Exception' is the base class for all non-system-exiting exceptions. 'BaseException' is the root class for everything, including 'SystemExit', 'KeyboardInterrupt', and 'GeneratorExit'. Catching 'BaseException' prevents the user from terminating the program via Ctrl+C, which is usually poor practice.

### Q11. Explain the full syntax of a 'try-except-else-finally' block.

**Answer:** 'try' runs guarded code. 'except' handles specified exceptions. 'else' runs ONLY if no exceptions were raised in the try block (ideal for code that shouldn't be protected by the except handler). 'finally' ALWAYS executes regardless of whether exceptions were raised, caught, or unhandled, making it essential for resource cleanup.

### Q12. What is exception chaining ('raise ... from ...')?

**Answer:** Exception chaining explicitly links a newly raised exception to an original underlying cause: 'raise CustomError("Failed") from orig_err'. Python populates '__cause__' and prints 'The above exception was the direct cause of the following exception', preserving the complete diagnostic traceback.

### Q13. How do you create a custom exception in Python?

**Answer:** Define a class that inherits from 'Exception' (or a domain-specific subclass): 'class ModelTrainingError(Exception): pass'. Custom exceptions allow callers to catch specific error conditions precisely without catching unrelated general exceptions.

### Q14. What is the 'csv' module and why shouldn't you parse CSVs using line.split(',')?

**Answer:** The 'csv' module handles quote escaping, embedded commas inside quotes (e.g. "Dallas, TX"), multiline fields, and custom delimiters via 'csv.reader' and 'csv.DictReader'. Splitting on raw commas naive splits embedded quoted commas, corrupting tabular data.

### Q15. Explain 'json.load()' vs 'json.loads()' and 'json.dump()' vs 'json.dumps()'.

**Answer:** Methods without 's' operate on file streams/descriptors: 'load(f)' reads from a file, 'dump(obj, f)' writes to a file. Methods with 's' operate on strings: 'loads(s)' parses a JSON string, 'dumps(obj)' serializes an object into a JSON string.

### Q16. What causes a TypeError: Object of type X is not JSON serializable, and how do you fix it?

**Answer:** JSON natively supports only dicts, lists, strings, numbers, booleans, and None. Non-standard types (like NumPy arrays, Pandas DataFrames, datetime, UUID) raise TypeError. Fixed by passing a custom serializer function to 'default=my_serializer' in 'json.dumps()' (e.g. converting numpy arrays via .tolist()).

### Q17. What is file locking and why is it important in multi-process AI pipelines?

**Answer:** File locking prevents race conditions and corrupted writes when multiple worker processes read/write the same file concurrently. In Unix, 'fcntl.flock(f, fcntl.LOCK_EX)' provides advisory file locking.

### Q18. Explain the difference between 'shutil.copy()' and 'shutil.copy2()'.

**Answer:** 'shutil.copy()' copies the file content and permissions. 'shutil.copy2()' copies content, permissions, and preserves complete file metadata including access and modification timestamps.

### Q19. What is an atomic file write and how do you implement it?

**Answer:** An atomic write ensures that a file is either completely written or completely untouched if the process crashes mid-write. Implemented by writing data to a temporary file on the same filesystem (e.g. 'tempfile.NamedTemporaryFile') and then atomically renaming it over the destination using 'os.replace()'.

### Q20. How does memory-mapped file I/O ('mmap') work in Python?

**Answer:** The 'mmap' module maps a file directly into the process's virtual address space, allowing files to be read and modified as if they were in-memory bytearrays. The OS loads pages lazily from disk on demand, enabling high-performance processing of multi-gigabyte datasets without loading the entire file into RAM.

### Q21. What is the purpose of the 'tempfile' module?

**Answer:** 'tempfile' securely creates temporary files and directories with unique names in the OS temporary directory (e.g. /tmp), preventing symlink race attacks and ensuring automatic deletion on close with 'NamedTemporaryFile(delete=True)'.

### Q22. How do you recursively search for files matching a pattern using 'pathlib'?

**Answer:** Use 'Path.rglob(pattern)': e.g., 'list(Path('.').rglob('*.parquet'))' recursively yields all Parquet files throughout all nested subdirectories efficiently.

### Q23. What is the difference between 'os.remove()' and 'shutil.rmtree()'?

**Answer:** 'os.remove(path)' deletes a single file (raises IsADirectoryError on folders). 'shutil.rmtree(path)' recursively deletes an entire directory tree and all its nested files and subfolders.

### Q24. What is standard I/O redirection in Python and how do you redirect stdout?

**Answer:** Python routes print statements to 'sys.stdout'. You can redirect it programmatically by reassigning 'sys.stdout = open("log.txt", "w")' or using the context manager 'contextlib.redirect_stdout(target_stream)'.

### Q25. Explain the difference between 'IOError', 'OSError', and 'FileNotFoundError'.

**Answer:** In Python 3.3+, 'IOError' and 'OSError' were unified under 'OSError'. 'FileNotFoundError' is a specific built-in subclass of 'OSError' raised when a requested file or directory path does not exist on the filesystem.

### Q26. How do you ensure proper UTF-8 encoding when opening files across different operating systems?

**Answer:** Always specify 'encoding="utf-8"' explicitly in 'open(filename, "r", encoding="utf-8")'. If omitted, Python defaults to the OS-dependent locale (e.g. Windows often defaults to 'cp1252'), leading to UnicodeDecodeError when encountering non-ASCII characters.

### Q27. What is the difference between 'pickle' and 'json' for serialization?

**Answer:** 'pickle' is a Python-specific binary protocol capable of serializing almost any arbitrary Python object (functions, classes, lambda closures, custom ML models). 'json' is a human-readable, cross-language plain-text standard. Crucially, unpickling untrusted data is insecure and can execute arbitrary shellcode.

### Q28. Why should you avoid using bare 'except:' clauses?

**Answer:** A bare 'except:' catches ALL exceptions, including 'KeyboardInterrupt' (preventing graceful exit via Ctrl+C) and 'SystemExit'. It also masks programming bugs like 'NameError' or 'TypoError', making debugging painful. Always catch specific exceptions or 'except Exception:'.

### Q29. How does 'traceback.format_exc()' help in production logging?

**Answer:** 'traceback.format_exc()' captures the complete multi-frame stack trace as a string without crashing the application, allowing background worker daemons to log complete diagnostic traces to monitoring systems like Datadog or Sentry.

### Q30. What is the 'io.StringIO' and 'io.BytesIO' module used for?

**Answer:** 'io.StringIO' and 'io.BytesIO' provide in-memory stream buffers that implement the standard file interface (read, write, seek, getvalue). They are widely used in unit testing to mock file I/O and in web services to generate in-memory PDFs, images, or CSV exports without touching physical disk.
