# Interview Q&A — Variables, Data Types & Operators

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the difference between mutable and immutable objects in Python.

**Answer:** Mutable objects (e.g. lists, dicts, sets) can be altered in-place after creation without changing their memory address (id()). Immutable objects (e.g. ints, floats, strings, tuples, frozensets) cannot be changed; any operation that modifies them instantiates a new object in memory.

### Q2. How does CPython handle memory management and garbage collection?

**Answer:** CPython primarily relies on reference counting: every object tracks how many references point to it. When the count drops to zero, its memory is immediately deallocated. To detect self-referential reference cycles (e.g. list A contains list B and vice versa), Python supplements reference counting with a generational cyclic garbage collector.

### Q3. Explain the difference between the '==' operator and the 'is' keyword.

**Answer:** '==' checks for value equality by invoking the object's __eq__() method (e.g. [1, 2] == [1, 2] is True). 'is' checks for object identity (memory address equality: id(a) == id(b)). Use 'is' strictly for singletons like 'None', 'True', and 'False'.

### Q4. What is dynamic typing and duck typing in Python?

**Answer:** Dynamic typing means variable types are checked at runtime rather than compile-time; a variable name is just a pointer that can bind to any object. Duck typing is the principle: 'If it walks like a duck and quacks like a duck, it's a duck'—code checks whether an object provides the required methods or attributes rather than checking its explicit class inheritance.

### Q5. What are Python's interned strings and small integer caching?

**Answer:** CPython pre-allocates and caches small integers between -5 and 256 at startup, so all references to 42 share the same memory location. String interning automatically caches string literals that look like Python identifiers to allow O(1) pointer comparison instead of O(N) character-by-character comparison.

### Q6. Explain how Python floating-point precision works and why 0.1 + 0.2 != 0.3.

**Answer:** Python floats are implemented as 64-bit IEEE 754 double-precision binary numbers. Fractions with denominators that are not powers of two (like 1/10 and 1/5) cannot be represented exactly in binary and produce repeating fractions. The resulting tiny rounding discrepancy causes 0.1 + 0.2 to evaluate to 0.30000000000000004. Use the decimal.Decimal class or math.isclose() for exact financial or scientific calculations.

### Q7. What is the Walrus operator (:=) and when should it be used?

**Answer:** The Walrus operator (assignment expression) allows assigning a value to a variable within an expression itself. For example: while (chunk := file.read(8192)): process(chunk). It eliminates duplicate function calls and verbose variable declarations before loops or if-conditions.

### Q8. Explain the difference between deepcopy and shallow copy.

**Answer:** A shallow copy (copy.copy) constructs a new collection object and inserts references to the original child objects. A deep copy (copy.deepcopy) recursively creates duplicate copies of every object found inside the original compound object, completely decoupling the clone from the original.

### Q9. What are the advantages of using namedtuple or dataclass over standard tuples/dictionaries?

**Answer:** Namedtuples provide lightweight, immutable tuple structures with human-readable dot notation access without dictionary memory overhead. Dataclasses (@dataclass) auto-generate boilerplate dunder methods (__init__, __repr__, __eq__) for mutable structured classes with native type hints, default values, and optional immutability (frozen=True).

### Q10. Why is type(x) == int generally discouraged compared to isinstance(x, int)?

**Answer:** isinstance(x, int) respects object-oriented inheritance: it returns True for instances of subclasses derived from int (such as bool). type(x) == int checks exact identity and fails if the variable is an instance of a subclass, breaking polymorphism.

### Q11. How does string concatenation using '+' in a loop compare to ''.join()?

**Answer:** Strings are immutable. Using '+' in a loop re-allocates a new string buffer and copies all characters on every iteration, scaling as O(N^2) time. ''.join(list_of_strings) pre-calculates the total string length, allocates a single memory buffer, and copies characters in O(N) linear time.

### Q12. What happens under the hood when you unpack variables like 'a, *b, c = [1, 2, 3, 4, 5]'?

**Answer:** Extended iterable unpacking assigns the first element to 'a' (1), the last element to 'c' (5), and collects all intermediate elements into a standard Python list assigned to 'b' ([2, 3, 4]).

### Q13. What is the difference between bytes and bytearray?

**Answer:** bytes is an immutable sequence of integers in the range 0 <= x < 256. bytearray is a mutable counterpart that allows in-place modifications, appending, and slicing, making it much more efficient for binary I/O, socket streaming, and audio/image buffers.

### Q14. What are Python's bitwise operators and when are they used in AI/ML?

**Answer:** Bitwise operators include & (AND), | (OR), ^ (XOR), ~ (NOT), << (left shift), and >> (right shift). In AI/ML and Pandas/NumPy, & and | are overloaded to perform element-wise boolean masking on arrays and DataFrames where Python's standard 'and'/'or' keywords would fail.

### Q15. How does Python represent arbitrarily large integers without integer overflow?

**Answer:** Unlike C/C++ which use fixed 32-bit or 64-bit words, Python 3 integers are arbitrary-precision objects ('bignums'). CPython allocates a variable array of 30-bit digits on the heap, dynamically expanding memory as numbers grow, preventing integer overflow at the cost of slight speed.

### Q16. Explain the concept of variable scoping and the LEGB rule.

**Answer:** Python resolves variable names following the LEGB hierarchy: Local (inside the current function), Enclosing (in enclosing functions for nested closures), Global (module level), and Built-in (pre-defined Python names like len, range). Use 'global' to rebind module variables and 'nonlocal' to rebind enclosing closure variables.

### Q17. What is an Iterable vs an Iterator?

**Answer:** An Iterable is any object implementing __iter__() that returns an iterator (e.g. lists, dicts, tuples, strings). An Iterator is an object with state implementing __next__() which returns the next item in sequence and raises StopIteration when exhausted.

### Q18. How does the 'hash()' function work and what makes an object hashable?

**Answer:** An object is hashable if it has a hash value that never changes during its lifetime (implements __hash__()) and can be compared to other objects (implements __eq__()). All immutable built-in types (int, float, str, tuple with hashable items) are hashable; mutable types (list, dict, set) are unhashable.

### Q19. What is None in Python and how is it represented in memory?

**Answer:** None is a singleton object of type NoneType used to signify the absence of a value or default return value of functions. Because only one instance exists across the Python process, you should always check identity using 'x is None' rather than 'x == None'.

### Q20. Why should default function arguments never be mutable objects like 'def func(x=[])'?

**Answer:** Default parameter expressions are evaluated once when the function definition is executed, not each time the function is called. If the default is a mutable list or dictionary, mutations persist across all future function invocations across the program.

### Q21. How do you properly implement an optional mutable default argument in Python?

**Answer:** Set the default parameter to None: 'def func(x=None): if x is None: x = []'. This ensures a new, fresh list is instantiated on every call where no argument is passed.

### Q22. What is the difference between 'del variable' and deleting an object?

**Answer:** 'del x' deletes the name binding 'x' from the current namespace and decrements the reference count of the object it pointed to. It does not directly delete the object itself; the garbage collector frees the object only when its reference count reaches zero.

### Q23. Explain Python's slice syntax [start:stop:step] and negative indexing.

**Answer:** Slice extraction extracts elements from 'start' up to but not including 'stop', incrementing index by 'step'. Negative indices count backward from the end of the sequence (-1 is the last item). 'arr[::-1]' creates a reversed copy of the sequence.

### Q24. What is the difference between str() and repr()?

**Answer:** str() returns an informal, readable string representation intended for human end users. repr() returns an unambiguous, formal string representation typically valid Python code that could recreate the object, primarily used for debugging and logging.

### Q25. What is a closure in Python and why is it useful?

**Answer:** A closure is a nested function that retains access to variables from its enclosing lexical scope even after the outer function has finished executing. Closures enable data hiding, factory functions, and function decorators.

### Q26. How do format strings (f-strings) differ from str.format() and % formatting?

**Answer:** f-strings (introduced in Python 3.6) are evaluated at runtime directly as expressions rather than parsed through constant string lookups. They are faster, more readable, support arbitrary Python expressions inside {}, and format inline specifiers like f'{val:.2f}'.

### Q27. What is the purpose of the 'pass' statement in Python?

**Answer:** 'pass' is a null operation (no-op). It serves as a syntactic placeholder in blocks where Python grammar requires a statement (such as empty functions, classes, or exception blocks) but no action is needed.

### Q28. What are dunder (double underscore) methods in Python?

**Answer:** Dunder methods (magic methods like __init__, __len__, __getitem__, __call__) allow custom classes to hook into Python's core protocols and operators, enabling object-oriented operator overloading, context managers, and iteration.

### Q29. Explain the difference between 'sys.getsizeof()' and actual memory consumption.

**Answer:** sys.getsizeof() returns the memory footprint allocated directly by the container object itself in bytes, excluding the memory of objects referenced inside the container. To calculate true recursive memory, one must recursively traverse all referenced child objects.

### Q30. What is the purpose of '__slots__' in a Python class?

**Answer:** By default, Python instances store instance attributes in a dynamic dictionary (__dict__). Defining __slots__ = ('x', 'y') replaces __dict__ with a fixed-size compact array, drastically reducing memory usage when creating millions of small objects and preventing dynamic attribute creation.
