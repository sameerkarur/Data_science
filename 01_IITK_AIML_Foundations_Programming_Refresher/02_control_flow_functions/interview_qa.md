# Interview Q&A — Control Flow & Functions

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. How does short-circuit evaluation work with 'and' and 'or' operators?

**Answer:** Python evaluates logical expressions from left to right and stops as soon as the outcome is determined. In 'A and B', if A is falsy, Python returns A immediately without evaluating B. In 'A or B', if A is truthy, Python returns A immediately without evaluating B. This enables safe guards like: 'user and user.is_authenticated'.

### Q2. Explain the difference between 'break', 'continue', and 'pass'.

**Answer:** 'break' terminates the nearest enclosing loop immediately. 'continue' skips the remainder of the current loop iteration and proceeds to the next iteration. 'pass' is a null statement that does nothing and acts as a syntactic placeholder.

### Q3. How does the 'else' block behave in 'for' and 'while' loops?

**Answer:** The 'else' block attached to a loop executes only if the loop completes all iterations naturally without encountering a 'break' statement. It is commonly used for search loops to execute fallback logic when no match is found.

### Q4. What are first-class functions in Python?

**Answer:** Functions in Python are first-class objects: they can be assigned to variables, passed as arguments to other functions, returned from other functions, and stored in data structures like lists and dictionaries.

### Q5. Explain the mechanics of a Python decorator.

**Answer:** A decorator is a callable that takes a function as input, extends or modifies its behavior via an internal wrapper function, and returns the modified function. Syntactic sugar '@my_decorator' above 'def foo()' is equivalent to 'foo = my_decorator(foo)'.

### Q6. Why should you use 'functools.wraps' inside custom decorators?

**Answer:** When a wrapper function replaces the original function, metadata such as the function name (__name__), docstring (__doc__), and signature are replaced by the wrapper's metadata. '@functools.wraps(fn)' copies this metadata back to the wrapper, preserving introspection and debugging integrity.

### Q7. What are *args and **kwargs in function definitions?

**Answer:** *args collects variable-length non-keyword positional arguments into a tuple. **kwargs collects arbitrary keyword arguments into a dictionary. They allow functions to accept dynamic signatures and pass arguments cleanly down to underlying APIs.

### Q8. Explain keyword-only arguments and how to enforce them.

**Answer:** Arguments defined after an asterisk ('*') in a function signature must be passed as keyword arguments: 'def query(sql, *, timeout=30)'. This prevents callers from accidentally passing positional values to sensitive options, improving readability.

### Q9. What is the difference between a generator function and a regular function?

**Answer:** A regular function runs to completion and returns a single value via 'return', destroying its stack frame. A generator function contains 'yield'; when called, it returns a generator iterator object that pauses execution state, yielding values lazily on-demand with O(1) memory.

### Q10. Explain the 'yield from' expression introduced in Python 3.3.

**Answer:** 'yield from iterable' delegates generation directly to a sub-generator or iterable, transparently yielding all values and establishing a bi-directional communication channel for .send(), .throw(), and .close() between caller and sub-generator.

### Q11. What is a lambda function and what are its architectural limitations?

**Answer:** A lambda is an anonymous, single-expression inline function: 'lambda x, y: x + y'. Limitations include: it can only contain a single expression (no multi-line statements, loops, or assignments), cannot have type annotations, and can degrade stack trace readability.

### Q12. How does Python's structural pattern matching (match-case) work?

**Answer:** Introduced in Python 3.10, 'match subject: case pattern:' allows matching on literal values, types, sequence structures, and object attributes with optional guards (if conditions). Unlike C switch statements, it unpacks and binds variables directly from complex nested structures.

### Q13. What causes a RecursionError in Python and how do you check the recursion limit?

**Answer:** A RecursionError occurs when recursive calls exceed the interpreter's maximum stack depth (typically 1000 frames in CPython) to protect against C stack overflow. Checked via 'sys.getrecursionlimit()' and adjusted via 'sys.setrecursionlimit(n)'.

### Q14. Explain tail-call optimization and does Python support it?

**Answer:** Tail-call optimization (TCO) allows a function's stack frame to be reused if the recursive call is the very last operation. Python deliberately does NOT support TCO to preserve complete stack traces for accurate debugging and profiling (as stated by Guido van Rossum).

### Q15. What is the difference between 'global' and 'nonlocal' keywords?

**Answer:** 'global var' tells Python that assignments to 'var' modify the module-level global variable. 'nonlocal var' tells Python to rebind a variable in the nearest enclosing non-global scope (essential for closures modifying state in outer functions).

### Q16. How does 'functools.lru_cache' work and what are its requirements?

**Answer:** lru_cache wraps a function with a Least-Recently-Used memoization cache, storing results of expensive function calls. Requirements: all arguments passed to the cached function must be hashable because cache keys are generated from argument tuples.

### Q17. Explain higher-order functions with examples (map, filter, reduce).

**Answer:** A higher-order function takes one or more functions as arguments or returns a function. 'map(fn, seq)' applies fn to each element; 'filter(pred, seq)' retains elements where pred is True; 'functools.reduce(fn, seq)' accumulates elements pairwise into a single result.

### Q18. Why are list comprehensions generally preferred over map() and filter() in modern Python?

**Answer:** List comprehensions are more idiomatic, readable, support simultaneous filtering and mapping without nested lambda calls, and execute faster in CPython because they avoid function call overhead for every element.

### Q19. What is function currying and partial function application?

**Answer:** Currying translates a function callable with N arguments into a chain of N functions that each take one argument. Partial application ('functools.partial(fn, *fixed_args)') pre-fills a subset of arguments, creating a new callable with a simpler signature.

### Q20. What is the difference between passing arguments by value vs by reference in Python?

**Answer:** Python uses 'pass-by-object-reference' (or 'pass-by-assignment'). The function receives a copy of the reference to the object. If the object is mutable (like a list), mutating it in-place reflects in the caller; if you rebind the variable name ('x = 10'), the caller's binding remains unaffected.

### Q21. How do generator expressions differ from list comprehensions?

**Answer:** List comprehensions '[x*2 for x in data]' allocate the entire list in memory immediately. Generator expressions '(x*2 for x in data)' produce items lazily one at a time using O(1) memory, making them ideal for processing gigabyte-scale datasets or infinite streams.

### Q22. What is a pure function and why is it desirable in data pipelines?

**Answer:** A pure function produces the same output for identical inputs and causes zero observable side effects (no mutation of global state, no disk/network I/O). Pure functions are easy to unit test, refactor, memoize, and parallelize across CPU cores.

### Q23. How does 'zip()' work and what happens with mismatched sequence lengths?

**Answer:** 'zip(a, b)' pairs corresponding elements from iterables into tuples until the shortest iterable is exhausted. To iterate until the longest sequence completes without truncating, use 'itertools.zip_longest(*iterables, fillvalue=None)'.

### Q24. Explain 'enumerate(iterable, start=0)' and why it replaces manual index tracking.

**Answer:** 'enumerate' yields (index, item) pairs directly in C, eliminating the need to manually initialize, increment, and index arrays ('arr[i]'), which prevents off-by-one errors and improves execution speed.

### Q25. What are function annotations / type hints and do they enforce types at runtime?

**Answer:** Type hints (e.g. 'def add(x: int, y: int) -> int:') document expected types and enable static analysis by tools like MyPy, IDE autocompletion, and Pydantic validation. They do NOT enforce type safety at runtime by default; Python remains dynamically typed.

### Q26. How do you inspect a function's parameters and annotations programmatically?

**Answer:** Using the 'inspect' module: 'inspect.signature(fn)' returns a Signature object containing parameter names, default values, and annotations, allowing runtime validation and dependency injection frameworks to bind arguments dynamically.

### Q27. What is the difference between 'any()' and 'all()' built-in functions?

**Answer:** 'any(iterable)' returns True if at least one element evaluates to truthy, short-circuiting on the first True. 'all(iterable)' returns True only if every element is truthy, short-circuiting on the first False. Both handle empty iterables according to formal logic (all([]) is True, any([]) is False).

### Q28. What is variable shadowing in Python?

**Answer:** Variable shadowing occurs when a variable declared within an inner scope (such as a local function variable) shares the same name as a variable in an outer scope, temporarily overriding access to the outer variable within that inner scope.

### Q29. Explain the 'itertools' module and name 3 high-performance iterators.

**Answer:** 'itertools' provides memory-efficient building blocks for iterators written in C. Key examples: 'count()' (infinite sequence), 'cycle()' (repeats an iterable indefinitely), 'chain()' (flattens multiple iterables sequentially), and 'combinations()'/'permutations()' (combinatorial generators).

### Q30. How does Python handle default positional argument ordering in function signatures?

**Answer:** Positional arguments without default values must precede arguments with default values. Defining 'def func(a=1, b):' raises a SyntaxError: non-default argument follows default argument.
