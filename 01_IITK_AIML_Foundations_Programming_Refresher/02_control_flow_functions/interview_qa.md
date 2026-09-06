# Interview Q&A — Control Flow & Functions

> **30 questions** — read aloud, then explain without looking.


## elif avoids re-checking earlier True branches; separate if allows multiple blocks to run.

### Q1. Control flow

When use elif vs separate if?


## for when iterations known/iterable; while when condition-driven.

### Q2. Control flow

for vs while?


## Runs if loop completes without break — useful for search-not-found.

### Q3. Control flow

What does loop else do?


## *args collects positional extras as tuple; **kwargs collects keyword extras as dict.

### Q4. Functions

*args vs **kwargs?


## Default evaluated once; shared list mutated across calls. Use None default.

### Q5. Functions

Default mutable argument bug?


## Single expression, no statements; fine for short callbacks.

### Q6. Functions

Lambda limitations?


## Local → Enclosing → Global → Built-in name lookup order.

### Q7. Scope

LEGB rule?


## global binds module-level; nonlocal binds nearest enclosing (non-global) scope.

### Q8. Scope

global vs nonlocal?


## Generator lazy, memory efficient; list stores all values.

### Q9. Advanced

Generator vs list?


## Function wrapping another to extend behavior without modifying source.

### Q10. Advanced

What is a decorator?


## sys.getrecursionlimit(); deep recursion risks stack overflow — prefer iteration.

### Q11. Advanced

Recursion limit?


## Not optimized by CPython; iteration preferred for deep calls.

### Q12. Advanced

Tail recursion in Python?


## Functions are objects — assign, pass, return, store in collections.

### Q13. Advanced

First-class functions?


## Comprehensions more Pythonic and often faster to read.

### Q14. Advanced

map/filter vs comprehensions?


## Structural pattern matching (3.10+) — cleaner than long if/elif chains.

### Q15. Advanced

match/case purpose?


## return ends function; yield makes generator pausing execution.

### Q16. Interview

Difference return vs yield?


## Inner function remembering enclosing scope variables.

### Q17. Interview

Closure definition?


## Bind loop variable via default arg: lambda x=i: x.

### Q18. Interview

Late binding closure fix?


## No at runtime by default; use mypy/pyright for static checking.

### Q19. Interview

Type hints enforced?


## PEP 257; first line summary; used by help() and Sphinx.

### Q20. Interview

Docstring conventions?


## pass noop; break exit loop; continue next iteration.

### Q21. Interview

pass, break, continue?


## `a if cond else b` — expression not statement.

### Q22. Interview

Ternary operator syntax?


## Provides index+value without manual counter.

### Q23. Interview

Enumerate benefit?


## Stops at shortest; use itertools.zip_longest for padding.

### Q24. Interview

Zip with unequal lengths?


## any True if one true; all True if all true — short-circuit.

### Q25. Interview

any() vs all()?


## `reversed(seq)` or `range(len-1,-1,-1)` for indices.

### Q26. Interview

How to reverse iterate?


## Has __name__, __doc__, __defaults__; can setattr on functions.

### Q27. Interview

Function as object attributes?


## functools.partial fixes subset of arguments.

### Q28. Interview

Partial application?


## Cache results dict keyed by args — functools.lru_cache built-in.

### Q29. Interview

Memoization pattern?


## Deep trees, performance critical paths, Python stack limits.

### Q30. Interview

When not to use recursion?
