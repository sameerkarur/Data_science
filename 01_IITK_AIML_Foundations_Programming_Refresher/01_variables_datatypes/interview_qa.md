# Interview Q&A — Variables, Data Types & Operators

> **30 questions** — read aloud, then explain without looking.


## **Mutable:** list, dict, set, bytearray. **Immutable:** int, float, str, tuple, frozenset, bytes. Immutable objects cannot change after creation; 'modification' creates new objects.

### Q1. Conceptual

Mutable vs immutable types in Python?


## `==` compares **values** (equality). `is` compares **identity** (same object in memory). Use `is` only for singletons like `None`.

### Q2. Conceptual

Difference between `==` and `is`?


## Variable names bind to objects at runtime; the same name can refer to different types over time without explicit declaration.

### Q3. Conceptual

What is dynamic typing?


## Prefer `isinstance(x, int)` over `type(x) == int` because isinstance respects inheritance.

### Q4. Conceptual

How do you check type safely?


## A singleton representing absence of value. Always test with `x is None`.

### Q5. Conceptual

What is `None`?


## Parentheses > exponent > unary > mult/div/mod > add/sub > comparisons > not > and > or. Use parentheses when unsure.

### Q6. Operators

Explain operator precedence.


## `and` stops at first falsy; `or` stops at first truthy. Side effects may not run.

### Q7. Operators

What is short-circuit evaluation?


## `/` is true division (float). `//` is floor division toward negative infinity.

### Q8. Operators

Difference between `/` and `//`?


## No. Methods like `.upper()` return **new** strings.

### Q9. Strings

Are strings mutable?


## f-strings (3.6+) are fastest and most readable. `.format()` flexible for templates. `%` is legacy.

### Q10. Strings

f-string vs .format() vs %?


## Floating-point binary representation cannot exactly store some decimals. Use `decimal.Decimal` for money.

### Q11. Casting

Why does `0.1 + 0.2 != 0.3`?


## Any non-empty string is truthy, including `'False'` and `'0'`.

### Q12. Casting

When is `bool('False')` True?


## CPython object address (identity). Same id means same object.

### Q13. Memory

What does `id()` return?


## Assignment copies reference. Use `.copy()` or `list()` for shallow copy.

### Q14. Memory

Copy vs reference for lists?


## `a, b = b, a` using tuple packing/unpacking.

### Q15. Practical

Swap two variables pythonically?


## Assign and use in expression: `if (n := len(data)) > 10:` avoids duplicate computation.

### Q16. Practical

Walrus operator use case?


## `a < b < c` equivalent to `a < b and b < c`, evaluates b once.

### Q17. Practical

Chained comparisons?


## Empty containers, zero numbers, None, False are falsy; most else truthy.

### Q18. Practical

What is truthiness?


## `isinstance(obj, Base)` True for subclasses. `type(obj) == Base` False for subclass instances.

### Q19. Interview

Explain `isinstance` vs `type` for subclasses.


## Small ints (-5 to 256) may be cached; `is` may appear True for equal small ints.

### Q20. Interview

Integer interning in CPython?


## `complex(real, imag)` with `.real`, `.imag`. Useful in signal processing.

### Q21. Interview

Complex numbers in Python?


## Syntax `x: int = 0` for hints; stored in `__annotations__` on functions/modules.

### Q22. Interview

What are variable annotations?


## `str` is Unicode text; `bytes` is raw binary. Encode/decode at I/O boundaries.

### Q23. Interview

Bytes vs str?


## `//` floors (-7//2 = -4). int(-7/2) truncates toward zero (-3).

### Q24. Interview

Floor vs truncation for negatives?


## When you need quotient and remainder together — time conversions, pagination.

### Q25. Interview

When use `divmod`?


## `x += 1` may call `__iadd__` in place for mutable types; not always same as `x = x + 1`.

### Q26. Interview

Explain augmented assignment.


## `1e-3` == 0.001; useful for ML learning rates.

### Q27. Interview

Scientific notation literals?


## `1_000_000` readable; ignored by parser.

### Q28. Interview

Underscores in numeric literals?


## Mutable defaults shared across calls — use `None` sentinel instead.

### Q29. Interview

None as default argument pitfall?


## Follows coercion rules; int + float → float. Don't rely on implicit coercion in production.

### Q30. Interview

How Python handles mixed-type ops?
