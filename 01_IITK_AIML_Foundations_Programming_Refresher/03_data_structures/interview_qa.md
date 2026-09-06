# Interview Q&A — Data Structures

> **30 questions** — read aloud, then explain without looking.


## List mutable, slower, more methods. Tuple immutable, hashable if elements hashable, faster.

### Q1. Lists

List vs tuple?


## Shallow copies nested references; deep copies nested objects recursively.

### Q2. Lists

Shallow vs deep copy?


## Unique membership, fast O(1) average lookup, set algebra.

### Q3. Sets

When use set?


## Python 3.7+ preserves insertion order officially.

### Q4. Dicts

Dict insertion order?


## Keys must be immutable/hashable — no list keys.

### Q5. Dicts

Hashable keys requirement?


## [...] builds list; (...) gen exp lazy.

### Q6. Comprehensions

List vs generator comp?


## deque O(1) pops from left; list O(n) pop(0).

### Q7. Performance

 deque vs list for queue?


## Multiset counts; most_common, arithmetic.

### Q8. Collections

Counter use?


## defaultdict auto-creates missing keys with factory.

### Q9. Collections

defaultdict vs dict?


## Lightweight immutable records with named fields.

### Q10. Collections

namedtuple benefit?


## Amortized O(1); insert at front O(n).

### Q11. Interview

Time complexity list append?


## Open addressing in CPython 3.7+; hash table.

### Q12. Interview

Dict collision handling?


## Rarely; dict ordered + move_to_end in OrderedDict for LRU.

### Q13. Interview

OrderedDict still needed?


## Min-heap on list — heappush/heappop for priority queues.

### Q14. Interview

Heapq module?


## Binary search insertion in sorted lists.

### Q15. Interview

bisect module?


## array stores homogeneous compact numeric types.

### Q16. Interview

Array module vs list?


## Can replace segment with different length list.

### Q17. Interview

Slice assignment behavior?


## * captures remainder; works in py3 extended unpacking.

### Q18. Interview

Tuple unpacking star?


## Average O(min(len(s), len(t))) for intersection.

### Q19. Interview

Set operations complexity?


## get cleaner for missing keys; try for exceptional cases.

### Q20. Interview

Dict get vs try/except?


## Stack of dicts for scoped lookups without copying.

### Q21. Interview

ChainMap use?


## Subclass wrappers when need to override methods cleanly.

### Q22. Interview

UserDict/UserList?


## types.MappingProxyType read-only view.

### Q23. Interview

Immutable dict alternative?


## dict, list, str, int, float, bool, None — not set/tuple directly.

### Q24. Interview

JSON serializable types?


## Average O(1), worst case O(n) under hash collisions.

### Q25. Interview

Big-O dict lookup?


## Nested deeply or side effects — use loop for clarity.

### Q26. Interview

When list comprehension too much?


## Python sort is stable — equal elements keep relative order.

### Q27. Interview

Sort stable?


## Computes sort key once per element — efficient pattern.

### Q28. Interview

key= lambda in sort?


## Must decide: overwrite, list values, or reject.

### Q29. Interview

Reverse dict duplicate values?


## __slots__ reduces per-instance dict overhead in classes.

### Q30. Interview

Memory view of slots?
