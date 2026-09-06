# Interview Q&A — Data Structures (Lists, Dicts, Tuples, Sets)

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the internal memory layout of a Python list.

**Answer:** A Python list is a dynamic contiguous array of pointers (references) to objects stored elsewhere on the heap, not a linked list or contiguous block of raw values. When resized, CPython over-allocates memory geometrically (0, 4, 8, 16, 24, 32...) to ensure appending achieves O(1) amortized time complexity.

### Q2. What is the time complexity of common list operations: append, insert(0), pop(), pop(0), and index search?

**Answer:** append() is O(1) amortized; pop() from end is O(1); insert(0) and pop(0) are O(N) because every existing pointer must be shifted in memory; in-operator linear search ('x in my_list') is O(N).

### Q3. Why is 'collections.deque' preferred over list for queue/FIFO operations?

**Answer:** deque (double-ended queue) is implemented as a doubly linked list of fixed-size memory blocks. Appending and popping from either end ('appendleft()', 'popleft()') is strictly O(1) time complexity, whereas list.pop(0) requires shifting all N elements (O(N)).

### Q4. How does a Python dictionary achieve O(1) average lookup time?

**Answer:** Python dictionaries use a hash table. When inserting a key-value pair, Python computes hash(key) to determine an index in an underlying sparse array. If no collision occurs, lookup and insertion take O(1) time. In Python 3.7+, dicts use a compact array representation that preserves insertion order while reducing memory by ~25%.

### Q5. How does CPython resolve hash collisions in dictionaries?

**Answer:** CPython uses open addressing with a pseudo-random probing sequence: index = (5*index + 1 + perturb) >> 5. This probing sequence visits all slots in the table systematically without the severe clustering typical of simple linear probing.

### Q6. What happens if you insert a mutable object as a dictionary key or set element?

**Answer:** Python raises a TypeError: unhashable type. Dictionary keys and set elements must implement __hash__() and __eq__() so their hash remains constant; mutating a key in-place would alter its hash code, making it impossible to locate in the hash table bucket.

### Q7. Explain the differences between sets and lists.

**Answer:** Sets contain unique, unordered, hashable elements implemented via hash tables, providing O(1) average membership tests ('x in s'). Lists maintain insertion order, allow duplicates, store arbitrary mutable/immutable objects, but require O(N) linear time for membership checks.

### Q8. What mathematical set operations are natively supported in Python?

**Answer:** Union ('s1 | s2' or s1.union(s2)), Intersection ('s1 & s2'), Difference ('s1 - s2'), Symmetric Difference ('s1 ^ s2' elements in either but not both), and subset/superset checks ('s1 <= s2').

### Q9. What is a 'collections.defaultdict' and how does it prevent KeyError?

**Answer:** defaultdict is a subclass of dict that overrides __missing__(key). When a requested key is missing, it calls the factory callable provided during instantiation (e.g. int, list, set) to generate and insert a default value automatically without throwing a KeyError.

### Q10. What is 'collections.Counter' and how is it used in data science?

**Answer:** Counter is a dict subclass designed for counting hashable elements. It maps items to their integer frequencies and provides convenience methods like 'most_common(k)' which uses a heap to retrieve top-k elements in O(N log k) time.

### Q11. Explain how 'heapq' works in Python and what heap invariant it maintains.

**Answer:** The 'heapq' module implements binary min-heaps over standard Python lists where heap[0] is always the smallest element (invariant: heap[k] <= heap[2*k+1] and heap[k] <= heap[2*k+2]). Pushing and popping take O(log N) time, making it ideal for priority queues and Dijkstra's algorithm.

### Q12. What is the difference between sorting with 'list.sort()' and the 'sorted()' built-in?

**Answer:** 'list.sort()' sorts the list in-place and returns None, modifying the original object with zero extra memory allocation. 'sorted(iterable)' accepts any iterable and returns a new sorted list, leaving the original sequence unchanged.

### Q13. What sorting algorithm does Python use and what is its time complexity?

**Answer:** Python uses Timsort (an adaptive stable merge/insertion sort hybrid). It achieves O(N log N) worst-case and average-case time complexity, but drops to O(N) linear time on data that is already sorted or partially sorted, with O(N) space complexity.

### Q14. What does it mean that Python's sorting algorithm is 'stable'?

**Answer:** A sorting algorithm is stable if elements with identical sort keys retain their original relative order in the sorted output. This allows multi-level sorting by sorting sequentially on secondary keys and then primary keys.

### Q15. Explain the 'key' argument in 'sorted()' or 'max()'.

**Answer:** The 'key' parameter accepts a single-argument callable applied to each element before comparison (e.g. 'sorted(students, key=lambda s: s['gpa'])'). The key function is called exactly once per element (unlike old cmp functions called O(N log N) times), maximizing speed.

### Q16. How does the 'bisect' module work and what is its time complexity?

**Answer:** The 'bisect' module provides binary search algorithms for sorted sequences: 'bisect_left()' and 'bisect_right()' locate insertion points in O(log N) time, while 'insort()' inserts items maintaining sorted order in O(N) time due to underlying array shifting.

### Q17. What is a tuple and why should you use it instead of a list when data is static?

**Answer:** A tuple is an immutable sequence of fixed length. Tuples consume less memory than lists because they do not require over-allocation for growth, are hashable (can be used as dict keys or set members), and convey architectural intent that the collection is read-only.

### Q18. Explain list slicing and whether it creates a copy or a view.

**Answer:** In standard Python, list slicing ('sub = my_list[1:5]') creates a brand-new shallow copy of the specified pointer slice. Mutating the sublist does not affect the original list. In contrast, NumPy array slicing creates a view sharing the same memory buffer.

### Q19. How do dictionary views ('keys()', 'values()', 'items()') behave in Python 3?

**Answer:** Dictionary views provide dynamic, read-only windows into the dictionary's current state: if the underlying dictionary is modified, the views reflect those changes immediately. 'dict.keys()' and 'dict.items()' support set-like operations (union, intersection).

### Q20. What is an OrderedDict and is it still necessary in Python 3.7+?

**Answer:** Since Python 3.7, standard dicts are guaranteed to maintain insertion order. However, 'collections.OrderedDict' remains useful because it provides order-sensitive equality comparisons and specialized methods like 'move_to_end(key, last=True)' and 'popitem(last=False)' (O(1) FIFO popping).

### Q21. What is a 'frozenset' and when should it be used?

**Answer:** A frozenset is an immutable, hashable version of a set. Because it cannot be modified after creation, it can be used as a dictionary key or nested as an element within another set (enabling sets of sets).

### Q22. How do you merge two dictionaries in Python 3.9+?

**Answer:** Using the merge union operator: 'merged = dict1 | dict2'. To update in-place, use the update operator: 'dict1 |= dict2'. Keys in dict2 overwrite keys in dict1 if collisions occur.

### Q23. Explain dictionary comprehension with an example.

**Answer:** Dict comprehension constructs a new dictionary concisely: '{k: v for k, v in iterable if condition}'. For example, inverting a dictionary: '{v: k for k, v in original.items()}' assuming all values are unique and hashable.

### Q24. What is the time complexity of checking membership in a list vs a set?

**Answer:** 'x in my_list' is O(N) because Python must scan elements sequentially until a match is found. 'x in my_set' is O(1) on average because Python computes hash(x) and jumps directly to the corresponding hash bucket.

### Q25. How does memory allocation differ between a list of 1,000,000 integers in Python vs a NumPy array?

**Answer:** A Python list of 1,000,000 ints stores 1,000,000 8-byte pointers pointing to separate 28-byte PyObject integer structs scattered across memory (~36 MB total). A NumPy 64-bit integer array stores 1,000,000 raw 8-byte binary integers contiguously in a single 8 MB buffer (~4.5x smaller, cache-coherent).

### Q26. What is 'collections.ChainMap' and how does it resolve lookups?

**Answer:** ChainMap groups multiple dictionaries into a single logical view without copying data. Lookups search each underlying mapping sequentially and return the value from the first dictionary containing the key. Ideal for managing scoped configuration (CLI args > env vars > defaults).

### Q27. How do you remove duplicates from a list while preserving element order?

**Answer:** In Python 3.7+, use 'list(dict.fromkeys(my_list))'. It runs in O(N) time and preserves insertion order because standard dictionaries are insertion-ordered and deduplicate keys automatically.

### Q28. Explain shallow copy vs deep copy in the context of nested lists: '[[0]*3]*3'.

**Answer:** '[[0]*3]*3' creates an outer list containing three references to the EXACT SAME inner list. Modifying matrix[0][0] = 1 will mutate all three rows simultaneously. The correct idiom is '[[0]*3 for _ in range(3)]', which instantiates three distinct inner list objects.

### Q29. What is the difference between 'my_list.clear()' and 'my_list = []'?

**Answer:** 'my_list.clear()' mutates the existing list object in-place, clearing all elements; any other variables referencing that same list will see the cleared list. 'my_list = []' creates a new empty list and rebinds the name, leaving other references to the original list untouched.

### Q30. How does Python implement hash tables to prevent Denial of Service (HashDoS) attacks?

**Answer:** Python uses SipHash with a randomized secret salt generated per process startup. This prevents malicious actors from crafting pre-computed collision inputs that force dictionary operations from O(1) down to O(N^2) worst-case degradation.
