# Interview Q&A — Object-Oriented Programming & Modules

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. What is the difference between a class and an instance in Python?

**Answer:** A class is a blueprint or prototype defining attributes, methods, and behaviors. An instance is a concrete realization of that class in memory created by calling the class constructor. Multiple instances maintain their own unique instance namespaces while sharing class-level attributes.

### Q2. Explain the difference between '__new__' and '__init__'.

**Answer:** '__new__' is a static method responsible for creating and returning a new instance of the class in memory (object allocation). '__init__' is an instance method called immediately after '__new__' to initialize instance attributes (object initialization). Override '__new__' for singletons or subclassing immutable types like tuple.

### Q3. What is the Method Resolution Order (MRO) and what algorithm determines it?

**Answer:** MRO is the linear sequence of classes Python searches when looking up a method or attribute in multiple inheritance hierarchies. CPython calculates MRO using the C3 Linearization algorithm, which guarantees monotonicity and preserves local precedence order. View it with 'Class.mro()'.

### Q4. What does 'super()' do and why should you avoid hardcoding parent class names?

**Answer:** 'super()' returns a proxy object delegating method calls to the next class in the instance's MRO sequence. Using 'super().method()' avoids hardcoded class references, guarantees proper cooperative multiple inheritance in diamond hierarchies, and prevents diamond methods from running multiple times.

### Q5. Explain the difference between '@classmethod', '@staticmethod', and regular instance methods.

**Answer:** Instance methods take 'self' as their first argument, binding to specific instances and accessing instance state. '@classmethod' takes 'cls' as its first argument, binding to the class itself and commonly used as alternative constructors. '@staticmethod' takes neither 'self' nor 'cls', functioning as a plain function namespaced within the class.

### Q6. What are property decorators ('@property', '@prop.setter') and why are they used?

**Answer:** Properties implement the descriptor protocol, allowing method calls to be accessed with attribute syntax ('obj.price'). They allow adding validation, lazy computation, or deprecation warnings to existing attributes without breaking existing public API contracts.

### Q7. What is the Python Descriptor Protocol?

**Answer:** A descriptor is an object attribute that overrides default attribute access behavior by implementing one or more of '__get__()', '__set__()', or '__delete__()'. Descriptors power properties, classmethods, staticmethods, and ORM fields (like Django/SQLAlchemy models).

### Q8. Explain Abstract Base Classes (ABCs) and how they enforce interface contracts.

**Answer:** Defined via the 'abc' module ('from abc import ABC, abstractmethod'), ABCs define interfaces that derived subclasses MUST implement. Attempting to instantiate a subclass that has not implemented all '@abstractmethod' methods raises a TypeError at instantiation time.

### Q9. What is the difference between '__getattr__' and '__getattribute__'?

**Answer:** '__getattribute__' is invoked unconditionally for EVERY attribute access on an instance. '__getattr__' is a fallback invoked ONLY if the attribute was not found through normal lookup (__dict__, class hierarchy). Overriding '__getattribute__' carelessly causes infinite recursion unless routed through 'super().__getattribute__()'.

### Q10. What are dunder methods '__str__' and '__repr__'?

**Answer:** '__str__' returns a user-friendly, human-readable string representation (invoked by print() and str()). '__repr__' returns an unambiguous, formal representation useful for developers and debugging (invoked by repr() and interactive REPL). Good rule: '__repr__' should ideally be executable Python code.

### Q11. How do you make a custom Python class support indexing ('obj[i]') and slicing?

**Answer:** Implement the '__getitem__(self, key)' and '__setitem__(self, key, value)' dunder methods. When slicing is used, Python passes a slice object ('slice(start, stop, step)') as the key parameter.

### Q12. What does '__call__' do in a Python class?

**Answer:** Implementing '__call__(self, *args, **kwargs)' allows instances of the class to be invoked directly like functions ('obj(x)'). This is commonly used to create stateful callable objects, custom ML layer estimators, and parameterized decorators.

### Q13. How do you make an instance hashable and comparable for use in sets or dict keys?

**Answer:** Implement '__hash__(self)' to return an integer hash and '__eq__(self, other)' to compare equality. Crucially, the attributes contributing to the hash must remain immutable after initialization.

### Q14. What is the purpose of 'hasattr()', 'getattr()', and 'setattr()'?

**Answer:** These built-in reflection functions allow dynamic attribute inspection and manipulation at runtime. 'getattr(obj, 'name', default)' retrieves attributes dynamically; 'hasattr(obj, 'name')' checks existence; 'setattr(obj, 'name', val)' assigns values dynamically.

### Q15. What is the difference between class attributes and instance attributes?

**Answer:** Class attributes are defined directly inside the class body and shared across all instances. Instance attributes are bound to specific instances (typically inside '__init__') via 'self.attr'. Mutating a class attribute via an instance creates a new shadowing instance attribute rather than modifying the shared class attribute.

### Q16. Explain the concept of operator overloading with an example.

**Answer:** Operator overloading allows custom classes to define how native Python operators (+, -, *, ==, <) behave on their instances by implementing corresponding dunder methods (e.g. '__add__' for +, '__sub__' for -, '__mul__' for *, '__lt__' for <).

### Q17. What is multiple inheritance and what is the 'Diamond Problem'?

**Answer:** Multiple inheritance allows a class to inherit from more than one parent. The Diamond Problem occurs when class D inherits from B and C, both of which inherit from A. If B and C override a method from A, ambiguity arises about which method D should execute. Python resolves this deterministically using C3 Linearization (MRO).

### Q18. How does Python implement encapsulation and private attributes?

**Answer:** Python relies on conventions rather than strict compiler enforcement. A single leading underscore ('_attr') indicates a private/internal attribute by convention. A double leading underscore ('__attr') triggers name mangling, transforming the name to '_ClassName__attr' to prevent accidental overrides in subclasses.

### Q19. What is the purpose of '__slots__' in optimizing OOP applications?

**Answer:** By default, Python stores instance attributes in a dynamic '__dict__'. '__slots__' restricts attributes to a predefined set and stores them in a compact C-level array. This eliminates '__dict__' overhead, saving up to 40-50% RAM when creating millions of small model instances or feature nodes.

### Q20. Explain the difference between composition and inheritance. Why is composition often favored?

**Answer:** Inheritance ('is-a' relationship) tightly couples subclasses to parent implementation details. Composition ('has-a' relationship) builds complex objects by assembling independent, decoupled components. Composition provides greater flexibility, prevents fragile base class problems, and makes testing simpler via dependency injection.

### Q21. What is a metaclass in Python?

**Answer:** A metaclass is the 'class of a class'—it defines how classes themselves are constructed, instantiated, and validated. Just as an object is an instance of a class, a class is an instance of a metaclass (by default, 'type'). Metaclasses intercept class creation to enforce schemas (e.g. in ORMs like Django and Pydantic).

### Q22. How does 'type()' function as both an introspection tool and a class factory?

**Answer:** When called with one argument, 'type(obj)' returns the class of obj. When called with three arguments, 'type(name, bases, dict)' dynamically creates and returns a brand-new class at runtime with the specified name, parent classes, and attribute dictionary.

### Q23. What is duck typing and how does 'collections.abc' facilitate it?

**Answer:** Duck typing prioritizes capability over class inheritance. 'collections.abc' provides abstract protocols (like Mapping, Sequence, Iterable). Using 'isinstance(obj, collections.abc.Mapping)' checks whether an object adheres to dictionary protocols without forcing it to inherit from dict.

### Q24. What happens when an exception is raised inside an '__init__' method?

**Answer:** The object's initialization fails, and the exception propagates upward. The partially initialized instance is not bound to the caller's target variable, and its reference count drops to zero, triggering garbage collection and calling '__del__' if '__new__' completed.

### Q25. What is the purpose of '__del__' and why is it tricky to use?

**Answer:** '__del__' is a finalizer called when an instance is about to be destroyed by the garbage collector. It is tricky because the exact timing of garbage collection is non-deterministic, exceptions raised inside '__del__' are ignored and printed to stderr, and circular references can delay cleanup.

### Q26. How do Python packages and modules differ?

**Answer:** A module is a single Python source file (e.g. 'utils.py') containing code. A package is a directory containing multiple modules and sub-packages. Prior to Python 3.3, a package required an '__init__.py' file; Python 3.3+ introduced namespace packages that do not strictly require '__init__.py'.

### Q27. What is the purpose of 'if __name__ == "__main__":'?

**Answer:** Every module in Python has a built-in '__name__' attribute. When a file is run directly from the command line, Python sets '__name__ = "__main__"'. When imported by another script, '__name__' equals the module name. This guard prevents test or execution code from running automatically upon import.

### Q28. How does Python's import system resolve modules via 'sys.path'?

**Answer:** When an 'import foo' statement executes, Python searches directories in 'sys.path' in order: (1) the current script's directory, (2) PYTHONPATH environment variable directories, (3) standard library directories, and (4) third-party site-packages. It caches imported modules in 'sys.modules'.

### Q29. What is a circular import in Python and how do you resolve it?

**Answer:** A circular import occurs when module A imports module B, which in turn imports module A before module A has finished initializing. Resolutions: (1) refactor shared dependencies into a third module C, (2) move the import statement inside the specific function that uses it, or (3) import the module itself rather than specific symbols.

### Q30. Explain the difference between absolute imports and relative imports.

**Answer:** Absolute imports specify the full path from the project's root directory ('from mypkg.subpkg import module'). Relative imports use leading dots to navigate from the current module's location ('.module' for same directory, '..module' for parent directory), making packages relocatable.
