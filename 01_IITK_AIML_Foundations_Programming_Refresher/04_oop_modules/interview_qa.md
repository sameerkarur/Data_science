# Interview Q&A — OOP & Modules

> **30 questions** — read aloud, then explain without looking.


## OOP basics

### Q1. What is a class vs object?

A **class** is a blueprint; an **object** is an instance with its own state and behavior.

### Q2. Explain encapsulation, inheritance, polymorphism.

Encapsulation hides internals; inheritance reuses/extends parent; polymorphism — same interface, different behavior.


## Methods

### Q3. @classmethod vs @staticmethod?

classmethod receives cls (alternate constructors); staticmethod is a namespaced utility without cls/self.

### Q4. What is `@property`?

Controlled attribute access with getter/setter validation.


## Design

### Q5. Composition vs inheritance?

Prefer has-a over is-a when reuse doesn't imply subtype.

### Q6. SOLID summary?

SRP, OCP, Liskov substitution, Interface segregation, Dependency inversion.

### Q7. When use dataclass?

Reduce boilerplate for data-holding classes with auto __init__/__repr__.


## Magic methods

### Q8. __str__ vs __repr__?

__repr__ developer-unambiguous; __str__ user-readable.


## Modules

### Q9. `if __name__ == '__main__'`?

Runs only when file executed as script, not imported.


## Patterns

### Q10. Decorator pattern in Python?

@decorator wraps functions to add logging, timing, auth, etc.


## Interview

### Q11. MRO?

Method resolution order — C3 linearization for multiple inheritance.

### Q12. __slots__?

Fixed attributes — saves memory, blocks arbitrary attrs.

### Q13. Iterator vs generator?

Both lazy; generator uses yield syntax, simpler for streams.

### Q14. Abstract base classes?

Enforce interface via @abstractmethod before instantiation.

### Q15. Testing classes?

Test public methods; mock external I/O and dependencies.

### Q16. Type hints on methods?

Document contracts; use mypy for static checks.

### Q17. Enum use case?

Named constants — status codes, categories, safe comparisons.

### Q18. Context manager on class?

Implement __enter__/__exit__ for resource cleanup.

### Q19. Namedtuple vs dataclass?

namedtuple immutable; dataclass flexible defaults/mutability.

### Q20. Factory function pattern?

Hide concrete class construction behind create_* API.

### Q21. Strategy pattern?

Swap algorithms at runtime via shared interface.

### Q22. Mixin classes?

Small reusable bases mixed into multiple subclasses.

### Q23. Private name mangling?

__attr triggers _ClassName__attr — convention not security.

### Q24. super() in diamond inheritance?

Follows MRO — calls next class in order, not only direct parent.

### Q25. Immutable objects?

tuple, frozenset, str — safe as dict keys when hashable.

### Q26. Dependency injection?

Pass dependencies into __init__ instead of hard-coding globals.

### Q27. Expenses/Task project OOP?

Model domain entities as classes; separate persistence layer.

### Q28. When NOT to use OOP?

Simple scripts, one-off ETL — functions may suffice.

### Q29. Duck typing?

If it quacks like a duck — behavior matters, not explicit type hierarchy.

### Q30. Metaclass (brief)?

Class of a class — advanced, rarely needed in DS day-to-day.
