# OOP & Modules — Practice Questions (50+)

## How to practice

1. **Start Jupyter from repo root** so dataset paths resolve:
   ```bash
   cd Data_science
   jupyter notebook
   ```
2. Open this notebook (`practice.ipynb`) in the subtopic folder.
3. Run the **Setup** cell — it finds `datasets/shared/` automatically.
4. Read `basics.md` for syntax, then solve **one question at a time** in the code cell below it.
5. Do **not** peek at `solutions.ipynb` until you have a working attempt (even if wrong).
6. After all questions, review `interview_qa.md` aloud as mock interview answers.
7. **Pro tip:** Re-solve Q41–Q50 from memory the next day without notes.

**Target:** 50 questions × 5 courses ≈ **1,250+ problems** in this repo (+ NumPy/Pandas/Matplotlib/Seaborn banks).



## Section 1: Classes

**Q1.** Define class `Person` with __init__(name, age) and method introduce().

**Q2.** Add class variable `species='Homo sapiens'` shared by all Person.

**Q3.** Implement `__str__` and `__repr__` for Person.

**Q4.** Property: make `age` with getter/setter validation >=0.

**Q5.** classmethod `from_birth_year` alternative constructor.

**Q6.** staticmethod `is_adult(age)` on Person.

**Q7.** Inheritance: Student(Person) adds student_id.

**Q8.** Method overriding: Student introduce includes sid.

**Q9.** Use isinstance and issubclass checks.

**Q10.** Abstract base: Animal with abstract speak using ABC.


## Section 2: Encapsulation

**Q11.** Private convention: _balance and name mangling __secret.

**Q12.** Implement deposit/withdraw with balance check.

**Q13.** Read-only property balance.

**Q14.** Composition: Car has Engine object.

**Q15.** Dataclass for Point x,y with auto __init__.

**Q16.** Slots to restrict attributes.

**Q17.** Enum for Status ACTIVE/INACTIVE.

**Q18.** Magic __len__ on custom collection.

**Q19.** Magic __eq__ compare two Points.

**Q20.** Iterator protocol __iter__/__next__.


## Section 3: Modules

**Q21.** Import math and use sqrt, pi.

**Q22.** from collections import Counter, defaultdict.

**Q23.** Create module-level __all__ = ['foo'] pattern demo.

**Q24.** Use if __name__ == '__main__' in module pattern.

**Q25.** Import with alias: import pandas as pd.

**Q26.** Relative import concept — package __init__.py role.

**Q27.** Use pathlib.Path in module style.

**Q28.** Virtual env purpose — print sys.prefix.

**Q29.** pip install -r requirements.txt purpose.

**Q30.** Use functools.lru_cache on recursive fib.


## Section 4: Polymorphism

**Q31.** Duck typing: classes with draw() method.

**Q32.** Operator overloading __add__ for Vector.

**Q33.** Multiple inheritance MRO print.

**Q34.** Protocol/duck: file-like read().

**Q35.** Template method pattern skeleton.

**Q36.** Strategy pattern with callables.

**Q37.** Singleton pattern (simple module-level).

**Q38.** Factory function create_shape('circle').

**Q39.** Mixin class LoggerMixin.

**Q40.** Compare @dataclass vs regular class boilerplate.


## Section 5: Projects

**Q41.** Design ExpenseTracker class: add, total, list.

**Q42.** Task class with priority enum and done flag.

**Q43.** TaskManager add/complete/list_pending.

**Q44.** Serialize ExpenseTracker to dict JSON-ready.

**Q45.** Validate input in setter (positive amount).

**Q46.** Unit test style assert for TaskManager.

**Q47.** Use typing.List, Optional in class hints.

**Q48.** Context manager class __enter__/__exit__.

**Q49.** Compare inheritance vs composition for TaskManager storage.

**Q50.** Design class diagram in comments for portfolio project.

**Q51.** Refactor procedural code into OOP — wrap sales summary.

---
**Total: 51 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.