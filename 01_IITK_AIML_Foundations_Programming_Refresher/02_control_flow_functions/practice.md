# Control Flow & Functions — Practice Questions (50+)

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



## Section 1: if/elif/else

**Q1.** Print 'pass' if score>=60 else 'fail' for score=75.

**Q2.** Classify age into child (<13), teen (<20), adult using if/elif/else.

**Q3.** Check if year is leap year using full rule.

**Q4.** Nested if: login if user=='admin' and pwd=='secret'.

**Q5.** Use ternary to pick max of a,b without max().

**Q6.** Grade letter from score: A>=90, B>=80, C>=70, else F.

**Q7.** Check if triangle sides a,b,c can form valid triangle.

**Q8.** Match sign of number: positive/negative/zero.

**Q9.** Validate email has '@' and '.' after @ (simple check).

**Q10.** Use `match/case` (3.10+) to map HTTP code 404 to 'Not Found'.


## Section 2: Loops

**Q11.** Print squares 1..10 with for loop.

**Q12.** Sum list `[1,2,3,4,5]` with while loop.

**Q13.** Print FizzBuzz 1..15.

**Q14.** Use `enumerate` on `['a','b','c']` print index and value.

**Q15.** Use `zip` to combine names and scores lists.

**Q16.** Break when sum exceeds 10 iterating `[3,4,5,6]`.

**Q17.** Continue skipping evens, print odds 1..10.

**Q18.** Nested loop multiplication table 1..3.

**Q19.** Loop `else`: search for 7 in list, print 'not found' in else.

**Q20.** Iterate dict `{'a':1,'b':2}` keys, values, items.


## Section 3: Functions basics

**Q21.** Define `add(a,b)` returning sum. Call with 3,4.

**Q22.** Function with default arg `greet(name='World')`.

**Q23.** Use *args to sum arbitrary numbers.

**Q24.** Use **kwargs to build formatted string.

**Q25.** Return multiple values: quotient and remainder.

**Q26.** Docstring: write function with docstring and print `help()`.

**Q27.** Lambda to square: apply to 5.

**Q28.** Nested function: outer returns inner that adds n.

**Q29.** Recursion: factorial of 5.

**Q30.** Recursion: Fibonacci nth (simple, n=10).


## Section 4: Scope & advanced

**Q31.** Demonstrate local vs global with `global` keyword.

**Q32.** Use `nonlocal` in nested function to modify outer x.

**Q33.** Function annotation: `def f(x: int) -> int: return x*2`.

**Q34.** Pass function as argument: apply twice.

**Q35.** Implement linear search function returning index or -1.

**Q36.** Write `is_palindrome(s)` ignoring case.

**Q37.** Decorator: simple timer wrapper (use time.sleep demo).

**Q38.** Generator function yielding squares up to n=5.

**Q39.** Use `map` and `filter` on list 1..10: evens squared.

**Q40.** Closure pitfall: fix late-binding in loop creating lambdas.


## Section 5: Challenges

**Q41.** Binary search iterative on sorted list.

**Q42.** Count vowels in string with function.

**Q43.** Merge two sorted lists into one sorted.

**Q44.** Implement `min` manually for list.

**Q45.** Prime check function.

**Q46.** GCD using Euclidean algorithm.

**Q47.** Matrix transpose 2x3 using nested loops.

**Q48.** Parse CSV line `'a,b,c'` into list without split (manual).

**Q49.** Retry decorator pattern: call fn up to 3 times until success.

**Q50.** Implement `chunk_list(lst, size)` → sublists.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.