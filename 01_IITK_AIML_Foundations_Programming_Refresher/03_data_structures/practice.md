# Data Structures — Practice Questions (50+)

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



## Section 1: Lists

**Q1.** Create list 1..10. Print first 3, last 2, reverse.

**Q2.** Append 99, extend [100,101], insert 0 at index 0.

**Q3.** Remove value 2, pop last, del index 0.

**Q4.** Sort descending without sort() — use sorted().

**Q5.** List comprehension: squares of evens 0..20.

**Q6.** Nested list 3x3 identity matrix.

**Q7.** Flatten nested list [[1,2],[3,4]].

**Q8.** Copy list shallow vs deep for nested.

**Q9.** Find index of 4 in list; count occurrences of 1.

**Q10.** Rotate list [1,2,3,4] left by 1.


## Section 2: Tuples & sets

**Q11.** Create tuple, unpack, swap with tuple.

**Q12.** Set from list with duplicates; union/intersection.

**Q13.** frozenset as dict key demo.

**Q14.** Check subset/superset.

**Q15.** Symmetric difference of sets.

**Q16.** Named tuple Point(x,y) distance from origin.

**Q17.** Tuple unpacking with *rest.

**Q18.** Set comprehension {x%3 for x in range(10)}.

**Q19.** Remove duplicates preserving order.

**Q20.** Merge two sets into sorted list.


## Section 3: Dictionaries

**Q21.** Create dict, access, get with default.

**Q22.** Loop items; build reverse dict values→keys.

**Q23.** Merge dicts with | (3.9+) or update.

**Q24.** defaultdict list — group items.

**Q25.** Counter on 'hello'.

**Q26.** Sort dict by value descending.

**Q27.** Dict comprehension: square keys 1..5.

**Q28.** Pop item with default.

**Q29.** Setdefault to accumulate counts manually.

**Q30.** Nested dict access with .get chain safe.


## Section 4: Comprehensions

**Q31.** Dict comp filter: even squares.

**Q32.** Set comp: len words > 3.

**Q33.** Nested list comp  multiplication table.

**Q34.** Replace map/filter with comprehension.

**Q35.** Transpose matrix via zip.

**Q36.** Enumerate into dict.

**Q37.** Filter None from list comp.

**Q38.** Walrus in comprehension (if valid).

**Q39.** Pairwise sums with zip.

**Q40.** Invert dict with duplicate values handling.


## Section 5: Challenges

**Q41.** Two-sum: find pair summing to target.

**Q42.** Anagram check using Counter.

**Q43.** LRU cache manual OrderedDict.

**Q44.** Stack using list push/pop.

**Q45.** Queue using collections.deque.

**Q46.** Frequency top-k words.

**Q47.** Merge overlapping intervals.

**Q48.** Group by key function.

**Q49.** Deep get path 'a.b.c' in nested dict.

**Q50.** Rotate matrix 90 degrees.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.