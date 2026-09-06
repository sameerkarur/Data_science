# Variables, Data Types & Operators — Practice Questions (50+)

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



## Section 1: Types & introspection

**Q1.** Create variables `a=10`, `b=3.14`, `c='hello'`, `d=True`. Print each value with its type.

**Q2.** Use `isinstance` to check if `x=42` is int or float. Print both checks.

**Q3.** Convert string `'123'` to int, float, and bool. Print results.

**Q4.** What happens converting `'0'` and `''` to bool? Demonstrate with print.

**Q5.** Create `n=None`. Check `n is None` vs `n == None`. Print both.

**Q6.** Print `id()` of two variables referencing same list vs two equal lists.

**Q7.** Use `type()` chaining: create `x=5`, then reassign `x='five'`. Print type before and after.

**Q8.** Demonstrate integer division `//` vs true division `/` for `7/2` and `7//2`.

**Q9.** Compute modulo, power: `17 % 5` and `2 ** 10`.

**Q10.** Use `divmod(17, 5)` and unpack quotient and remainder.


## Section 2: Operators

**Q11.** Evaluate and print: `3 + 4 * 2`, `(3+4)*2`, and operator precedence demo.

**Q12.** Compare `==` vs `is` for two strings `'aiml'` created separately.

**Q13.** Use chained comparison: check if `1 < x < 10` for `x=5`.

**Q14.** Demonstrate short-circuit: `False and print('no')` — explain via safe version.

**Q15.** Use walrus operator `:=` to assign and test `n=10` in one expression inside print.

**Q16.** Bitwise AND/OR on `5 & 3` and `5 | 3`. Print binary with `bin()`.

**Q17.** Shift bits: `8 << 2` and `8 >> 1`.

**Q18.** Use `round(3.14159, 2)` and `round(2.675, 2)` — print both.

**Q19.** Compute absolute value with `abs(-42)` and `pow(2,8)`.

**Q20.** Assign augmented ops: `x=10`, then `x+=5`, `x*=2`. Print final x.


## Section 3: Strings

**Q21.** Create multi-line string with triple quotes containing your name and course.

**Q22.** Slice `'Data Science'` to get `'Data'`, `'Science'`, and reverse full string.

**Q23.** Use f-string to print `Sales: $1234.5` formatted to 2 decimals.

**Q24.** Split `'a,b,c,d'` on comma and join back with `'-'`.

**Q25.** Strip whitespace from `'  hello  '` and convert to upper/lower.

**Q26.** Check if `'python'` starts with `'py'` and ends with `'on'`.

**Q27.** Replace `'cat'` with `'dog'` in `'the cat sat'`.

**Q28.** Find index of `'a'` in `'banana'` using `.find()` and `.index()`.

**Q29.** Use `in` to test substring `'data' in 'metadata'`.

**Q30.** Format with `.format()`: `'{} scored {}'`.format('Alice', 99).


## Section 4: Casting & validation

**Q31.** Safely convert user string `'42.5'` to float; handle `'bad'` with try/except.

**Q32.** Check if string `'12345'` is numeric using `.isdigit()`.

**Q33.** Convert list of string numbers `['1','2','3']` to ints with list comprehension.

**Q34.** Use `bool()` on `[0], [1], {}, {0}, ''`. Print all.

**Q35.** Swap two variables `a=1, b=2` without temp variable.

**Q36.** Unpack tuple `(2024, 9, 4)` into year, month, day.

**Q37.** Create constants PI=3.14159 (convention) and compute circle area r=5.

**Q38.** Demonstrate float precision issue: `0.1 + 0.2 == 0.3` and fix with round.

**Q39.** Use `complex(2,3)` and print real/imag parts.

**Q40.** Convert int 255 to hex and binary strings.


## Section 5: Mixed challenges

**Q41.** Write expression for Celsius to Fahrenheit: C=25.

**Q42.** Compute BMI given weight=70kg height=1.75m. Print rounded to 1 decimal.

**Q43.** Given `seconds=3661`, print hours, minutes, seconds using divmod.

**Q44.** Check leap year rule for year=2024 ( divisible by 4, except centuries unless /400).

**Q45.** Build string `'Item: {}, Price: ${:.2f}'` for 3 products in a loop.

**Q46.** Use `ord('A')` and `chr(65)` to show ASCII mapping.

**Q47.** Evaluate truthiness of `None or 0 or '' or 'ok'`.

**Q48.** Create variable annotations: `count: int = 0` and print annotation.

**Q49.** Simulate `input` with variable `user_name = 'Alex'` and greet.

**Q50.** Combine types in f-string: int, float, bool in one line.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.