# Interview Q&A — NumPy

| # | Question | Answer |
|---|----------|--------|
| 1 | Why NumPy over Python lists? | **Speed** (C backend), **memory efficiency**, **vectorized ops**, foundation for pandas/sklearn. |
| 2 | What is broadcasting? | Operating on arrays of **different shapes** by stretching dimensions rules. |
| 3 | `array.shape` vs `array.size`? | **shape** = tuple of dimensions; **size** = total number of elements. |
| 4 | View vs copy in NumPy? | **View** shares memory; **copy** is independent. Slicing often returns a view. |
| 5 | How to handle missing values in NumPy? | `np.nan`, `np.isnan()`, `np.nanmean()`, etc. |
