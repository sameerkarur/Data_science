# Interview Q&A — Pandas Data Wrangling & Analysis

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Explain the internal memory layout of a Pandas DataFrame.

**Answer:** A Pandas DataFrame is organized into a 2D tabular structure backed by a BlockManager (or PyArrow Array in Pandas 2.0+). The BlockManager groups columns of identical datatypes (e.g. all float64 columns) into contiguous 2D NumPy arrays, while Series are 1D arrays with labeled indices.

### Q2. What is the difference between 'loc' and 'iloc' in Pandas?

**Answer:** 'loc' is label-based indexing: it selects rows and columns using index labels and column names (inclusive of both start and end endpoints). 'iloc' is integer-position-based indexing: it selects elements using 0-based integer coordinates (exclusive of the stop endpoint, following Python slice conventions).

### Q3. What are 'at' and 'iat' accessors and when should you use them?

**Answer:** 'at[row_label, col_label]' and 'iat[row_idx, col_idx]' provide ultra-fast scalar access to a single element. Because they bypass Series construction and overhead checks in loc/iloc, scalar access is up to 10x faster inside performance-critical loops.

### Q4. Explain the 'Split-Apply-Combine' strategy in Pandas 'groupby()'.

**Answer:** Split: the DataFrame is partitioned into groups based on key columns. Apply: an aggregation (e.g. sum, mean), transformation (e.g. z-score per group), or filter is applied to each independent group. Combine: the individual group results are concatenated back into a single aligned DataFrame/Series.

### Q5. What is the difference between 'df.groupby().aggregate()', 'transform()', and 'filter()'?

**Answer:** 'aggregate()' reduces each group to a single summary row per group. 'transform()' returns an object of the EXACT SAME row count as the original DataFrame, broadcasting group statistics back to individual rows. 'filter()' discards entire groups that fail a boolean predicate.

### Q6. How does 'pd.merge()' perform inner, outer, left, and right joins?

**Answer:** Inner: retains only rows with matching keys in both DataFrames. Left: retains all rows from the left DataFrame, filling unmatched right columns with NaN. Right: retains all rows from the right DataFrame. Outer: retains all rows from both DataFrames, filling missing values with NaN on either side.

### Q7. What is MultiIndex (hierarchical index) and how do you unstack it?

**Answer:** A MultiIndex allows indexing along multiple hierarchical dimensions (levels). 'df.unstack(level=-1)' pivots an inner row index level into columns; 'df.stack()' pivots column headers into the innermost row index.

### Q8. How does 'pivot_table()' differ from 'pivot()' in Pandas?

**Answer:** 'pivot()' reshapes data without aggregation; it throws a ValueError if duplicate key combinations exist. 'pivot_table()' handles duplicates by applying an aggregation function ('aggfunc='mean'' by default) and supports margins (subtotals).

### Q9. Explain 'pd.melt()' and why it is essential for tidying wide data.

**Answer:** 'pd.melt()' unpivots a DataFrame from wide format (many columns) to long format (few columns with identifier variables and value-variable pairs), which is the standard format required for tidy data analysis, Seaborn plotting, and relational modeling.

### Q10. What is the difference between 'df.dropna(how="any")' and 'df.dropna(how="all")'?

**Answer:** 'how="any"' drops a row/column if even a single NaN value is present. 'how="all"' drops a row/column ONLY if every single value along that axis is NaN. The 'subset' parameter restricts checks to designated columns.

### Q11. How does 'df.fillna(method="ffill")' differ from 'bfill'?

**Answer:** 'ffill' (forward-fill) propagates the last known valid observation forward until the next valid value is found (essential for non-stationary time series). 'bfill' (backward-fill) propagates the next valid value backward.

### Q12. What is interpolation in Pandas and when is it preferred over simple fillna?

**Answer:** 'df.interpolate(method="linear" or "spline")' estimates missing values by fitting continuous mathematical curves between surrounding points, preserving smooth trends in continuous time-series sensor or financial data.

### Q13. Explain Categorical dtypes and how they conserve RAM.

**Answer:** Categorical columns store data as an integer array of codes (0, 1, 2) pointing to a small index of unique categories. For low-cardinality string columns repeated across millions of rows, memory drops by up to 90% and grouping speed increases significantly.

### Q14. What is the difference between 'df.isin()' and 'df.query()'?

**Answer:** 'df[col.isin([val1, val2])]' filters for membership using a list. 'df.query("col in [val1, val2]")' evaluates expressions using NumExpr, avoiding intermediate memory allocations and offering cleaner syntax for complex compound filters.

### Q15. Why should you avoid using Python's 'for index, row in df.iterrows():'?

**Answer:** 'iterrows()' converts every row into a new Pandas Series object, which discards dtypes (casting mixed columns to object) and incurs massive Python function overhead. Instead, use vectorized operations, list comprehensions, or 'itertuples()'.

### Q16. What does 'df.itertuples(index=False, name="Row")' return and why is it faster?

**Answer:** 'itertuples()' generates lightweight namedtuples directly in C, preserving native column data types and running ~50x to 100x faster than iterrows().

### Q17. Explain how to downcast numerical data types to optimize memory.

**Answer:** Use 'pd.to_numeric(col, downcast="integer")' or 'downcast="float"'. It automatically downcasts 64-bit types (int64, float64) to the smallest lossless integer (int8, int16, int32) or float (float32), dramatically reducing memory footprint.

### Q18. What is the purpose of 'df.memory_usage(deep=True)'?

**Answer:** Standard memory_usage() reports only pointers for object columns. 'deep=True' inspects the system heap recursively to measure the exact byte size of underlying string and Python objects, providing true RAM metrics.

### Q19. How do window functions work in Pandas (rolling, expanding, exponential)?

**Answer:** 'df.rolling(window=7).mean()' computes moving averages over fixed sliding windows. 'df.expanding().sum()' computes cumulative metrics from the start of data. 'df.ewm(span=20).mean()' computes exponential moving averages with decay weights.

### Q20. What is the difference between 'df.corr()' methods ('pearson', 'spearman', 'kendall')?

**Answer:** 'pearson' evaluates linear correlation on continuous data. 'spearman' evaluates monotonic relationships on ranked values. 'kendall' (Kendall's tau) evaluates concordance based on pairwise ranks, offering superior robustness for small datasets with ties.

### Q21. Explain the difference between 'pd.concat(axis=0)' and 'pd.concat(axis=1)'.

**Answer:** 'axis=0' appends rows vertically, aligning columns by name and filling missing columns with NaN. 'axis=1' concatenates columns horizontally, aligning rows by index labels.

### Q22. What is 'df.duplicated(keep="first")' vs 'keep=False'?

**Answer:** 'keep="first"' flags duplicates as True starting from the second occurrence onward. 'keep=False' flags ALL occurrences of duplicated rows as True, enabling complete extraction or deletion of all ambiguous non-unique records.

### Q23. How does 'pd.crosstab()' differ from 'pivot_table()'?

**Answer:** 'pd.crosstab()' calculates frequency count contingency tables between two or more categorical factors by default, optimized for categorical interactions and Chi-Square preparation. 'pivot_table()' is designed for general numeric aggregations.

### Q24. Explain how string operations work via the '.str' accessor.

**Answer:** The '.str' accessor provides vectorized string operations that handle NaNs safely: '.str.strip()', '.str.lower()', '.str.extract(r'(\d+)')', and '.str.contains(regex)'. It executes without manual Python string iteration.

### Q25. How do datetime accessors work via the '.dt' accessor?

**Answer:** When a column is datetime64, '.dt' exposes vectorized calendar operations: '.dt.year', '.dt.month', '.dt.day_name()', '.dt.quarter', and '.dt.hour', enabling rapid feature engineering for seasonal trends.

### Q26. What is 'pd.DateOffset' and how does it support business calendar arithmetic?

**Answer:** DateOffset allows adding/subtracting semantic business intervals (e.g. 'df["Date"] + pd.DateOffset(months=3)' or 'pd.offsets.BDay(5)' for 5 business days, skipping weekends and holidays automatically).

### Q27. What is the difference between 'df.eval()' and 'df.query()'?

**Answer:** 'df.query()' filters rows using string boolean expressions. 'df.eval()' executes column arithmetic (e.g. 'df.eval("C = A + B * 2")') using NumExpr, accelerating matrix math on large DataFrames by computing directly in CPU cache.

### Q28. How does 'df.replace()' differ from 'df.fillna()'?

**Answer:** 'fillna()' replaces ONLY NaN/null values. 'replace(to_replace, value)' replaces arbitrary specific values, regex patterns, or lists of values throughout the DataFrame.

### Q29. What is 'pd.get_dummies()' and what is the purpose of 'drop_first=True'?

**Answer:** 'pd.get_dummies()' one-hot encodes categorical variables into binary indicator columns. 'drop_first=True' removes the first category column, preventing the 'Dummy Variable Trap' (perfect multicollinearity where the dropped category is represented when all other indicators are 0).

### Q30. What is the Arrow engine in Pandas 2.0 and what are its advantages?

**Answer:** Pandas 2.0 allows DataFrames to be backed by Apache Arrow memory buffers instead of NumPy. Arrow provides native support for missing data without type coercion, high-performance string manipulation, PyCapsule zero-copy data sharing with DuckDB/Polars, and multithreaded I/O.
