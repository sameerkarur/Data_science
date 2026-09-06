# Interview Q&A — Python Essentials for Data Science

> **30 High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.

### Q1. Why is vectorization in NumPy/Pandas faster than pure Python for-loops?

**Answer:** Vectorization delegates array computations to pre-compiled C/Fortran routines that operate on contiguous memory buffers, leveraging hardware-level SIMD (Single Instruction, Multiple Data) CPU vector registers and eliminating Python bytecode interpretation, dynamic type checking, and reference counting per element.

### Q2. Explain the memory difference between a Python list and a NumPy array.

**Answer:** A Python list stores references (pointers) to disparate PyObject instances on the heap, incurring 28+ bytes per integer and destroying CPU L1/L2 cache locality. A NumPy array stores raw binary data contiguously in a single contiguous memory block with uniform data types, maximizing memory efficiency and cache hits.

### Q3. What are NumPy Universal Functions (ufuncs)?

**Answer:** Ufuncs are vectorized wrappers around functions that operate element-wise on ndarrays in compiled C code, supporting broadcasting, type casting, and reduction operations (e.g. np.add.reduce()).

### Q4. How does broadcasting allow operations on arrays of differing shapes?

**Answer:** Broadcasting stretches smaller dimensions to match larger dimensions without copying data if, starting from trailing dimensions: (1) dimensions are equal, or (2) one of the dimensions is 1.

### Q5. What is the difference between an in-place operation and returning a new array in Pandas/NumPy?

**Answer:** In-place operations ('inplace=True' or 'arr += 1') modify the existing memory buffer directly, saving memory allocation. Returning a new array leaves the original intact and allocates a separate block of memory, which is safer for functional immutability.

### Q6. Explain method chaining in Pandas and why it is used.

**Answer:** Method chaining combines multiple data transformations in a single fluent expression (e.g. 'df.query().assign().groupby().mean()'), improving readability, avoiding intermediate variable pollution, and simplifying debugging.

### Q7. What is the difference between 'apply()' and vectorized Pandas operations?

**Answer:** Vectorized Pandas operations (built on C/NumPy) process arrays in compiled code. 'apply()' iterates over rows/columns in Python bytecode, executing a Python function call per row/column and running 10x–100x slower.

### Q8. How do categorical data types in Pandas optimize memory?

**Answer:** Categorical types replace repetitive string objects with integer codes referencing a small dictionary of unique categories. For low-cardinality string columns (e.g. State, Department), this can reduce memory usage by 80–90%.

### Q9. What is the difference between shallow copy and deep copy in Pandas?

**Answer:** 'df.copy(deep=False)' copies DataFrame indices and structure but shares underlying data buffers. 'df.copy(deep=True)' creates an independent copy of both structure and data.

### Q10. How does Pandas handle datetime operations and time series resampling?

**Answer:** Pandas converts date strings to 64-bit nanosecond integers ('datetime64[ns]'). The 'dt' accessor provides vectorized date arithmetic, and '.resample()' enables frequency conversion (e.g. daily to monthly aggregation).

### Q11. Explain the difference between 'merge()', 'join()', and 'concat()' in Pandas.

**Answer:** 'merge()' performs relational SQL-like joins on arbitrary columns. 'join()' joins DataFrames primarily on their indices. 'concat()' stacks DataFrames along an axis (0 for rows, 1 for columns).

### Q12. How does Pandas resolve missing values ('None', 'np.nan', 'pd.NA')?

**Answer:** 'np.nan' is a float representing IEEE 754 NaN. In older Pandas, integer columns with missing values were forced to float64. Modern Pandas introduces 'pd.NA' and nullable data types (Int64, boolean, string) to preserve true types.

### Q13. What is multi-indexing (hierarchical indexing) in Pandas?

**Answer:** Multi-indexing allows DataFrames and Series to maintain multiple index levels along rows or columns, enabling representation of higher-dimensional data in 2D tabular form.

### Q14. What is the purpose of 'df.itertuples()' over 'df.iterrows()'?

**Answer:** 'df.itertuples()' yields namedtuples directly in C, running significantly faster than 'df.iterrows()', which creates a Series object per row and does not preserve data types.

### Q15. Explain memory-efficient techniques for reading massive CSV files in Pandas.

**Answer:** Techniques: specify 'usecols' to load only required columns, explicitly declare 'dtype' (e.g. downcasting float64 to float32), use 'pd.read_csv(chunksize=...)' to stream rows in batches, or convert files to Parquet.

### Q16. Why is Parquet preferred over CSV for data science storage?

**Answer:** Parquet is a columnar, compressed binary format supporting schema enforcement, dictionary encoding, and column projection/predicate pushdown, making reads 5x–10x faster and reducing storage size by up to 80%.

### Q17. What is the difference between filtering with boolean masking vs 'df.query()'?

**Answer:** Boolean masking evaluates Python expressions in memory. 'df.query()' uses NumExpr under the hood to compile string expressions into multithreaded C code without allocating intermediate boolean array buffers.

### Q18. How do you detect and fix memory leaks in long-running Python data pipelines?

**Answer:** Use 'tracemalloc' to track memory allocations across code lines, 'gc.collect()' to force garbage collection of cyclic references, avoid appending to global lists, and delete unneeded large DataFrames using 'del df'.

### Q19. What is the difference between 'map()', 'applymap()' (now 'map()'), and 'transform()' in Pandas?

**Answer:** 'map()' applies element-wise mapping on a Series. 'map()' on a DataFrame operates on every element. 'transform()' inside groupby returns an aligned Series of the same length as the original DataFrame.

### Q20. How does multiprocessing in Python differ from multithreading for data tasks?

**Answer:** Due to the GIL, multithreading runs on a single core and is effective only for I/O-bound tasks (API calls, disk reads). Multiprocessing spawns separate OS processes with independent GILs, enabling true multicore parallel execution for CPU-heavy data transformations.

### Q21. What is Joblib and how is it used in data science?

**Answer:** Joblib provides lightweight pipelining, disk-based caching of function results ('joblib.Memory'), and transparent parallel loops ('Parallel(n_jobs=-1)') for embarrassing parallel tasks like cross-validation.

### Q22. Explain the difference between 'dropna()' and 'fillna()'.

**Answer:** 'dropna()' removes rows or columns containing missing values. 'fillna()' imputes missing values using constants, summary statistics (mean, median), or propagation methods ('ffill', 'bfill').

### Q23. How do you handle duplicate rows in Pandas?

**Answer:** Use 'df.duplicated(subset=..., keep='first')' to locate duplicates, and 'df.drop_duplicates(subset=..., keep='first')' to eliminate duplicate rows.

### Q24. What is method chaining with '.pipe()' in Pandas?

**Answer:** '.pipe(func, *args, **kwargs)' applies custom functions to the entire DataFrame in a chain, enabling clean functional data cleaning pipelines without nested function calls.

### Q25. What are the advantages of PyArrow backend in Pandas 2.0+?

**Answer:** Pandas 2.0+ integrates Apache Arrow for in-memory column buffers, offering faster string operations, standardized null handling, shared zero-copy memory between languages, and reduced RAM usage.

### Q26. Explain how string methods work in Pandas via the '.str' accessor.

**Answer:** The '.str' accessor exposes vectorized string operations (e.g. .str.lower(), .str.contains(), .str.extract()) that execute across an entire Series while handling missing values safely.

### Q27. What is the difference between 'qcut()' and 'cut()' in Pandas?

**Answer:** 'cut()' divides data into equal-width bins based on value range. 'qcut()' divides data into equal-frequency quantiles (e.g. quartiles, deciles) based on sample distribution.

### Q28. How do you profile code performance in Jupyter notebooks?

**Answer:** Use magic commands: '%time' for single execution time, '%timeit' for averaged benchmark runs, '%prun' for cProfile function call profiler, and '%%memit' for peak memory usage.

### Q29. What is the difference between Series and 1D NumPy arrays?

**Answer:** A Series is a labeled 1D NumPy array wrapped with an explicit index and optional name, enabling automatic data alignment during mathematical operations.

### Q30. What is defensive programming in data engineering?

**Answer:** Defensive programming anticipates corrupted inputs by enforcing strict schema validation (using Pydantic, Great Expectations), assertion checks on row counts, handling missing files gracefully, and logging execution metrics.
