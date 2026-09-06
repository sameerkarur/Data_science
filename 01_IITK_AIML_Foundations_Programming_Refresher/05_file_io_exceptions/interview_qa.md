# Interview Q&A — File I/O & Exceptions

> **30 questions** — read aloud, then explain without looking.


## pathlib

### Q1. pathlib vs os.path?

pathlib is OOP, composable with /, cross-platform.


## Files

### Q2. Why `with open`?

Guarantees file close on exceptions.

### Q3. Text vs binary mode?

Text decodes Unicode; binary for bytes/images.


## CSV

### Q4. csv vs pandas?

csv lightweight streaming; pandas analytics at scale.


## JSON

### Q5. JSON types?

dict, list, str, int, float, bool, None.


## pandas I/O

### Q6. read_csv essentials?

usecols, dtype, parse_dates, nrows, na_values.


## Exceptions

### Q7. try/except/else/finally?

else runs if no error; finally always cleanup.


## Workflow

### Q8. DATA_DIR pattern?

Central datasets/shared; notebooks find repo root dynamically.

### Q9. Atomic writes?

Write temp file then rename to avoid corruption.


## Interview

### Q10. Large file strategies?

chunksize, nrows, pyarrow, parquet, mmap.

### Q11. Encoding issues?

Always specify utf-8; handle errors='replace' if needed.

### Q12. Git LFS?

For files >100MB — pointer in git, blob stored separately.

### Q13. Pickle risks?

Never unpickle untrusted data — RCE risk.

### Q14. Parquet vs CSV?

Parquet columnar compressed typed — faster for analytics.

### Q15. Environment path override?

AIML_DATA_DIR env var for CI/prod flexibility.

### Q16. Validate before ETL?

Check columns, dtypes, row counts early.

### Q17. Logging vs print?

Logging levels, files, production pipelines.

### Q18. shutil.copy2?

Copies file + metadata.

### Q19. Path.rglob?

Recursive glob from directory tree.

### Q20. Custom exceptions?

Domain errors users can catch specifically.

### Q21. raise from?

Chains exceptions preserving root cause.

### Q22. Compression gzip?

Smaller disk, more CPU on read/write.

### Q23. Excel multiple sheets?

read_excel(sheet_name=None) → dict of DataFrames.

### Q24. Round-trip integrity?

Hash/checksum and assert shape dtypes after write.

### Q25. on_bad_lines?

Skip or warn on malformed CSV rows (pandas).

### Q26. Config JSON pattern?

Store seed, paths, hyperparams outside code.

### Q27. Lineage documentation?

Track source → transforms → output for audits.

### Q28. Retry on I/O?

Transient network/disk errors — bounded retries with backoff.

### Q29. Contextlib suppress?

Cleanly ignore expected exceptions in narrow scope.

### Q30. Home loan sample vs full?

Sample in repo; full 158MB local/gitignored — document in README.
