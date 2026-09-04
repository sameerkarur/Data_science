# Interview Q&A — Pandas

| # | Question | Answer |
|---|----------|--------|
| 1 | Series vs DataFrame? | **Series** = 1D labeled array; **DataFrame** = 2D table of Series. |
| 2 | `loc` vs `iloc`? | **loc** = label-based indexing; **iloc** = integer position indexing. |
| 3 | How to handle missing data? | `dropna()`, `fillna()`, `interpolate()`, or replace with median/mode. |
| 4 | `groupby` does what? | Split-apply-combine: group rows by key, apply aggregation/transform. |
| 5 | `merge` vs `concat`? | **merge** = SQL-like join on keys; **concat** = stack along axis 0/1. |
