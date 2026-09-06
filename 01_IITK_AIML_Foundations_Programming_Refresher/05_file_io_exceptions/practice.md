# File I/O & Exceptions — Practice Questions (50+)

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


### Dataset for this topic: `sales`
Practice reading/writing CSV, JSON, and shared datasets.

```python
import pandas as pd
import numpy as np

sales_path = DATA_DIR / 'AusApparalSales4thQrt2020.csv'
df = pd.read_csv(sales_path)
print(df.shape)
df.head()
```


## Section 1: pathlib

**Q1.** List all files in repo root using Path.

**Q2.** Check if DATA_DIR exists and is directory.

**Q3.** Join path to read sales CSV with Path / operator.

**Q4.** Create temp folder under subtopic if not exists.

**Q5.** Glob all *.csv in DATA_DIR.

**Q6.** Read text file basics with Path.read_text.

**Q7.** Write lines with Path.write_text.

**Q8.** Get file size in bytes.

**Q9.** Resolve absolute path.

**Q10.** Rename file with Path.rename.


## Section 2: read/write

**Q11.** Open file with `with open` read mode.

**Q12.** Read line by line into list.

**Q13.** Append to file mode 'a'.

**Q14.** Write CSV manually without pandas.

**Q15.** Read CSV with csv.DictReader.

**Q16.** Use json.dump and json.load.

**Q17.** Pretty print JSON indent=2.

**Q18.** Read first 5 lines of large CSV without loading all.

**Q19.** Copy file with shutil.copy.

**Q20.** Delete temp file with Path.unlink.


## Section 3: pandas I/O

**Q21.** Load sales CSV with pd.read_csv from DATA_DIR.

**Q22.** Save dataframe head to csv index=False.

**Q23.** Read only first 1000 rows with nrows.

**Q24.** Specify dtype on read for memory.

**Q25.** Handle parse_dates on Date column.

**Q26.** Export to parquet if pyarrow available (try/except).

**Q27.** Read Excel if openpyxl — marketing xlsx in project optional.

**Q28.** Load HR dataset; save cleaned subset.

**Q29.** Use compression gzip on to_csv.

**Q30.** Verify round-trip: read written csv equals head.


## Section 4: Exceptions

**Q31.** Try/except FileNotFoundError.

**Q32.** Raise ValueError with message.

**Q33.** Custom exception class AppError.

**Q34.** else clause in try/except.

**Q35.** finally always runs demo.

**Q36.** Chained exception raise from.

**Q37.** Assert for development checks.

**Q38.** Logging instead of print for errors.

**Q39.** Retry reading file 3 times.

**Q40.** Validate CSV columns before processing.


## Section 5: Real workflows

**Q41.** Build load_dataset(name) with friendly errors.

**Q42.** Log processing steps to file.

**Q43.** Config JSON: read/write settings dict.

**Q44.** Walk directory tree with os.walk or Path.rglob.

**Q45.** Environment variable for data path override.

**Q46.** Context manager for temp write then read.

**Q47.** Merge multiple CSV chunks with concat.

**Q48.** Document data lineage in markdown file.

**Q49.** Checksum file size before/after copy.

**Q50.** Safe write: write temp then replace.

---
**Total: 50 questions**

Stuck? → `basics.md` → then `solutions.ipynb` → then `interview_qa.md`.