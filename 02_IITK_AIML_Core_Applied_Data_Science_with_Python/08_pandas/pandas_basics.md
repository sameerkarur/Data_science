# Pandas: Complete Step-by-Step Tutorial & Data Wrangling Handbook
**Official Tutorial & Practical Analytics Guide (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Pandas & Why Use It?](#1-what-is-pandas--why-use-it)
2. [Installation & Importing](#2-installation--importing)
3. [Pandas Data Structures: Series (1D) & DataFrame (2D)](#3-pandas-data-structures-series-1d--dataframe-2d)
4. [Creating DataFrames from Dictionaries, Lists & CSV](#4-creating-dataframes-from-dictionaries-lists--csv)
5. [Viewing & Inspecting Data (Head, Tail, Info, Describe)](#5-viewing--inspecting-data-head-tail-info-describe)
6. [Selection & Slicing (loc, iloc & Boolean Filtering)](#6-selection--slicing-loc-iloc--boolean-filtering)
7. [Data Cleaning (Missing Values, Duplicates & Types)](#7-data-cleaning-missing-values-duplicates--types)
8. [Data Transformation & Feature Engineering](#8-data-transformation--feature-engineering)
9. [GroupBy & Aggregations (Split-Apply-Combine)](#9-groupby--aggregations-split-apply-combine)
10. [Merging, Joining & Concatenating (Inner, Outer, Left, Right)](#10-merging-joining--concatenating)
11. [Pivot Tables & Cross-Tabulations](#11-pivot-tables--cross-tabulations)
12. [Reading & Writing External Files (CSV, Excel, JSON)](#12-reading--writing-external-files)
13. [Try It Yourself! (Hands-On Practice Exercises)](#13-try-it-yourself-hands-on-practice-exercises)
14. [Quick Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What is Pandas & Why Use It?

**Pandas** is the premiere Python library for data manipulation and tabular data analysis. It provides fast, flexible, and expressive data structures designed to make working with "relational" or "labeled" data intuitive and natural.

### Why use Pandas?
- **Excel on Steroids:** Easily handle millions of rows with high performance.
- **Missing Data Handling:** Detect, drop, or impute missing values seamlessly (`NaN` / `None`).
- **Flexible Reshaping:** Pivot, melt, stack, and aggregate multi-dimensional tables.
- **SQL-like Joins:** Execute lightning-fast inner, outer, left, and cross joins between datasets.
- **Time Series Ready:** Specialized frequency conversion, date shifting, and rolling statistics.

---

## 2. Installation & Importing

Install Pandas via `pip`:
```bash
pip install pandas
```

Standard industry convention is to import Pandas as `pd`:
```python
import pandas as pd
print(f"Pandas Version: {pd.__version__}")
```

#### Output:
```text
Pandas Version: 2.2.2
```

---

## 3. Pandas Data Structures: Series (1D) & DataFrame (2D)

Pandas provides two foundational data structures:
1. **`Series`:** A one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.).
2. **`DataFrame`:** A two-dimensional tabular data structure with labeled axes (rows and columns). A DataFrame is essentially a collection of Series sharing a common index.

### Visual Representation of Series vs DataFrame:

```
        PANDAS SERIES (1D)                         PANDAS DATAFRAME (2D)
                                                 Columns ──► ['Name', 'Age', 'City']
     Index ──► Data Values                         Index      Col 0    Col 1    Col 2
    ┌───────┬─────────────┐                       ┌───────┬─────────┬──────┬─────────┐
    │   0   │    10.5     │                       │   0   │  Alice  │  25  │   NYC   │
    ├───────┼─────────────┤                       ├───────┼─────────┼──────┼─────────┤
    │   1   │    20.8     │                       │   1   │   Bob   │  30  │   LA    │
    ├───────┼─────────────┤                       ├───────┼─────────┼──────┼─────────┤
    │   2   │    35.2     │                       │   2   │ Charlie │  35  │ Chicago │
    └───────┴─────────────┘                       └───────┴─────────┴──────┴─────────┘
     dtype: float64                                Row 0 ──► Series: [Alice, 25, NYC]
                                                   Col 0 ──► Series: [Alice, Bob, Charlie]
```

---

## 4. Creating DataFrames from Dictionaries, Lists & CSV

### Example 1: Creating a Series (GeeksforGeeks Style)
```python
import pandas as pd
import numpy as np

# From a Python list
fruits = pd.Series(['Apple', 'Banana', 'Cherry'], index=['a', 'b', 'c'])
print("Pandas Series with Custom Index:\n", fruits)
```

#### Output:
```text
Pandas Series with Custom Index:
 a     Apple
 b    Banana
 c    Cherry
 dtype: object
```

### Example 2: Creating a DataFrame from a Dictionary
```python
import pandas as pd

employee_data = {
    'EmpID': [101, 102, 103, 104],
    'Name': ['Sarah', 'David', 'Elena', 'Michael'],
    'Department': ['Engineering', 'Marketing', 'Engineering', 'Finance'],
    'Salary': [85000, 62000, 92000, 78000],
    'Experience': [4, 2, 7, 5]
}

df = pd.DataFrame(employee_data)
print("Employee DataFrame:\n", df)
```

#### Output:
```text
Employee DataFrame:
    EmpID     Name   Department  Salary  Experience
0    101    Sarah  Engineering   85000           4
1    102    David    Marketing   62000           2
2    103    Elena  Engineering   92000           7
3    104  Michael      Finance   78000           5
```

---

## 5. Viewing & Inspecting Data (Head, Tail, Info, Describe)

When exploring a new dataset, always execute these diagnostic inspections:

```python
import pandas as pd

# 1. View first 2 rows
print("--- df.head(2) ---\n", df.head(2))

# 2. View shape and column names
print("\nShape (Rows, Columns):", df.shape)
print("Column Names:         ", df.columns.tolist())
print("Data Types:\n", df.dtypes)

# 3. Comprehensive Statistical Summary
print("\n--- df.describe() Numerical Summary ---\n", df.describe())
```

#### Output:
```text
--- df.head(2) ---
    EmpID   Name   Department  Salary  Experience
0    101  Sarah  Engineering   85000           4
1    102  David    Marketing   62000           2

Shape (Rows, Columns): (4, 5)
Column Names:          ['EmpID', 'Name', 'Department', 'Salary', 'Experience']
Data Types:
 EmpID          int64
Name          object
Department    object
Salary         int64
Experience     int64
dtype: object

--- df.describe() Numerical Summary ---
             EmpID        Salary  Experience
count     4.000000      4.000000    4.000000
mean    102.500000  79250.000000    4.500000
std       1.290994  12816.005618    2.081666
min     101.000000  62000.000000    2.000000
25%     101.750000  74000.000000    3.500000
50%     102.500000  81500.000000    4.500000
75%     103.250000  86750.000000    5.500000
max     104.000000  92000.000000    7.000000
```

---

## 6. Selection & Slicing (loc, iloc & Boolean Filtering)

Accessing subsets of data is the most common operation in Pandas.

### Visual Diagram: `.loc` vs `.iloc`

```
  df.loc[row_label, col_label]        vs        df.iloc[row_integer, col_integer]
  (Explicit Label / Name Based)                 (Pure 0-Indexed Position Based)
  
  df.loc[1:2, 'Name':'Salary']                  df.iloc[1:3, 1:4]
  (INCLUSIVE of endpoint 'Salary'!)             (EXCLUSIVE of endpoint index 3 & 4!)
```

```python
import pandas as pd

# 1. Select single column as Series
names = df['Name']

# 2. Select multiple columns as DataFrame
subset = df[['Name', 'Salary']]
print("Multiple Columns:\n", subset)

# 3. .iloc: Select rows 0 to 1, columns 1 to 3 by integer index
print("\n--- df.iloc[0:2, 1:4] ---")
print(df.iloc[0:2, 1:4])

# 4. .loc: Select by column names and condition
print("\n--- High Earners (Salary >= 80,000) ---")
high_earners = df.loc[df['Salary'] >= 80000, ['Name', 'Department', 'Salary']]
print(high_earners)
```

#### Output:
```text
Multiple Columns:
       Name  Salary
0    Sarah   85000
1    David   62000
2    Elena   92000
3  Michael   78000

--- df.iloc[0:2, 1:4] ---
    Name   Department  Salary
0  Sarah  Engineering   85000
1  David    Marketing   62000

--- High Earners (Salary >= 80,000) ---
    Name   Department  Salary
0  Sarah  Engineering   85000
2  Elena  Engineering   92000
```

---

## 7. Data Cleaning (Missing Values, Duplicates & Types)

In the real world, data is messy. Here is the canonical W3Schools cleaning workflow:

```python
import pandas as pd
import numpy as np

# Sample dataset with missing values and duplicates
raw_records = pd.DataFrame({
    'TransactionID': [1, 2, 3, 3, 4],
    'Customer': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David'],
    'Amount': [250.0, np.nan, 150.0, 150.0, 420.0],
    'Date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-03', 'InvalidDate']
})

print("Raw Dirty Data:\n", raw_records)

# 1. Identify missing values
print("\nMissing Values Count:\n", raw_records.isna().sum())

# 2. Impute missing numeric values with column median
median_amount = raw_records['Amount'].median()
raw_records['Amount'] = raw_records['Amount'].fillna(median_amount)

# 3. Remove duplicate rows
clean_df = raw_records.drop_duplicates()

# 4. Clean dates using errors='coerce' to turn bad dates into NaT
clean_df['Date'] = pd.to_datetime(clean_df['Date'], errors='coerce')

print("\n--- Cleaned DataFrame ---\n", clean_df)
```

#### Output:
```text
Raw Dirty Data:
    TransactionID Customer  Amount         Date
0              1    Alice   250.0   2026-01-01
1              2      Bob     NaN   2026-01-02
2              3  Charlie   150.0   2026-01-03
3              3  Charlie   150.0   2026-01-03
4              4    David   420.0  InvalidDate

Missing Values Count:
 TransactionID    0
Customer         0
Amount           1
Date             0
dtype: int64

--- Cleaned DataFrame ---
    TransactionID Customer  Amount       Date
0              1    Alice   250.0 2026-01-01
1              2      Bob   200.0 2026-01-02
2              3  Charlie   150.0 2026-01-03
4              4    David   420.0        NaT
```

---

## 8. Data Transformation & Feature Engineering

Transforming raw columns into predictive features:

```python
import pandas as pd

df = pd.DataFrame({
    'Product': ['Laptop Pro', 'Wireless Mouse', 'Mechanical Keyboard'],
    'UnitPrice': [1200, 35, 120],
    'Quantity': [2, 10, 4]
})

# 1. Vectorized Column Creation
df['TotalRevenue'] = df['UnitPrice'] * df['Quantity']

# 2. Custom Function Application with .apply()
def categorize_tier(price):
    if price > 500:
        return 'Premium'
    elif price > 50:
        return 'Mid-Range'
    return 'Budget'

df['Tier'] = df['UnitPrice'].apply(categorize_tier)

# 3. String Methods with .str accessor
df['Product_Upper'] = df['Product'].str.upper()

print("Engineered DataFrame:\n", df[['Product', 'TotalRevenue', 'Tier', 'Product_Upper']])
```

#### Output:
```text
Engineered DataFrame:
                Product  TotalRevenue       Tier        Product_Upper
0           Laptop Pro          2400    Premium           LAPTOP PRO
1       Wireless Mouse           350     Budget       WIRELESS MOUSE
2  Mechanical Keyboard           480  Mid-Range  MECHANICAL KEYBOARD
```

---

## 9. GroupBy & Aggregations (Split-Apply-Combine)

The **Split-Apply-Combine** strategy is the foundation of group aggregations:

```
                  SPLIT-APPLY-COMBINE PIPELINE
       Input Table ──► SPLIT by Department:
                          ├── Engineering Sub-table
                          ├── Marketing Sub-table
                          └── Finance Sub-table
                                    │
                       APPLY Aggregation: sum(Salary), mean(Experience)
                                    │
                       COMBINE into Summary Table:
                          Department     TotalSalary  AvgExp
                          Engineering       177,000     5.5
                          Marketing          62,000     2.0
                          Finance            78,000     5.0
```

```python
import pandas as pd

sales_data = pd.DataFrame({
    'Region': ['North', 'South', 'North', 'South', 'North', 'West'],
    'Rep': ['Alex', 'Brian', 'Alex', 'David', 'Elena', 'Fiona'],
    'Units': [50, 40, 65, 30, 80, 45],
    'Revenue': [5000, 4200, 6800, 3100, 8400, 4700]
})

# Group by Region with multiple aggregations
region_summary = sales_data.groupby('Region').agg(
    TotalRevenue=('Revenue', 'sum'),
    AvgUnits=('Units', 'mean'),
    TotalTransactions=('Rep', 'count')
).reset_index()

print("Regional Performance Summary:\n", region_summary)
```

#### Output:
```text
Regional Performance Summary:
   Region  TotalRevenue   AvgUnits  TotalTransactions
0  North         20200  65.000000                  3
1  South          7300  35.000000                  2
2   West          4700  45.000000                  1
```

---

## 10. Merging, Joining & Concatenating

Combining distinct relational tables using primary keys:

```
                      VISUALIZING SQL-STYLE JOINS
      INNER JOIN                      LEFT JOIN                     OUTER JOIN
    ┌────┬─────────┐                ┌────┬─────────┐              ┌────┬─────────┐
    │ ID │ Shared  │                │ ID │ All Left│              │ ID │ All Rows│
    └────┴─────────┘                └────┴─────────┘              └────┴─────────┘
  (Keys in BOTH tables)         (All Left + Matching Right)   (Union of all keys)
```

```python
import pandas as pd

customers = pd.DataFrame({
    'CustID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
})

orders = pd.DataFrame({
    'OrderID': [501, 502, 503],
    'CustID': [1, 2, 4],  # Customer 4 does not exist in customers table
    'Amount': [350, 120, 890]
})

# Inner Merge (Only matching keys)
inner_df = pd.merge(customers, orders, on='CustID', how='inner')
print("--- Inner Join ---\n", inner_df)

# Left Merge (Preserves all customers)
left_df = pd.merge(customers, orders, on='CustID', how='left')
print("\n--- Left Join ---\n", left_df)
```

#### Output:
```text
--- Inner Join ---
    CustID   Name  OrderID  Amount
0       1  Alice      501     350
1       2    Bob      502     120

--- Left Join ---
    CustID     Name  OrderID  Amount
0       1    Alice    501.0   350.0
1       2      Bob    502.0   120.0
2       3  Charlie      NaN     NaN
```

---

## 11. Pivot Tables & Cross-Tabulations

Pivot tables summarize complex multi-dimensional tables into presentation grids:

```python
import pandas as pd

orders_df = pd.DataFrame({
    'Year': [2025, 2025, 2026, 2026, 2026],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Electronics', 'Clothing'],
    'Sales': [1500, 400, 2200, 1800, 650]
})

pivot = pd.pivot_table(
    orders_df,
    values='Sales',
    index='Category',
    columns='Year',
    aggfunc='sum',
    fill_value=0
)

print("Sales Pivot Table:\n", pivot)
```

#### Output:
```text
Sales Pivot Table:
 Year          2025  2026
Category                 
Clothing       400   650
Electronics   1500  4000
```

---

## 12. Reading & Writing External Files

```python
import pandas as pd
import tempfile
import os

# Create sample DataFrame
df = pd.DataFrame({'Model': ['ResNet50', 'BERT', 'GPT-4'], 'Parameters_M': [25.6, 110, 175000]})

# Write to temporary CSV
with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
    tmp_path = tmp.name

df.to_csv(tmp_path, index=False)
print(f"Written to CSV: {tmp_path}")

# Read CSV back into DataFrame
df_read = pd.read_csv(tmp_path)
print("Read DataFrame:\n", df_read)

# Clean up
if os.path.exists(tmp_path):
    os.remove(tmp_path)
```

#### Output:
```text
Written to CSV: /var/folders/.../temp.csv
Read DataFrame:
       Model  Parameters_M
0  ResNet50          25.6
1      BERT         110.0
2     GPT-4      175000.0
```

---

## 13. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding Top Customers by Expenditure
**Task:** Given a DataFrame of e-commerce orders, compute the total expenditure per customer and find the top 2 customers with the highest spending:
```python
orders = pd.DataFrame({
    'Customer': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
    'Spend': [120, 450, 80, 210, 310, 400]
})
```

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import pandas as pd

orders = pd.DataFrame({
    'Customer': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Alice'],
    'Spend': [120, 450, 80, 210, 310, 400]
})

top_spenders = (orders.groupby('Customer')['Spend']
                .sum()
                .sort_values(ascending=False)
                .head(2)
                .reset_index())

print("Top 2 Customers by Total Spend:\n", top_spenders)
```
#### Output:
```text
Top 2 Customers by Total Spend:
   Customer  Spend
0      Bob    760
1    Alice    600
```
</details>

---

## 14. Quick Reference Cheat Sheet

| Task | Pandas Command | Description |
|---|---|---|
| **Read CSV** | `pd.read_csv('file.csv')` | Ingests CSV to DataFrame |
| **Inspect Data** | `df.head()`, `df.info()`, `df.describe()` | Examines structure & statistics |
| **Filter Rows** | `df[df['age'] > 30]`, `df.query('age > 30')` | Boolean conditional selection |
| **Select Columns**| `df[['name', 'salary']]` | Extracts column subset |
| **Label Slice** | `df.loc[0:5, ['name', 'age']]` | Label-based row and column slice |
| **Positional Slice**| `df.iloc[0:5, 0:2]` | 0-indexed integer slice |
| **Fill Missing** | `df['col'].fillna(df['col'].median())` | Imputes missing values |
| **Drop Missing** | `df.dropna(subset=['id', 'date'])` | Removes records with NaNs |
| **Drop Duplicates**| `df.drop_duplicates()` | Eliminates duplicate rows |
| **GroupBy** | `df.groupby('dept')['salary'].mean()` | Aggregates across categories |
| **Merge / Join** | `pd.merge(df1, df2, on='key', how='inner')` | SQL-style relational merge |
| **Pivot Table** | `pd.pivot_table(df, values='x', index='y', columns='z')`| 2D multi-index summary |
| **Export CSV** | `df.to_csv('output.csv', index=False)` | Writes DataFrame to disk |
