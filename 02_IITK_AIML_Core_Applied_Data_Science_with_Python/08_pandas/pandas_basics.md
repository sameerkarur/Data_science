# Pandas - Python Data Analysis Library

## What is Pandas?
Pandas is a powerful, open-source library for data manipulation and analysis. It provides two primary data structures — **Series** (1D) and **DataFrame** (2D) — that make working with structured/tabular data fast and intuitive.

**Why use Pandas?**
- Read/write data from CSV, Excel, SQL, JSON, and more
- Handle missing data gracefully
- Powerful grouping, merging, and reshaping operations
- Built on top of NumPy (fast performance)
- Essential for data cleaning, exploration, and preparation

## Installation
```bash
pip install pandas
```

## Importing
```python
import pandas as pd
```

---

## 1. Data Structures

### Series (1D)
```python
# From a list
s = pd.Series([10, 20, 30, 40])
# 0    10
# 1    20
# 2    30
# 3    40

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# a    10
# b    20
# c    30

# From a dictionary
s = pd.Series({'a': 100, 'b': 200, 'c': 300})
```

### DataFrame (2D)
```python
# From a dictionary
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NYC', 'LA', 'Chicago']
})
#       Name  Age     City
# 0    Alice   25      NYC
# 1      Bob   30       LA
# 2  Charlie   35  Chicago

# From a list of lists
df = pd.DataFrame(
    [[1, 'Alice', 25], [2, 'Bob', 30]],
    columns=['ID', 'Name', 'Age']
)

# From NumPy array
import numpy as np
df = pd.DataFrame(np.random.rand(3, 4), columns=['A', 'B', 'C', 'D'])
```

---

## 2. Reading & Writing Data

### Reading
```python
# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=';', header=0, index_col='ID')

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df = pd.read_json('data.json')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)

# From clipboard
df = pd.read_clipboard()
```

### Writing
```python
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json')
df.to_sql('table_name', conn, if_exists='replace')
```

---

## 3. Exploring Data

```python
df.head()           # first 5 rows
df.head(10)         # first 10 rows
df.tail()           # last 5 rows
df.shape            # (rows, columns)
df.info()           # column names, types, non-null counts
df.describe()       # statistics for numeric columns
df.dtypes           # data types of each column
df.columns          # list of column names
df.index            # index info
df.nunique()        # unique values per column
df.value_counts()   # frequency counts (for Series)
df.sample(5)        # 5 random rows
```

---

## 4. Selecting Data

### Selecting Columns
```python
df['Name']              # single column (returns Series)
df[['Name', 'Age']]    # multiple columns (returns DataFrame)
```

### Selecting Rows
```python
# By index position (iloc - integer location)
df.iloc[0]              # first row
df.iloc[0:3]            # rows 0, 1, 2
df.iloc[0, 1]           # row 0, column 1
df.iloc[:, 0:2]         # all rows, first 2 columns

# By label (loc)
df.loc[0]               # row with index label 0
df.loc[0:2, 'Name']     # rows 0-2, column 'Name'
df.loc[:, ['Name', 'Age']]  # all rows, specific columns
```

### Conditional Selection (Filtering)
```python
df[df['Age'] > 25]                          # rows where Age > 25
df[df['City'] == 'NYC']                     # rows where City is NYC
df[(df['Age'] > 25) & (df['City'] == 'LA')] # multiple conditions (AND)
df[(df['Age'] < 25) | (df['Age'] > 35)]     # multiple conditions (OR)
df[df['Name'].isin(['Alice', 'Bob'])]        # values in a list
df[df['Name'].str.contains('li')]            # string contains
df.query('Age > 25 and City == "NYC"')       # query method
```

---

## 5. Adding & Modifying Data

### Adding Columns
```python
df['Salary'] = [50000, 60000, 70000]         # from a list
df['Senior'] = df['Age'] > 30                 # computed column
df['Full'] = df['Name'] + ' - ' + df['City'] # string concatenation
df['Tax'] = df['Salary'] * 0.3               # arithmetic
```

### Modifying Values
```python
df.loc[0, 'Age'] = 26                        # specific cell
df['Age'] = df['Age'] + 1                    # entire column
df.loc[df['City'] == 'NYC', 'City'] = 'New York'  # conditional update
```

### Renaming Columns
```python
df.rename(columns={'Name': 'Full_Name', 'Age': 'Years'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']        # rename all at once
```

### Dropping
```python
df.drop('Salary', axis=1, inplace=True)       # drop column
df.drop([0, 1], axis=0, inplace=True)         # drop rows by index
df.drop(columns=['col1', 'col2'], inplace=True)
```

---

## 6. Handling Missing Data

```python
# Detect
df.isnull()              # True where NaN
df.notnull()             # True where NOT NaN
df.isnull().sum()        # count NaN per column
df.isnull().sum().sum()  # total NaN in entire DataFrame

# Drop
df.dropna()              # drop rows with any NaN
df.dropna(axis=1)        # drop columns with any NaN
df.dropna(subset=['Age']) # drop rows where Age is NaN
df.dropna(thresh=2)      # keep rows with at least 2 non-NaN values

# Fill
df.fillna(0)                          # fill NaN with 0
df['Age'].fillna(df['Age'].mean())    # fill with mean
df.fillna(method='ffill')             # forward fill
df.fillna(method='bfill')             # backward fill
df.interpolate()                      # interpolate missing values
```

---

## 7. Grouping & Aggregation

```python
# Basic groupby
df.groupby('City')['Salary'].mean()
df.groupby('City')['Salary'].sum()
df.groupby('City').count()

# Multiple aggregations
df.groupby('City')['Salary'].agg(['mean', 'sum', 'count', 'min', 'max'])

# Group by multiple columns
df.groupby(['City', 'Department'])['Salary'].mean()

# Custom aggregation
df.groupby('City').agg({
    'Salary': 'mean',
    'Age': ['min', 'max'],
    'Name': 'count'
})

# Transform (returns same-sized DataFrame)
df['Salary_zscore'] = df.groupby('City')['Salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## 8. Sorting

```python
df.sort_values('Age')                          # ascending
df.sort_values('Age', ascending=False)         # descending
df.sort_values(['City', 'Age'], ascending=[True, False])  # multiple columns
df.sort_index()                                # sort by index
df.nlargest(5, 'Salary')                       # top 5 by Salary
df.nsmallest(3, 'Age')                         # bottom 3 by Age
```

---

## 9. Merging, Joining & Concatenating

### Concatenate
```python
# Stack vertically (row-wise)
pd.concat([df1, df2], ignore_index=True)

# Stack horizontally (column-wise)
pd.concat([df1, df2], axis=1)
```

### Merge (like SQL JOIN)
```python
# Inner join (default)
pd.merge(df1, df2, on='ID')

# Left join
pd.merge(df1, df2, on='ID', how='left')

# Right join
pd.merge(df1, df2, on='ID', how='right')

# Outer join
pd.merge(df1, df2, on='ID', how='outer')

# Merge on different column names
pd.merge(df1, df2, left_on='emp_id', right_on='employee_id')
```

### Join (index-based)
```python
df1.join(df2, how='left')
```

---

## 10. Pivot Tables & Cross Tabs

```python
# Pivot table
pd.pivot_table(df, values='Salary', index='City', columns='Department', aggfunc='mean')

# Cross tabulation
pd.crosstab(df['City'], df['Department'])

# Melt (wide to long format)
pd.melt(df, id_vars=['Name'], value_vars=['Math', 'Science'], var_name='Subject', value_name='Score')
```

---

## 11. String Operations

```python
df['Name'].str.lower()              # lowercase
df['Name'].str.upper()              # uppercase
df['Name'].str.title()              # title case
df['Name'].str.strip()              # remove whitespace
df['Name'].str.replace('old', 'new')
df['Name'].str.contains('pattern')  # returns boolean
df['Name'].str.startswith('A')
df['Name'].str.split(' ')           # split into list
df['Name'].str.len()                # length of each string
df['Name'].str.extract(r'(\d+)')    # regex extract
```

---

## 12. Date & Time

```python
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.day_name()
df['Hour'] = df['Date'].dt.hour

# Date range
pd.date_range(start='2024-01-01', periods=10, freq='D')   # daily
pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')  # monthly

# Resample time series
df.set_index('Date').resample('M').mean()    # monthly average
df.set_index('Date').resample('W').sum()     # weekly sum

# Time difference
df['Duration'] = df['End'] - df['Start']
df['Days'] = df['Duration'].dt.days
```

---

## 13. Apply & Map

```python
# Apply function to a column
df['Age_group'] = df['Age'].apply(lambda x: 'Senior' if x > 30 else 'Junior')

# Apply function to entire DataFrame
df[['A', 'B']].apply(np.sum, axis=0)    # column-wise
df[['A', 'B']].apply(np.sum, axis=1)    # row-wise

# Map (for Series - replace values)
df['City'].map({'NYC': 'New York', 'LA': 'Los Angeles'})

# Applymap (element-wise for entire DataFrame) — renamed to map() in newer pandas
df[['A', 'B']].applymap(lambda x: round(x, 2))
```

---

## 14. Useful Methods

```python
# Duplicates
df.duplicated()                    # boolean mask
df.drop_duplicates()               # remove duplicates
df.drop_duplicates(subset=['Name'])

# Replace values
df.replace({'NYC': 'New York', 'LA': 'Los Angeles'})
df['Age'].replace({25: 26, 30: 31})

# Rank
df['Salary_rank'] = df['Salary'].rank(ascending=False)

# Clip values
df['Age'].clip(lower=20, upper=40)

# Binning
df['Age_bin'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Middle', 'Senior'])
df['Age_bin'] = pd.qcut(df['Age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # quantile-based

# Reset index
df.reset_index(drop=True, inplace=True)

# Set index
df.set_index('ID', inplace=True)
```

---

## Quick Reference Table

| Operation | Syntax |
|---|---|
| Read CSV | `pd.read_csv('file.csv')` |
| Write CSV | `df.to_csv('file.csv', index=False)` |
| First N rows | `df.head(n)` |
| Shape | `df.shape` |
| Info | `df.info()` |
| Statistics | `df.describe()` |
| Select column | `df['col']` or `df[['col1','col2']]` |
| Filter rows | `df[df['col'] > value]` |
| Add column | `df['new'] = values` |
| Drop column | `df.drop('col', axis=1)` |
| Fill NaN | `df.fillna(value)` |
| Drop NaN | `df.dropna()` |
| Group by | `df.groupby('col')['val'].mean()` |
| Sort | `df.sort_values('col')` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Concat | `pd.concat([df1, df2])` |
| Pivot | `pd.pivot_table(df, values, index, columns)` |
| Apply | `df['col'].apply(func)` |
| Duplicates | `df.drop_duplicates()` |
| To datetime | `pd.to_datetime(df['col'])` |
