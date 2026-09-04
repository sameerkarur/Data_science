# Pandas Practice Questions
### Dataset: AusApparalSales4thQrt2020.csv (Australian Apparel Sales Q4 2020)

**Columns:** Date, Time, State, Group, Unit, Sales

> **Setup:**
```python
import pandas as pd
import numpy as np

df = pd.read_csv('AusApparalSales4thQrt2020.csv')
```

---

## Section 1: Loading & Exploring Data

**Q1.** Load the CSV file into a DataFrame. Display the first 10 rows and last 5 rows.

**Q2.** Print the shape of the DataFrame. How many rows and columns are there?

**Q3.** Use `.info()` to check column data types and non-null counts. Are there any missing values?

**Q4.** Use `.describe()` to get summary statistics for numeric columns. What is the mean, min, and max of Sales?

**Q5.** Print all unique values in the `State`, `Group`, and `Time` columns. How many unique values does each have?

**Q6.** Display 10 random rows from the DataFrame using `.sample()`.

**Q7.** Check the data types of each column. Is the `Date` column a datetime type? If not, convert it.

---

## Section 2: Selecting & Filtering Data

**Q8.** Select only the `State` and `Sales` columns. Display the first 10 rows.

**Q9.** Select rows 100 to 110 using `.iloc[]`.

**Q10.** Select all rows where `State` is `'WA'`.

**Q11.** Select all rows where `Sales` is greater than 30000.

**Q12.** Select all rows where `Group` is `'Women'` AND `Time` is `'Morning'`.

**Q13.** Select all rows where `State` is either `'WA'`, `'NT'`, or `'SA'` using `.isin()`.

**Q14.** Select all rows where `Sales` is between 15000 and 35000 (inclusive).

**Q15.** Use `.query()` method to find all rows where `Unit > 15` and `Group == 'Kids'`.

**Q16.** Find all rows where the `Group` column contains the letter `'e'` (using `.str.contains()`).

---

## Section 3: Adding, Modifying & Removing Columns

**Q17.** Add a new column `Revenue_Per_Unit` which is `Sales / Unit`.

**Q18.** Add a new column `Sales_Category` that labels each row as:
- `'Low'` if Sales < 15000
- `'Medium'` if Sales between 15000 and 30000
- `'High'` if Sales > 30000

**Q19.** Add a new column `Month` extracted from the `Date` column (after converting to datetime).

**Q20.** Rename the column `Unit` to `Units_Sold` and `Sales` to `Total_Sales`.

**Q21.** Drop the `Revenue_Per_Unit` column you created in Q17.

**Q22.** Reorder the columns so that `Date` is first, followed by `State`, `Group`, `Time`, `Unit`, `Sales`.

---

## Section 4: Sorting

**Q23.** Sort the DataFrame by `Sales` in descending order. Display the top 10 highest sales.

**Q24.** Sort by `State` (ascending) and then by `Sales` (descending).

**Q25.** Find the top 5 rows with the highest `Unit` values using `.nlargest()`.

**Q26.** Find the bottom 5 rows with the lowest `Sales` values using `.nsmallest()`.

**Q27.** Sort by `Date` (chronological order). You may need to convert `Date` to datetime first.

---

## Section 5: Grouping & Aggregation

**Q28.** Find the total sales for each `State`. Which state has the highest total sales?

**Q29.** Find the average units sold per `Group` (Kids, Men, Women, Seniors).

**Q30.** Find the total sales for each `Time` period (Morning, Afternoon, Evening).

**Q31.** Group by `State` and `Group`, then find the mean sales for each combination.

**Q32.** Group by `State` and calculate multiple aggregations on `Sales`: mean, sum, count, min, max.

**Q33.** Find the total units sold per state per month. (Hint: extract month from Date first)

**Q34.** Use `.transform()` to add a column showing each state's average sales alongside each row.

**Q35.** For each `Group`, find the state with the highest average sales.

**Q36.** Calculate the percentage contribution of each `State` to total sales.

---

## Section 6: Handling Missing Data

**Q37.** Check if there are any missing values in the DataFrame. Show the count per column.

**Q38.** Intentionally set 50 random `Sales` values to `NaN`. Then:
- Count the total NaN values
- Fill NaN with the mean of `Sales`
- Alternatively, fill NaN with forward fill (`ffill`)

**Q39.** From the modified DataFrame (with NaN), drop all rows that have any missing values. How many rows remain?

**Q40.** Fill missing `Sales` values with the mean sales of their respective `State` group.

---

## Section 7: Pivot Tables & Cross Tabs

**Q41.** Create a pivot table showing average `Sales` for each `State` (rows) and `Time` (columns).

**Q42.** Create a pivot table showing total `Unit` sold for each `Group` (rows) and `State` (columns).

**Q43.** Create a cross-tabulation of `State` and `Group` (count of occurrences).

**Q44.** Create a pivot table showing average `Sales` by `Month` (rows) and `Group` (columns).

**Q45.** From the pivot table in Q41, which State-Time combination has the highest average sales?

---

## Section 8: Merging & Concatenating

**Q46.** Split the DataFrame into two: `df_morning` (Time == Morning) and `df_evening` (Time == Evening). Concatenate them back together vertically.

**Q47.** Split the DataFrame into `df_wa` (State == WA) and `df_nt` (State == NT). Concatenate them and reset the index.

**Q48.** Create a small DataFrame with State abbreviations and full names:
```python
state_names = pd.DataFrame({
    'State': ['WA', 'NT', 'SA', 'TAS'],
    'Full_Name': ['Western Australia', 'Northern Territory', 'South Australia', 'Tasmania']
})
```
Merge this with the main DataFrame to add full state names.

**Q49.** Split the data by month into 3 separate DataFrames (Oct, Nov, Dec). Then concatenate them back and verify the shape matches the original.

---

## Section 9: String Operations & Data Cleaning

**Q50.** Check if there are any leading/trailing spaces in the `State` or `Group` columns. If yes, strip them.

**Q51.** Convert all values in the `Group` column to uppercase.

**Q52.** Create a new column `State_Group` that combines `State` and `Group` with a hyphen (e.g., `'WA-Kids'`).

**Q53.** Find all rows where `Time` starts with `'M'`.

**Q54.** Replace `'WA'` with `'Western Australia'` in the `State` column.

---

## Section 10: Date & Time Operations

**Q55.** Convert the `Date` column to datetime format. Extract `Year`, `Month`, `Day`, and `Day_Name` into separate columns.

**Q56.** Find the total sales for each day of the week (Monday, Tuesday, etc.). Which day has the highest sales?

**Q57.** Find the total sales per week number.

**Q58.** Filter data for only the month of November 2020.

**Q59.** Find the date with the single highest total daily sales (sum of all sales on that date).

**Q60.** Calculate the 7-day rolling average of daily total sales.

---

## Section 11: Apply & Map

**Q61.** Use `.apply()` with a lambda function to create a column that doubles the `Sales` value.

**Q62.** Use `.apply()` to categorize `Unit` as:
- `'Low Demand'` if Unit <= 5
- `'Medium Demand'` if Unit between 6-15
- `'High Demand'` if Unit > 15

**Q63.** Use `.map()` to replace `Time` values: `Morning → AM`, `Afternoon → PM`, `Evening → EVE`.

**Q64.** Write a custom function that takes a row and returns `Sales * Unit`. Apply it row-wise using `.apply(axis=1)`.

---

## Section 12: Advanced Analysis

**Q65.** Find the top 3 states by total sales for each month.

**Q66.** Calculate the month-over-month sales growth percentage for each state.

**Q67.** Find the state where the difference between max and min daily sales is the largest.

**Q68.** Rank all states by their total sales using `.rank()`.

**Q69.** Create a new column showing the cumulative sales for each state (sorted by date).

**Q70.** Use `pd.cut()` to bin `Sales` into 4 categories: Low, Medium, High, Premium. Show the count per bin.

---

### Difficulty Guide
- **Q1-Q7:** Beginner (Loading, exploring)
- **Q8-Q16:** Beginner (Selecting, filtering)
- **Q17-Q27:** Intermediate (Modifying, sorting)
- **Q28-Q36:** Intermediate (Grouping, aggregation)
- **Q37-Q49:** Intermediate-Advanced (Missing data, pivots, merging)
- **Q50-Q64:** Advanced (String ops, datetime, apply)
- **Q65-Q70:** Advanced (Real-world analysis)
