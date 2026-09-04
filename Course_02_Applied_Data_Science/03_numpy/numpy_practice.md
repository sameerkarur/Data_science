# NumPy Practice Questions
### Dataset: AusApparalSales4thQrt2020.csv (Australian Apparel Sales Q4 2020)

**Columns:** Date, Time, State, Group, Unit, Sales

> **Setup:** Load the data first, then extract numeric columns (Unit, Sales) as NumPy arrays.
```python
import numpy as np
import pandas as pd

df = pd.read_csv('AusApparalSales4thQrt2020.csv')
units = df['Unit'].to_numpy()
sales = df['Sales'].to_numpy()
```

---

## Section 1: Array Basics

**Q1.** Print the shape, dimensions, size, and data type of the `sales` array.

**Q2.** Print the first 10 elements, last 5 elements, and every 3rd element of the `units` array.

**Q3.** Reshape the first 60 elements of `sales` into a 2D array of shape (10, 6). Print the result.

**Q4.** Flatten the reshaped array from Q3 back to 1D and verify it matches the original 60 elements.

**Q5.** Create a copy of the `units` array. Modify the first element of the copy to 999. Verify the original array is unchanged.

---

## Section 2: Indexing & Slicing

**Q6.** Extract all sales values from index 100 to 120 (inclusive).

**Q7.** Extract every 50th element from the `sales` array.

**Q8.** Using boolean indexing, find all `units` values that are greater than 20.

**Q9.** Using boolean indexing, find all `sales` values where `units` is exactly 10.

**Q10.** Use fancy indexing to extract sales at indices [0, 100, 500, 1000, 5000].

---

## Section 3: Mathematical & Statistical Operations

**Q11.** Calculate the mean, median, standard deviation, and variance of the `sales` array.

**Q12.** Find the minimum and maximum values in `units`. Also find their index positions using `argmin()` and `argmax()`.

**Q13.** Calculate the total (sum) of all sales. What is the cumulative sum of the first 20 sales values?

**Q14.** Calculate the average sales per unit. (Hint: divide `sales` array by `units` array — handle division by zero if any)

**Q15.** What percentage of total sales does each individual sale represent? Store the result in a new array.

---

## Section 4: Array Operations

**Q16.** Add 500 to every element in the `sales` array (broadcasting). Print the first 10 results.

**Q17.** Multiply all `units` values by 2.5 (simulating a price increase). Print the first 10 results.

**Q18.** Create a boolean array where `sales > 30000`. How many True values are there? What percentage of total rows is that?

**Q19.** Use `np.where()` to create a new array that labels each sale as `'High'` if sales > 25000, else `'Low'`.

**Q20.** Clip the `sales` array so that all values are between 10000 and 40000. Print the first 20 values.

---

## Section 5: Sorting & Searching

**Q21.** Sort the `sales` array in ascending order. What are the 5 smallest and 5 largest sales values?

**Q22.** Use `np.argsort()` on `sales` to find the indices of the top 10 highest sales.

**Q23.** Find all unique values in the `units` array. How many unique unit values exist?

**Q24.** Use `np.unique()` with `return_counts=True` on `units` to find the frequency of each unit value.

**Q25.** Use `np.searchsorted()` on a sorted `sales` array to find where the value 25000 would be inserted.

---

## Section 6: Reshaping & Stacking

**Q26.** Take the first 120 sales values. Reshape them into a (10, 12) matrix. Find the sum of each row and each column.

**Q27.** Split the first 100 elements of `units` into 5 equal arrays using `np.split()`.

**Q28.** Create two arrays: one with the first 50 sales values and another with the next 50. Stack them vertically using `np.vstack()` and horizontally using `np.hstack()`.

**Q29.** Take the (10, 12) matrix from Q26. Transpose it. What is the new shape?

**Q30.** Add a new axis to the `units` array to make it a column vector. Print the shape.

---

## Section 7: Aggregation with Axis

**Q31.** Reshape the first 200 sales values into a (10, 20) matrix. Calculate:
- Sum along axis=0 (column-wise sum)
- Sum along axis=1 (row-wise sum)
- Mean along axis=0
- Mean along axis=1

**Q32.** For the same matrix, find the max value in each row and the min value in each column.

**Q33.** For the same matrix, use `np.argmax(axis=1)` to find which column has the highest value in each row.

---

## Section 8: Random & Simulation

**Q34.** Set a random seed of 42. Generate a random array of 100 values from a normal distribution with mean=25000 and std=5000. Compare its mean and std with the actual `sales` array.

**Q35.** Use `np.random.choice()` to randomly sample 50 values from the `sales` array (without replacement).

**Q36.** Simulate 1000 random daily sales by sampling from the `units` array with replacement. What is the mean of this simulated data?

**Q37.** Generate a 5x5 random integer matrix with values between 1 and 50. Find its determinant and inverse (if it exists).

---

## Section 9: Linear Algebra (Bonus)

**Q38.** Create a 3x3 matrix from the first 9 sales values. Calculate its:
- Transpose
- Determinant
- Trace (sum of diagonal)

**Q39.** Create two 3x3 matrices from sales data. Perform matrix multiplication using `@` operator and `np.dot()`.

**Q40.** Solve the system of equations:
```
2x + 3y = 25000
4x + y  = 30000
```
Use `np.linalg.solve()`.

---

## Section 10: Real-World Analysis with NumPy

**Q41.** Calculate the z-score for each value in the `sales` array. How many sales are more than 2 standard deviations from the mean?

**Q42.** Calculate the correlation coefficient between `units` and `sales` using `np.corrcoef()`.

**Q43.** Use `np.percentile()` to find the 25th, 50th, 75th, and 90th percentiles of `sales`.

**Q44.** Bin the `sales` data into 5 equal-width bins using `np.histogram()`. Print the bin edges and counts.

**Q45.** Calculate the moving average of `sales` with a window size of 7 using `np.convolve()`.

---

### Difficulty Guide
- **Q1-Q10:** Beginner (Array creation, indexing, slicing)
- **Q11-Q20:** Intermediate (Math operations, broadcasting, filtering)
- **Q21-Q30:** Intermediate (Sorting, reshaping, stacking)
- **Q31-Q37:** Advanced (Axis operations, random, simulation)
- **Q38-Q45:** Advanced (Linear algebra, real-world analysis)
