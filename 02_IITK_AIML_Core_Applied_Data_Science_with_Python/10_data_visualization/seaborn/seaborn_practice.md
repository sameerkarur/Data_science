# Seaborn Practice Questions
### Dataset: AusApparalSales4thQrt2020.csv (Australian Apparel Sales Q4 2020)

**Columns:** Date, Time, State, Group, Unit, Sales

> **Setup:**
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('AusApparalSales4thQrt2020.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
df['Month'] = df['Date'].dt.month_name()

# Strip any whitespace from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

sns.set_theme(style='whitegrid')
```

---

## Section 1: Distribution Plots

**Q1.** Create a histogram of `Sales` using `sns.histplot()`. Add a KDE curve overlay.

**Q2.** Create a histogram of `Sales` colored by `Group` (hue). Use `multiple='stack'` to stack them.

**Q3.** Create a KDE plot of `Sales` for each `Time` period (Morning, Afternoon, Evening) with filled areas.

**Q4.** Create a 2D KDE plot (bivariate) of `Unit` vs `Sales` using `sns.kdeplot()` with `fill=True`.

**Q5.** Use `sns.displot()` (figure-level) to create histograms of `Sales` faceted by `State` (one column per state). Add KDE.

**Q6.** Create an ECDF (Empirical Cumulative Distribution) plot of `Sales` grouped by `Group` using `sns.displot(kind='ecdf')`.

**Q7.** Create a histogram of `Unit` for each `Time` period using `sns.displot()` with `col='Time'`.

---

## Section 2: Categorical Plots — Bar & Count

**Q8.** Create a bar plot showing the average `Sales` per `State` using `sns.barplot()`. Which state has the highest average?

**Q9.** Create a bar plot of average `Sales` per `State`, grouped by `Group` (use `hue='Group'`).

**Q10.** Create a count plot showing the number of records per `State`.

**Q11.** Create a count plot of `Group` colored by `Time` period.

**Q12.** Create a bar plot showing the average `Sales` per `Group` for each `Time` period. Use `hue='Time'`.

**Q13.** Create a horizontal bar plot of average `Sales` per `State` using `sns.barplot()` with `orient='h'`.

**Q14.** Use `sns.catplot()` (figure-level) to create bar plots of average `Sales` per `Group`, faceted by `State` (one subplot per state).

---

## Section 3: Categorical Plots — Box & Violin

**Q15.** Create a box plot of `Sales` for each `State` using `sns.boxplot()`.

**Q16.** Create a box plot of `Sales` for each `Group`, colored by `Time` (use `hue='Time'`).

**Q17.** Create a violin plot of `Sales` for each `State`.

**Q18.** Create a split violin plot comparing `Sales` for `Morning` vs `Evening` across each `State`. 
(Hint: filter data to only Morning & Evening, use `hue='Time'`, `split=True`)

**Q19.** Create a box plot of `Unit` per `Group` and overlay individual data points using `sns.stripplot()` on top.

**Q20.** Use `sns.catplot(kind='violin')` to create violin plots of `Sales` per `Group`, faceted by `Time`.

---

## Section 4: Categorical Plots — Strip & Swarm

**Q21.** Create a strip plot of `Sales` for each `State` with `jitter=True` and `alpha=0.4`.

**Q22.** Create a swarm plot of `Sales` for each `Group`. Color by `Time` period.

**Q23.** Create a combined plot: box plot + strip plot of `Sales` per `State` (overlay strip on box).

**Q24.** Use `sns.catplot(kind='swarm')` to show `Unit` values per `Group`, faceted by `Time`.

---

## Section 5: Relational Plots (Scatter & Line)

**Q25.** Create a scatter plot of `Unit` (x) vs `Sales` (y) using `sns.scatterplot()`. Color by `Group`.

**Q26.** Create a scatter plot of `Unit` vs `Sales`, colored by `State` and sized by `Sales`.

**Q27.** Create a scatter plot of `Unit` vs `Sales` with different marker styles for each `Time` period.

**Q28.** Calculate daily total sales per state. Create a line plot of daily sales over time, with one line per `State` using `sns.lineplot()`.

**Q29.** Create a line plot of monthly average `Sales` per `Group` using `sns.lineplot()`.

**Q30.** Use `sns.relplot()` (figure-level) to create scatter plots of `Unit` vs `Sales`, faceted by `Time` (columns) and `Group` (rows).

---

## Section 6: Regression Plots

**Q31.** Create a regression plot of `Unit` vs `Sales` using `sns.regplot()`. Is there a linear relationship?

**Q32.** Create a regression plot with `ci=None` (no confidence interval) and `order=2` (polynomial fit).

**Q33.** Use `sns.lmplot()` to create regression plots of `Unit` vs `Sales`, colored by `Group` (one regression line per group).

**Q34.** Use `sns.lmplot()` to create regression plots of `Unit` vs `Sales`, faceted by `State` (one subplot per state).

**Q35.** Use `sns.lmplot()` with `col='Time'` and `hue='Group'` to see how the Unit-Sales relationship varies by time and group.

---

## Section 7: Matrix Plots (Heatmaps)

**Q36.** Calculate the correlation matrix of numeric columns (`Unit`, `Sales`). Create a heatmap with `sns.heatmap()` and annotate values.

**Q37.** Create a pivot table of average `Sales` by `State` (rows) and `Time` (columns). Plot it as a heatmap with annotations.

**Q38.** Create a pivot table of total `Sales` by `State` (rows) and `Group` (columns). Plot as a heatmap using the `'YlOrRd'` colormap.

**Q39.** Create a pivot table of average `Unit` by `Group` (rows) and `Month` (columns). Plot as an annotated heatmap.

**Q40.** Create a clustermap of the State-Group sales pivot table from Q38 using `sns.clustermap()`.

---

## Section 8: Pair Plots & Joint Plots

**Q41.** Create a pair plot of the numeric columns (`Unit`, `Sales`) colored by `Group` using `sns.pairplot()`.

**Q42.** Create a pair plot colored by `State`. Use `diag_kind='kde'` for the diagonal.

**Q43.** Create a joint plot of `Unit` vs `Sales` using `sns.jointplot()` with default settings (scatter + histograms).

**Q44.** Create a joint plot of `Unit` vs `Sales` with `kind='kde'` (2D density).

**Q45.** Create a joint plot of `Unit` vs `Sales` with `kind='hex'` (hexbin).

**Q46.** Create a joint plot of `Unit` vs `Sales` colored by `Group` using `hue='Group'`.

---

## Section 9: FacetGrid & Custom Multi-Panel Plots

**Q47.** Use `sns.FacetGrid` to create a grid of histograms of `Sales`, one per `State`.

**Q48.** Use `sns.FacetGrid` with `col='Time'` and `row='Group'` to create a grid of scatter plots (Unit vs Sales).

**Q49.** Use `sns.catplot()` to create box plots of `Sales` per `State`, faceted by `Group` (one subplot per group).

**Q50.** Use `sns.displot()` to create KDE plots of `Sales`, faceted by `State` (columns) and `Time` (rows).

---

## Section 10: Themes, Styles & Customization

**Q51.** Create the same bar plot (average Sales per State) using 4 different Seaborn styles:
- `whitegrid`, `darkgrid`, `white`, `ticks`
Display them in a 2x2 subplot grid.

**Q52.** Create a bar plot of average Sales per Group using 3 different color palettes:
- `'deep'`, `'pastel'`, `'colorblind'`
Display them in a 1x3 subplot grid.

**Q53.** Create a bar plot using `sns.set_context('talk')` for presentation-ready sizing. Add a custom title and labels.

**Q54.** Create a scatter plot and customize it with Matplotlib:
- Set figure size to (12, 7)
- Remove top and right spines
- Add a custom title with bold font
- Rotate x-tick labels 45 degrees
- Save as PNG at 300 dpi

---

## Section 11: Real-World Data Analysis Visualizations

**Q55.** **Sales Performance Dashboard:** Create a 2x2 figure with:
- Top-left: Bar plot of total sales per state
- Top-right: Box plot of sales distribution per group
- Bottom-left: Heatmap of avg sales by State vs Time
- Bottom-right: Line plot of monthly sales trend per group

**Q56.** **State Comparison:** For each state, create a violin plot of Sales split by Time (Morning vs Evening). Arrange all states in a single row.

**Q57.** **Group Analysis:** Create a catplot showing box plots of Sales for each Group, faceted by Month. Which group performs best in which month?

**Q58.** **Correlation Deep Dive:** Add `Revenue_Per_Unit` (Sales/Unit) column. Create a pair plot of `Unit`, `Sales`, and `Revenue_Per_Unit` colored by `Group`.

**Q59.** **Time-of-Day Analysis:** Create a heatmap showing average Sales for each `State` (rows) vs `Time` (columns) for each month separately (3 heatmaps side by side).

**Q60.** **Complete EDA Report:** Create a single figure with 6 subplots that tells the complete story of this dataset:
1. Overall sales distribution (histogram + KDE)
2. Sales by state (bar plot)
3. Sales by group (box plot)
4. Sales by time of day (violin plot)
5. Unit vs Sales relationship (scatter + regression)
6. State vs Group heatmap (average sales)
Add a main suptitle: "Australian Apparel Sales - Q4 2020 EDA"

---

### Difficulty Guide
- **Q1-Q7:** Beginner (Distribution plots)
- **Q8-Q14:** Beginner (Bar & count plots)
- **Q15-Q24:** Intermediate (Box, violin, strip, swarm)
- **Q25-Q35:** Intermediate (Scatter, line, regression)
- **Q36-Q46:** Intermediate-Advanced (Heatmaps, pair plots, joint plots)
- **Q47-Q54:** Advanced (FacetGrid, themes, customization)
- **Q55-Q60:** Advanced (Real-world dashboard & EDA visualizations)
