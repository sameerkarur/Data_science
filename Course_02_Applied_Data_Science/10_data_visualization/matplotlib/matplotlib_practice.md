# Matplotlib Practice Questions
### Dataset: AusApparalSales4thQrt2020.csv (Australian Apparel Sales Q4 2020)

**Columns:** Date, Time, State, Group, Unit, Sales

> **Setup:**
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('AusApparalSales4thQrt2020.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
```

---

## Section 1: Basic Line Plots

**Q1.** Calculate the total daily sales (sum of all sales per date). Plot a line chart of daily total sales over time.
- Add a title: "Daily Total Sales - Q4 2020"
- Label x-axis: "Date", y-axis: "Total Sales ($)"

**Q2.** Plot the daily total sales for each `State` on the same line chart (one line per state). Add a legend.

**Q3.** Plot the daily total sales for the `Kids` group only. Use a red dashed line with circle markers.

**Q4.** Plot the monthly total sales (Oct, Nov, Dec) as a line chart with markers. Annotate the highest month with an arrow.

**Q5.** Plot the 7-day rolling average of daily total sales. Overlay it on top of the actual daily sales (use different colors and alpha).

---

## Section 2: Bar Charts

**Q6.** Create a vertical bar chart showing total sales per `State`. 
- Use different colors for each bar
- Add value labels on top of each bar

**Q7.** Create a horizontal bar chart showing total sales per `Group` (Kids, Men, Women, Seniors). Sort bars by value.

**Q8.** Create a grouped bar chart comparing total sales per `State` for each `Time` period (Morning, Afternoon, Evening).

**Q9.** Create a stacked bar chart showing total sales per `State`, stacked by `Group`.

**Q10.** Create a bar chart showing the average units sold per `Group` for each month (Oct, Nov, Dec) — grouped bars.

---

## Section 3: Scatter Plots

**Q11.** Create a scatter plot of `Unit` (x-axis) vs `Sales` (y-axis). What relationship do you observe?

**Q12.** Create a scatter plot of `Unit` vs `Sales`, colored by `Group`. Add a legend and use different colors for each group.

**Q13.** Create a scatter plot of `Unit` vs `Sales`, where the size of each point represents the `Sales` value. Use alpha=0.5 for transparency.

**Q14.** Create a scatter plot of average daily `Unit` vs average daily `Sales` per state. Label each point with the state name.

**Q15.** Create a scatter plot of `Unit` vs `Sales` colored by `Time` period. Add a colorbar or legend.

---

## Section 4: Histograms

**Q16.** Plot a histogram of the `Sales` column with 30 bins. Add a vertical line at the mean sales value (use `axvline`).

**Q17.** Plot overlapping histograms of `Sales` for `Morning`, `Afternoon`, and `Evening` (use alpha=0.5 for each).

**Q18.** Plot a histogram of `Unit` values with 20 bins. Use `density=True` to show probability density.

**Q19.** Plot side-by-side histograms of `Sales` for `Kids` and `Seniors` groups using subplots.

**Q20.** Plot a histogram of `Sales` for each `State` in a 2x2 subplot grid.

---

## Section 5: Pie Charts

**Q21.** Create a pie chart showing the percentage of total sales contributed by each `State`. Add percentage labels.

**Q22.** Create a pie chart showing the sales distribution across `Group` (Kids, Men, Women, Seniors). Explode the largest slice.

**Q23.** Create a pie chart showing the sales distribution across `Time` periods.

**Q24.** Create a donut chart (pie chart with a white circle in the center) for sales by `State`.

---

## Section 6: Box Plots

**Q25.** Create a box plot of `Sales` values for each `State`.

**Q26.** Create a box plot of `Sales` for each `Group`, with filled colors (use `patch_artist=True`).

**Q27.** Create a box plot of `Unit` for each `Time` period. Add a horizontal line at the overall median.

**Q28.** Create side-by-side box plots: one for `Sales` by `State` and one for `Units` by `State` (use subplots).

---

## Section 7: Subplots & Multi-Panel Figures

**Q29.** Create a 2x2 subplot figure with:
- Top-left: Line plot of daily total sales
- Top-right: Bar chart of total sales per state
- Bottom-left: Histogram of sales
- Bottom-right: Pie chart of sales by group

**Q30.** Create a 1x3 subplot figure showing bar charts of total sales per `Group` for each `Time` period (Morning, Afternoon, Evening).

**Q31.** Create a 2x2 subplot figure with histograms of `Sales` for each state (pick any 4 states). Use shared y-axis.

**Q32.** Create a figure with 3 subplots stacked vertically showing daily sales trends for October, November, and December separately.

---

## Section 8: Customization & Styling

**Q33.** Take the bar chart from Q6 and customize it:
- Use `plt.style.use('ggplot')`
- Add a grid (y-axis only)
- Remove top and right spines
- Use a custom color palette
- Set font sizes for title (16), labels (12), ticks (10)

**Q34.** Create a line plot of daily sales for WA state with:
- Custom line color and width
- Shaded area between the line and x-axis (`fill_between`)
- Annotate the peak sales day with text and an arrow

**Q35.** Create a professional-looking bar chart comparing states. Add:
- Error bars showing standard deviation
- A horizontal line showing the overall average
- Custom tick labels rotated 45 degrees

**Q36.** Create any plot of your choice and save it as:
- PNG (300 dpi)
- PDF
- SVG

---

## Section 9: Heatmaps & Advanced Plots

**Q37.** Create a heatmap (using `imshow`) showing average sales for each `State` (rows) vs `Time` (columns). Add text annotations with values.

**Q38.** Create a heatmap showing average sales for each `Group` (rows) vs `Month` (columns).

**Q39.** Create a stacked area chart showing daily total sales by `Group` over time.

**Q40.** Create a plot with twin y-axes: total `Sales` on the left y-axis and total `Units` on the right y-axis, both plotted against `Date`.

---

## Section 10: Real-World Data Analysis Visualizations

**Q41.** Create a "Sales Dashboard" figure with 4 subplots:
- Total sales trend over time (line)
- Sales breakdown by state (bar)
- Sales distribution (histogram)
- Top group performance (horizontal bar)
Add a main title: "Q4 2020 Australian Apparel Sales Dashboard"

**Q42.** Plot the daily sales for each state in separate subplots (one per state) arranged in a grid. Use shared x and y axes for easy comparison.

**Q43.** Create a bar chart showing the top 10 dates with the highest total sales. Highlight the #1 date in a different color.

**Q44.** Create a grouped bar chart comparing Morning vs Evening sales for each state. Add percentage difference labels.

**Q45.** Create a monthly comparison chart: for each state, show 3 bars (Oct, Nov, Dec) side by side. Use a professional color scheme.

---

### Difficulty Guide
- **Q1-Q5:** Beginner (Line plots)
- **Q6-Q10:** Beginner-Intermediate (Bar charts)
- **Q11-Q15:** Intermediate (Scatter plots)
- **Q16-Q24:** Intermediate (Histograms, pie charts)
- **Q25-Q32:** Intermediate (Box plots, subplots)
- **Q33-Q40:** Advanced (Customization, heatmaps)
- **Q41-Q45:** Advanced (Dashboard-style visualizations)
