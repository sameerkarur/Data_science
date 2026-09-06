# Seaborn - Statistical Data Visualization

## What is Seaborn?
Seaborn is a Python data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive, informative statistical graphics with minimal code.

**Why use Seaborn?**
- Beautiful default themes and color palettes
- Built-in support for statistical plots (regression, distribution, categorical)
- Works seamlessly with Pandas DataFrames
- Less code than Matplotlib for complex plots
- Automatic estimation and plotting of statistical relationships

## Installation
```bash
pip install seaborn
```

## Importing
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## Built-in Datasets (for practice)
```python
# List all available datasets
sns.get_dataset_names()

# Load a dataset
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
penguins = sns.load_dataset('penguins')
flights = sns.load_dataset('flights')
```

---

## 1. Setting Themes & Styles

```python
# Set theme (applies globally)
sns.set_theme()                        # default seaborn theme
sns.set_theme(style='whitegrid')       # white background with grid
sns.set_theme(style='darkgrid')        # dark background with grid
sns.set_theme(style='white')           # white, no grid
sns.set_theme(style='dark')            # dark, no grid
sns.set_theme(style='ticks')           # ticks on axes

# Set context (controls scale of elements)
sns.set_context('notebook')    # default
sns.set_context('paper')      # smaller
sns.set_context('talk')       # larger (for presentations)
sns.set_context('poster')     # largest

# Color palettes
sns.set_palette('deep')       # default
sns.set_palette('muted')
sns.set_palette('pastel')
sns.set_palette('bright')
sns.set_palette('dark')
sns.set_palette('colorblind')

# Custom palette
sns.color_palette('husl', 8)           # 8 evenly spaced colors
sns.color_palette('coolwarm', 6)       # diverging palette
sns.color_palette(['#FF6B6B', '#4ECDC4', '#45B7D1'])  # custom colors
```

---

## 2. Distribution Plots

### Histogram (histplot)
```python
tips = sns.load_dataset('tips')

# Basic histogram
sns.histplot(data=tips, x='total_bill')

# With KDE (Kernel Density Estimate) overlay
sns.histplot(data=tips, x='total_bill', kde=True, bins=20)

# Colored by category
sns.histplot(data=tips, x='total_bill', hue='sex', kde=True)

# Stacked
sns.histplot(data=tips, x='total_bill', hue='sex', multiple='stack')

plt.show()
```

### KDE Plot (kdeplot)
```python
# Basic KDE
sns.kdeplot(data=tips, x='total_bill')

# Filled KDE
sns.kdeplot(data=tips, x='total_bill', fill=True, alpha=0.5)

# Multiple groups
sns.kdeplot(data=tips, x='total_bill', hue='sex', fill=True)

# 2D KDE (bivariate)
sns.kdeplot(data=tips, x='total_bill', y='tip', fill=True, cmap='Blues')

plt.show()
```

### Distribution Plot (displot) — Figure-level
```python
# Histogram
sns.displot(data=tips, x='total_bill', kind='hist', kde=True)

# KDE
sns.displot(data=tips, x='total_bill', kind='kde')

# ECDF (Empirical Cumulative Distribution)
sns.displot(data=tips, x='total_bill', kind='ecdf')

# Faceted by column
sns.displot(data=tips, x='total_bill', col='sex', kde=True)

plt.show()
```

---

## 3. Categorical Plots

### Bar Plot (barplot) — shows mean with confidence interval
```python
tips = sns.load_dataset('tips')

# Basic bar plot
sns.barplot(data=tips, x='day', y='total_bill')

# Grouped by hue
sns.barplot(data=tips, x='day', y='total_bill', hue='sex')

# Horizontal
sns.barplot(data=tips, y='day', x='total_bill')

# Custom estimator
sns.barplot(data=tips, x='day', y='total_bill', estimator=np.median)

plt.show()
```

### Count Plot (countplot) — counts occurrences
```python
# Count of each category
sns.countplot(data=tips, x='day')

# Grouped
sns.countplot(data=tips, x='day', hue='sex')

# Ordered
sns.countplot(data=tips, x='day', order=['Thur', 'Fri', 'Sat', 'Sun'])

plt.show()
```

### Box Plot (boxplot)
```python
# Basic box plot
sns.boxplot(data=tips, x='day', y='total_bill')

# Grouped
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')

# Horizontal
sns.boxplot(data=tips, y='day', x='total_bill')

# Show all points
sns.boxplot(data=tips, x='day', y='total_bill')
sns.stripplot(data=tips, x='day', y='total_bill', color='black', alpha=0.3, size=3)

plt.show()
```

### Violin Plot (violinplot)
```python
# Basic violin plot
sns.violinplot(data=tips, x='day', y='total_bill')

# Split violin (compare two groups)
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)

# Inner options: 'box', 'quartile', 'point', 'stick', None
sns.violinplot(data=tips, x='day', y='total_bill', inner='quartile')

plt.show()
```

### Strip Plot (stripplot) — individual data points
```python
sns.stripplot(data=tips, x='day', y='total_bill', jitter=True, alpha=0.6)
plt.show()
```

### Swarm Plot (swarmplot) — non-overlapping points
```python
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()
```

### Categorical Plot (catplot) — Figure-level
```python
# kind: 'strip', 'swarm', 'box', 'violin', 'bar', 'count', 'point'
sns.catplot(data=tips, x='day', y='total_bill', kind='box')

# Faceted
sns.catplot(data=tips, x='day', y='total_bill', kind='bar', col='sex')

plt.show()
```

---

## 4. Relational Plots

### Scatter Plot (scatterplot)
```python
tips = sns.load_dataset('tips')

# Basic scatter
sns.scatterplot(data=tips, x='total_bill', y='tip')

# Color by category
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')

# Size by value
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', size='size')

# Style by category
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex', style='smoker')

plt.show()
```

### Line Plot (lineplot)
```python
flights = sns.load_dataset('flights')

# Basic line plot
sns.lineplot(data=flights, x='year', y='passengers')

# Grouped
sns.lineplot(data=flights, x='year', y='passengers', hue='month')

# With confidence interval
sns.lineplot(data=flights, x='year', y='passengers', ci=95)

plt.show()
```

### Relational Plot (relplot) — Figure-level
```python
# kind: 'scatter' or 'line'
sns.relplot(data=tips, x='total_bill', y='tip', kind='scatter', hue='sex', col='time')

plt.show()
```

---

## 5. Regression Plots

### regplot — single axes
```python
# Linear regression with scatter
sns.regplot(data=tips, x='total_bill', y='tip')

# Polynomial regression
sns.regplot(data=tips, x='total_bill', y='tip', order=2)

# Without confidence interval
sns.regplot(data=tips, x='total_bill', y='tip', ci=None)

# Scatter only (no regression line)
sns.regplot(data=tips, x='total_bill', y='tip', fit_reg=False)

plt.show()
```

### lmplot — Figure-level (supports faceting)
```python
# Basic
sns.lmplot(data=tips, x='total_bill', y='tip')

# Grouped by hue
sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex')

# Faceted
sns.lmplot(data=tips, x='total_bill', y='tip', col='sex', row='smoker')

plt.show()
```

---

## 6. Matrix Plots

### Heatmap
```python
# Correlation heatmap
tips_numeric = tips.select_dtypes(include=[np.number])
corr = tips_numeric.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr,
            annot=True,          # show values
            fmt='.2f',           # format
            cmap='coolwarm',     # colormap
            center=0,            # center colormap at 0
            square=True,         # square cells
            linewidths=0.5,      # line between cells
            vmin=-1, vmax=1)     # value range

plt.title('Correlation Heatmap')
plt.show()
```

### Clustermap (hierarchical clustering)
```python
sns.clustermap(corr,
               annot=True,
               cmap='coolwarm',
               figsize=(8, 8))
plt.show()
```

---

## 7. Pair Plot (pairplot) — All pairwise relationships

```python
iris = sns.load_dataset('iris')

# Basic pair plot
sns.pairplot(iris)

# Colored by species
sns.pairplot(iris, hue='species')

# With regression lines on scatter plots
sns.pairplot(iris, hue='species', kind='reg')

# Select specific columns
sns.pairplot(iris, vars=['sepal_length', 'sepal_width', 'petal_length'], hue='species')

# Customize diagonal
sns.pairplot(iris, hue='species', diag_kind='kde')   # 'hist' or 'kde'

plt.show()
```

---

## 8. Joint Plot (jointplot) — Bivariate + marginal distributions

```python
# Basic (scatter + histograms)
sns.jointplot(data=tips, x='total_bill', y='tip')

# KDE
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde', fill=True)

# Hex
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')

# Regression
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')

# With hue
sns.jointplot(data=tips, x='total_bill', y='tip', hue='sex')

plt.show()
```

---

## 9. FacetGrid — Multi-plot grids

```python
tips = sns.load_dataset('tips')

# Create grid
g = sns.FacetGrid(tips, col='sex', row='smoker', height=4)

# Map a plot to each facet
g.map(sns.scatterplot, 'total_bill', 'tip')

# Map with hue
g = sns.FacetGrid(tips, col='time', hue='sex', height=5)
g.map(sns.scatterplot, 'total_bill', 'tip')
g.add_legend()

plt.show()
```

---

## 10. Customizing Seaborn Plots

### Since Seaborn returns Matplotlib objects, you can customize with Matplotlib:
```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=tips, x='day', y='total_bill', ax=ax)

# Matplotlib customization
ax.set_title('Average Bill by Day', fontsize=16, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Total Bill ($)', fontsize=12)
ax.set_ylim(0, 30)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Figure-level plots return a FacetGrid object:
```python
g = sns.catplot(data=tips, x='day', y='total_bill', kind='bar')
g.fig.set_size_inches(10, 6)
g.set_axis_labels('Day', 'Total Bill ($)')
g.set_titles('Bill Distribution')
plt.show()
```

---

## 11. Axes-level vs Figure-level Functions

| Axes-level (use with ax=) | Figure-level (creates own figure) |
|---|---|
| `scatterplot()` | `relplot(kind='scatter')` |
| `lineplot()` | `relplot(kind='line')` |
| `histplot()` | `displot(kind='hist')` |
| `kdeplot()` | `displot(kind='kde')` |
| `boxplot()` | `catplot(kind='box')` |
| `violinplot()` | `catplot(kind='violin')` |
| `barplot()` | `catplot(kind='bar')` |
| `countplot()` | `catplot(kind='count')` |
| `regplot()` | `lmplot()` |
| `heatmap()` | — |

**Axes-level**: Pass `ax=` to plot on existing axes. Good for subplots.
**Figure-level**: Creates its own figure. Supports `col=` and `row=` for faceting.

---

## Quick Reference Table

| Plot Type | Function | Use Case |
|---|---|---|
| Histogram | `sns.histplot()` | Distribution of a single variable |
| KDE | `sns.kdeplot()` | Smooth density estimate |
| Box plot | `sns.boxplot()` | Distribution summary (median, quartiles) |
| Violin plot | `sns.violinplot()` | Distribution shape + box plot |
| Bar plot | `sns.barplot()` | Mean of a variable per category |
| Count plot | `sns.countplot()` | Count per category |
| Scatter plot | `sns.scatterplot()` | Relationship between two variables |
| Line plot | `sns.lineplot()` | Trends over time |
| Regression | `sns.regplot()` | Scatter + fitted regression line |
| Heatmap | `sns.heatmap()` | Matrix/correlation visualization |
| Pair plot | `sns.pairplot()` | All pairwise relationships |
| Joint plot | `sns.jointplot()` | Bivariate + marginal distributions |
| Strip plot | `sns.stripplot()` | Individual data points by category |
| Swarm plot | `sns.swarmplot()` | Non-overlapping categorical scatter |
| Facet grid | `sns.FacetGrid()` | Multi-panel plots by category |
