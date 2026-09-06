# Seaborn: Complete Step-by-Step Statistical Graphics Handbook
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Seaborn & Why Use It Over Raw Matplotlib?](#1-what-is-seaborn--why-use-it-over-raw-matplotlib)
2. [Installation & Built-in Datasets](#2-installation--built-in-datasets)
3. [The Three Seaborn Plotting Families (Relational, Categorical, Distributions)](#3-the-three-seaborn-plotting-families)
4. [Relational Plots: `scatterplot()` & `lineplot()`](#4-relational-plots-scatterplot--lineplot)
5. [Categorical Plots: `barplot()`, `countplot()`, `boxplot()` & `violinplot()`](#5-categorical-plots)
6. [Distribution Plots: `histplot()`, `kdeplot()` & `displot()`](#6-distribution-plots)
7. [Matrix Plots: Correlation `heatmap()` & Hierarchical `clustermap()`](#7-matrix-plots-correlation-heatmap--clustermap)
8. [Linear Regression Plots: `regplot()` & `lmplot()`](#8-linear-regression-plots)
9. [Multi-Plot Grids: `pairplot()` & `FacetGrid()`](#9-multi-plot-grids-pairplot--facetgrid)
10. [Aesthetics, Themes & Color Palettes](#10-aesthetics-themes--color-palettes)
11. [Try It Yourself! (Hands-On Practice Exercises)](#11-try-it-yourself-hands-on-practice-exercises)
12. [Quick Reference Cheat Sheet](#12-quick-reference-cheat-sheet)

---

## 1. What is Seaborn & Why Use It Over Raw Matplotlib?

**Seaborn** is a statistical data visualization library based on Matplotlib. It provides a high-level, declarative API for drawing attractive and informative statistical graphics.

### Key Advantages:
- **Automatic Statistical Estimations:** Automatically computes confidence intervals, error bars, regression trendlines, and probability densities.
- **Direct DataFrame Integration:** Accepts native Pandas DataFrames directly via column name strings (`x='total_bill', y='tip', hue='smoker'`).
- **Semantic Mapping (`hue`, `size`, `style`):** Map multiple data dimensions onto colors, marker shapes, and sizes in one line.
- **Modern Aesthetics:** Elegant default themes, colorblind-friendly palettes, and clean typography.

---

## 2. Installation & Built-in Datasets

```bash
pip install seaborn
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set standard Seaborn theme
sns.set_theme(style="darkgrid", palette="muted")

# Load famous built-in datasets
tips = sns.load_dataset("tips")
print("Tips Dataset Head:\n", tips.head(3))
```

#### Output:
```text
Tips Dataset Head:
    total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
```

---

## 3. The Three Seaborn Plotting Families

Seaborn organizes all plots into three distinct figure-level paradigms:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SEABORN PLOTTING TAXONOMY                       │
├───────────────────┬──────────────────────┬─────────────────────────────┤
│ 1. RELATIONAL     │ 2. CATEGORICAL       │ 3. DISTRIBUTIONS            │
│   relplot()       │   catplot()          │   displot()                 │
│   ├── scatterplot │   ├── barplot        │   ├── histplot              │
│   └── lineplot    │   ├── boxplot        │   ├── kdeplot               │
│                   │   ├── violinplot     │   └── ecdfplot              │
│                   │   └── countplot      │                             │
└───────────────────┴──────────────────────┴─────────────────────────────┘
```

---

## 4. Relational Plots: `scatterplot()` & `lineplot()`

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(7, 4.5))

# Scatterplot with color (hue) and size semantics
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="time",
    style="smoker",
    size="size",
    sizes=(20, 200),
    palette="deep",
    ax=ax
)

ax.set_title("Total Bill vs Tip Amount by Dining Time & Party Size", fontweight='bold')
plt.show()
```

#### Output:
```text
[Rendered Graphic: Multi-dimensional scatter plot with distinct Lunch/Dinner colors and variable circle sizes]
```

---

## 5. Categorical Plots

### Box Plots & Violin Plots (Examining Dispersion)
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Boxplot
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex", palette="Set2", ax=ax1)
ax1.set_title("Total Bill Distribution by Day (Box Plot)")

# Violinplot with split hue
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True, palette="Pastel1", ax=ax2)
ax2.set_title("Density & Quartiles by Day (Split Violin)")

plt.tight_layout()
plt.show()
```

#### Output:
```text
[Rendered Graphic: Dual-panel comparison showing outlier quartiles (Box) and probability density curves (Violin)]
```

---

## 6. Distribution Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")

fig, ax = plt.subplots(figsize=(8, 4))

# Kernel Density Estimation (KDE) with hue
sns.kdeplot(
    data=penguins,
    x="flipper_length_mm",
    hue="species",
    fill=True,
    common_norm=False,
    palette="crest",
    alpha=0.5,
    linewidth=2,
    ax=ax
)

ax.set_title("Flipper Length Probability Density by Penguin Species", fontweight='bold')
plt.show()
```

#### Output:
```text
[Rendered Graphic: Smooth overlapping colored probability distributions showing species separation]
```

---

## 7. Matrix Plots: Correlation Heatmap & Clustermap

Heatmaps visualize numerical correlation matrices:

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
corr_matrix = iris.drop(columns='species').corr()

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=1.5,
    cbar_kws={"shrink": 0.8},
    ax=ax
)

ax.set_title("Iris Feature Pearson Correlation Matrix", fontweight='bold')
plt.show()
```

#### Output:
```text
[Rendered Graphic: 4x4 annotated color grid showing strong positive correlation between petal length and width]
```

---

## 8. Linear Regression Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# Regplot fits and plots a linear regression model with 95% bootstrap confidence band
g = sns.lmplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="smoker",
    col="time",
    height=4,
    aspect=1.2,
    palette="Dark2"
)

g.set_axis_labels("Total Bill ($)", "Tip ($)")
plt.show()
```

#### Output:
```text
[Rendered Graphic: Two regression panels (Lunch and Dinner) with fitted trendlines and shaded 95% CI bands]
```

---

## 9. Multi-Plot Grids: `pairplot()`

`pairplot` generates pairwise bivariate distributions across all numerical variables in a single function call:

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

# Pairwise scatter plots and diagonal KDEs
g = sns.pairplot(iris, hue="species", palette="Set1", corner=True)
plt.show()
```

#### Output:
```text
[Rendered Graphic: Lower-triangle pairwise scatter matrix highlighting distinct separation of Iris-setosa]
```

---

## 10. Aesthetics, Themes & Color Palettes

Seaborn provides 5 built-in themes: `darkgrid`, `whitegrid`, `dark`, `white`, `ticks`.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Preview color palettes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2.5))

sns.palplot(sns.color_palette("mako", 8))
sns.palplot(sns.color_palette("flare", 8))
plt.show()
```

---

## 11. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Finding High-Value Customers
**Task:** Using the `tips` dataset, plot a grouped bar chart comparing the mean tip percentage `(tip / total_bill) * 100` across days of the week, broken down by smoker status.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
tips['tip_pct'] = (tips['tip'] / tips['total_bill']) * 100

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=tips, x="day", y="tip_pct", hue="smoker", errorbar="sd", palette="Blues", ax=ax)

ax.set_title("Average Tip Percentage by Day and Smoking Status")
ax.set_ylabel("Tip Percentage (%)")
plt.show()
```
</details>

---

## 12. Quick Reference Cheat Sheet

| Function | Primary Use Case | Key Parameters |
|---|---|---|
| `sns.scatterplot()` | Bivariate continuous points | `hue`, `style`, `size`, `palette` |
| `sns.lineplot()` | Time series / continuous curves | `hue`, `errorbar='ci'`, `estimator='mean'` |
| `sns.barplot()` | Mean estimates with error bars | `x`, `y`, `hue`, `estimator`, `ci` |
| `sns.boxplot()` | Quartiles, median, and outliers | `notch=True`, `whis`, `palette` |
| `sns.violinplot()` | Kernel density + box representation | `split=True`, `inner='quartile'` |
| `sns.histplot()` | Binned counts with KDE overlay | `kde=True`, `bins`, `stat='density'` |
| `sns.heatmap()` | 2D color matrix of correlation | `annot=True`, `cmap='coolwarm'`, `vmin`, `vmax` |
| `sns.pairplot()` | Pairwise grid across all features | `hue`, `corner=True`, `diag_kind='kde'` |
| `sns.lmplot()` | Linear regression with facets | `col`, `row`, `order=2` (polynomial) |
