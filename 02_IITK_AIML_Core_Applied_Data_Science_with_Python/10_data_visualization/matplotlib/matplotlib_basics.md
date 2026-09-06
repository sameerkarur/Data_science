# Matplotlib - Data Visualization Library

## What is Matplotlib?
Matplotlib is the most widely used Python library for creating static, animated, and interactive visualizations. It gives you full control over every aspect of a figure — from line styles to axis labels to layout.

**Why use Matplotlib?**
- Create publication-quality plots
- Highly customizable (colors, fonts, sizes, layouts)
- Supports many plot types (line, bar, scatter, histogram, pie, etc.)
- Works well with NumPy and Pandas
- Foundation for Seaborn and other visualization libraries

## Installation
```bash
pip install matplotlib
```

## Importing
```python
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. Basic Plot Structure

```python
# Simple line plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.title('My First Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
```

### Figure and Axes (Object-Oriented Approach - Recommended)
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('My Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
plt.show()
```

### Figure Size
```python
fig, ax = plt.subplots(figsize=(10, 6))   # width=10, height=6 inches
```

---

## 2. Line Plot

```python
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(8, 5))

# Basic line
ax.plot(x, np.sin(x), label='sin(x)')
ax.plot(x, np.cos(x), label='cos(x)')

# Customization
ax.plot(x, np.sin(x),
        color='red',          # line color
        linewidth=2,          # line width
        linestyle='--',       # line style: '-', '--', '-.', ':'
        marker='o',           # marker style
        markersize=4,         # marker size
        alpha=0.7,            # transparency (0-1)
        label='sin(x)')

ax.set_title('Trigonometric Functions', fontsize=16)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend()                   # show legend
ax.grid(True, alpha=0.3)     # show grid
plt.tight_layout()
plt.show()
```

### Line Styles & Markers
```
Line styles: '-' (solid), '--' (dashed), '-.' (dash-dot), ':' (dotted)
Markers: 'o' (circle), 's' (square), '^' (triangle), 'D' (diamond),
         '*' (star), '+' (plus), 'x' (cross), '.' (point)
Colors: 'r' (red), 'g' (green), 'b' (blue), 'k' (black),
        'c' (cyan), 'm' (magenta), 'y' (yellow), 'w' (white)
        Or use hex: '#FF5733', or named: 'steelblue'
```

### Shorthand Format String
```python
plt.plot(x, y, 'ro--')   # red, circle markers, dashed line
plt.plot(x, y, 'b^-')    # blue, triangle markers, solid line
plt.plot(x, y, 'gs:')    # green, square markers, dotted line
```

---

## 3. Scatter Plot

```python
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = np.random.rand(50) * 500

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y,
                     c=colors,         # color by value
                     s=sizes,          # size by value
                     cmap='viridis',   # colormap
                     alpha=0.6,
                     edgecolors='black')

plt.colorbar(scatter, label='Color Scale')
ax.set_title('Scatter Plot')
plt.show()
```

### Common Colormaps
```
'viridis', 'plasma', 'inferno', 'magma', 'cividis'
'Blues', 'Reds', 'Greens', 'Purples', 'Oranges'
'coolwarm', 'RdYlBu', 'RdYlGn', 'Spectral'
'Set1', 'Set2', 'Set3', 'tab10', 'tab20'
```

---

## 4. Bar Chart

### Vertical Bar Chart
```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, values,
              color='steelblue',
              edgecolor='black',
              width=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}', ha='center', va='bottom')

ax.set_title('Bar Chart')
ax.set_ylabel('Values')
plt.show()
```

### Horizontal Bar Chart
```python
ax.barh(categories, values, color='coral')
```

### Grouped Bar Chart
```python
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, values1, width, label='Group 1')
ax.bar(x + width/2, values2, width, label='Group 2')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.show()
```

### Stacked Bar Chart
```python
ax.bar(categories, values1, label='Group 1')
ax.bar(categories, values2, bottom=values1, label='Group 2')
ax.legend()
```

---

## 5. Histogram

```python
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data,
        bins=30,              # number of bins
        color='steelblue',
        edgecolor='black',
        alpha=0.7,
        density=True)         # normalize to probability density

ax.set_title('Histogram')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
plt.show()
```

### Multiple Histograms
```python
ax.hist(data1, bins=30, alpha=0.5, label='Data 1')
ax.hist(data2, bins=30, alpha=0.5, label='Data 2')
ax.legend()
```

---

## 6. Pie Chart

```python
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [40, 25, 20, 15]
explode = (0.05, 0, 0, 0)   # explode first slice

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes,
       explode=explode,
       labels=labels,
       autopct='%1.1f%%',    # show percentage
       startangle=90,
       colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
       shadow=True)

ax.set_title('Programming Languages')
plt.show()
```

---

## 7. Box Plot

```python
data = [np.random.randn(100) for _ in range(4)]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data,
                labels=['A', 'B', 'C', 'D'],
                patch_artist=True,       # fill with color
                notch=True)              # notched box

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_title('Box Plot')
plt.show()
```

---

## 8. Heatmap (using imshow)

```python
data = np.random.rand(5, 5)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='YlOrRd')

# Add colorbar
plt.colorbar(im)

# Add text annotations
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center')

ax.set_title('Heatmap')
plt.show()
```

---

## 9. Subplots (Multiple Plots)

### Basic Grid
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, y)
axes[0, 0].set_title('Plot 1')

axes[0, 1].scatter(x, y)
axes[0, 1].set_title('Plot 2')

axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 30])
axes[1, 0].set_title('Plot 3')

axes[1, 1].hist(np.random.randn(100), bins=20)
axes[1, 1].set_title('Plot 4')

plt.tight_layout()    # prevent overlap
plt.show()
```

### Unequal Subplots
```python
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)    # 1 row, 2 cols, position 1
ax2 = fig.add_subplot(2, 2, 2)    # 2 rows, 2 cols, position 2
ax3 = fig.add_subplot(2, 2, 4)    # 2 rows, 2 cols, position 4
```

### Shared Axes
```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
```

---

## 10. Customization

### Titles & Labels
```python
ax.set_title('Title', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X Label', fontsize=12, labelpad=10)
ax.set_ylabel('Y Label', fontsize=12)
```

### Axis Limits & Ticks
```python
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['Zero', 'Two', 'Four', 'Six', 'Eight', 'Ten'], rotation=45)
ax.tick_params(axis='both', labelsize=10)
```

### Legend
```python
ax.legend(loc='upper right')       # location
ax.legend(loc='best')              # auto best location
ax.legend(fontsize=10, frameon=True, shadow=True, ncol=2)
# loc options: 'upper left', 'upper right', 'lower left', 'lower right',
#              'center', 'best'
```

### Grid
```python
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.grid(axis='y')    # only horizontal grid lines
```

### Annotations
```python
ax.annotate('Peak', xy=(3, 10), xytext=(4, 12),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

ax.text(5, 5, 'Some text', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
```

### Spines (Borders)
```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
```

---

## 11. Styles

```python
# List available styles
print(plt.style.available)

# Use a style
plt.style.use('seaborn-v0_8')
plt.style.use('ggplot')
plt.style.use('dark_background')
plt.style.use('fivethirtyeight')
plt.style.use('bmh')

# Temporary style
with plt.style.context('ggplot'):
    plt.plot(x, y)
    plt.show()
```

---

## 12. Saving Figures

```python
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
fig.savefig('plot.pdf', format='pdf')
fig.savefig('plot.svg', format='svg')
fig.savefig('plot.jpg', dpi=150, quality=95)

# Transparent background
fig.savefig('plot.png', dpi=300, transparent=True)
```

---

## 13. Plotting with Pandas

```python
import pandas as pd

df = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [100, 150, 130, 170, 200],
    'Profit': [20, 35, 25, 40, 50]
})

# Quick plots directly from DataFrame
df.plot(x='Month', y='Sales', kind='line')
df.plot(x='Month', y='Sales', kind='bar')
df.plot(x='Sales', y='Profit', kind='scatter')
df['Sales'].plot(kind='hist', bins=10)
df.plot(kind='box')

# kind options: 'line', 'bar', 'barh', 'scatter', 'hist', 'box', 'pie', 'area'
plt.show()
```

---

## Quick Reference Table

| Plot Type | Syntax |
|---|---|
| Line plot | `ax.plot(x, y)` |
| Scatter plot | `ax.scatter(x, y)` |
| Bar chart | `ax.bar(x, height)` |
| Horizontal bar | `ax.barh(y, width)` |
| Histogram | `ax.hist(data, bins=n)` |
| Pie chart | `ax.pie(sizes, labels=labels)` |
| Box plot | `ax.boxplot(data)` |
| Heatmap | `ax.imshow(data, cmap='...')` |
| Fill between | `ax.fill_between(x, y1, y2)` |
| Error bars | `ax.errorbar(x, y, yerr=err)` |
| Subplots | `fig, axes = plt.subplots(rows, cols)` |
| Title | `ax.set_title('...')` |
| Labels | `ax.set_xlabel('...')` / `ax.set_ylabel('...')` |
| Legend | `ax.legend()` |
| Grid | `ax.grid(True)` |
| Save | `fig.savefig('file.png', dpi=300)` |
| Style | `plt.style.use('style_name')` |
