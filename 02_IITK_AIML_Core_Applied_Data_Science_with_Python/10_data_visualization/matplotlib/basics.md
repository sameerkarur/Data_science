# Matplotlib: Complete Step-by-Step Tutorial & Data Visualization Guide
**Official Tutorial & Visual Architecture Handbook (W3Schools & GeeksforGeeks Style)**

---

## 📑 Table of Contents (On this page)
1. [What is Matplotlib & Why Use It?](#1-what-is-matplotlib--why-use-it)
2. [Installation & Importing (Pyplot)](#2-installation--importing-pyplot)
3. [The Matplotlib Figure & Artist Hierarchy (Visual Architecture)](#3-the-matplotlib-figure--artist-hierarchy-visual-architecture)
4. [Line Plots (Markers, Line Styles & Colors)](#4-line-plots-markers-line-styles--colors)
5. [Labels, Titles, Grids & Legends](#5-labels-titles-grids--legends)
6. [Subplots & Multi-Panel Figures (`plt.subplots`)](#6-subplots--multi-panel-figures-pltsubplots)
7. [Scatter Plots & Color Mapping (Colormaps)](#7-scatter-plots--color-mapping-colormaps)
8. [Bar Charts (Vertical, Horizontal & Grouped)](#8-bar-charts-vertical-horizontal--grouped)
9. [Histograms & Density Distributions](#9-histograms--density-distributions)
10. [Pie Charts & Donut Charts](#10-pie-charts--donut-charts)
11. [3D Surface Plots (Viridis Colormap as in NumPy Docs)](#11-3d-surface-plots-viridis-colormap)
12. [Saving High-Resolution Figures (DPI, PDF, PNG)](#12-saving-high-resolution-figures)
13. [Try It Yourself! (Hands-On Practice Exercises)](#13-try-it-yourself-hands-on-practice-exercises)
14. [Quick Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What is Matplotlib & Why Use It?

**Matplotlib** is the comprehensive library for creating static, animated, and interactive visualizations in Python. Created by John D. Hunter in 2003, it is the underlying rendering engine powering Seaborn, Pandas plotting, and scientific Python visualization.

### Why use Matplotlib?
- **Granular Control:** You can control every single pixel, spine, tick mark, label, and legend.
- **Publication-Quality Graphics:** Exports crisp vector graphics in PDF, SVG, EPS, and high-DPI PNG formats.
- **Ecosystem Integration:** Works seamlessly with NumPy arrays and Pandas DataFrames.
- **3D & Specialized Projections:** Built-in support for 3D surfaces, polar charts, and geographic projections.

---

## 2. Installation & Importing (Pyplot)

Install Matplotlib via `pip`:
```bash
pip install matplotlib
```

Import the pyplot module as `plt`:
```python
import matplotlib.pyplot as plt
import numpy as np

print(f"Matplotlib Version: {plt.matplotlib.__version__}")
```

#### Output:
```text
Matplotlib Version: 3.9.0
```

---

## 3. The Matplotlib Figure & Artist Hierarchy (Visual Architecture)

Understanding Matplotlib requires understanding its two-tier object hierarchy:
1. **Figure (`fig`):** The entire top-level canvas/window holding everything.
2. **Axes (`ax`):** The actual plot/coordinate space with x-axis, y-axis, title, and plotted data. (A Figure can contain multiple Axes).

### Visual Hierarchy Diagram:

```
┌───────────────────────────────────────────────────────────────┐
│ FIGURE (Canvas)                                               │
│                                                               │
│   Title: "Model Performance Comparison"                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ AXES (Plot Area)                                        │  │
│  │  Y-Axis Label                                           │  │
│  │    ▲                                                    │  │
│  │    │     Legend: [── Epoch Loss   - - Val Loss]        │  │
│  │    │                                                    │  │
│  │    │        \                                           │  │
│  │    │         \                                          │  │
│  │    │          \_________                                │  │
│  │    │                    \---------                      │  │
│  │    └─────────────────────────────────────►              │  │
│  │           Spines (Borders) & Ticks        X-Axis Label  │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Line Plots (Markers, Line Styles & Colors)

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample training progression data
epochs = np.arange(1, 6)
train_acc = [0.65, 0.78, 0.85, 0.91, 0.96]
val_acc =   [0.62, 0.74, 0.80, 0.86, 0.89]

fig, ax = plt.subplots(figsize=(7, 4))

# Plot lines with custom markers and styles
ax.plot(epochs, train_acc, marker='o', color='#2563eb', linewidth=2.5, label='Train Accuracy')
ax.plot(epochs, val_acc, marker='s', linestyle='--', color='#10b981', linewidth=2.0, label='Val Accuracy')

ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch Number", fontsize=11)
ax.set_ylabel("Accuracy Score", fontsize=11)
ax.set_ylim([0.5, 1.0])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.show()
print("Plot successfully rendered.")
```

#### Output:
```text
Plot successfully rendered.
[Interactive Canvas Displayed: 2 Lines with circle & square markers, gridlines, and legend]
```

---

## 5. Labels, Titles, Grids & Legends

You can customize typography and annotations with clean styling:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(x, y1, color='#ef4444', label='Sin(x)')
ax.plot(x, y2, color='#3b82f6', linestyle='-.', label='Cos(x)')

# Highlight maximum point
max_idx = np.argmax(y1)
ax.annotate('Global Peak (x=π/2, y=1.0)',
            xy=(x[max_idx], y1[max_idx]),
            xytext=(x[max_idx]+1.2, y1[max_idx]-0.2),
            arrowprops=dict(facecolor='#10b981', shrink=0.05, width=1.5, headwidth=8),
            fontweight='bold', color='#10b981')

ax.set_title("Trigonometric Waveforms & Mathematical Annotations", fontsize=13)
ax.set_xlabel("Radian Angle θ")
ax.set_ylabel("Amplitude")
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.legend()
plt.show()
```

#### Output:
```text
[Rendered Graphic: Dual sinusoidal curves with green callout arrow pointing to peak]
```

---

## 6. Subplots & Multi-Panel Figures (`plt.subplots`)

Multi-panel plots allow comparing related experiments side-by-side:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 50)

# Create 1 row, 2 columns of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left Subplot: Linear vs Quadratic
ax1.plot(x, x, label='Linear (O(N))', color='#38bdf8')
ax1.plot(x, x**2, label='Quadratic (O(N²))', color='#f59e0b')
ax1.set_title("Computational Growth")
ax1.legend()
ax1.grid(True)

# Right Subplot: Exponential
ax2.plot(x, np.exp(x), label='Exponential (O(2ᴺ))', color='#ef4444')
ax2.set_title("Exponential Growth Explosion")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

#### Output:
```text
[Rendered Graphic: 2 distinct side-by-side panels showing polynomial and exponential curves]
```

---

## 7. Scatter Plots & Color Mapping (Colormaps)

Scatter plots visualize relationships between continuous variables, using color and size to represent additional dimensions:

```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Generate synthetic housing dataset
square_feet = np.random.randint(800, 3500, 60)
prices = square_feet * 250 + np.random.normal(0, 50000, 60)
bedrooms = np.random.randint(1, 5, 60)

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter with color mapping by bedrooms and point size
scatter = ax.scatter(square_feet, prices / 1000,
                     c=bedrooms, cmap='viridis',
                     s=bedrooms * 40, alpha=0.8, edgecolors='black')

ax.set_title("Housing Market: Square Footage vs Price", fontsize=13, fontweight='bold')
ax.set_xlabel("Property Size (sq ft)")
ax.set_ylabel("Price ($ in Thousands)")

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Number of Bedrooms", rotation=270, labelpad=15)

plt.show()
```

#### Output:
```text
[Rendered Graphic: Multi-colored scatter points with Viridis colorbar showing bedroom scale]
```

---

## 8. Bar Charts (Vertical, Horizontal & Grouped)

```python
import matplotlib.pyplot as plt
import numpy as np

frameworks = ['PyTorch', 'TensorFlow', 'Scikit-Learn', 'XGBoost', 'Keras']
popularity_scores = [88, 72, 65, 80, 58]

fig, ax = plt.subplots(figsize=(7, 4))

colors = ['#f97316', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899']
bars = ax.bar(frameworks, popularity_scores, color=colors, edgecolor='#1e293b', width=0.6)

# Add numerical values above bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
            f'{height}%', ha='center', va='bottom', fontweight='bold')

ax.set_title("AIML Framework Industry Adoption Rate", fontsize=13, fontweight='bold')
ax.set_ylabel("Adoption Percentage (%)")
ax.set_ylim([0, 100])
plt.show()
```

#### Output:
```text
[Rendered Graphic: 5 colorful vertical bars with percentage badges placed directly above each bar]
```

---

## 9. Histograms & Density Distributions

```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Draw 10,000 samples from normal distribution
latency_ms = np.random.normal(loc=120, scale=25, size=10000)

fig, ax = plt.subplots(figsize=(8, 4))

n, bins, patches = ax.hist(latency_ms, bins=40, color='#0284c7', edgecolor='#0369a1', alpha=0.85)

# Highlight SLA threshold (180ms)
ax.axvline(180, color='#ef4444', linestyle='--', linewidth=2, label='P99 SLA Threshold (180ms)')

ax.set_title("API Server Latency Distribution (10,000 Requests)", fontsize=13)
ax.set_xlabel("Response Latency (ms)")
ax.set_ylabel("Frequency Count")
ax.legend()
plt.show()
```

#### Output:
```text
[Rendered Graphic: Bell-shaped histogram with red vertical dashed SLA threshold line]
```

---

## 10. Pie Charts & Donut Charts

```python
import matplotlib.pyplot as plt

categories = ['Direct Search', 'Referral', 'Social Media', 'Organic Video', 'Email']
shares = [40, 25, 15, 12, 8]
explode = (0.08, 0, 0, 0, 0)  # Explode the 1st slice

fig, ax = plt.subplots(figsize=(6, 6))

colors = ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185']
ax.pie(shares, labels=categories, explode=explode, autopct='%1.1f%%',
       startangle=140, colors=colors, wedgeprops=dict(width=0.6, edgecolor='white'))

ax.set_title("Acquisition Traffic Sources (Donut Chart)", fontsize=13, fontweight='bold')
plt.show()
```

#### Output:
```text
[Rendered Graphic: Elegant donut chart with exploded primary slice and percentage callouts]
```

---

## 11. 3D Surface Plots (Viridis Colormap)

*(As shown in the official NumPy and Matplotlib documentation from your screenshot)*

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(projection='3d')

# Create meshgrid
X = np.arange(-5, 5, 0.15)
Y = np.arange(-5, 5, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the 3D surface with viridis colormap
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax.set_title("3D Mathematical Surface: Z = sin(√(X² + Y²))", fontsize=14, fontweight='bold')
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_zlabel("Z amplitude")

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
```

#### Output:
```text
[Rendered Graphic: 3D wireframe surface with concentric ripples styled with Viridis yellow-to-purple colormap]
```

---

## 12. Saving High-Resolution Figures

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot([1, 2, 3], [10, 20, 15], color='#3b82f6', marker='o')
ax.set_title("Export Example")

# Save in high-DPI raster format and vector PDF
fig.savefig("sample_plot.png", dpi=300, bbox_inches='tight')
fig.savefig("sample_plot.pdf", bbox_inches='tight')
print("Successfully exported sample_plot.png (300 DPI) and sample_plot.pdf")
plt.close(fig)
```

#### Output:
```text
Successfully exported sample_plot.png (300 DPI) and sample_plot.pdf
```

---

## 13. Try It Yourself! (Hands-On Practice Exercises)

### Exercise 1: Cumulative Sales Chart with Dual Axis
**Task:** Create a dual-axis chart where the primary Y-axis shows monthly sales as a bar chart, and the secondary Y-axis shows cumulative revenue as a line plot.

<details>
<summary>👉 Click to Reveal Solution</summary>

```python
import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [12000, 15000, 18000, 14000, 22000, 25000]
cumulative = np.cumsum(sales)

fig, ax1 = plt.subplots(figsize=(8, 4.5))

# Primary Axis: Bars
ax1.bar(months, sales, color='#38bdf8', alpha=0.7, label='Monthly Sales')
ax1.set_ylabel("Monthly Sales ($)", color='#0284c7', fontweight='bold')
ax1.set_ylim([0, 30000])

# Secondary Axis: Line
ax2 = ax1.twinx()
ax2.plot(months, cumulative, color='#ef4444', marker='o', linewidth=2.5, label='Cumulative Revenue')
ax2.set_ylabel("Cumulative Revenue ($)", color='#dc2626', fontweight='bold')

plt.title("Dual-Axis Financial Performance", fontsize=13, fontweight='bold')
plt.show()
```
#### Output:
```text
[Rendered Graphic: Combined bar chart with red overlay line using twinx dual Y-axis]
```
</details>

---

## 14. Quick Reference Cheat Sheet

| Plot Type | Matplotlib Syntax | Key Parameters |
|---|---|---|
| **Line Plot** | `ax.plot(x, y)` | `color`, `linestyle='--'`, `marker='o'`, `linewidth` |
| **Scatter** | `ax.scatter(x, y)` | `c=colors`, `s=sizes`, `cmap='viridis'`, `alpha` |
| **Bar Chart** | `ax.bar(x, heights)` | `color`, `width`, `edgecolor`, `align` |
| **Histogram** | `ax.hist(data, bins=30)` | `density=True`, `cumulative=True`, `edgecolor` |
| **Pie Chart** | `ax.pie(values, labels=lbls)` | `autopct='%1.1f%%'`, `explode`, `startangle` |
| **Subplots** | `fig, (ax1, ax2) = plt.subplots(1, 2)` | `figsize=(w, h)`, `sharex=True`, `sharey=True` |
| **Save Figure**| `fig.savefig('name.png', dpi=300)` | `bbox_inches='tight'`, `transparent=False` |
