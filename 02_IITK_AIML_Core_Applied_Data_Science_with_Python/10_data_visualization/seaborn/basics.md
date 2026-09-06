# Chapter 10: Data Visualization & Visual Architecture
**Comprehensive Textbook Guide — Advanced Applied Data Science**

---

## 1. Executive Overview & Mental Models

Data visualization translates high-dimensional relationships into visual encodings (position, length, angle, hue, saturation). Matplotlib provides the foundational rendering canvas and Artist object hierarchy, while Seaborn provides statistical aggregation abstractions.

```
                    MATPLOTLIB ARTIST OBJECT TREE
    ┌────────────────────────────────────────────────────────┐
    │ Figure (The Canvas Container)                          │
    │  └── Axes (The Actual Plotting Area / Coordinate Space)│
    │       ├── XAxis & YAxis (Ticks, TickLabels, Scale)     │
    │       ├── Line2D / BarContainer (The Data Plots)       │
    │       ├── Legend & Title Text Objects                  │
    └────────────────────────────────────────────────────────┘
```

---

## 2. Deep Theoretical Foundations

### 1. The Grammar of Graphics (Wilkinson / Wickham)
Data graphics decompose into independent orthogonal layers:
1. **Data:** Raw tabular dataset.
2. **Aesthetic Mapping (`aes`):** Mapping variables to visual channels (X, Y, Color, Size, Shape).
3. **Geometric Objects (`geom`):** The physical marks (points, lines, bars, ribbons).
4. **Statistical Transformations (`stat`):** Binning, smoothing, quantile estimation.
5. **Coordinate Systems (`coord`):** Cartesian, logarithmic, polar.
6. **Faceting (`facet`):** Conditioning subplots across discrete categories.

### 2. Kernel Density Estimation (KDE) Mechanics
Seaborn density plots approximate continuous probability distributions non-parametrically using kernel smoothing:
$$\hat{f}_h(x) = \frac{1}{n h} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$$
Where $K(u)$ is typically the standard Gaussian kernel $\frac{1}{\sqrt{2\pi}}e^{-u^2 / 2}$ and $h$ is the bandwidth smoothing factor determined via Silverman's rule of thumb.

---

## 3. Production Implementation: Publication-Grade Multi-Panel Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_publication_diagnostics(actual: np.ndarray, predicted: np.ndarray, residuals: np.ndarray) -> plt.Figure:
    """Generates publication-grade model diagnostic dashboard with dark aesthetic."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Panel 1: Prediction vs Ground Truth
    axes[0].scatter(actual, predicted, alpha=0.4, color='#38bdf8', edgecolors='none', s=20)
    ideal_line = [min(actual), max(actual)]
    axes[0].plot(ideal_line, ideal_line, color='#f43f5e', linestyle='--', linewidth=1.5, label='Identity (y=x)')
    axes[0].set_title("Parity Plot: Actual vs Predicted", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Ground Truth Target", fontsize=10)
    axes[0].set_ylabel("Model Prediction", fontsize=10)
    axes[0].legend(frameon=True, facecolor='#1e293b', edgecolor='#334155')
    axes[0].grid(True, linestyle=':', alpha=0.3)
    
    # Panel 2: Residual Distribution with KDE
    sns.histplot(residuals, kde=True, ax=axes[1], color='#10b981', stat='density', bins=30)
    axes[1].axvline(0, color='#f43f5e', linestyle='--', linewidth=1.5, label='Zero Residual')
    axes[1].set_title("Residual Error Distribution (KDE)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Residual Error (Actual - Pred)", fontsize=10)
    axes[1].legend(frameon=True, facecolor='#1e293b', edgecolor='#334155')
    axes[1].grid(True, linestyle=':', alpha=0.3)
    
    fig.tight_layout()
    return fig
```
