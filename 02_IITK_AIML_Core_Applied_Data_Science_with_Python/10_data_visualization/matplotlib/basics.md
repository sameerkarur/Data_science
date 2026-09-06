# Matplotlib Visual Architecture, Figure Tree & Low-Level Primitives
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 MATPLOTLIB COMPOSITION & ARTIST HIERARCHY
    Figure (Top-level canvas)
      └── Axes (Data projection area)
           ├── XAxis / YAxis (Tick locators, formatters, labels)
           ├── Collections (PathCollection for scatter plots)
           ├── Lines (Line2D objects for continuous series)
           └── Patches (Rectangle, Polygon for histograms & bars)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Stateful vs Object-Oriented Interface
- **Stateful (`pyplot`):** `plt.plot()` relies on a global current figure and axes state machine. Prone to side effects and silent cross-talk across function boundaries.
- **Object-Oriented (OO):** `fig, ax = plt.subplots()` explicitly creates and binds the `Figure` and `Axes` instances. Mandatory for production pipelines, multi-panel layouts, and custom twin axes.

### 2. Layout Optimization
Always apply `fig.tight_layout()` or `constrained_layout=True` to prevent title and tick label clipping when exporting high-DPI vector PDF/SVG figures for reports.
