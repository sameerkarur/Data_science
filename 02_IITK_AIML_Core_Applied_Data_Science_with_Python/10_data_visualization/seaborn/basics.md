# Seaborn Statistical Graphics & High-Level Mapping
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 SEABORN HIGH-LEVEL STATISTICAL MAPPING
    Tidy DataFrame (Long Form)
              │
    sns.FacetGrid / PairGrid / CatPlot
              │
    Computes Statistical Estimations (Bootstrap Confidence Intervals, KDEs, Medians)
              │
    Draws into Underlying Matplotlib Axes Subplots!
```

---

## 🧭 Deep Theoretical Foundations

### 1. Figure-Level vs Axes-Level Functions
- **Axes-Level (`sns.scatterplot`, `sns.histplot`, `sns.boxplot`):** Draw directly into a specified Matplotlib `ax` and return the `Axes` object. Ideal for composing custom multi-plot dashboards.
- **Figure-Level (`sns.relplot`, `sns.displot`, `sns.catplot`):** Create and manage an entire `FacetGrid` figure, controlling dimensions via `height` and `aspect`.

### 2. Kernel Density Estimation (KDE)
Estimates the continuous probability density function of a random variable non-parametrically using Gaussian kernels with Scott's or Silverman's rule of thumb for bandwidth selection.
