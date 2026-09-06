"""
Orchestrates writing comprehensive, production-grade basics.md files across all 33 curriculum topics.
"""

from pathlib import Path
from basics_c01 import C01_BASICS
from basics_c02 import C02_BASICS
from basics_c03 import C03_BASICS
from basics_c04 import C04_BASICS
from basics_c05 import C05_BASICS
from basics_c06 import C06_BASICS

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_BASICS = {}
ALL_BASICS.update(C01_BASICS)
ALL_BASICS.update(C02_BASICS)
ALL_BASICS.update(C03_BASICS)
ALL_BASICS.update(C04_BASICS)
ALL_BASICS.update(C05_BASICS)
ALL_BASICS.update(C06_BASICS)

# Also copy visualization basics to matplotlib and seaborn subfolders if present
mpl_folder = "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib"
sns_folder = "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn"

ALL_BASICS[mpl_folder] = """# Matplotlib Visual Architecture, Figure Tree & Low-Level Primitives
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
"""

ALL_BASICS[sns_folder] = """# Seaborn Statistical Graphics & High-Level Mapping
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
"""

written_count = 0
for rel_path, markdown_content in ALL_BASICS.items():
    target_dir = REPO_ROOT / rel_path
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_dir / "basics.md"
    target_file.write_text(markdown_content.strip() + "\n", encoding="utf-8")
    written_count += 1

print(f"🎉 Successfully wrote {written_count} comprehensive basics.md guides across the repository!")
