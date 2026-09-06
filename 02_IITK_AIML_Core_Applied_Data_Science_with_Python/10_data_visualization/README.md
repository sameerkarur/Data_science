# Interview Q&A — Data Visualization

| # | Question | Answer |
|---|----------|--------|
| 1 | Matplotlib vs Seaborn? | **Matplotlib** = low-level, full control. **Seaborn** = statistical plots on top of Matplotlib, faster EDA. |
| 2 | When use `plt.subplot` vs `plt.subplots`? | **subplot** = one at a time; **subplots** = grid of axes returned as array. |
| 3 | Seaborn `displot` vs `histplot`? | **displot** = figure-level (FacetGrid); **histplot** = axes-level histogram/KDE. |
| 4 | What is a heatmap used for? | Visualize **correlation matrix** or 2D aggregated values. |
| 5 | Plotly/Bokeh vs Matplotlib? | **Interactive** (zoom, pan, hover) vs mostly **static** plots. |

## Matplotlib & Seaborn practice
- [Matplotlib — 45 questions](matplotlib/matplotlib_practice.ipynb)
- [Seaborn — 60 questions](seaborn/seaborn_practice.ipynb)
