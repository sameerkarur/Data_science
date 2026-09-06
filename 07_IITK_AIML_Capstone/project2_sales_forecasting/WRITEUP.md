# Sales Forecasting Capstone — STAR Write-up

**Project:** Course 7 Capstone Project 2 — Sales Forecasting  
**Client framing:** Fresh Analytics | Multi-restaurant demand prediction  
**Stack:** Python, pandas, seaborn/matplotlib, scikit-learn, XGBoost  
**Period analyzed:** 2019-01-01 → 2021-12-31

---

## Situation

Fresh Analytics supports restaurant operators who must plan production, staffing, and inventory under volatile day-of-week and seasonal demand. Decisions that depend on demand — prep quantities, labor hours, purchasing — degrade quickly when forecasts are off. The business needed a clear picture of *how* sales behave across six restaurants and a reliable way to project **next-year daily demand** from historical item-level sales.

The available data formed a three-table panel:
- **sales.csv** — daily item sales (`date`, `item_id`, `price`, `item_count`)
- **items.csv** — item catalog (`id`, `store_id`, `name`, `kcal`, `cost`)
- **resturants.csv** — store directory (`id`, `name`; filename typo retained from source)

Roughly **109.6k** item–day rows spanning **100 items** and **6 restaurants**, with many structural zero-count days (full item×day panel rather than missing data).

---

## Task

Deliver an end-to-end analytics and forecasting package that:

1. **Merges** sales, items, and restaurants into one analysis-ready dataset (date, item id/name, price, item count, kcal, store id/name).
2. Completes **EDA** covering overall patterns, weekday/month/quarter seasonality, restaurant performance, item popularity, volume-vs-revenue leadership, and most expensive items by store (with calories).
3. Builds and compares **Linear Regression, Random Forest, and XGBoost** using calendar features (day of week, quarter, month, year, day of month, and related flags), with the **last six months** held out for testing.
4. Selects the best model by **RMSE** and produces a **365-day forward forecast**.
5. Packages a polished notebook, publication-grade charts, and executive analytical write-up.

---

## Action

**Preliminary analysis.** Profiled shapes, dtypes, missingness, and duplicates; screened outliers on positive `item_count` with Tukey fences (right-skewed high-volume days retained as demand signal). Merged the three sources and derived `revenue = price × item_count`.

**Exploratory analysis.** Aggregated to daily system volume/revenue; broke demand down by weekday, month, and quarter; compared restaurants by year/month/day; ranked popular items overall and per store; tested whether the highest-volume store also earns the most money per day; identified the priciest item and kcal at each restaurant.

**Forecasting.** Built a daily series of total `item_count`. Engineered calendar features (`year`, `month`, `day`, `dayofweek`, `quarter`, `dayofyear`, `weekofyear`, `is_weekend`, month-start/end). Split train vs last-6-month test. Fit Linear Regression (scaled), Random Forest, and XGBoost; compared RMSE / MAE / R². Retrained the winner on all history and forecasted the next 365 days. Saved plots and metrics under `outputs/`.

---

## Result

### Model performance (holdout = last 6 months)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **XGBoost (best)** | **57.90** | **45.80** | **0.948** |
| Random Forest | 59.96 | 47.30 | 0.944 |
| Linear Regression | 230.76 | 185.51 | 0.175 |

**Best model: XGBoost** — ~4× lower RMSE than linear regression and a slight edge over Random Forest, with strong holdout R² (~0.95).

**Next-year outlook (XGBoost):** mean ~**679 units/day**, ~**247.7k units** over 365 days (2022-01-01 → 2022-12-31).

### Key business insights

- **Seasonality is real.** Weekday and monthly cycles are strong and stable enough to power calendar-feature models.
- **Concentration risk.** **Bob’s Diner** dominates both volume and average daily revenue; other stores contribute a thin long tail — forecasts and ops plans should weight that store heavily.
- **SKU skew.** Items like *Strawberry Smoothy* at Bob’s Diner account for a disproportionate share of units — inventory focus belongs on a short list of heroes.
- **Volume ≠ automatic revenue elsewhere.** Across the chain, Bob’s leads both metrics, but average revenue per unit differs sharply by store mix (e.g., Fou Cher is low volume but higher ticket).

### Artifacts

- Executed notebook: `sales_forecasting.ipynb`
- Charts & metrics: `outputs/`
- Deliverables: Executable notebook, 365-day forecasting models, evaluation plots, and executive writeup.

---

*Prepared for IITK AIML Capstone (Course 7) · Sales Forecasting*
