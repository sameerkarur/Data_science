# Data Wrangling, Imputation & Outlier Engineering
**Comprehensive Architectural Guide & Execution Foundations**

---

## 📌 Executive Architecture & Visual Flowchart

```
                 DATA CLEANING & IMPUTATION LIFECYCLE
       Raw Data ──► Detect Missingness Patterns:
                      ├── MCAR (Missing Completely at Random)
                      ├── MAR  (Missing at Random)
                      └── MNAR (Missing Not at Random)
                           │
       Screen Outliers via IQR Tukey Fences / Z-score
                           │
       Apply Domain Scalers (StandardScaler / RobustScaler)
```

---

## 🧭 Deep Theoretical Foundations

### 1. Tukey's Fences for Outlier Detection
$$	ext{IQR} = Q_3 - Q_1$$
$$	ext{Lower Bound} = Q_1 - 1.5 	imes 	ext{IQR}, \quad 	ext{Upper Bound} = Q_3 + 1.5 	imes 	ext{IQR}$$
Points outside these bounds are flagged as distributional anomalies.
