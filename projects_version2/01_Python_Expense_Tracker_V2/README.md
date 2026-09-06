# Project 01 (V2): Event-Sourced Personal Expense Engine
**Next-Generation Python Architecture**  
*Curriculum: Foundations — Programming Refresher*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Feature | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Storage Engine** | Flat CSV file parsing (`expenses.csv`) | ACID-compliant relational SQLite database |
| **Data Integrity** | String manipulation, prone to race conditions | Parameterized SQL queries, constraints & atomic commits |
| **Budget Monitoring** | Static total threshold comparison | Real-time hierarchical category budgets with warning/critical tiers |
| **Statistical Intelligence**| None | Dynamic Z-score outlier detection identifying anomalous spending spikes |
| **Extensibility** | Monolithic procedural script | Clean Object-Oriented Dataclass & Service pattern |

---

## 🚀 How to Run

```bash
python projects_version2/01_Python_Expense_Tracker_V2/expense_tracker_v2.py
```
