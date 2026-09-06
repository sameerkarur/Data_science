# Project 04 (V2): Uplift Modeling & Multi-Touch Attribution
**Next-Generation Marketing Analytics Architecture**  
*Curriculum: Core — Applied Data Science with Python*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Dimension | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Analytical Philosophy** | Observational correlation (Who converted?) | Causal inference (Who converted *because* of marketing?) |
| **Uplift Modeling** | None | Two-Model incremental response analysis (Treatment vs Control) |
| **Attribution Model** | Single channel / aggregate counts | First-Touch, Last-Touch & Markov Chain State Transition matrices |
| **Removal Effect** | Not evaluated | Quantifies marginal drop in conversions when removing a channel |
| **Budget Allocation** | Retrospective reporting | Algorithmic media mix rebalancing based on incremental uplift |

---

## 🚀 How to Run

```bash
python projects_version2/04_DS_Marketing_Campaign_Uplift_Attribution_V2/marketing_uplift_attribution_v2.py
```
