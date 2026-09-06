# Project 02 (V2): DAG Task Scheduler & RBAC Engine
**Next-Generation Python Architecture**  
*Curriculum: Foundations — Programming Refresher*

---

## 🏗️ Architectural Differences (V1 vs V2)

| Feature | Version 1 Baseline | Version 2 Advanced Implementation |
|---|---|---|
| **Task Flow Structure** | Flat list of unrelated tasks | Directed Acyclic Graph (DAG) with prerequisites and dependency tracking |
| **Deadlock Prevention** | None | Depth-First Search (DFS) Cycle Detection rejecting circular dependencies |
| **Execution Ordering** | Chronological creation order | Topological Sorting combined with Priority-Queue scheduling |
| **Authentication & Security** | Plain SHA-256 without salt | PBKDF2-HMAC-SHA256 with 100,000 iterations & 16-byte random salts |
| **Access Control** | Single-tier access | Granular Role-Based Access Control (Admin, Contributor, Viewer) |

---

## 🚀 How to Run

```bash
python projects_version2/02_Python_Task_Manager_V2/task_scheduler_dag_v2.py
```
