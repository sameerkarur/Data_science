"""
Projects Version 2: Master Test & Execution Orchestrator
Runs all 15 next-generation project implementations sequentially and outputs a unified health report.
Author: Sameer Karur
"""

import sys
import subprocess
import time
from pathlib import Path

PROJECTS = [
    ("01_Python_Expense_Tracker_V2", "01_Python_Expense_Tracker_V2/expense_tracker_v2.py"),
    ("02_Python_Task_Manager_V2", "02_Python_Task_Manager_V2/task_scheduler_dag_v2.py"),
    ("03_DS_Sales_Analysis_Cohort_RFM_V2", "03_DS_Sales_Analysis_Cohort_RFM_V2/sales_cohort_rfm_v2.py"),
    ("04_DS_Marketing_Campaign_Uplift_Attribution_V2", "04_DS_Marketing_Campaign_Uplift_Attribution_V2/marketing_uplift_attribution_v2.py"),
    ("05_ML_Spotify_Cohorts_HDBSCAN_PCA_V2", "05_ML_Spotify_Cohorts_HDBSCAN_PCA_V2/spotify_hdbscan_pca_v2.py"),
    ("06_ML_Employee_Attrition_Explainable_Boosting_V2", "06_ML_Employee_Attrition_Explainable_Boosting_V2/employee_attrition_xgboost_shap_v2.py"),
    ("07_DL_Lending_Club_Tabular_ResNet_V2", "07_DL_Lending_Club_Tabular_ResNet_V2/lending_club_tabular_resnet_v2.py"),
    ("08_DL_Home_Loan_Risk_TabNet_V2", "08_DL_Home_Loan_Risk_TabNet_V2/home_loan_tabnet_v2.py"),
    ("09_GenAI_Storytelling_MultiAgent_StateGraph_V2", "09_GenAI_Storytelling_MultiAgent_StateGraph_V2/storytelling_stategraph_v2.py"),
    ("10_GenAI_Virtual_PMO_Swarm_Simulation_V2", "10_GenAI_Virtual_PMO_Swarm_Simulation_V2/virtual_pmo_swarm_v2.py"),
    ("11_AdvGenAI_Hybrid_Dense_Sparse_RAG_V2", "11_AdvGenAI_Hybrid_Dense_Sparse_RAG_V2/hybrid_dense_sparse_rag_v2.py"),
    ("12_AdvGenAI_Multimodal_Creative_Studio_V2", "12_AdvGenAI_Multimodal_Creative_Studio_V2/multimodal_creative_studio_v2.py"),
    ("13_Capstone1_Autonomous_Perception_ViT_EfficientNet_V2", "13_Capstone1_Autonomous_Perception_ViT_EfficientNet_V2/autonomous_perception_v2.py"),
    ("14_Capstone2_Hierarchical_Probabilistic_Demand_Forecasting_V2", "14_Capstone2_Hierarchical_Probabilistic_Demand_Forecasting_V2/sales_demand_forecasting_v2.py"),
    ("15_Capstone3_Dual_Vision_BPR_Recommender_V2", "15_Capstone3_Dual_Vision_BPR_Recommender_V2/heritage_vision_bpr_recommender_v2.py"),
]

def main():
    root = Path(__file__).resolve().parent
    py_bin = sys.executable

    print("=" * 80)
    print("🌟 Master Execution Suite: Projects Version 2 (15 Alternative Architectures)")
    print(f"🐍 Python Interpreter: {py_bin}")
    print("=" * 80)

    results = []

    for idx, (p_name, script_rel) in enumerate(PROJECTS, 1):
        script_path = root / script_rel
        print(f"\n[{idx:02d}/15] Executing: {p_name} ...")
        start = time.time()
        try:
            res = subprocess.run([py_bin, str(script_path)], capture_output=True, text=True, check=True)
            elapsed = time.time() - start
            print(f"  ✅ SUCCESS ({elapsed:.2f}s)")
            results.append((idx, p_name, "PASSED", elapsed, ""))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"  ❌ FAILED ({elapsed:.2f}s)")
            results.append((idx, p_name, "FAILED", elapsed, e.stderr[:200]))

    print("\n" + "=" * 80)
    print("📋 SUMMARY REPORT: PROJECTS VERSION 2 EXECUTION")
    print("=" * 80)
    for idx, name, status, el, err in results:
        status_icon = "🟢" if status == "PASSED" else "🔴"
        print(f"{status_icon} [{idx:02d}] {name:55} : {status} ({el:.2f}s)")
        if err:
            print(f"    ↳ Error: {err}")

    passed_count = sum(1 for r in results if r[2] == "PASSED")
    print("\n" + "-" * 80)
    print(f"✨ Final Result: {passed_count}/{len(PROJECTS)} Projects Passed Execution.")
    print("=" * 80)

if __name__ == "__main__":
    main()
