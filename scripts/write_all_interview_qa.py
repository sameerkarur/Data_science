"""
Orchestrator to write all authentic, high-yield interview questions & model answers
into every subtopic directory across Courses 1 to 5.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa_data_c01 import C1_01_VARS, C1_02_CONTROL, C1_03_STRUCTURES, C1_04_OOP, C1_05_FILES
from qa_data_c02 import C2_01_INTRO, C2_02_PYTHON_DS
from qa_data_c02_part2 import C2_03_NUMPY, C2_04_LINALG
from qa_data_c02_part3 import C2_05_STATS, C2_06_PROB
from qa_data_c02_part4 import C2_07_ADVSTATS, C2_08_PANDAS
from qa_data_c02_part5 import C2_09_WRANGLING, C2_10_VIZ, C2_11_REGEX_APIS
from qa_data_c03 import C3_01_FE, C3_02_CLUSTERING, C3_03_CLASSIFICATION, C3_04_IMBALANCE, C3_05_EVAL
from qa_data_c04 import C4_01_NN_BASICS, C4_02_KERAS_TF, C4_03_DL_PREPROC_IMBALANCE, C4_04_DL_MODEL_EVAL
from qa_data_c05 import C5_01_PROMPT, C5_02_CHATGPT, C5_03_OPTIMIZATION

TOPIC_MAPPING = {
    # Course 1
    "01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes": ("Variables, Data Types & Operators", C1_01_VARS),
    "01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions": ("Control Flow & Functions", C1_02_CONTROL),
    "01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures": ("Data Structures (Lists, Dicts, Tuples, Sets)", C1_03_STRUCTURES),
    "01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules": ("Object-Oriented Programming & Modules", C1_04_OOP),
    "01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions": ("File I/O, Exceptions & Context Managers", C1_05_FILES),
    
    # Course 2
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science": ("Introduction to Data Science & CRISP-DM", C2_01_INTRO),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials": ("Python Essentials for Data Science", C2_02_PYTHON_DS),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy": ("NumPy & High-Performance Array Computing", C2_03_NUMPY),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra": ("Linear Algebra for Machine Learning", C2_04_LINALG),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals": ("Statistics Fundamentals & EDA", C2_05_STATS),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions": ("Probability Distributions & Bayes", C2_06_PROB),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics": ("Advanced Hypothesis Testing & Statistical Inference", C2_07_ADVSTATS),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas": ("Pandas Data Wrangling & Analysis", C2_08_PANDAS),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling": ("Data Cleaning, Imputation & Wrangling", C2_09_WRANGLING),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization": ("Data Visualization & Grammar of Graphics", C2_10_VIZ),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib": ("Matplotlib Visualizations", C2_10_VIZ),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn": ("Seaborn Statistical Graphics", C2_10_VIZ),
    "02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis": ("Regular Expressions, JSON & REST APIs", C2_11_REGEX_APIS),
    
    # Course 3
    "03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering": ("EDA & Feature Engineering", C3_01_FE),
    "03_IITK_AIML_Core_Machine_Learning/02_clustering": ("Unsupervised Clustering Algorithms", C3_02_CLUSTERING),
    "03_IITK_AIML_Core_Machine_Learning/03_classification": ("Supervised Classification & Ensembles", C3_03_CLASSIFICATION),
    "03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data": ("Handling Imbalanced Datasets & Anomaly Detection", C3_04_IMBALANCE),
    "03_IITK_AIML_Core_Machine_Learning/05_model_evaluation": ("Model Evaluation & Cross-Validation Strategies", C3_05_EVAL),
    
    # Course 4
    "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics": ("Neural Network Foundations & Backpropagation", C4_01_NN_BASICS),
    "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow": ("Keras & TensorFlow 2 Architecture", C4_02_KERAS_TF),
    "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance": ("Deep Learning Regularization & Preprocessing", C4_03_DL_PREPROC_IMBALANCE),
    "04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl": ("Deep Learning Model Evaluation & Interpretability", C4_04_DL_MODEL_EVAL),
    
    # Course 5
    "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering": ("Prompt Engineering & In-Context Learning", C5_01_PROMPT),
    "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications": ("ChatGPT Applications & Tool Orchestration", C5_02_CHATGPT),
    "05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization": ("GenAI Optimization & Fine-Tuning Strategies", C5_03_OPTIMIZATION),
}


def build_markdown(title: str, qa_pairs: list[tuple[str, str]]) -> str:
    lines = [
        f"# Interview Q&A — {title}",
        "",
        f"> **{len(qa_pairs)} High-Yield Questions & Model Answers** for AI/ML and Data Science Technical Interviews.",
        "",
    ]
    for idx, (q, a) in enumerate(qa_pairs, start=1):
        lines.append(f"### Q{idx}. {q}")
        lines.append("")
        lines.append(f"**Answer:** {a}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main():
    total_written = 0
    total_questions = 0
    for rel_folder, (title, qa_list) in TOPIC_MAPPING.items():
        folder = REPO_ROOT / rel_folder
        folder.mkdir(parents=True, exist_ok=True)
        target_file = folder / "interview_qa.md"
        content = build_markdown(title, qa_list)
        target_file.write_text(content, encoding="utf-8")
        total_written += 1
        total_questions += len(qa_list)
        print(f"✅ Written {len(qa_list)} Q&As -> {target_file.relative_to(REPO_ROOT)}")

    print(f"\n🎉 Successfully updated {total_written} interview banks with {total_questions} authentic questions!")


if __name__ == "__main__":
    main()
