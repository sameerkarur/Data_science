"""
Generate full-featured, interactive index.html
"""

import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "sameerkarur/Data_science"
BRANCH = "main"

def colab_link(rel_path: str) -> str:
    clean = rel_path.lstrip("./")
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/{clean}"

def gh_link(rel_path: str) -> str:
    clean = rel_path.lstrip("./")
    return f"https://github.com/{GITHUB_REPO}/blob/{BRANCH}/{clean}"

def topic_card(title: str, folder_rel: str, practice="practice.ipynb", solutions="solutions.ipynb", basics="basics.md", qa="interview_qa.md") -> str:
    p_path = f"{folder_rel}/{practice}"
    s_path = f"{folder_rel}/{solutions}"
    b_path = f"{folder_rel}/{basics}"
    q_path = f"{folder_rel}/{qa}"
    
    c_url = colab_link(p_path)
    s_url = colab_link(s_path)
    
    return f"""          <div class="topic-card">
            <div class="topic-header">
              <span class="topic-title">{html.escape(title)}</span>
            </div>
            <div class="topic-actions">
              <a class="btn-colab" href="{c_url}" target="_blank" rel="noopener noreferrer" title="Run and practice live in Google Colab">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run in Colab">
              </a>
              <button class="btn-preview" onclick="openPreview('{p_path}', '{html.escape(title)}', '{s_path}', '{b_path}', '{q_path}')">
                👁️ Preview
              </button>
            </div>
            <div class="files">
              <a href="{b_path}" target="_blank">📖 Basics</a> · 
              <a href="{s_url}" target="_blank" title="Open complete solutions in Colab">💡 Solutions</a> · 
              <a href="{q_path}" target="_blank">❓ Interview Q&amp;A</a>
            </div>
          </div>"""

def proj_card(title: str, nb_path: str, writeup_path: str = None) -> str:
    c_url = colab_link(nb_path)
    g_url = gh_link(nb_path)
    writeup_html = f' · <a class="proj-writeup" href="{writeup_path}" target="_blank">📄 Writeup</a>' if writeup_path else ""
    return f"""          <div class="project-card">
            <span class="proj-title">{html.escape(title)}</span>
            <div class="proj-actions">
              <a class="btn-colab-small" href="{c_url}" target="_blank" rel="noopener noreferrer" title="Run project notebook in Google Colab">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
              </a>
              <a class="proj-gh" href="{g_url}" target="_blank">View on GitHub</a>{writeup_html}
            </div>
          </div>"""

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AIML Practice Hub — Sameer Karur</title>
  <link rel="icon" href="https://colab.research.google.com/assets/colab-badge.svg" type="image/svg+xml">
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --card-hover: #1f242c;
      --accent: #10a37f;
      --accent-hover: #1a7f64;
      --colab-orange: #f9ab00;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
      --code-bg: #0b0e14;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    
    /* Header */
    header {{
      background: linear-gradient(135deg, #161b22 0%, #090d13 100%);
      padding: 2.75rem 2rem 2.25rem;
      border-bottom: 1px solid var(--border);
      text-align: center;
    }}
    header h1 {{
      font-size: 2.1rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 0.5rem;
      background: linear-gradient(90deg, #58a6ff, #10a37f);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    header p {{
      color: var(--muted);
      max-width: 680px;
      margin: 0 auto 1.25rem;
      font-size: 0.95rem;
    }}
    .badge {{
      display: inline-block;
      background: rgba(16, 163, 127, 0.15);
      border: 1px solid var(--accent);
      color: #2dd4bf;
      padding: 0.25rem 0.85rem;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
    }}

    /* Container */
    main {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 2rem 1.25rem;
    }}

    /* Interactive Banner */
    .hero-banner {{
      background: linear-gradient(135deg, rgba(249, 171, 0, 0.08) 0%, rgba(16, 163, 127, 0.08) 100%);
      border: 1px solid rgba(249, 171, 0, 0.3);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
      display: flex;
      align-items: center;
      gap: 1.25rem;
    }}
    .pulse-indicator {{
      width: 14px;
      height: 14px;
      background-color: #10a37f;
      border-radius: 50%;
      box-shadow: 0 0 0 rgba(16, 163, 127, 0.5);
      animation: pulse 2s infinite;
      flex-shrink: 0;
    }}
    @keyframes pulse {{
      0% {{ box-shadow: 0 0 0 0 rgba(16, 163, 127, 0.6); }}
      70% {{ box-shadow: 0 0 0 10px rgba(16, 163, 127, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(16, 163, 127, 0); }}
    }}
    .hero-text strong {{ color: #58a6ff; }}
    .hero-text p {{ font-size: 0.92rem; color: #c9d1d9; margin-top: 0.2rem; }}

    /* Quick Start */
    .start {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
    }}
    .start pre {{
      background: var(--code-bg);
      padding: 0.85rem 1rem;
      border-radius: 8px;
      overflow-x: auto;
      margin-top: 0.65rem;
      font-size: 0.85rem;
      color: #79c0ff;
      border: 1px solid #21262d;
    }}

    /* Section Headings */
    h2 {{
      font-size: 1.35rem;
      font-weight: 600;
      margin: 2.25rem 0 1rem;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* Courses Accordion */
    .course {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 1.25rem;
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .course:hover {{ border-color: #484f58; }}
    .course-head {{
      padding: 1.1rem 1.5rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }}
    .course-head:hover {{ background: var(--card-hover); }}
    .course-head h3 {{ font-size: 1.05rem; font-weight: 600; color: #f0f6fc; }}
    .course-head span {{ color: var(--muted); font-size: 0.88rem; }}
    .course-body {{
      padding: 0 1.5rem 1.5rem;
      display: none;
      border-top: 1px solid var(--border);
      background: rgba(13, 17, 23, 0.4);
    }}
    .course.open .course-body {{ display: block; }}

    /* Subtopics Grid */
    .subtopics {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 0.85rem;
      margin-top: 1.25rem;
    }}
    .topic-card {{
      background: #11161d;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 0.15s, border-color 0.15s;
    }}
    .topic-card:hover {{
      transform: translateY(-2px);
      border-color: #58a6ff;
    }}
    .topic-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 0.75rem;
    }}
    .topic-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #e6edf3;
    }}
    .topic-actions {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.75rem;
    }}
    .btn-colab {{
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      transition: opacity 0.15s;
    }}
    .btn-colab:hover {{ opacity: 0.85; }}
    .btn-colab img {{ height: 22px; }}
    
    .btn-preview {{
      background: #21262d;
      border: 1px solid var(--border);
      color: #c9d1d9;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.2rem 0.65rem;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.25rem;
      transition: all 0.15s;
    }}
    .btn-preview:hover {{
      background: #30363d;
      color: #fff;
      border-color: #8b949e;
    }}

    .files {{
      font-size: 0.82rem;
      color: var(--muted);
      border-top: 1px solid #21262d;
      padding-top: 0.6rem;
      margin-top: 0.25rem;
    }}
    .files a {{
      color: #58a6ff;
      text-decoration: none;
    }}
    .files a:hover {{ text-decoration: underline; }}

    /* Projects Section */
    .projects {{
      border-top: 1px solid var(--border);
      padding-top: 1.25rem;
      margin-top: 1.25rem;
    }}
    .projects h4 {{
      font-size: 0.92rem;
      margin-bottom: 0.75rem;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .projects-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 0.75rem;
    }}
    .project-card {{
      background: #11161d;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .proj-title {{
      font-size: 0.9rem;
      font-weight: 600;
      color: #e6edf3;
    }}
    .proj-actions {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      font-size: 0.82rem;
    }}
    .btn-colab-small img {{ height: 18px; vertical-align: middle; }}
    .proj-gh {{
      color: #58a6ff;
      text-decoration: none;
    }}
    .proj-gh:hover {{ text-decoration: underline; }}
    .proj-writeup {{
      color: #2dd4bf;
      text-decoration: none;
    }}
    .proj-writeup:hover {{ text-decoration: underline; }}

    /* Interactive Preview Modal */
    .modal-overlay {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.78);
      backdrop-filter: blur(5px);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      padding: 1.5rem;
    }}
    .modal-dialog {{
      background: #161b22;
      border: 1px solid #484f58;
      border-radius: 14px;
      width: 100%;
      max-width: 900px;
      height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      overflow: hidden;
    }}
    .modal-header {{
      padding: 1.1rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #11161d;
    }}
    .modal-header h3 {{
      font-size: 1.15rem;
      color: #f0f6fc;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .modal-close {{
      background: transparent;
      border: none;
      color: #8b949e;
      font-size: 1.75rem;
      line-height: 1;
      cursor: pointer;
      padding: 0 0.5rem;
    }}
    .modal-close:hover {{ color: #fff; }}

    .modal-toolbar {{
      padding: 0.85rem 1.5rem;
      background: #161b22;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
    }}
    .btn-modal-colab {{
      background: linear-gradient(135deg, #f9ab00 0%, #e37400 100%);
      color: #000 !important;
      font-weight: 700;
      font-size: 0.88rem;
      padding: 0.45rem 1.1rem;
      border-radius: 6px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      box-shadow: 0 2px 8px rgba(249, 171, 0, 0.3);
      transition: transform 0.15s;
    }}
    .btn-modal-colab:hover {{ transform: scale(1.02); }}
    
    .btn-modal-secondary {{
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .btn-modal-secondary:hover {{
      background: #30363d;
      color: #fff;
      border-color: #8b949e;
    }}

    .modal-body {{
      padding: 1.5rem;
      overflow-y: auto;
      flex: 1;
      font-size: 0.92rem;
    }}
    .cell-container {{
      margin-bottom: 1.25rem;
      border: 1px solid #21262d;
      border-radius: 8px;
      overflow: hidden;
    }}
    .cell-md {{
      padding: 1rem;
      background: #11161d;
      line-height: 1.6;
    }}
    .cell-md h1, .cell-md h2, .cell-md h3 {{
      color: #58a6ff;
      margin-bottom: 0.5rem;
    }}
    .cell-code-header {{
      background: #161b22;
      border-bottom: 1px solid #21262d;
      padding: 0.35rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      color: #8b949e;
    }}
    .copy-btn {{
      background: transparent;
      border: 1px solid #30363d;
      color: #8b949e;
      font-size: 0.75rem;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      cursor: pointer;
    }}
    .copy-btn:hover {{
      color: #fff;
      border-color: #8b949e;
    }}
    .cell-code {{
      background: var(--code-bg);
      padding: 0.85rem 1rem;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.85rem;
      overflow-x: auto;
      color: #c9d1d9;
    }}
    .fallback-card {{
      background: #11161d;
      border: 1px dashed #484f58;
      border-radius: 10px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      margin: 2rem auto;
      max-width: 600px;
    }}
    .fallback-card h4 {{ font-size: 1.2rem; color: #58a6ff; margin-bottom: 0.75rem; }}
    .fallback-card p {{ color: #8b949e; margin-bottom: 1.5rem; font-size: 0.95rem; }}

    /* Footer */
    footer {{
      text-align: center;
      padding: 2.5rem 1rem;
      color: var(--muted);
      font-size: 0.9rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }}
  </style>
</head>
<body>
  <header>
    <span class="badge">IITK AIML · LMS-Aligned · Dual Architecture</span>
    <h1>Generative AI &amp; Machine Learning Hub</h1>
    <p>Professional Certificate Course — E&amp;ICT Academy, IIT Kanpur &amp; Simplilearn.<br>Curated and engineered by <a href="https://github.com/sameerkarur" style="color:#58a6ff;text-decoration:none">Sameer Karur</a>.</p>
  </header>

  <main>
    <!-- Interactive Cloud Banner -->
    <div class="hero-banner">
      <div class="pulse-indicator"></div>
      <div class="hero-text">
        <strong>🚀 100% Interactive Cloud Practice Mode Activated:</strong>
        <p>No more raw JSON text! Click <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="height:18px;vertical-align:middle;margin:0 4px"> on any topic below to run, edit, and test Jupyter notebooks in <strong>Google Colab</strong> with free Python compute. Or click <strong>👁️ Preview</strong> to inspect exercises directly in your browser.</p>
      </div>
    </div>

    <!-- Quick Start -->
    <div class="start">
      <strong>💻 Local Offline Setup (Optional)</strong>
      <pre>git clone https://github.com/{GITHUB_REPO}.git
cd Data_science
pip install -r requirements.txt
jupyter notebook</pre>
      <p style="margin-top:0.75rem;color:var(--muted);font-size:0.88rem">Open any subtopic → read <code>basics.md</code> → solve <code>practice.ipynb</code> → compare with <code>solutions.ipynb</code> → review <code>interview_qa.md</code></p>
    </div>

    <h2>📚 Mandatory Courses (1 to 7)</h2>

    <!-- Induction -->
    <section class="course" id="ind">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Induction Session</h3><span>Program induction</span>
      </div>
      <div class="course-body">
        <div style="margin-top:1rem"><a href="00_Induction_Session/README.md" style="color:#58a6ff;font-weight:600">Induction Session for Professional Certificate Course in AI and Machine Learning</a></div>
      </div>
    </section>

    <!-- Course 1 -->
    <section class="course open" id="c1">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 1 — IITK AIML Foundations: Programming Refresher</h3><span>5 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Variables, Data Types & Operators', '01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes')}
{topic_card('Control Flow & Functions', '01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions')}
{topic_card('Data Structures (Lists, Dicts, Tuples)', '01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures')}
{topic_card('OOP & Modules', '01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules')}
{topic_card('File I/O & Exceptions', '01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions')}
        </div>
        <div class="projects">
          <h4>Course 1 Projects</h4>
          <div class="projects-grid">
            <div class="project-card">
              <span class="proj-title">Personal Expense Tracker</span>
              <div class="proj-actions">
                <a class="proj-gh" href="01_IITK_AIML_Foundations_Programming_Refresher/projects/Personal_Expense_Tracker/" target="_blank">View Project Folder</a>
              </div>
            </div>
            <div class="project-card">
              <span class="proj-title">Task Manager (CLI &amp; Data Pipeline)</span>
              <div class="proj-actions">
                <a class="proj-gh" href="01_IITK_AIML_Foundations_Programming_Refresher/projects/Task_Manager/" target="_blank">View Project Folder</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Course 2 -->
    <section class="course" id="c2">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 2 — IITK AIML Core: Applied Data Science with Python</h3><span>11 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Intro to Data Science', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science')}
{topic_card('Python Essentials for Data Science', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials')}
{topic_card('NumPy Mastery (45 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy', practice='numpy_practice.ipynb', solutions='numpy_solutions.ipynb', basics='numpy_basics.md')}
{topic_card('Linear Algebra for Machine Learning', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra')}
{topic_card('Statistics Fundamentals', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals')}
{topic_card('Probability & Distributions', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions')}
{topic_card('Advanced Statistics & Hypothesis Testing', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics')}
{topic_card('Pandas Data Analysis (70 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas', practice='pandas_practice.ipynb', solutions='pandas_solutions.ipynb', basics='pandas_basics.md')}
{topic_card('Data Wrangling & Cleaning', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling')}
{topic_card('Matplotlib Visualization (45 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib', practice='matplotlib_practice.ipynb', solutions='matplotlib_solutions.ipynb', basics='matplotlib_basics.md')}
{topic_card('Seaborn Statistical Plots (60 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn', practice='seaborn_practice.ipynb', solutions='seaborn_solutions.ipynb', basics='seaborn_basics.md')}
{topic_card('Regex, JSON & Web APIs', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis')}
        </div>
        <div class="projects">
          <h4>Course 2 Projects</h4>
          <div class="projects-grid">
{proj_card('Marketing Campaigns Analysis (EDA &amp; Hypothesis Testing)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Marketing_Campaigns/Marketing_Campaigns_Analysis.ipynb')}
{proj_card('Sales Analysis (Seasonal Decomposition &amp; RFM)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Sales_Analysis/Sales_Analysis.ipynb')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 3 -->
    <section class="course" id="c3">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 3 — IITK AIML Core: Machine Learning</h3><span>5 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('EDA & Feature Engineering', '03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering')}
{topic_card('Unsupervised Clustering (K-Means, DBSCAN)', '03_IITK_AIML_Core_Machine_Learning/02_clustering')}
{topic_card('Supervised Classification (Trees, Ensembles)', '03_IITK_AIML_Core_Machine_Learning/03_classification')}
{topic_card('Imbalanced Data Handling & SMOTE', '03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data')}
{topic_card('Model Evaluation & Hyperparameter Tuning', '03_IITK_AIML_Core_Machine_Learning/05_model_evaluation')}
        </div>
        <div class="projects">
          <h4>Course 3 Projects</h4>
          <div class="projects-grid">
{proj_card('Song Cohorts Clustering (Rolling Stone Top 500 Spotify)', '03_IITK_AIML_Core_Machine_Learning/projects/Creating_Cohorts_of_Songs/Creating_Cohorts_of_Songs.ipynb')}
{proj_card('Employee Turnover Analytics (GBDT &amp; Explainability)', '03_IITK_AIML_Core_Machine_Learning/projects/Employee_Turnover_Analytics/Employee_Turnover_Analytics.ipynb')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 4 -->
    <section class="course" id="c4">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 4 — IITK AIML - Core: Deep Learning with Keras and TensorFlow</h3><span>4 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Neural Network Basics (Forward/Backprop, Activation)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics')}
{topic_card('Keras & TensorFlow Architecture', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow')}
{topic_card('DL Preprocessing & Class Imbalance (Focal Loss)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance')}
{topic_card('DL Model Evaluation (ROC-AUC, Calibration)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl')}
        </div>
        <div class="projects">
          <h4>Course 4 Projects</h4>
          <div class="projects-grid">
{proj_card('Lending Club Loan Default Prediction (Deep MLP)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Lending_Club_Loan_Analysis/Lending_Club_Loan_Analysis.ipynb')}
{proj_card('Home Loan Credit Risk Modeling (Keras Tabular Net)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Home_Loan_Data_Analysis/Home_Loan_Data_Analysis.ipynb')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 5 -->
    <section class="course" id="c5">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 5 — IITK AIML Core: Essentials of Generative AI, Prompt Engineering &amp; ChatGPT</h3><span>3 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Prompt Engineering Fundamentals', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering')}
{topic_card('ChatGPT Enterprise Applications', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications')}
{topic_card('GenAI Optimization & Guardrails', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization')}
        </div>
        <div class="projects">
          <h4>Course 5 Projects</h4>
          <div class="projects-grid">
{proj_card('ChatGPT-Based Interactive Storytelling Engine', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/ChatGPT_Based_Storytelling/ChatGPT_Based_Storytelling.ipynb')}
{proj_card('Virtual Project Management Consultant (Multi-Role AI)', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/Virtual_Project_Management_Consultant/Virtual_Project_Management_Consultant.ipynb')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 6 -->
    <section class="course open" id="c6">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 6 — IITK AIML - Advanced Generative AI</h3><span>3 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('RAG Architectures & Retrieval Engineering (50 Q)', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures')}
{topic_card('Vector Databases & ChromaDB (50 Q)', '06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma')}
{topic_card('Multimodal Generative Models & Vision AI (50 Q)', '06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models')}
        </div>
        <div class="projects">
          <h4>Course 6 Projects</h4>
          <div class="projects-grid">
{proj_card('Nestlé HR Policy Assistant (LangChain RAG &amp; ChromaDB)', '06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/hr_assistant.ipynb', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/WRITEUP.md')}
{proj_card('Netflix Marketing Creative Studio (DALL·E &amp; Gradio UI)', '06_IITK_AIML_Advanced_Generative_AI/project2_designs/netflix_design_generator.ipynb', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project2_designs/WRITEUP.md')}
          </div>
        </div>
        <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem"><a href="06_IITK_AIML_Advanced_Generative_AI/README.md" style="color:var(--accent)">Course 6 Overview &amp; LMS Upload Checklist</a></p>
      </div>
    </section>

    <!-- Course 7 -->
    <section class="course open" id="c7">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 7 — IITK AIML - Capstone Projects</h3><span>3 Capstone Deliverables Completed &amp; Packaged</span>
      </div>
      <div class="course-body">
        <div class="projects" style="border-top:none;padding-top:0.5rem">
          <div class="projects-grid">
{proj_card('Capstone 1: Autonomous Driving Perception (Transfer Learning MobileNetV2)', '07_IITK_AIML_Capstone/project1_autonomous_driving/lms_upload/autonomous_driving.ipynb')}
{proj_card('Capstone 2: Restaurant Demand &amp; Sales Forecasting (XGBoost Ensemble)', '07_IITK_AIML_Capstone/project2_sales_forecasting/lms_upload/sales_forecasting.ipynb')}
{proj_card('Capstone 3: Cultural Heritage Tourism AI (ResNet Landmark Classifier &amp; SVD)', '07_IITK_AIML_Capstone/project3_preserving_heritage/lms_upload/preserving_heritage.ipynb')}
          </div>
        </div>
        <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem"><a href="07_IITK_AIML_Capstone/README.md" style="color:var(--accent)">Capstone Overview &amp; LMS Upload Instructions</a></p>
      </div>
    </section>

    <!-- Dual Architecture & Theory -->
    <h2>⚡ Next-Gen Architectures &amp; Master Guides</h2>
    <section class="course open" id="v2-and-guides">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Projects Version 2 &amp; Master Learning Guide</h3><span>Alternative Engineering Paradigms</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
          <div class="topic-card">
            <div class="topic-header"><span class="topic-title">Master AI/ML Learning Guide</span></div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Theoretical foundations, derivations, linear algebra, loss functions &amp; 100+ interview Q&amp;As.</p>
            <div class="files"><a href="LEARNING_GUIDE.md" target="_blank" style="font-weight:600;color:#58a6ff">📖 Read LEARNING_GUIDE.md</a></div>
          </div>
          <div class="topic-card">
            <div class="topic-header"><span class="topic-title">Projects Version 2 (Suite of 15)</span></div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Advanced alternative architectures (Event-sourcing, DAG schedulers, TabNet, Hybrid RAG, EfficientNet, BPR).</p>
            <div class="files"><a href="projects_version2/README.md" target="_blank" style="font-weight:600;color:#58a6ff">📂 Explore projects_version2/</a></div>
          </div>
          <div class="topic-card">
            <div class="topic-header"><span class="topic-title">Practice Guide &amp; Study Workflow</span></div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">8-week mastery calendar, question banks distribution &amp; problem-solving methodology.</p>
            <div class="files"><a href="PRACTICE_GUIDE.md" target="_blank" style="font-weight:600;color:#58a6ff">📋 Read PRACTICE_GUIDE.md</a></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Electives -->
    <h2>🎓 Advanced Electives</h2>
    <section class="course" id="e1">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 1 — Advanced Deep Learning &amp; Computer Vision</h3><span>Confirm with POC</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_01_ADL_and_Computer_Vision/README.md" style="color:#58a6ff">IITK AIML Advanced: ADL &amp; Computer Vision</a></div></div>
    </section>
    <section class="course" id="e2">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 2 — NLP and Speech Recognition</h3><span>Registered · Live Classes Sep 19–Nov 1</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_02_NLP_and_Speech_Recognition/README.md" style="color:#58a6ff">IITK AIML Advanced: NLP and Speech Recognition</a></div></div>
    </section>
    <section class="course" id="e3">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 3 — Reinforcement Learning</h3><span>Registered · Live Classes Sep 19–Nov 1</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_03_Reinforcement_Learning/README.md" style="color:#58a6ff">IITK AIML Advanced: Reinforcement Learning</a></div></div>
    </section>
    <section class="course" id="e4">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 4 — Microsoft Azure AI Fundamentals</h3><span>Confirm with POC</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_04_Microsoft_Azure_AI_Fundamentals/README.md" style="color:#58a6ff">IITK AIML - Microsoft Azure AI Fundamentals</a></div></div>
    </section>
    <section class="course" id="e5">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 5 — Academic Masterclass by IIT Kanpur</h3><span>IITK Faculty Masterclass</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_05_Academic_Masterclass_by_IIT_Kanpur/README.md" style="color:#58a6ff">Academic Masterclass by IIT Kanpur</a></div></div>
    </section>

    <h2>📊 Shared Datasets</h2>
    <p style="color:var(--muted);margin-bottom:1rem">All datasets included in <a href="datasets/README.md" style="color:var(--accent)">datasets/shared/</a> — ready to clone and practice.</p>
  </main>

  <!-- Interactive Modal Viewer -->
  <div id="nb-modal" class="modal-overlay" style="display:none;" onclick="closeModal(event)">
    <div class="modal-dialog" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 id="modal-title">Interactive Notebook Preview</h3>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-toolbar">
        <a id="modal-btn-colab" class="btn-modal-colab" href="#" target="_blank" rel="noopener noreferrer">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
          <span>🚀 Run &amp; Practice Live in Colab</span>
        </a>
        <a id="modal-btn-sol" class="btn-modal-secondary" href="#" target="_blank">💡 Solutions (Colab)</a>
        <a id="modal-btn-gh" class="btn-modal-secondary" href="#" target="_blank">👁️ View on GitHub</a>
        <a id="modal-btn-basics" class="btn-modal-secondary" href="#" target="_blank">📖 Basics</a>
        <a id="modal-btn-qa" class="btn-modal-secondary" href="#" target="_blank">❓ Interview Q&amp;A</a>
      </div>
      <div id="modal-body" class="modal-body">
        <div class="fallback-card">
          <h4>Loading Notebook Cells...</h4>
          <p>Fetching notebook structure from repository...</p>
        </div>
      </div>
    </div>
  </div>

  <footer>
    Built with ❤️ by <a href="https://github.com/sameerkarur" style="color:var(--accent);text-decoration:none">Sameer Karur</a> · IITK AIML Program · Star ⭐ on GitHub
  </footer>

  <script>
    const GITHUB_REPO = "{GITHUB_REPO}";
    const BRANCH = "{BRANCH}";

    function openPreview(practicePath, title, solutionsPath, basicsPath, qaPath) {{
      const modal = document.getElementById('nb-modal');
      const modalTitle = document.getElementById('modal-title');
      const btnColab = document.getElementById('modal-btn-colab');
      const btnSol = document.getElementById('modal-btn-sol');
      const btnGh = document.getElementById('modal-btn-gh');
      const btnBasics = document.getElementById('modal-btn-basics');
      const btnQa = document.getElementById('modal-btn-qa');
      const modalBody = document.getElementById('modal-body');

      modalTitle.textContent = title;
      
      const colabUrl = `https://colab.research.google.com/github/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{practicePath}}`;
      const solColabUrl = `https://colab.research.google.com/github/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{solutionsPath}}`;
      const ghUrl = `https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{practicePath}}`;

      btnColab.href = colabUrl;
      btnSol.href = solColabUrl;
      btnGh.href = ghUrl;
      btnBasics.href = basicsPath || '#';
      btnQa.href = qaPath || '#';

      modalBody.innerHTML = `
        <div class="fallback-card">
          <h4>🚀 Ready to Practice "${{title}}"!</h4>
          <p>Click below to execute this notebook interactively in Google Colab with free GPU/CPU, or inspect questions right here.</p>
          <a class="btn-modal-colab" href="${{colabUrl}}" target="_blank" style="margin: 0 auto; display: inline-flex;">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
            <span>Launch Notebook in Colab</span>
          </a>
        </div>
      `;

      modal.style.display = 'flex';

      // Attempt to fetch notebook JSON
      const rawUrl = `https://raw.githubusercontent.com/${{GITHUB_REPO}}/${{BRANCH}}/${{practicePath}}`;
      fetch(rawUrl)
        .then(res => {{
          if (!res.ok) throw new Error('Network error');
          return res.json();
        }})
        .then(nb => {{
          renderNotebook(nb, modalBody, colabUrl);
        }})
        .catch(err => {{
          // If raw github fetch fails (e.g. offline/CORS), try relative path
          fetch(practicePath)
            .then(res => res.json())
            .then(nb => renderNotebook(nb, modalBody, colabUrl))
            .catch(() => {{
              // Keep friendly launch card
            }});
        }});
    }}

    function renderNotebook(nb, container, colabUrl) {{
      const cells = nb.cells || [];
      if (!cells.length) return;
      
      let htmlContent = `
        <div style="background: rgba(16,163,127,0.1); border: 1px solid var(--accent); padding: 0.85rem 1rem; border-radius: 8px; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 0.9rem; color: #2dd4bf;">💡 <strong>${{cells.length}} Notebook Cells</strong> loaded. Run code interactively:</span>
          <a class="btn-modal-colab" href="${{colabUrl}}" target="_blank" style="padding: 0.35rem 0.85rem; font-size: 0.8rem;">
            Run in Colab 🚀
          </a>
        </div>
      `;

      cells.forEach((cell, idx) => {{
        const rawSource = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        // Skip colab badge cell
        if (rawSource.includes('colab-badge.svg')) return;

        if (cell.cell_type === 'markdown') {{
          let cleanMd = escapeHtml(rawSource)
            .replace(/^# (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h4 style="color:#58a6ff;margin-top:0.5rem">$1</h4>')
            .replace(/^### (.*$)/gim, '<h5 style="color:#2dd4bf;margin-top:0.4rem">$1</h5>')
            .replace(/\\*\\*(.*?)\\*\\*/gim, '<strong>$1</strong>')
            .replace(/`(.*?)`/gim, '<code style="background:#0b0e14;padding:0.1rem 0.3rem;border-radius:3px;color:#79c0ff">$1</code>')
            .replace(/\\n/g, '<br>');

          htmlContent += `
            <div class="cell-container">
              <div class="cell-md">${{cleanMd}}</div>
            </div>
          `;
        }} else if (cell.cell_type === 'code') {{
          const codeText = escapeHtml(rawSource);
          htmlContent += `
            <div class="cell-container">
              <div class="cell-code-header">
                <span>[${{cell.execution_count || ' '}}] Python Code</span>
                <button class="copy-btn" onclick="copyCode(this)">📋 Copy</button>
              </div>
              <pre class="cell-code"><code>${{codeText || '# Code cell'}}</code></pre>
            </div>
          `;
        }}
      }});

      container.innerHTML = htmlContent;
    }}

    function copyCode(btn) {{
      const code = btn.closest('.cell-container').querySelector('code').innerText;
      navigator.clipboard.writeText(code).then(() => {{
        btn.textContent = '✅ Copied!';
        setTimeout(() => btn.textContent = '📋 Copy', 2000);
      }});
    }}

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }}

    function closeModal(e) {{
      if (!e || e.target.classList.contains('modal-overlay') || e.target.classList.contains('modal-close')) {{
        document.getElementById('nb-modal').style.display = 'none';
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});

    // Universal interceptor: NEVER dump raw JSON for any .ipynb link
    document.addEventListener('click', function(e) {{
      const a = e.target.closest('a');
      if (!a) return;
      let href = a.getAttribute('href');
      if (!href) return;
      
      // If it is a relative or raw .ipynb link (not already routed to Colab)
      if (href.endsWith('.ipynb') && !href.includes('colab.research.google.com') && !href.includes('github.com/')) {{
        e.preventDefault();
        let clean = href.replace(/^(\\.\\/|\\/)/, '');
        clean = clean.replace(/.*githubusercontent\\.com\\/[^\\/]+\\/[^\\/]+\\/[^\\/]+\\//, '');
        const colabUrl = `https://colab.research.google.com/github/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{clean}}`;
        window.open(colabUrl, '_blank');
      }}
    }});
  </script>
</body>
</html>
"""

def main():
    target = REPO_ROOT / "index.html"
    target.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"🎉 Successfully built interactive index.html at {target}!")

if __name__ == "__main__":
    main()
