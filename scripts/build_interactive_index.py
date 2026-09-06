"""
Generate premier, state-of-the-art interactive index.html
Features:
- 1-Click Interactive Google Colab launch buttons
- Instant In-Page Code & Exercise Previewer
- In-Page Markdown Reader Modal for Basics, Interview Q&As, and Learning Guides (NO RAW TEXT!)
- Live Real-Time Search across all 1,650+ topics and projects
- Category Filter Pills (Python, DS, ML, DL, GenAI, Capstone, V2)
- Sleek stats header and cleaned banner (Removed "No more raw JSON text!")
- Universal link safety interceptor
"""

import html
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "sameerkarur/Data_science"
BRANCH = "main"

def colab_link(rel_path: str) -> str:
    clean = rel_path.lstrip("./")
    return f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/{clean}"

def gh_blob_link(rel_path: str) -> str:
    clean = rel_path.lstrip("./")
    return f"https://github.com/{GITHUB_REPO}/blob/{BRANCH}/{clean}"

def topic_card(title: str, folder_rel: str, category: str, q_count: str, practice="practice.ipynb", solutions="solutions.ipynb", basics="basics.md", qa="interview_qa.md") -> str:
    p_path = f"{folder_rel}/{practice}"
    s_path = f"{folder_rel}/{solutions}"
    b_path = f"{folder_rel}/{basics}"
    q_path = f"{folder_rel}/{qa}"
    
    c_url = colab_link(p_path)
    s_url = colab_link(s_path)
    
    return f"""          <div class="topic-card" data-category="{category}" data-title="{html.escape(title.lower())}">
            <div class="topic-header">
              <span class="topic-title">{html.escape(title)}</span>
              <span class="q-badge">{html.escape(q_count)}</span>
            </div>
            <div class="topic-actions">
              <a class="btn-colab" href="{c_url}" target="_blank" rel="noopener noreferrer" title="Execute & practice live in Google Colab">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run in Colab">
              </a>
              <button class="btn-preview" onclick="openPreview('{p_path}', '{html.escape(title)}', '{s_path}', '{b_path}', '{q_path}')">
                👁️ Code Preview
              </button>
            </div>
            <div class="files">
              <a href="{b_path}" onclick="openMdModal('{b_path}', '{html.escape(title)} — Basics &amp; Concepts'); return false;">📖 Basics</a> · 
              <a href="{s_url}" target="_blank" title="Open complete solutions in Colab">💡 Solutions</a> · 
              <a href="{q_path}" onclick="openMdModal('{q_path}', '{html.escape(title)} — Interview Q&amp;A'); return false;">❓ Interview Q&amp;A</a>
            </div>
          </div>"""

def proj_card(title: str, nb_path: str, category: str, writeup_path: str = None) -> str:
    c_url = colab_link(nb_path)
    g_url = gh_blob_link(nb_path)
    writeup_html = f' · <a class="proj-writeup" href="{writeup_path}" onclick="openMdModal(\'{writeup_path}\', \'{html.escape(title)} — Project Writeup\'); return false;">📄 Writeup</a>' if writeup_path else ""
    return f"""          <div class="project-card" data-category="{category}" data-title="{html.escape(title.lower())}">
            <span class="proj-title">{html.escape(title)}</span>
            <div class="proj-actions">
              <a class="btn-colab-small" href="{c_url}" target="_blank" rel="noopener noreferrer" title="Run project notebook in Google Colab">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
              </a>
              <a class="proj-gh" href="{g_url}" target="_blank">View on GitHub</a>{writeup_html}
            </div>
          </div>"""

HTML_CONTENT = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AIML Practice Hub — Sameer Karur</title>
  <link rel="icon" href="https://colab.research.google.com/assets/colab-badge.svg" type="image/svg+xml">
  <!-- Markdown parser for instant in-page rendering without raw text -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --card-hover: #1f242c;
      --accent: #10a37f;
      --accent-hover: #1a7f64;
      --accent-glow: rgba(16, 163, 127, 0.25);
      --colab-orange: #f9ab00;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
      --code-bg: #0b0e14;
      --blue: #58a6ff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}

    /* Header & Navigation */
    header {{
      background: linear-gradient(135deg, #161b22 0%, #090d13 100%);
      padding: 3rem 1.5rem 2rem;
      border-bottom: 1px solid var(--border);
      text-align: center;
      position: relative;
    }}
    .badge {{
      display: inline-block;
      background: rgba(16, 163, 127, 0.12);
      border: 1px solid var(--accent);
      color: #2dd4bf;
      padding: 0.25rem 0.95rem;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      letter-spacing: 0.3px;
    }}
    header h1 {{
      font-size: 2.25rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 0.5rem;
      background: linear-gradient(90deg, #58a6ff, #2dd4bf, #10a37f);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    header p {{
      color: var(--muted);
      max-width: 680px;
      margin: 0 auto 1.5rem;
      font-size: 0.98rem;
    }}

    /* Key Metrics Bar */
    .stats-bar {{
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 1.25rem;
      margin: 1.5rem auto 0.5rem;
      max-width: 950px;
    }}
    .stat-item {{
      background: rgba(22, 27, 34, 0.7);
      border: 1px solid var(--border);
      padding: 0.6rem 1.2rem;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-width: 130px;
    }}
    .stat-num {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #58a6ff;
    }}
    .stat-label {{
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}

    /* Main Container */
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 2rem 1.25rem;
    }}

    /* Hero Banner */
    .hero-banner {{
      background: linear-gradient(135deg, rgba(249, 171, 0, 0.08) 0%, rgba(16, 163, 127, 0.08) 100%);
      border: 1px solid rgba(249, 171, 0, 0.35);
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
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

    /* Real-Time Search & Category Filter */
    .search-container {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.25rem;
      margin-bottom: 2.25rem;
    }}
    .search-box {{
      position: relative;
      display: flex;
      align-items: center;
      margin-bottom: 1rem;
    }}
    .search-icon {{
      position: absolute;
      left: 1rem;
      font-size: 1rem;
      color: var(--muted);
      pointer-events: none;
    }}
    #topic-search {{
      width: 100%;
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 2.5rem 0.75rem 2.75rem;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}
    #topic-search:focus {{
      border-color: #58a6ff;
      box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
    }}
    #search-clear {{
      position: absolute;
      right: 1rem;
      font-size: 1.25rem;
    }}
    .filter-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .pill {{
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .pill:hover {{
      background: #30363d;
      color: #fff;
    }}
    .pill.active {{
      background: var(--accent);
      color: #000;
      border-color: var(--accent);
    }}

    /* Section Headings */
    h2 {{
      font-size: 1.35rem;
      font-weight: 600;
      margin: 2.5rem 0 1rem;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* Course Accordion */
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
      padding: 1.15rem 1.5rem;
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
      gap: 0.5rem;
    }}
    .topic-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #e6edf3;
    }}
    .q-badge {{
      background: rgba(88, 166, 255, 0.12);
      border: 1px solid rgba(88, 166, 255, 0.3);
      color: #79c0ff;
      font-size: 0.75rem;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      white-space: nowrap;
      font-weight: 600;
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
      cursor: pointer;
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
      cursor: pointer;
    }}
    .proj-writeup:hover {{ text-decoration: underline; }}

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

    /* Interactive Modals */
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
      max-width: 960px;
      height: 88vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px rgba(0,0,0,0.65);
      overflow: hidden;
    }}
    .modal-header {{
      padding: 1.15rem 1.5rem;
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
      padding: 1.75rem;
      overflow-y: auto;
      flex: 1;
      font-size: 0.95rem;
      line-height: 1.7;
    }}
    .modal-body h1, .modal-body h2, .modal-body h3 {{
      color: #58a6ff;
      margin: 1.25rem 0 0.6rem;
    }}
    .modal-body h1 {{ font-size: 1.6rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    .modal-body h2 {{ font-size: 1.35rem; }}
    .modal-body h3 {{ font-size: 1.15rem; color: #2dd4bf; }}
    .modal-body p {{ margin-bottom: 0.85rem; }}
    .modal-body ul, .modal-body ol {{ margin: 0.5rem 0 1rem 1.5rem; }}
    .modal-body li {{ margin-bottom: 0.35rem; }}
    .modal-body table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.25rem 0;
      font-size: 0.88rem;
    }}
    .modal-body th, .modal-body td {{
      padding: 0.6rem 0.85rem;
      border: 1px solid var(--border);
      text-align: left;
    }}
    .modal-body th {{ background: #11161d; color: #58a6ff; }}
    .modal-body pre {{
      background: var(--code-bg);
      padding: 1rem;
      border-radius: 8px;
      overflow-x: auto;
      margin: 0.85rem 0;
      border: 1px solid var(--border);
      color: #c9d1d9;
    }}
    .modal-body code {{
      background: rgba(110, 118, 129, 0.2);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      color: #79c0ff;
      font-size: 0.88rem;
    }}

    /* Cell Container for Notebook Viewer */
    .cell-container {{
      margin-bottom: 1.25rem;
      border: 1px solid #21262d;
      border-radius: 8px;
      overflow: hidden;
    }}
    .cell-md {{
      padding: 1rem;
      background: #11161d;
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

    <!-- Metrics Bar -->
    <div class="stats-bar">
      <div class="stat-item"><span class="stat-num">1,650+</span><span class="stat-label">Practice Problems</span></div>
      <div class="stat-item"><span class="stat-num">300+</span><span class="stat-label">Interview Q&amp;As</span></div>
      <div class="stat-item"><span class="stat-num">15</span><span class="stat-label">Next-Gen V2 Projects</span></div>
      <div class="stat-item"><span class="stat-num">3</span><span class="stat-label">LMS Capstones</span></div>
      <div class="stat-item"><span class="stat-num">100%</span><span class="stat-label">Interactive Colab</span></div>
    </div>
  </header>

  <main>
    <!-- Interactive Cloud Banner -->
    <div class="hero-banner">
      <div class="pulse-indicator"></div>
      <div class="hero-text">
        <strong>⚡ 1-Click Interactive Cloud Execution Ready:</strong>
        <p>Click <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="height:18px;vertical-align:middle;margin:0 4px"> on any topic or project to run, edit, and test Jupyter notebooks in <strong>Google Colab</strong> with free GPU/CPU compute. Or click <strong>👁️ Code Preview</strong> to inspect exercises directly in your browser.</p>
      </div>
    </div>

    <!-- Real-Time Search & Category Filters -->
    <div class="search-container">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="topic-search" placeholder="Search 1,650+ topics, algorithms, datasets, or projects (e.g. 'RAG', 'K-Means', 'XGBoost', 'NumPy', 'Colab')..." oninput="handleSearch(this.value)">
        <span id="search-clear" onclick="clearSearch()">&times;</span>
      </div>
      <div class="filter-pills">
        <button class="pill active" onclick="setFilter('all', this)">All (32 Topics · 15 Projects)</button>
        <button class="pill" onclick="setFilter('python', this)">Python (250+ Q)</button>
        <button class="pill" onclick="setFilter('ds', this)">Data Science (670+ Q)</button>
        <button class="pill" onclick="setFilter('ml', this)">Machine Learning (250+ Q)</button>
        <button class="pill" onclick="setFilter('dl', this)">Deep Learning (200+ Q)</button>
        <button class="pill" onclick="setFilter('genai', this)">GenAI &amp; ChatGPT (150+ Q)</button>
        <button class="pill" onclick="setFilter('c6', this)">Advanced GenAI &amp; RAG (150+ Q)</button>
        <button class="pill" onclick="setFilter('capstone', this)">Capstones</button>
        <button class="pill" onclick="setFilter('v2', this)">Version 2 Suite</button>
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
    <section class="course" id="ind" data-course="python">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Induction Session</h3><span>Program induction</span>
      </div>
      <div class="course-body">
        <div style="margin-top:1rem"><a href="00_Induction_Session/README.md" onclick="openMdModal('00_Induction_Session/README.md', 'Induction Session Overview'); return false;" style="color:#58a6ff;font-weight:600">📖 Induction Session for Professional Certificate Course in AI and Machine Learning</a></div>
      </div>
    </section>

    <!-- Course 1 -->
    <section class="course open" id="c1" data-course="python">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 1 — IITK AIML Foundations: Programming Refresher</h3><span>5 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Variables, Data Types & Operators', '01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes', 'python', '50 Problems')}
{topic_card('Control Flow & Functions', '01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions', 'python', '50 Problems')}
{topic_card('Data Structures (Lists, Dicts, Tuples)', '01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures', 'python', '50 Problems')}
{topic_card('OOP & Modules', '01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules', 'python', '50 Problems')}
{topic_card('File I/O & Exceptions', '01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions', 'python', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 1 Projects</h4>
          <div class="projects-grid">
            <div class="project-card" data-category="python" data-title="personal expense tracker">
              <span class="proj-title">Personal Expense Tracker</span>
              <div class="proj-actions">
                <a class="proj-gh" href="01_IITK_AIML_Foundations_Programming_Refresher/projects/Personal_Expense_Tracker/" target="_blank">View Project Folder</a>
              </div>
            </div>
            <div class="project-card" data-category="python" data-title="task manager cli">
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
    <section class="course" id="c2" data-course="ds">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 2 — IITK AIML Core: Applied Data Science with Python</h3><span>11 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Intro to Data Science', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/01_intro_data_science', 'ds', '50 Problems')}
{topic_card('Python Essentials for Data Science', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/02_python_essentials', 'ds', '50 Problems')}
{topic_card('NumPy Mastery (45 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/03_numpy', 'ds', '45 Problems', practice='numpy_practice.ipynb', solutions='numpy_solutions.ipynb', basics='numpy_basics.md')}
{topic_card('Linear Algebra for Machine Learning', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/04_linear_algebra', 'ds', '50 Problems')}
{topic_card('Statistics Fundamentals', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/05_statistics_fundamentals', 'ds', '50 Problems')}
{topic_card('Probability & Distributions', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/06_probability_distributions', 'ds', '50 Problems')}
{topic_card('Advanced Statistics & Hypothesis Testing', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/07_advanced_statistics', 'ds', '50 Problems')}
{topic_card('Pandas Data Analysis (70 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/08_pandas', 'ds', '70 Problems', practice='pandas_practice.ipynb', solutions='pandas_solutions.ipynb', basics='pandas_basics.md')}
{topic_card('Data Wrangling & Cleaning', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/09_data_wrangling', 'ds', '50 Problems')}
{topic_card('Matplotlib Visualization (45 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/matplotlib', 'ds', '45 Problems', practice='matplotlib_practice.ipynb', solutions='matplotlib_solutions.ipynb', basics='matplotlib_basics.md')}
{topic_card('Seaborn Statistical Plots (60 Questions)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/10_data_visualization/seaborn', 'ds', '60 Problems', practice='seaborn_practice.ipynb', solutions='seaborn_solutions.ipynb', basics='seaborn_basics.md')}
{topic_card('Regex, JSON & Web APIs', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/11_regex_json_apis', 'ds', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 2 Projects</h4>
          <div class="projects-grid">
{proj_card('Marketing Campaigns Analysis (EDA &amp; Hypothesis Testing)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Marketing_Campaigns/Marketing_Campaigns_Analysis.ipynb', 'ds')}
{proj_card('Sales Analysis (Seasonal Decomposition &amp; RFM)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Sales_Analysis/Sales_Analysis.ipynb', 'ds')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 3 -->
    <section class="course" id="c3" data-course="ml">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 3 — IITK AIML Core: Machine Learning</h3><span>5 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('EDA & Feature Engineering', '03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering', 'ml', '50 Problems')}
{topic_card('Unsupervised Clustering (K-Means, DBSCAN)', '03_IITK_AIML_Core_Machine_Learning/02_clustering', 'ml', '50 Problems')}
{topic_card('Supervised Classification (Trees, Ensembles)', '03_IITK_AIML_Core_Machine_Learning/03_classification', 'ml', '50 Problems')}
{topic_card('Imbalanced Data Handling & SMOTE', '03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data', 'ml', '50 Problems')}
{topic_card('Model Evaluation & Hyperparameter Tuning', '03_IITK_AIML_Core_Machine_Learning/05_model_evaluation', 'ml', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 3 Projects</h4>
          <div class="projects-grid">
{proj_card('Song Cohorts Clustering (Rolling Stone Top 500 Spotify)', '03_IITK_AIML_Core_Machine_Learning/projects/Creating_Cohorts_of_Songs/Creating_Cohorts_of_Songs.ipynb', 'ml')}
{proj_card('Employee Turnover Analytics (GBDT &amp; Explainability)', '03_IITK_AIML_Core_Machine_Learning/projects/Employee_Turnover_Analytics/Employee_Turnover_Analytics.ipynb', 'ml')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 4 -->
    <section class="course" id="c4" data-course="dl">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 4 — IITK AIML - Core: Deep Learning with Keras and TensorFlow</h3><span>4 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Neural Network Basics (Forward/Backprop, Activation)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics', 'dl', '50 Problems')}
{topic_card('Keras & TensorFlow Architecture', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow', 'dl', '50 Problems')}
{topic_card('DL Preprocessing & Class Imbalance (Focal Loss)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance', 'dl', '50 Problems')}
{topic_card('DL Model Evaluation (ROC-AUC, Calibration)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl', 'dl', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 4 Projects</h4>
          <div class="projects-grid">
{proj_card('Lending Club Loan Default Prediction (Deep MLP)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Lending_Club_Loan_Analysis/Lending_Club_Loan_Analysis.ipynb', 'dl')}
{proj_card('Home Loan Credit Risk Modeling (Keras Tabular Net)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Home_Loan_Data_Analysis/Home_Loan_Data_Analysis.ipynb', 'dl')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 5 -->
    <section class="course" id="c5" data-course="genai">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 5 — IITK AIML Core: Essentials of Generative AI, Prompt Engineering &amp; ChatGPT</h3><span>3 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('Prompt Engineering Fundamentals', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering', 'genai', '50 Problems')}
{topic_card('ChatGPT Enterprise Applications', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications', 'genai', '50 Problems')}
{topic_card('GenAI Optimization & Guardrails', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization', 'genai', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 5 Projects</h4>
          <div class="projects-grid">
{proj_card('ChatGPT-Based Interactive Storytelling Engine', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/ChatGPT_Based_Storytelling/ChatGPT_Based_Storytelling.ipynb', 'genai')}
{proj_card('Virtual Project Management Consultant (Multi-Role AI)', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/Virtual_Project_Management_Consultant/Virtual_Project_Management_Consultant.ipynb', 'genai')}
          </div>
        </div>
      </div>
    </section>

    <!-- Course 6 -->
    <section class="course open" id="c6" data-course="c6">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 6 — IITK AIML - Advanced Generative AI</h3><span>3 practice topics · 2 projects</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
{topic_card('RAG Architectures & Retrieval Engineering', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures', 'c6', '50 Problems')}
{topic_card('Vector Databases & ChromaDB', '06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma', 'c6', '50 Problems')}
{topic_card('Multimodal Generative Models & Vision AI', '06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models', 'c6', '50 Problems')}
        </div>
        <div class="projects">
          <h4>Course 6 Projects</h4>
          <div class="projects-grid">
{proj_card('Nestlé HR Policy Assistant (LangChain RAG &amp; ChromaDB)', '06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/hr_assistant.ipynb', 'c6', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/WRITEUP.md')}
{proj_card('Netflix Marketing Creative Studio (DALL·E &amp; Gradio UI)', '06_IITK_AIML_Advanced_Generative_AI/project2_designs/netflix_design_generator.ipynb', 'c6', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project2_designs/WRITEUP.md')}
          </div>
        </div>
        <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem"><a href="06_IITK_AIML_Advanced_Generative_AI/README.md" onclick="openMdModal('06_IITK_AIML_Advanced_Generative_AI/README.md', 'Course 6 Overview &amp; LMS Checklist'); return false;" style="color:var(--accent)">Course 6 Overview &amp; LMS Upload Checklist</a></p>
      </div>
    </section>

    <!-- Course 7 -->
    <section class="course open" id="c7" data-course="capstone">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Course 7 — IITK AIML - Capstone Projects</h3><span>3 Capstone Deliverables Completed &amp; Packaged</span>
      </div>
      <div class="course-body">
        <div class="projects" style="border-top:none;padding-top:0.5rem">
          <div class="projects-grid">
{proj_card('Capstone 1: Autonomous Driving Perception (Transfer Learning MobileNetV2)', '07_IITK_AIML_Capstone/project1_autonomous_driving/lms_upload/autonomous_driving.ipynb', 'capstone')}
{proj_card('Capstone 2: Restaurant Demand &amp; Sales Forecasting (XGBoost Ensemble)', '07_IITK_AIML_Capstone/project2_sales_forecasting/lms_upload/sales_forecasting.ipynb', 'capstone')}
{proj_card('Capstone 3: Cultural Heritage Tourism AI (ResNet Landmark Classifier &amp; SVD)', '07_IITK_AIML_Capstone/project3_preserving_heritage/lms_upload/preserving_heritage.ipynb', 'capstone')}
          </div>
        </div>
        <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem"><a href="07_IITK_AIML_Capstone/README.md" onclick="openMdModal('07_IITK_AIML_Capstone/README.md', 'Capstone Overview &amp; LMS Instructions'); return false;" style="color:var(--accent)">Capstone Overview &amp; LMS Upload Instructions</a></p>
      </div>
    </section>

    <!-- Dual Architecture & Theory -->
    <h2>⚡ Next-Gen Architectures &amp; Master Guides</h2>
    <section class="course open" id="v2-and-guides" data-course="v2">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Projects Version 2 &amp; Master Learning Guide</h3><span>Alternative Engineering Paradigms</span>
      </div>
      <div class="course-body">
        <div class="subtopics">
          <div class="topic-card" data-category="v2" data-title="master aiml learning guide">
            <div class="topic-header">
              <span class="topic-title">Master AI/ML Learning Guide</span>
              <span class="q-badge">Master Theory</span>
            </div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Theoretical foundations, derivations, linear algebra, loss functions &amp; 100+ interview Q&amp;As.</p>
            <div class="files"><a href="LEARNING_GUIDE.md" onclick="openMdModal('LEARNING_GUIDE.md', 'Master AI/ML Learning Guide'); return false;" style="font-weight:600;color:#58a6ff">📖 Read LEARNING_GUIDE.md</a></div>
          </div>
          <div class="topic-card" data-category="v2" data-title="projects version 2 portfolio suite of 15">
            <div class="topic-header">
              <span class="topic-title">Projects Version 2 (Suite of 15)</span>
              <span class="q-badge">15 V2 Projects</span>
            </div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Advanced alternative architectures (Event-sourcing, DAG schedulers, TabNet, Hybrid RAG, EfficientNet, BPR).</p>
            <div class="files"><a href="projects_version2/README.md" onclick="openMdModal('projects_version2/README.md', 'Projects Version 2 Architectural Suite'); return false;" style="font-weight:600;color:#58a6ff">📂 Explore projects_version2/</a></div>
          </div>
          <div class="topic-card" data-category="v2" data-title="practice guide and study workflow">
            <div class="topic-header">
              <span class="topic-title">Practice Guide &amp; Study Workflow</span>
              <span class="q-badge">8-Week Plan</span>
            </div>
            <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">8-week mastery calendar, question banks distribution &amp; problem-solving methodology.</p>
            <div class="files"><a href="PRACTICE_GUIDE.md" onclick="openMdModal('PRACTICE_GUIDE.md', 'Practice Guide &amp; Workflow'); return false;" style="font-weight:600;color:#58a6ff">📋 Read PRACTICE_GUIDE.md</a></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Electives -->
    <h2>🎓 Advanced Electives</h2>
    <section class="course" id="e1" data-course="electives">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 1 — Advanced Deep Learning &amp; Computer Vision</h3><span>Confirm with POC</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_01_ADL_and_Computer_Vision/README.md" onclick="openMdModal('Elective_01_ADL_and_Computer_Vision/README.md', 'Elective 1: ADL &amp; Computer Vision'); return false;" style="color:#58a6ff">📖 Elective 1 Overview</a></div></div>
    </section>
    <section class="course" id="e2" data-course="electives">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 2 — NLP and Speech Recognition</h3><span>Registered · Live Classes Sep 19–Nov 1</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_02_NLP_and_Speech_Recognition/README.md" onclick="openMdModal('Elective_02_NLP_and_Speech_Recognition/README.md', 'Elective 2: NLP and Speech Recognition'); return false;" style="color:#58a6ff">📖 Elective 2 Overview</a></div></div>
    </section>
    <section class="course" id="e3" data-course="electives">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 3 — Reinforcement Learning</h3><span>Registered · Live Classes Sep 19–Nov 1</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_03_Reinforcement_Learning/README.md" onclick="openMdModal('Elective_03_Reinforcement_Learning/README.md', 'Elective 3: Reinforcement Learning'); return false;" style="color:#58a6ff">📖 Elective 3 Overview</a></div></div>
    </section>
    <section class="course" id="e4" data-course="electives">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 4 — Microsoft Azure AI Fundamentals</h3><span>Confirm with POC</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_04_Microsoft_Azure_AI_Fundamentals/README.md" onclick="openMdModal('Elective_04_Microsoft_Azure_AI_Fundamentals/README.md', 'Elective 4: Azure AI Fundamentals'); return false;" style="color:#58a6ff">📖 Elective 4 Overview</a></div></div>
    </section>
    <section class="course" id="e5" data-course="electives">
      <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
        <h3>Elective 5 — Academic Masterclass by IIT Kanpur</h3><span>IITK Faculty Masterclass</span>
      </div>
      <div class="course-body"><div style="margin-top:1rem"><a href="Elective_05_Academic_Masterclass_by_IIT_Kanpur/README.md" onclick="openMdModal('Elective_05_Academic_Masterclass_by_IIT_Kanpur/README.md', 'Academic Masterclass by IIT Kanpur'); return false;" style="color:#58a6ff">📖 Elective 5 Overview</a></div></div>
    </section>

    <h2>📊 Shared Datasets</h2>
    <p style="color:var(--muted);margin-bottom:1rem">All datasets included in <a href="datasets/README.md" onclick="openMdModal('datasets/README.md', 'Shared Datasets Reference'); return false;" style="color:var(--accent)">datasets/shared/</a> — ready to clone and practice.</p>
  </main>

  <!-- Interactive Modal Viewer (Notebooks) -->
  <div id="nb-modal" class="modal-overlay" style="display:none;" onclick="closeModal(event, 'nb-modal')">
    <div class="modal-dialog" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 id="modal-title">Interactive Notebook Preview</h3>
        <button class="modal-close" onclick="closeModal(null, 'nb-modal')">&times;</button>
      </div>
      <div class="modal-toolbar">
        <a id="modal-btn-colab" class="btn-modal-colab" href="#" target="_blank" rel="noopener noreferrer">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
          <span>🚀 Run &amp; Practice Live in Colab</span>
        </a>
        <a id="modal-btn-sol" class="btn-modal-secondary" href="#" target="_blank">💡 Solutions (Colab)</a>
        <a id="modal-btn-gh" class="btn-modal-secondary" href="#" target="_blank">👁️ View on GitHub</a>
        <a id="modal-btn-basics" class="btn-modal-secondary" href="#" onclick="switchModalToMd(this.dataset.path, this.dataset.title); return false;">📖 Basics</a>
        <a id="modal-btn-qa" class="btn-modal-secondary" href="#" onclick="switchModalToMd(this.dataset.path, this.dataset.title); return false;">❓ Interview Q&amp;A</a>
      </div>
      <div id="modal-body" class="modal-body">
        <div class="fallback-card">
          <h4>Loading Notebook Cells...</h4>
          <p>Fetching notebook structure from repository...</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Interactive Modal Viewer (Markdown Reader) -->
  <div id="md-modal" class="modal-overlay" style="display:none;" onclick="closeModal(event, 'md-modal')">
    <div class="modal-dialog" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 id="md-modal-title">Document Reader</h3>
        <button class="modal-close" onclick="closeModal(null, 'md-modal')">&times;</button>
      </div>
      <div class="modal-toolbar">
        <a id="md-modal-gh" class="btn-modal-secondary" href="#" target="_blank">👁️ Open Formatted on GitHub</a>
        <button class="btn-modal-secondary" onclick="copyCurrentMd()">📋 Copy Markdown</button>
      </div>
      <div id="md-modal-body" class="modal-body">
        <div class="fallback-card">
          <h4>Loading Document...</h4>
          <p>Rendering formatted markdown...</p>
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
    let currentRawMdText = "";

    // Open Notebook Preview Modal
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
      
      btnBasics.dataset.path = basicsPath || '';
      btnBasics.dataset.title = `${{title}} — Concepts & Reference`;
      btnQa.dataset.path = qaPath || '';
      btnQa.dataset.title = `${{title}} — Interview Q&A`;

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

      const rawUrl = `https://raw.githubusercontent.com/${{GITHUB_REPO}}/${{BRANCH}}/${{practicePath}}`;
      fetch(rawUrl)
        .then(res => {{
          if (!res.ok) throw new Error('Network error');
          return res.json();
        }})
        .then(nb => {{
          renderNotebook(nb, modalBody, colabUrl);
        }})
        .catch(() => {{
          fetch(practicePath)
            .then(res => res.json())
            .then(nb => renderNotebook(nb, modalBody, colabUrl))
            .catch(() => {{}});
        }});
    }}

    function renderNotebook(nb, container, colabUrl) {{
      const cells = nb.cells || [];
      if (!cells.length) return;
      
      let htmlContent = `
        <div style="background: rgba(16,163,127,0.1); border: 1px solid var(--accent); padding: 0.85rem 1rem; border-radius: 8px; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
          <span style="font-size: 0.9rem; color: #2dd4bf;">💡 <strong>${{cells.length}} Notebook Cells</strong> loaded. Run code interactively:</span>
          <a class="btn-modal-colab" href="${{colabUrl}}" target="_blank" style="padding: 0.35rem 0.85rem; font-size: 0.82rem;">
            Run in Colab 🚀
          </a>
        </div>
      `;

      cells.forEach((cell) => {{
        const rawSource = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        if (rawSource.includes('colab-badge.svg')) return;

        if (cell.cell_type === 'markdown') {{
          let renderedMd = typeof marked !== 'undefined' ? marked.parse(rawSource) : escapeHtml(rawSource).replace(/\\n/g, '<br>');
          htmlContent += `
            <div class="cell-container">
              <div class="cell-md">${{renderedMd}}</div>
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

    // Open Markdown Reader Modal
    function openMdModal(mdPath, title) {{
      if (!mdPath) return;
      const modal = document.getElementById('md-modal');
      const modalTitle = document.getElementById('md-modal-title');
      const modalBody = document.getElementById('md-modal-body');
      const ghBtn = document.getElementById('md-modal-gh');

      modalTitle.textContent = title;
      ghBtn.href = `https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{mdPath}}`;
      modalBody.innerHTML = `
        <div class="fallback-card">
          <h4>Loading Document...</h4>
          <p>Fetching and formatting ${{title}}...</p>
        </div>
      `;
      modal.style.display = 'flex';

      const rawUrl = `https://raw.githubusercontent.com/${{GITHUB_REPO}}/${{BRANCH}}/${{mdPath}}`;
      fetch(rawUrl)
        .then(res => {{
          if (!res.ok) throw new Error('Fetch failed');
          return res.text();
        }})
        .then(text => renderMarkdown(text, modalBody))
        .catch(() => {{
          fetch(mdPath)
            .then(res => res.text())
            .then(text => renderMarkdown(text, modalBody))
            .catch(err => {{
              modalBody.innerHTML = `<div class="fallback-card"><h4>Could not load document</h4><p>You can view it directly on <a href="${{ghBtn.href}}" target="_blank" style="color:#58a6ff">GitHub</a>.</p></div>`;
            }});
        }});
    }}

    function renderMarkdown(text, container) {{
      currentRawMdText = text;
      if (typeof marked !== 'undefined') {{
        container.innerHTML = marked.parse(text);
      }} else {{
        container.innerHTML = `<pre style="white-space:pre-wrap;font-family:inherit;">${{escapeHtml(text)}}</pre>`;
      }}
    }}

    function switchModalToMd(path, title) {{
      closeModal(null, 'nb-modal');
      openMdModal(path, title);
    }}

    function copyCurrentMd() {{
      if (currentRawMdText) {{
        navigator.clipboard.writeText(currentRawMdText).then(() => alert('Markdown copied to clipboard!'));
      }}
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

    function closeModal(e, modalId) {{
      if (!modalId) {{
        document.querySelectorAll('.modal-overlay').forEach(m => m.style.display = 'none');
        return;
      }}
      if (!e || e.target.classList.contains('modal-overlay') || e.target.classList.contains('modal-close')) {{
        document.getElementById(modalId).style.display = 'none';
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
      if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {{
        e.preventDefault();
        document.getElementById('topic-search').focus();
      }}
    }});

    // Live Search Logic
    function handleSearch(query) {{
      const q = query.toLowerCase().trim();
      const clearBtn = document.getElementById('search-clear');
      clearBtn.style.display = q ? 'block' : 'none';

      const cards = document.querySelectorAll('.topic-card, .project-card');
      const courses = document.querySelectorAll('.course');

      if (!q) {{
        cards.forEach(c => c.style.display = '');
        courses.forEach(c => c.style.display = '');
        return;
      }}

      courses.forEach(c => {{
        const courseCards = c.querySelectorAll('.topic-card, .project-card');
        let matchedInCourse = 0;
        courseCards.forEach(card => {{
          const title = card.getAttribute('data-title') || '';
          const cardText = card.textContent.toLowerCase();
          if (title.includes(q) || cardText.includes(q)) {{
            card.style.display = '';
            matchedInCourse++;
          }} else {{
            card.style.display = 'none';
          }}
        }});

        if (matchedInCourse > 0 || c.querySelector('h3').textContent.toLowerCase().includes(q)) {{
          c.style.display = '';
          c.classList.add('open');
        }} else {{
          c.style.display = 'none';
        }}
      }});
    }}

    function clearSearch() {{
      const input = document.getElementById('topic-search');
      input.value = '';
      handleSearch('');
      input.focus();
    }}

    // Filter Pills Logic
    function setFilter(cat, pillBtn) {{
      document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      pillBtn.classList.add('active');

      const courses = document.querySelectorAll('.course');
      const cards = document.querySelectorAll('.topic-card, .project-card');

      if (cat === 'all') {{
        courses.forEach(c => c.style.display = '');
        cards.forEach(card => card.style.display = '');
        return;
      }}

      courses.forEach(c => {{
        const courseCat = c.getAttribute('data-course') || '';
        if (courseCat === cat || (cat === 'capstone' && c.id === 'c7') || (cat === 'v2' && c.id === 'v2-and-guides') || (cat === 'c6' && c.id === 'c6')) {{
          c.style.display = '';
          c.classList.add('open');
          c.querySelectorAll('.topic-card, .project-card').forEach(card => card.style.display = '');
        }} else {{
          c.style.display = 'none';
        }}
      }});
    }}

    // Intercept any direct raw markdown or ipynb link clicks
    document.addEventListener('click', function(e) {{
      const a = e.target.closest('a');
      if (!a) return;
      const href = a.getAttribute('href');
      if (!href) return;
      
      // If user clicks a relative .md link not handled inline
      if (href.endsWith('.md') && !href.startsWith('http') && !a.getAttribute('onclick')) {{
        e.preventDefault();
        openMdModal(href, a.textContent.trim() || 'Document View');
      }}

      // If user clicks an unhandled .ipynb link
      if (href.endsWith('.ipynb') && !href.includes('colab.research.google.com') && !href.includes('github.com/')) {{
        e.preventDefault();
        let clean = href.replace(/^(\\.\\/|\\/)/, '');
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
    target.write_text(HTML_CONTENT, encoding="utf-8")
    print(f"🎉 Successfully built premier interactive index.html at {target}!")

if __name__ == "__main__":
    main()
