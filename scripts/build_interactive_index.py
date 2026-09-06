"""
Generate the Premier Next-Generation Learning Platform SPA index.html
Inspired by modern developer platforms (OpenCareerAI, Vercel, Linear, LeetCode).

Key Features:
1. 100% Self-Contained Built-in Zero-Dependency Fast Markdown & Interactive Q&A Engine (No external CDNs that can fail on htmlpreview or behind CSP!)
2. Interactive Q&A Flashcard Studio: Question badges, "Reveal Answer" toggles, "Mark Mastered" progress tracker (stored in localStorage), Q&A search filter.
3. Clean Document Reader: Renders tables, callout blocks, code blocks with copy buttons, math, typography — ZERO unparsed raw hashtags (#), asterisks (**), or symbols.
4. Unified Multi-Tab Topic Studio Modal:
   - 🎯 Tab 1: Interview Q&A Flashcard Studio
   - 📖 Tab 2: Basics & Conceptual Guide
   - 💻 Tab 3: Interactive Jupyter Notebook & Cell Viewer
   - 💡 Tab 4: 1-Click Solutions in Colab
5. Top Navigation App Bar with Category Views:
   - 📚 Courses & Labs
   - ⚡ Projects V2 Suite (15 Architectures)
   - 🏆 Capstones
   - 🎯 Interview Flashcards Hub
   - 📖 Master Guides & Theory
6. Custom Domain & 100% Free Forever Hosting Support
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
    
    escaped_title = html.escape(title)
    
    return f"""          <div class="topic-card" data-category="{category}" data-title="{html.escape(title.lower())}">
            <div class="topic-header">
              <span class="topic-title">{escaped_title}</span>
              <span class="q-badge">{html.escape(q_count)}</span>
            </div>
            <div class="topic-actions">
              <a class="btn-colab" href="{c_url}" target="_blank" rel="noopener noreferrer" title="Execute & practice live in Google Colab">
                <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run in Colab">
              </a>
              <button class="btn-studio" onclick="openStudio('{p_path}', '{escaped_title}', '{s_path}', '{b_path}', '{q_path}', 'notebook')">
                💻 Practice Lab
              </button>
            </div>
            <div class="files">
              <a href="#" onclick="openStudio('{p_path}', '{escaped_title}', '{s_path}', '{b_path}', '{q_path}', 'basics'); return false;">📖 Basics</a> · 
              <a href="{s_url}" target="_blank" title="Launch complete solutions in Colab">💡 Solutions</a> · 
              <a href="#" onclick="openStudio('{p_path}', '{escaped_title}', '{s_path}', '{b_path}', '{q_path}', 'qa'); return false;">🎯 Interview Q&amp;A</a>
            </div>
          </div>"""

def proj_card(title: str, nb_path: str, category: str, writeup_path: str = None) -> str:
    c_url = colab_link(nb_path)
    g_url = gh_blob_link(nb_path)
    escaped_title = html.escape(title)
    writeup_html = f' · <a class="proj-writeup" href="#" onclick="openStandaloneDoc(\'{writeup_path}\', \'{escaped_title} — Deliverable Writeup\'); return false;">📄 Writeup</a>' if writeup_path else ""
    return f"""          <div class="project-card" data-category="{category}" data-title="{html.escape(title.lower())}">
            <span class="proj-title">{escaped_title}</span>
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
  <title>AI/ML Practice Academy — Sameer Karur</title>
  <link rel="icon" href="https://colab.research.google.com/assets/colab-badge.svg" type="image/svg+xml">
  <style>
    :root {{
      --bg: #090d13;
      --card: #131922;
      --card-hover: #1b2330;
      --border: #263040;
      --border-focus: #3b82f6;
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.2);
      --blue: #3b82f6;
      --blue-glow: rgba(59, 130, 246, 0.2);
      --purple: #8b5cf6;
      --amber: #f59e0b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --code-bg: #06090e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      overflow-x: hidden;
    }}

    /* Top App Navigation Bar */
    .app-navbar {{
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0.75rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .nav-brand {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: #fff;
    }}
    .brand-icon {{
      width: 34px;
      height: 34px;
      background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      box-shadow: 0 0 15px var(--accent-glow);
    }}
    .brand-text h1 {{
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: -0.3px;
    }}
    .brand-text span {{
      font-size: 0.75rem;
      color: var(--muted);
      display: block;
    }}

    .nav-tabs {{
      display: flex;
      align-items: center;
      gap: 0.35rem;
      background: rgba(0, 0, 0, 0.35);
      padding: 0.25rem;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}
    .nav-tab {{
      background: transparent;
      border: none;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.4rem 0.85rem;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.15s;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .nav-tab:hover {{ color: #fff; background: rgba(255, 255, 255, 0.05); }}
    .nav-tab.active {{
      background: var(--blue);
      color: #fff;
      box-shadow: 0 0 12px var(--blue-glow);
    }}

    .nav-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .btn-domain-info {{
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid var(--accent);
      color: #34d399;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s;
    }}
    .btn-domain-info:hover {{
      background: var(--accent);
      color: #000;
    }}

    /* Hero Section */
    .hero {{
      padding: 2.75rem 1.5rem 1.75rem;
      text-align: center;
      position: relative;
    }}
    .hero-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #93c5fd;
      padding: 0.25rem 0.95rem;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      margin-bottom: 1rem;
    }}
    .hero h2 {{
      font-size: 2.35rem;
      font-weight: 800;
      letter-spacing: -0.8px;
      margin-bottom: 0.6rem;
      background: linear-gradient(90deg, #60a5fa, #34d399, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .hero p {{
      color: var(--muted);
      max-width: 720px;
      margin: 0 auto 1.5rem;
      font-size: 1rem;
    }}

    /* Stats Ribbon */
    .stats-ribbon {{
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 1rem;
      max-width: 1000px;
      margin: 0 auto;
    }}
    .stat-card {{
      background: var(--card);
      border: 1px solid var(--border);
      padding: 0.7rem 1.25rem;
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-width: 140px;
    }}
    .stat-val {{
      font-size: 1.35rem;
      font-weight: 800;
      color: #60a5fa;
    }}
    .stat-lbl {{
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 600;
    }}

    /* Global Search & Filter Bar */
    .controls-panel {{
      max-width: 1140px;
      margin: 2rem auto 1.5rem;
      padding: 0 1.25rem;
    }}
    .search-wrapper {{
      position: relative;
      margin-bottom: 1rem;
    }}
    .search-icon {{
      position: absolute;
      left: 1.1rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--muted);
      font-size: 1.05rem;
    }}
    #global-search {{
      width: 100%;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 3rem 0.85rem 2.85rem;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s;
    }}
    #global-search:focus {{
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px var(--blue-glow);
    }}
    .search-hint {{
      position: absolute;
      right: 1.1rem;
      top: 50%;
      transform: translateY(-50%);
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 0.1rem 0.45rem;
      font-size: 0.75rem;
      color: var(--muted);
    }}

    .category-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .cat-pill {{
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 600;
      padding: 0.35rem 0.85rem;
      border-radius: 999px;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .cat-pill:hover {{ color: #fff; border-color: #475569; }}
    .cat-pill.active {{
      background: var(--accent);
      color: #000;
      border-color: var(--accent);
      font-weight: 700;
    }}

    /* Main Container */
    main {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 0 1.25rem 3rem;
    }}

    /* Course Cards */
    .course-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      margin-bottom: 1.25rem;
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .course-section:hover {{ border-color: #475569; }}
    .course-head {{
      padding: 1.15rem 1.5rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
      background: rgba(19, 25, 34, 0.6);
    }}
    .course-head:hover {{ background: var(--card-hover); }}
    .course-head-left {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .course-head-left h3 {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #f8fafc;
    }}
    .course-count-badge {{
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #93c5fd;
      font-size: 0.75rem;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-weight: 600;
    }}
    .course-body {{
      padding: 1.25rem 1.5rem 1.5rem;
      display: none;
      border-top: 1px solid var(--border);
      background: rgba(9, 13, 19, 0.4);
    }}
    .course-section.open .course-body {{ display: block; }}

    /* Subtopics Grid */
    .subtopics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 0.85rem;
    }}
    .topic-card {{
      background: #0f151e;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s;
    }}
    .topic-card:hover {{
      transform: translateY(-2px);
      border-color: #3b82f6;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
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
      color: #f1f5f9;
    }}
    .q-badge {{
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
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
    .btn-colab img {{ height: 22px; }}
    .btn-studio {{
      background: #1e293b;
      border: 1px solid var(--border);
      color: #e2e8f0;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.25rem 0.7rem;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.15s;
    }}
    .btn-studio:hover {{
      background: #334155;
      color: #fff;
      border-color: #64748b;
    }}

    .files {{
      font-size: 0.82rem;
      color: var(--muted);
      border-top: 1px solid #1e293b;
      padding-top: 0.6rem;
      margin-top: 0.25rem;
    }}
    .files a {{
      color: #60a5fa;
      text-decoration: none;
      cursor: pointer;
    }}
    .files a:hover {{ text-decoration: underline; }}

    /* Projects Grid */
    .projects-container {{
      border-top: 1px solid var(--border);
      padding-top: 1.25rem;
      margin-top: 1.25rem;
    }}
    .projects-container h4 {{
      font-size: 0.85rem;
      margin-bottom: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      font-weight: 700;
    }}
    .projects-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 0.75rem;
    }}
    .project-card {{
      background: #0f151e;
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
      color: #f1f5f9;
    }}
    .proj-actions {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      font-size: 0.82rem;
    }}
    .btn-colab-small img {{ height: 18px; vertical-align: middle; }}
    .proj-gh {{
      color: #60a5fa;
      text-decoration: none;
    }}
    .proj-gh:hover {{ text-decoration: underline; }}
    .proj-writeup {{
      color: #34d399;
      text-decoration: none;
      cursor: pointer;
    }}
    .proj-writeup:hover {{ text-decoration: underline; }}

    /* SECTION HEADERS */
    .section-title {{
      font-size: 1.25rem;
      font-weight: 700;
      margin: 2.5rem 0 1rem;
      color: #60a5fa;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    /* UNIFIED STUDIO MODAL */
    .studio-modal {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(3, 7, 18, 0.82);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 1.5rem;
    }}
    .studio-dialog {{
      background: #111827;
      border: 1px solid #374151;
      border-radius: 16px;
      width: 100%;
      max-width: 1020px;
      height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
      overflow: hidden;
    }}
    .studio-header {{
      padding: 1rem 1.5rem;
      background: #0f172a;
      border-bottom: 1px solid #1e293b;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .studio-header-title {{
      display: flex;
      flex-direction: column;
    }}
    .studio-header-title h3 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #f8fafc;
    }}
    .studio-header-title span {{
      font-size: 0.8rem;
      color: var(--muted);
    }}
    .studio-header-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    .studio-close-btn {{
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 1.75rem;
      cursor: pointer;
      padding: 0 0.5rem;
      line-height: 1;
    }}
    .studio-close-btn:hover {{ color: #fff; }}

    /* Studio Sub-Tabs */
    .studio-tabs-bar {{
      padding: 0.6rem 1.5rem;
      background: #1e293b;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .studio-nav-tabs {{
      display: flex;
      gap: 0.5rem;
    }}
    .s-tab {{
      background: transparent;
      border: 1px solid transparent;
      color: #94a3b8;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.4rem 0.85rem;
      border-radius: 6px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}
    .s-tab:hover {{ color: #fff; }}
    .s-tab.active {{
      background: #0f172a;
      color: #38bdf8;
      border-color: #38bdf8;
    }}

    .studio-quick-links {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}
    .btn-colab-launch {{
      background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
      color: #000 !important;
      font-weight: 700;
      font-size: 0.82rem;
      padding: 0.35rem 0.85rem;
      border-radius: 6px;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      box-shadow: 0 2px 6px rgba(245, 158, 11, 0.3);
    }}
    .btn-colab-launch:hover {{ transform: scale(1.02); }}

    .studio-body {{
      padding: 1.75rem;
      overflow-y: auto;
      flex: 1;
      background: #090d13;
    }}

    /* Q&A FLASHCARD STUDIO COMPONENTS */
    .qa-studio-controls {{
      background: #131922;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1.25rem;
      margin-bottom: 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
    }}
    .qa-filter-input {{
      background: var(--code-bg);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.45rem 0.85rem;
      border-radius: 6px;
      font-size: 0.85rem;
      width: 260px;
      outline: none;
    }}
    .qa-filter-input:focus {{ border-color: var(--blue); }}
    .qa-ctrl-btns {{
      display: flex;
      gap: 0.5rem;
    }}
    .qa-ctrl-btn {{
      background: #1e293b;
      border: 1px solid var(--border);
      color: #cbd5e1;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
    }}
    .qa-ctrl-btn:hover {{ background: #334155; color: #fff; }}

    .qa-card {{
      background: #131922;
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 1rem;
      overflow: hidden;
      transition: all 0.2s;
    }}
    .qa-card:hover {{ border-color: #475569; }}
    .qa-card-head {{
      padding: 1rem 1.25rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      background: #161e2a;
    }}
    .qa-card-head-left {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex: 1;
    }}
    .qa-num-badge {{
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%);
      border: 1px solid rgba(59, 130, 246, 0.4);
      color: #60a5fa;
      font-size: 0.75rem;
      font-weight: 800;
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      white-space: nowrap;
    }}
    .qa-question-text {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #f8fafc;
    }}
    .qa-reveal-btn {{
      background: rgba(59, 130, 246, 0.1);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #93c5fd;
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      white-space: nowrap;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .qa-card.open .qa-reveal-btn {{
      background: #1e293b;
      color: #94a3b8;
    }}
    .qa-card-body {{
      padding: 1.25rem;
      display: none;
      border-top: 1px solid var(--border);
      background: #0b0f17;
    }}
    .qa-card.open .qa-card-body {{ display: block; }}
    .qa-answer-box {{
      background: rgba(16, 185, 129, 0.04);
      border-left: 3px solid #10b981;
      padding: 1rem 1.25rem;
      border-radius: 0 8px 8px 0;
      margin-bottom: 0.85rem;
      font-size: 0.92rem;
      line-height: 1.7;
      color: #e2e8f0;
    }}
    .qa-answer-tag {{
      font-size: 0.72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #34d399;
      margin-bottom: 0.4rem;
    }}
    .qa-card-footer {{
      display: flex;
      justify-content: flex-end;
      gap: 0.5rem;
      padding-top: 0.5rem;
    }}
    .qa-action-btn {{
      background: transparent;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.75rem;
      padding: 0.25rem 0.65rem;
      border-radius: 4px;
      cursor: pointer;
    }}
    .qa-action-btn:hover {{ color: #fff; border-color: #64748b; }}
    .qa-action-btn.mastered {{
      background: rgba(245, 158, 11, 0.15);
      border-color: #f59e0b;
      color: #fbbf24;
    }}

    /* CLEAN DOCUMENT VIEWER (MARKDOWN STYLED) */
    .doc-viewer {{
      line-height: 1.75;
      font-size: 0.95rem;
      color: #cbd5e1;
      max-width: 900px;
      margin: 0 auto;
    }}
    .doc-viewer h1 {{
      font-size: 1.65rem;
      font-weight: 800;
      color: #f8fafc;
      margin: 1.5rem 0 0.75rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .doc-viewer h2 {{
      font-size: 1.35rem;
      font-weight: 700;
      color: #60a5fa;
      margin: 1.75rem 0 0.6rem;
    }}
    .doc-viewer h3 {{
      font-size: 1.12rem;
      font-weight: 600;
      color: #34d399;
      margin: 1.25rem 0 0.5rem;
    }}
    .doc-viewer p {{ margin-bottom: 0.85rem; }}
    .doc-viewer ul, .doc-viewer ol {{
      margin: 0.5rem 0 1rem 1.5rem;
    }}
    .doc-viewer li {{ margin-bottom: 0.35rem; }}
    .doc-callout {{
      background: rgba(59, 130, 246, 0.08);
      border-left: 4px solid #3b82f6;
      border-radius: 0 8px 8px 0;
      padding: 0.85rem 1.2rem;
      margin: 1.25rem 0;
      display: flex;
      gap: 0.75rem;
      align-items: flex-start;
    }}
    .callout-icon {{ font-size: 1.2rem; line-height: 1; }}
    .callout-content {{ color: #93c5fd; font-size: 0.92rem; }}
    .doc-table-wrap {{
      overflow-x: auto;
      margin: 1.25rem 0;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .doc-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .doc-table th, .doc-table td {{
      padding: 0.65rem 0.95rem;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    .doc-table th {{
      background: #161e2a;
      color: #60a5fa;
      font-weight: 700;
    }}
    .doc-code-block {{
      background: #06090e;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 1rem 0;
      overflow: hidden;
    }}
    .doc-code-head {{
      background: #111827;
      padding: 0.3rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      font-size: 0.75rem;
      color: var(--muted);
    }}
    .doc-code-pre {{
      padding: 0.85rem 1rem;
      overflow-x: auto;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.85rem;
      color: #e2e8f0;
      margin: 0;
    }}
    .doc-code-inline {{
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      padding: 0.15rem 0.4rem;
      font-size: 0.85rem;
      color: #93c5fd;
    }}

    /* NOTEBOOK CELL VIEWER */
    .cell-box {{
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 1.25rem;
      overflow: hidden;
    }}
    .cell-md-box {{
      padding: 1rem 1.25rem;
      background: #0f151e;
    }}
    .cell-code-top {{
      background: #161e2a;
      border-bottom: 1px solid var(--border);
      padding: 0.35rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .cell-code-content {{
      background: #06090e;
      padding: 0.85rem 1rem;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.85rem;
      overflow-x: auto;
      color: #e2e8f0;
    }}

    /* DOMAIN CONFIG MODAL */
    .domain-modal {{
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(3, 7, 18, 0.82);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 2000;
      padding: 1.5rem;
    }}
    .domain-dialog {{
      background: #111827;
      border: 1px solid #374151;
      border-radius: 14px;
      width: 100%;
      max-width: 680px;
      max-height: 85vh;
      overflow-y: auto;
      padding: 2rem;
      box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
    }}
    .domain-dialog h3 {{
      font-size: 1.35rem;
      color: #60a5fa;
      margin-bottom: 0.5rem;
    }}
    .domain-dialog p {{
      color: #cbd5e1;
      font-size: 0.92rem;
      margin-bottom: 1rem;
    }}
    .domain-step {{
      background: #161e2a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-bottom: 0.85rem;
    }}
    .domain-step strong {{ color: #34d399; display: block; margin-bottom: 0.35rem; }}
    .domain-step pre {{
      background: #06090e;
      padding: 0.65rem 0.85rem;
      border-radius: 6px;
      font-size: 0.82rem;
      color: #93c5fd;
      overflow-x: auto;
      margin-top: 0.5rem;
    }}

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

  <!-- Top App Navigation Bar -->
  <nav class="app-navbar">
    <a href="#" class="nav-brand">
      <div class="brand-icon">⚡</div>
      <div class="brand-text">
        <h1>AI/ML Academy Hub</h1>
        <span>IITK AIML Professional Certificate · Sameer Karur</span>
      </div>
    </a>

    <div class="nav-tabs">
      <button class="nav-tab active" onclick="switchNavSection('all', this)">🌐 All Modules</button>
      <button class="nav-tab" onclick="switchNavSection('courses', this)">📚 Courses (1–7)</button>
      <button class="nav-tab" onclick="switchNavSection('v2', this)">⚡ Projects V2</button>
      <button class="nav-tab" onclick="switchNavSection('capstones', this)">🏆 Capstones</button>
      <button class="nav-tab" onclick="switchNavSection('interview', this)">🎯 Interview Hub</button>
      <button class="nav-tab" onclick="switchNavSection('theory', this)">📖 Master Guides</button>
    </div>

    <div class="nav-actions">
      <button class="btn-domain-info" onclick="openDomainModal()">🌐 Custom Domain &amp; Free Hosting</button>
    </div>
  </nav>

  <!-- Hero Section -->
  <section class="hero">
    <div class="hero-badge">
      <span>🚀 IIT Kanpur LMS-Aligned</span> · <span>100% Free Forever Cloud Compute</span>
    </div>
    <h2>Machine Learning &amp; Generative AI Practice Studio</h2>
    <p>Complete curriculum from Python foundations to Deep Learning, Large Language Models, RAG Architectures, and Multi-Modal Vision AI.</p>

    <div class="stats-ribbon">
      <div class="stat-card"><span class="stat-val">1,650+</span><span class="stat-lbl">Practice Problems</span></div>
      <div class="stat-card"><span class="stat-val">300+</span><span class="stat-lbl">Interview Flashcards</span></div>
      <div class="stat-card"><span class="stat-val">15</span><span class="stat-lbl">Next-Gen V2 Projects</span></div>
      <div class="stat-card"><span class="stat-val">3</span><span class="stat-lbl">Capstone Deliverables</span></div>
      <div class="stat-card"><span class="stat-val">100%</span><span class="stat-lbl">Free Google Colab</span></div>
    </div>
  </section>

  <!-- Global Search & Category Filters -->
  <div class="controls-panel">
    <div class="search-wrapper">
      <span class="search-icon">🔍</span>
      <input type="text" id="global-search" placeholder="Search 1,650+ problems, algorithms, or projects (e.g. 'RAG', 'K-Means', 'XGBoost', 'NumPy', 'Colab')..." oninput="handleGlobalSearch(this.value)">
      <span class="search-hint">Press / to focus</span>
    </div>

    <div class="category-pills">
      <button class="cat-pill active" onclick="setCategoryFilter('all', this)">All Categories</button>
      <button class="cat-pill" onclick="setCategoryFilter('python', this)">Python (250+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('ds', this)">Data Science (670+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('ml', this)">Machine Learning (250+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('dl', this)">Deep Learning (200+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('genai', this)">Generative AI (150+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('c6', this)">Advanced GenAI &amp; RAG (150+ Q)</button>
      <button class="cat-pill" onclick="setCategoryFilter('capstone', this)">Capstones</button>
      <button class="cat-pill" onclick="setCategoryFilter('v2', this)">Version 2 Suite</button>
    </div>
  </div>

  <main>
    <div id="section-courses-group">
      <h3 class="section-title">📚 Mandatory Courses (1 to 7)</h3>

      <!-- Induction -->
      <section class="course-section" id="ind" data-course="python">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Induction Session</h3>
            <span class="course-count-badge">Program induction</span>
          </div>
        </div>
        <div class="course-body">
          <div style="margin-top:0.5rem">
            <a href="#" onclick="openStandaloneDoc('00_Induction_Session/README.md', 'Induction Session Overview'); return false;" style="color:#60a5fa;font-weight:600">📖 Induction Session for Professional Certificate Course in AI and Machine Learning</a>
          </div>
        </div>
      </section>

      <!-- Course 1 -->
      <section class="course-section open" id="c1" data-course="python">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 1 — IITK AIML Foundations: Programming Refresher</h3>
            <span class="course-count-badge">5 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
{topic_card('Variables, Data Types & Operators', '01_IITK_AIML_Foundations_Programming_Refresher/01_variables_datatypes', 'python', '50 Problems')}
{topic_card('Control Flow & Functions', '01_IITK_AIML_Foundations_Programming_Refresher/02_control_flow_functions', 'python', '50 Problems')}
{topic_card('Data Structures (Lists, Dicts, Tuples)', '01_IITK_AIML_Foundations_Programming_Refresher/03_data_structures', 'python', '50 Problems')}
{topic_card('OOP & Modules', '01_IITK_AIML_Foundations_Programming_Refresher/04_oop_modules', 'python', '50 Problems')}
{topic_card('File I/O & Exceptions', '01_IITK_AIML_Foundations_Programming_Refresher/05_file_io_exceptions', 'python', '50 Problems')}
          </div>
          <div class="projects-container">
            <h4>Course 1 Deliverables</h4>
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
      <section class="course-section open" id="c2" data-course="ds">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 2 — IITK AIML Core: Applied Data Science with Python</h3>
            <span class="course-count-badge">11 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
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
          <div class="projects-container">
            <h4>Course 2 Deliverables</h4>
            <div class="projects-grid">
{proj_card('Marketing Campaigns Analysis (EDA &amp; Hypothesis Testing)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Marketing_Campaigns/Marketing_Campaigns_Analysis.ipynb', 'ds')}
{proj_card('Sales Analysis (Seasonal Decomposition &amp; RFM)', '02_IITK_AIML_Core_Applied_Data_Science_with_Python/projects/Sales_Analysis/Sales_Analysis.ipynb', 'ds')}
            </div>
          </div>
        </div>
      </section>

      <!-- Course 3 -->
      <section class="course-section" id="c3" data-course="ml">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 3 — IITK AIML Core: Machine Learning</h3>
            <span class="course-count-badge">5 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
{topic_card('EDA & Feature Engineering', '03_IITK_AIML_Core_Machine_Learning/01_eda_feature_engineering', 'ml', '50 Problems')}
{topic_card('Unsupervised Clustering (K-Means, DBSCAN)', '03_IITK_AIML_Core_Machine_Learning/02_clustering', 'ml', '50 Problems')}
{topic_card('Supervised Classification (Trees, Ensembles)', '03_IITK_AIML_Core_Machine_Learning/03_classification', 'ml', '50 Problems')}
{topic_card('Imbalanced Data Handling & SMOTE', '03_IITK_AIML_Core_Machine_Learning/04_imbalanced_data', 'ml', '50 Problems')}
{topic_card('Model Evaluation & Hyperparameter Tuning', '03_IITK_AIML_Core_Machine_Learning/05_model_evaluation', 'ml', '50 Problems')}
          </div>
          <div class="projects-container">
            <h4>Course 3 Deliverables</h4>
            <div class="projects-grid">
{proj_card('Song Cohorts Clustering (Rolling Stone Top 500 Spotify)', '03_IITK_AIML_Core_Machine_Learning/projects/Creating_Cohorts_of_Songs/Creating_Cohorts_of_Songs.ipynb', 'ml')}
{proj_card('Employee Turnover Analytics (GBDT &amp; Explainability)', '03_IITK_AIML_Core_Machine_Learning/projects/Employee_Turnover_Analytics/Employee_Turnover_Analytics.ipynb', 'ml')}
            </div>
          </div>
        </div>
      </section>

      <!-- Course 4 -->
      <section class="course-section" id="c4" data-course="dl">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 4 — IITK AIML - Core: Deep Learning with Keras and TensorFlow</h3>
            <span class="course-count-badge">4 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
{topic_card('Neural Network Basics (Forward/Backprop, Activation)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/01_neural_network_basics', 'dl', '50 Problems')}
{topic_card('Keras & TensorFlow Architecture', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/02_keras_tensorflow', 'dl', '50 Problems')}
{topic_card('DL Preprocessing & Class Imbalance (Focal Loss)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/03_preprocessing_imbalance', 'dl', '50 Problems')}
{topic_card('DL Model Evaluation (ROC-AUC, Calibration)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/04_model_evaluation_dl', 'dl', '50 Problems')}
          </div>
          <div class="projects-container">
            <h4>Course 4 Deliverables</h4>
            <div class="projects-grid">
{proj_card('Lending Club Loan Default Prediction (Deep MLP)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Lending_Club_Loan_Analysis/Lending_Club_Loan_Analysis.ipynb', 'dl')}
{proj_card('Home Loan Credit Risk Modeling (Keras Tabular Net)', '04_IITK_AIML_Core_Deep_Learning_with_Keras_and_TensorFlow/projects/Home_Loan_Data_Analysis/Home_Loan_Data_Analysis.ipynb', 'dl')}
            </div>
          </div>
        </div>
      </section>

      <!-- Course 5 -->
      <section class="course-section" id="c5" data-course="genai">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 5 — IITK AIML Core: Essentials of Generative AI, Prompt Engineering &amp; ChatGPT</h3>
            <span class="course-count-badge">3 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
{topic_card('Prompt Engineering Fundamentals', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/01_prompt_engineering', 'genai', '50 Problems')}
{topic_card('ChatGPT Enterprise Applications', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/02_chatgpt_applications', 'genai', '50 Problems')}
{topic_card('GenAI Optimization & Guardrails', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/03_genai_optimization', 'genai', '50 Problems')}
          </div>
          <div class="projects-container">
            <h4>Course 5 Deliverables</h4>
            <div class="projects-grid">
{proj_card('ChatGPT-Based Interactive Storytelling Engine', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/ChatGPT_Based_Storytelling/ChatGPT_Based_Storytelling.ipynb', 'genai')}
{proj_card('Virtual Project Management Consultant (Multi-Role AI)', '05_IITK_AIML_Core_Essentials_of_Generative_AI_Prompt_Engineering_ChatGPT/projects/Virtual_Project_Management_Consultant/Virtual_Project_Management_Consultant.ipynb', 'genai')}
            </div>
          </div>
        </div>
      </section>

      <!-- Course 6 -->
      <section class="course-section open" id="c6" data-course="c6">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Course 6 — IITK AIML - Advanced Generative AI</h3>
            <span class="course-count-badge">3 Practice Topics · 2 Projects</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
{topic_card('RAG Architectures & Retrieval Engineering', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures', 'c6', '50 Problems')}
{topic_card('Vector Databases & ChromaDB', '06_IITK_AIML_Advanced_Generative_AI/02_vector_databases_chroma', 'c6', '50 Problems')}
{topic_card('Multimodal Generative Models & Vision AI', '06_IITK_AIML_Advanced_Generative_AI/03_multimodal_generative_models', 'c6', '50 Problems')}
          </div>
          <div class="projects-container">
            <h4>Course 6 Deliverables</h4>
            <div class="projects-grid">
{proj_card('Nestlé HR Policy Assistant (LangChain RAG &amp; ChromaDB)', '06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/hr_assistant.ipynb', 'c6', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project1_hr_assistant/WRITEUP.md')}
{proj_card('Netflix Marketing Creative Studio (DALL·E &amp; Gradio UI)', '06_IITK_AIML_Advanced_Generative_AI/project2_designs/netflix_design_generator.ipynb', 'c6', writeup_path='06_IITK_AIML_Advanced_Generative_AI/project2_designs/WRITEUP.md')}
            </div>
          </div>
          <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem">
            <a href="#" onclick="openStandaloneDoc('06_IITK_AIML_Advanced_Generative_AI/README.md', 'Course 6 Overview &amp; LMS Upload Checklist'); return false;" style="color:#34d399">📖 Course 6 Overview &amp; LMS Upload Checklist</a>
          </p>
        </div>
      </section>
    </div>

    <!-- Course 7 Capstones -->
    <div id="section-capstones-group">
      <h3 class="section-title">🏆 Course 7 — IITK AIML Capstone Deliverables</h3>
      <section class="course-section open" id="c7" data-course="capstone">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Capstone Projects Portfolio</h3>
            <span class="course-count-badge">3 Completed Capstones</span>
          </div>
        </div>
        <div class="course-body">
          <div class="projects-grid">
{proj_card('Capstone 1: Autonomous Driving Perception (Transfer Learning MobileNetV2)', '07_IITK_AIML_Capstone/project1_autonomous_driving/lms_upload/autonomous_driving.ipynb', 'capstone')}
{proj_card('Capstone 2: Restaurant Demand &amp; Sales Forecasting (XGBoost Ensemble)', '07_IITK_AIML_Capstone/project2_sales_forecasting/lms_upload/sales_forecasting.ipynb', 'capstone')}
{proj_card('Capstone 3: Cultural Heritage Tourism AI (ResNet Landmark Classifier &amp; SVD)', '07_IITK_AIML_Capstone/project3_preserving_heritage/lms_upload/preserving_heritage.ipynb', 'capstone')}
          </div>
          <p style="color:var(--muted);margin-top:0.85rem;font-size:0.88rem">
            <a href="#" onclick="openStandaloneDoc('07_IITK_AIML_Capstone/README.md', 'Capstone Overview &amp; Submission Instructions'); return false;" style="color:#34d399">📖 Capstone Overview &amp; LMS Upload Instructions</a>
          </p>
        </div>
      </section>
    </div>

    <!-- Dual Architecture & Projects V2 -->
    <div id="section-v2-group">
      <h3 class="section-title">⚡ Projects Version 2 Suite (15 Next-Gen Architectures)</h3>
      <section class="course-section open" id="v2-and-guides" data-course="v2">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left">
            <h3>Alternative Architectural Paradigms</h3>
            <span class="course-count-badge">15 V2 Architectures</span>
          </div>
        </div>
        <div class="course-body">
          <div class="subtopics-grid">
            <div class="topic-card" data-category="v2" data-title="master aiml learning guide">
              <div class="topic-header">
                <span class="topic-title">Master AI/ML Learning Guide</span>
                <span class="q-badge">Comprehensive Theory</span>
              </div>
              <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Theoretical derivations, linear algebra, loss functions, optimizer math &amp; 100+ interview Q&amp;As.</p>
              <div class="files"><a href="#" onclick="openStandaloneDoc('LEARNING_GUIDE.md', 'Master AI/ML Learning Guide'); return false;" style="font-weight:600;color:#60a5fa">📖 Read LEARNING_GUIDE.md</a></div>
            </div>
            <div class="topic-card" data-category="v2" data-title="projects version 2 portfolio suite of 15">
              <div class="topic-header">
                <span class="topic-title">Projects Version 2 (Full Suite)</span>
                <span class="q-badge">15 Next-Gen Projects</span>
              </div>
              <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Advanced paradigms (Event-sourcing, DAG schedulers, TabNet, Hybrid RAG, EfficientNet, BPR).</p>
              <div class="files"><a href="#" onclick="openStandaloneDoc('projects_version2/README.md', 'Projects Version 2 Architectural Suite'); return false;" style="font-weight:600;color:#60a5fa">📂 Explore projects_version2/</a></div>
            </div>
            <div class="topic-card" data-category="v2" data-title="practice guide and study workflow">
              <div class="topic-header">
                <span class="topic-title">Practice Guide &amp; Study Workflow</span>
                <span class="q-badge">8-Week Roadmap</span>
              </div>
              <p style="font-size:0.85rem;color:var(--muted);margin-bottom:0.75rem">Mastery calendar, question banks distribution &amp; problem-solving methodology.</p>
              <div class="files"><a href="#" onclick="openStandaloneDoc('PRACTICE_GUIDE.md', 'Practice Guide &amp; Workflow'); return false;" style="font-weight:600;color:#60a5fa">📋 Read PRACTICE_GUIDE.md</a></div>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- Electives Section -->
    <div id="section-electives-group">
      <h3 class="section-title">🎓 Advanced Electives (Fall 2026)</h3>
      <section class="course-section" id="e1" data-course="electives">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left"><h3>Elective 1 — Advanced Deep Learning &amp; Computer Vision</h3></div>
        </div>
        <div class="course-body"><div style="margin-top:0.5rem"><a href="#" onclick="openStandaloneDoc('Elective_01_ADL_and_Computer_Vision/README.md', 'Elective 1: ADL &amp; Computer Vision'); return false;" style="color:#60a5fa">📖 Elective 1 Overview</a></div></div>
      </section>
      <section class="course-section" id="e2" data-course="electives">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left"><h3>Elective 2 — NLP and Speech Recognition</h3><span class="course-count-badge">Registered · Sep 19–Nov 1</span></div>
        </div>
        <div class="course-body"><div style="margin-top:0.5rem"><a href="#" onclick="openStandaloneDoc('Elective_02_NLP_and_Speech_Recognition/README.md', 'Elective 2: NLP and Speech Recognition'); return false;" style="color:#60a5fa">📖 Elective 2 Overview</a></div></div>
      </section>
      <section class="course-section" id="e3" data-course="electives">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left"><h3>Elective 3 — Reinforcement Learning</h3><span class="course-count-badge">Registered · Sep 19–Nov 1</span></div>
        </div>
        <div class="course-body"><div style="margin-top:0.5rem"><a href="#" onclick="openStandaloneDoc('Elective_03_Reinforcement_Learning/README.md', 'Elective 3: Reinforcement Learning'); return false;" style="color:#60a5fa">📖 Elective 3 Overview</a></div></div>
      </section>
      <section class="course-section" id="e4" data-course="electives">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left"><h3>Elective 4 — Microsoft Azure AI Fundamentals</h3></div>
        </div>
        <div class="course-body"><div style="margin-top:0.5rem"><a href="#" onclick="openStandaloneDoc('Elective_04_Microsoft_Azure_AI_Fundamentals/README.md', 'Elective 4: Azure AI Fundamentals'); return false;" style="color:#60a5fa">📖 Elective 4 Overview</a></div></div>
      </section>
      <section class="course-section" id="e5" data-course="electives">
        <div class="course-head" onclick="this.parentElement.classList.toggle('open')">
          <div class="course-head-left"><h3>Elective 5 — Academic Masterclass by IIT Kanpur</h3><span class="course-count-badge">Faculty Masterclass</span></div>
        </div>
        <div class="course-body"><div style="margin-top:0.5rem"><a href="#" onclick="openStandaloneDoc('Elective_05_Academic_Masterclass_by_IIT_Kanpur/README.md', 'Academic Masterclass by IIT Kanpur'); return false;" style="color:#60a5fa">📖 Elective 5 Overview</a></div></div>
      </section>
    </div>
  </main>

  <!-- UNIFIED WORKSPACE STUDIO MODAL -->
  <div id="studio-modal" class="studio-modal" onclick="closeStudioModal(event)">
    <div class="studio-dialog" onclick="event.stopPropagation()">
      <div class="studio-header">
        <div class="studio-header-title">
          <h3 id="studio-title">Topic Studio</h3>
          <span id="studio-subtitle">IITK AIML Interactive Learning Environment</span>
        </div>
        <div class="studio-header-actions">
          <button class="studio-close-btn" onclick="closeStudioModal()">&times;</button>
        </div>
      </div>

      <div class="studio-tabs-bar">
        <div class="studio-nav-tabs">
          <button id="tab-btn-qa" class="s-tab active" onclick="switchStudioTab('qa')">🎯 Interview Q&amp;A Flashcards</button>
          <button id="tab-btn-basics" class="s-tab" onclick="switchStudioTab('basics')">📖 Basics &amp; Concepts</button>
          <button id="tab-btn-notebook" class="s-tab" onclick="switchStudioTab('notebook')">💻 Practice Notebook</button>
        </div>
        <div class="studio-quick-links">
          <a id="studio-colab-link" class="btn-colab-launch" href="#" target="_blank" rel="noopener noreferrer">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
            <span>🚀 Run in Colab</span>
          </a>
          <a id="studio-sol-link" class="qa-ctrl-btn" href="#" target="_blank">💡 Solutions (Colab)</a>
          <a id="studio-gh-link" class="qa-ctrl-btn" href="#" target="_blank">👁️ GitHub</a>
        </div>
      </div>

      <div id="studio-body" class="studio-body">
        <!-- Content dynamically injected here -->
      </div>
    </div>
  </div>

  <!-- STANDALONE DOCUMENT MODAL -->
  <div id="standalone-modal" class="studio-modal" onclick="closeStandaloneModal(event)">
    <div class="studio-dialog" onclick="event.stopPropagation()">
      <div class="studio-header">
        <div class="studio-header-title">
          <h3 id="standalone-title">Document Reader</h3>
          <span id="standalone-path">Formatted Markdown View</span>
        </div>
        <div class="studio-header-actions">
          <button class="qa-ctrl-btn" onclick="copyCurrentDocument()">📋 Copy Document</button>
          <a id="standalone-gh-btn" class="qa-ctrl-btn" href="#" target="_blank">👁️ GitHub</a>
          <button class="studio-close-btn" onclick="closeStandaloneModal()">&times;</button>
        </div>
      </div>
      <div id="standalone-body" class="studio-body">
        <!-- Markdown injected here -->
      </div>
    </div>
  </div>

  <!-- CUSTOM DOMAIN & FREE HOSTING MODAL -->
  <div id="domain-modal" class="domain-modal" onclick="closeDomainModal(event)">
    <div class="domain-dialog" onclick="event.stopPropagation()">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
        <div>
          <h3>🌐 Run This Platform on Your Domain for Free Forever</h3>
          <p>You can connect your own domain (e.g. <code>learn.yourdomain.com</code> or <code>yourdomain.com</code>) with 100% free hosting and free auto-renewing SSL certificate.</p>
        </div>
        <button class="studio-close-btn" onclick="closeDomainModal()">&times;</button>
      </div>

      <div class="domain-step">
        <strong>Step 1: Free Hosting &amp; Zero Backend Bills</strong>
        <p>This entire academy runs as a high-performance Client-Side Single Page Application (SPA). GitHub Pages hosts it with global CDN caching for $0 forever. Google Colab provides the free cloud GPUs and compute. There is never any monthly server or maintenance cost.</p>
      </div>

      <div class="domain-step">
        <strong>Step 2: Add CNAME in Repository Root</strong>
        <p>Create a file named <code>CNAME</code> in your repository root with your domain name:</p>
        <pre>learn.yourdomain.com</pre>
      </div>

      <div class="domain-step">
        <strong>Step 3: Point Your DNS Record (GoDaddy, Namecheap, Cloudflare, etc.)</strong>
        <p>In your domain registrar DNS settings:</p>
        <pre>Type: CNAME
Host: learn (or @ for root domain)
Points to: sameerkarur.github.io</pre>
      </div>

      <div class="domain-step">
        <strong>Step 4: Enable Free HTTPS in GitHub</strong>
        <p>Go to GitHub Repo Settings &rarr; <strong>Pages</strong> &rarr; enter your custom domain &rarr; check <strong>"Enforce HTTPS"</strong>. GitHub will automatically provision and renew a free Let's Encrypt SSL certificate!</p>
      </div>

      <button class="qa-ctrl-btn" style="width:100%; padding:0.65rem; margin-top:0.5rem;" onclick="closeDomainModal()">Got it! Close</button>
    </div>
  </div>

  <footer>
    Built with ❤️ by <a href="https://github.com/sameerkarur" style="color:#34d399;text-decoration:none">Sameer Karur</a> · IITK AIML Program · Star ⭐ on GitHub
  </footer>

  <script>
    const GITHUB_REPO = "{GITHUB_REPO}";
    const BRANCH = "{BRANCH}";

    // Studio State
    let currentTopicState = {{
      practicePath: '',
      title: '',
      solutionsPath: '',
      basicsPath: '',
      qaPath: '',
      activeTab: 'qa',
      cache: {{}}
    }};

    let currentStandaloneText = '';

    // ==========================================
    // 1. BUILT-IN PURE JAVASCRIPT MARKDOWN PARSER (ZERO EXTERNAL DEPENDENCY)
    // ==========================================
    function parseMarkdown(md) {{
      if (!md) return '';
      let text = md.replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n');

      // 1. Code blocks (```lang ... ```)
      const codeBlocks = [];
      text = text.replace(/```([a-zA-Z0-9_-]*)\\n([\\s\\S]*?)```/g, (match, lang, code) => {{
        const placeholder = `__CODE_BLOCK_${{codeBlocks.length}}__`;
        const langName = lang ? lang.toUpperCase() : 'CODE';
        const escaped = escapeHtml(code.trim());
        const blockHtml = `
          <div class="doc-code-block">
            <div class="doc-code-head">
              <span>${{langName}}</span>
              <button class="qa-action-btn" onclick="copyCodeText(this)">📋 Copy</button>
            </div>
            <pre class="doc-code-pre"><code>${{escaped}}</code></pre>
          </div>
        `;
        codeBlocks.push(blockHtml);
        return placeholder;
      }});

      // 2. Tables
      text = text.replace(/((?:\\|[^\\n]+\\|\\n)+)/g, (match) => {{
        const lines = match.trim().split('\\n');
        if (lines.length < 2) return match;
        // Check if second line is divider
        if (!lines[1].includes('---')) return match;

        const parseRow = (line) => {{
          return line.split('|').slice(1, -1).map(c => c.trim());
        }};

        const headers = parseRow(lines[0]);
        let tableHtml = '<div class="doc-table-wrap"><table class="doc-table"><thead><tr>';
        headers.forEach(h => {{
          tableHtml += `<th>${{parseInline(h)}}</th>`;
        }});
        tableHtml += '</tr></thead><tbody>';

        for (let i = 2; i < lines.length; i++) {{
          const cols = parseRow(lines[i]);
          tableHtml += '<tr>';
          cols.forEach(c => {{
            tableHtml += `<td>${{parseInline(c)}}</td>`;
          }});
          tableHtml += '</tr>';
        }}
        tableHtml += '</tbody></table></div>';
        return tableHtml;
      }});

      // 3. Blockquotes (> ...)
      text = text.replace(/^(?:>\\s*(.*)\\n?)+/gm, (match) => {{
        const content = match.replace(/^>\\s?/gm, '').trim();
        return `<div class="doc-callout"><div class="callout-icon">💡</div><div class="callout-content">${{parseInline(content)}}</div></div>`;
      }});

      // 4. Headers (#, ##, ###, ####)
      text = text.replace(/^#### (.*$)/gim, '<h4 class="doc-h4">$1</h4>');
      text = text.replace(/^### (.*$)/gim, '<h3 class="doc-h3">$1</h3>');
      text = text.replace(/^## (.*$)/gim, '<h2 class="doc-h2">$1</h2>');
      text = text.replace(/^# (.*$)/gim, '<h1 class="doc-h1">$1</h1>');

      // 5. Horizontal rules
      text = text.replace(/^---$/gm, '<hr style="border:none;border-top:1px solid #1e293b;margin:1.5rem 0;">');

      // 6. Lists
      text = text.replace(/^(?:[*-]\\s+(.*)\\n?)+/gm, (match) => {{
        const items = match.trim().split('\\n').map(li => `<li>${{parseInline(li.replace(/^[*-]\\s+/, ''))}}</li>`).join('');
        return `<ul>${{items}}</ul>`;
      }});
      text = text.replace(/^(?:\\d+\\.\\s+(.*)\\n?)+/gm, (match) => {{
        const items = match.trim().split('\\n').map(li => `<li>${{parseInline(li.replace(/^\\d+\\.\\s+/, ''))}}</li>`).join('');
        return `<ol>${{items}}</ol>`;
      }});

      // 7. Paragraphs
      const paragraphs = text.split(/\\n{2,}/);
      const parsedP = paragraphs.map(p => {{
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<h') || p.startsWith('<div') || p.startsWith('<ul') || p.startsWith('<ol') || p.startsWith('__CODE_BLOCK_') || p.startsWith('<hr')) {{
          return p;
        }}
        return `<p>${{parseInline(p)}}</p>`;
      }}).join('\\n');

      // Restore code blocks
      let finalHtml = parsedP;
      codeBlocks.forEach((block, idx) => {{
        finalHtml = finalHtml.replace(`__CODE_BLOCK_${{idx}}__`, block);
      }});

      return `<div class="doc-viewer">${{finalHtml}}</div>`;
    }}

    function parseInline(str) {{
      if (!str) return '';
      return str
        .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="doc-code-inline">$1</code>')
        .replace(/\\[([^\\]]+)\\]\\(([^\\)]+)\\)/g, '<a href="$2" target="_blank" style="color:#60a5fa">$1 ↗</a>');
    }}

    // ==========================================
    // 2. INTERACTIVE INTERVIEW Q&A FLASHCARD PARSER
    // ==========================================
    function parseInterviewQA(rawText, topicTitle) {{
      if (!rawText) return '<div class="qa-card"><div class="qa-card-body">No questions found.</div></div>';
      
      const lines = rawText.split('\\n');
      const questions = [];
      let currentQ = null;

      lines.forEach((line) => {{
        const trimmed = line.trim();
        // Match ### Q1. or ### Q1: or ### Q1
        const qMatch = trimmed.match(/^###\\s+Q(\\d+)[.:]?\\s*(.*)/i);
        if (qMatch) {{
          if (currentQ) questions.push(currentQ);
          currentQ = {{
            num: qMatch[1],
            question: qMatch[2] || `Question ${{qMatch[1]}}`,
            answerLines: []
          }};
        }} else if (currentQ) {{
          currentQ.answerLines.push(line);
        }}
      }});
      if (currentQ) questions.push(currentQ);

      // If no ### Q format found (e.g. Course 1 alternative format)
      if (questions.length === 0) {{
        return parseMarkdown(rawText);
      }}

      let html = `
        <div class="qa-studio-controls">
          <input type="text" class="qa-filter-input" placeholder="🔍 Filter ${{questions.length}} interview questions..." oninput="filterQAQuestions(this.value)">
          <div class="qa-ctrl-btns">
            <button class="qa-ctrl-btn" onclick="toggleAllQACards(true)">👁️ Expand All</button>
            <button class="qa-ctrl-btn" onclick="toggleAllQACards(false)">🔒 Collapse All</button>
            <button class="qa-ctrl-btn" onclick="copyAllQA('${{escapeHtml(topicTitle)}}')">📋 Copy All</button>
          </div>
        </div>
        <div id="qa-cards-list">
      `;

      questions.forEach((q) => {{
        const rawAns = q.answerLines.join('\\n').trim();
        // Clean up "**Answer:**" tag if present
        const cleanAns = rawAns.replace(/^\\*\\*Answer:\\*\\*\\s*/i, '');
        const parsedAns = parseMarkdown(cleanAns);
        const cardId = `${{topicTitle.replace(/[^a-zA-Z0-9]/g, '_')}}_q${{q.num}}`;
        const isMastered = localStorage.getItem(`mastered_${{cardId}}`) === 'true';

        html += `
          <div class="qa-card" id="qa-card-${{q.num}}" data-qnum="${{q.num}}" data-text="${{escapeHtml((q.question + ' ' + cleanAns).toLowerCase())}}">
            <div class="qa-card-head" onclick="toggleQACard(this.parentElement)">
              <div class="qa-card-head-left">
                <span class="qa-num-badge">Q${{q.num}}</span>
                <span class="qa-question-text">${{escapeHtml(q.question)}}</span>
              </div>
              <button class="qa-reveal-btn">👁️ Reveal Answer</button>
            </div>
            <div class="qa-card-body">
              <div class="qa-answer-box">
                <div class="qa-answer-tag">💡 Model Answer &amp; Conceptual Breakdown</div>
                ${{parsedAns}}
              </div>
              <div class="qa-card-footer">
                <button class="qa-action-btn" onclick="copySingleQAPair(this)">📋 Copy Question &amp; Answer</button>
                <button class="qa-action-btn ${{isMastered ? 'mastered' : ''}}" onclick="toggleMasterCard(this, '${{cardId}}')">
                  ${{isMastered ? '⭐ Mastered' : '☆ Mark as Mastered'}}
                </button>
              </div>
            </div>
          </div>
        `;
      }});

      html += '</div>';
      return html;
    }}

    // ==========================================
    // 3. WORKSPACE STUDIO MODAL CONTROLLER
    // ==========================================
    function openStudio(practicePath, title, solutionsPath, basicsPath, qaPath, defaultTab) {{
      currentTopicState = {{
        practicePath,
        title,
        solutionsPath,
        basicsPath,
        qaPath,
        activeTab: defaultTab || 'qa',
        cache: {{}}
      }};

      document.getElementById('studio-title').textContent = title;
      document.getElementById('studio-subtitle').textContent = `IITK AIML Interactive Studio · ${{practicePath.split('/')[0]}}`;

      const colabUrl = `https://colab.research.google.com/github/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{practicePath}}`;
      const solColabUrl = `https://colab.research.google.com/github/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{solutionsPath}}`;
      const ghUrl = `https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{practicePath}}`;

      document.getElementById('studio-colab-link').href = colabUrl;
      document.getElementById('studio-sol-link').href = solColabUrl;
      document.getElementById('studio-gh-link').href = ghUrl;

      document.getElementById('studio-modal').style.display = 'flex';
      switchStudioTab(currentTopicState.activeTab);
    }}

    function switchStudioTab(tab) {{
      currentTopicState.activeTab = tab;
      document.querySelectorAll('.s-tab').forEach(t => t.classList.remove('active'));
      const activeBtn = document.getElementById(`tab-btn-${{tab}}`);
      if (activeBtn) activeBtn.classList.add('active');

      const body = document.getElementById('studio-body');
      body.innerHTML = `
        <div style="text-align:center; padding:3rem; color:var(--muted)">
          <div style="font-size:2rem; margin-bottom:0.5rem">⚡</div>
          <h4>Loading ${{tab.toUpperCase()}} Environment...</h4>
        </div>
      `;

      if (tab === 'qa') {{
        loadStudioQA();
      }} else if (tab === 'basics') {{
        loadStudioBasics();
      }} else if (tab === 'notebook') {{
        loadStudioNotebook();
      }}
    }}

    function loadStudioQA() {{
      const path = currentTopicState.qaPath;
      const body = document.getElementById('studio-body');
      if (!path) {{
        body.innerHTML = '<div class="qa-card"><div class="qa-card-body">Interview Q&A not found for this topic.</div></div>';
        return;
      }}
      if (currentTopicState.cache[path]) {{
        body.innerHTML = currentTopicState.cache[path];
        return;
      }}
      fetchFile(path).then(text => {{
        const html = parseInterviewQA(text, currentTopicState.title);
        currentTopicState.cache[path] = html;
        body.innerHTML = html;
      }}).catch(err => {{
        body.innerHTML = `<div class="qa-card"><div class="qa-card-body">Failed to load Interview Q&A. You can view on <a href="https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{path}}" target="_blank" style="color:#60a5fa">GitHub</a>.</div></div>`;
      }});
    }}

    function loadStudioBasics() {{
      const path = currentTopicState.basicsPath;
      const body = document.getElementById('studio-body');
      if (!path) {{
        body.innerHTML = '<div class="qa-card"><div class="qa-card-body">Basics documentation not found.</div></div>';
        return;
      }}
      if (currentTopicState.cache[path]) {{
        body.innerHTML = currentTopicState.cache[path];
        return;
      }}
      fetchFile(path).then(text => {{
        const html = parseMarkdown(text);
        currentTopicState.cache[path] = html;
        body.innerHTML = html;
      }}).catch(err => {{
        body.innerHTML = `<div class="qa-card"><div class="qa-card-body">Failed to load basics. View on <a href="https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{path}}" target="_blank" style="color:#60a5fa">GitHub</a>.</div></div>`;
      }});
    }}

    function loadStudioNotebook() {{
      const path = currentTopicState.practicePath;
      const body = document.getElementById('studio-body');
      const colabUrl = document.getElementById('studio-colab-link').href;
      
      fetchFile(path).then(text => {{
        try {{
          const nb = JSON.parse(text);
          renderNotebookInStudio(nb, body, colabUrl);
        }} catch(e) {{
          body.innerHTML = '<p>Unable to parse notebook structure.</p>';
        }}
      }}).catch(err => {{
        body.innerHTML = `
          <div style="background:#131922; border:1px solid var(--border); border-radius:12px; padding:2.5rem; text-align:center;">
            <h4>🚀 Ready to Practice Live in Google Colab</h4>
            <p style="color:var(--muted); margin:0.75rem 0 1.5rem">Click below to open the complete notebook with cloud GPU/CPU compute:</p>
            <a class="btn-colab-launch" href="${{colabUrl}}" target="_blank">Launch in Google Colab 🚀</a>
          </div>
        `;
      }});
    }}

    function renderNotebookInStudio(nb, container, colabUrl) {{
      const cells = nb.cells || [];
      let html = `
        <div style="background:rgba(59,130,246,0.1); border:1px solid #3b82f6; border-radius:8px; padding:0.85rem 1.25rem; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
          <span style="color:#93c5fd; font-size:0.9rem">💡 <strong>${{cells.length}} Notebook Cells</strong> available. Run code interactively:</span>
          <a class="btn-colab-launch" href="${{colabUrl}}" target="_blank">Run in Colab 🚀</a>
        </div>
      `;

      cells.forEach((cell, idx) => {{
        const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        if (src.includes('colab-badge.svg')) return;

        if (cell.cell_type === 'markdown') {{
          html += `
            <div class="cell-box">
              <div class="cell-md-box">${{parseMarkdown(src)}}</div>
            </div>
          `;
        }} else if (cell.cell_type === 'code') {{
          html += `
            <div class="cell-box">
              <div class="cell-code-top">
                <span>[${{cell.execution_count || ' '}}] Python Code</span>
                <button class="qa-action-btn" onclick="copyCodeText(this)">📋 Copy</button>
              </div>
              <pre class="cell-code-content"><code>${{escapeHtml(src)}}</code></pre>
            </div>
          `;
        }}
      }});

      container.innerHTML = html;
    }}

    // ==========================================
    // 4. STANDALONE DOCUMENT MODAL
    // ==========================================
    function openStandaloneDoc(path, title) {{
      const modal = document.getElementById('standalone-modal');
      const titleElem = document.getElementById('standalone-title');
      const pathElem = document.getElementById('standalone-path');
      const ghBtn = document.getElementById('standalone-gh-btn');
      const body = document.getElementById('standalone-body');

      titleElem.textContent = title;
      pathElem.textContent = path;
      ghBtn.href = `https://github.com/${{GITHUB_REPO}}/blob/${{BRANCH}}/${{path}}`;

      body.innerHTML = `
        <div style="text-align:center; padding:3rem; color:var(--muted)">
          <div style="font-size:2rem; margin-bottom:0.5rem">⚡</div>
          <h4>Loading Document...</h4>
        </div>
      `;
      modal.style.display = 'flex';

      fetchFile(path).then(text => {{
        currentStandaloneText = text;
        body.innerHTML = parseMarkdown(text);
      }}).catch(err => {{
        body.innerHTML = `<div class="qa-card"><div class="qa-card-body">Could not load document. View on <a href="${{ghBtn.href}}" target="_blank" style="color:#60a5fa">GitHub</a>.</div></div>`;
      }});
    }}

    // Utility fetcher (supports raw github with relative fallback)
    function fetchFile(path) {{
      const rawUrl = `https://raw.githubusercontent.com/${{GITHUB_REPO}}/${{BRANCH}}/${{path}}`;
      return fetch(rawUrl).then(res => {{
        if (!res.ok) throw new Error('Network error');
        return res.text();
      }}).catch(() => {{
        return fetch(path).then(res => res.text());
      }});
    }}

    // ==========================================
    // 5. INTERACTION & EVENT HANDLERS
    // ==========================================
    function toggleQACard(card) {{
      card.classList.toggle('open');
      const btn = card.querySelector('.qa-reveal-btn');
      if (btn) {{
        btn.textContent = card.classList.contains('open') ? '🔒 Hide Answer' : '👁️ Reveal Answer';
      }}
    }}

    function toggleAllQACards(expand) {{
      document.querySelectorAll('.qa-card').forEach(card => {{
        if (expand) {{
          card.classList.add('open');
          const btn = card.querySelector('.qa-reveal-btn');
          if (btn) btn.textContent = '🔒 Hide Answer';
        }} else {{
          card.classList.remove('open');
          const btn = card.querySelector('.qa-reveal-btn');
          if (btn) btn.textContent = '👁️ Reveal Answer';
        }}
      }});
    }}

    function filterQAQuestions(q) {{
      const query = q.toLowerCase().trim();
      document.querySelectorAll('.qa-card').forEach(card => {{
        const text = card.getAttribute('data-text') || '';
        if (!query || text.includes(query)) {{
          card.style.display = '';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }}

    function toggleMasterCard(btn, cardId) {{
      const isMastered = btn.classList.contains('mastered');
      if (isMastered) {{
        btn.classList.remove('mastered');
        btn.textContent = '☆ Mark as Mastered';
        localStorage.removeItem(`mastered_${{cardId}}`);
      }} else {{
        btn.classList.add('mastered');
        btn.textContent = '⭐ Mastered';
        localStorage.setItem(`mastered_${{cardId}}`, 'true');
      }}
    }}

    function copySingleQAPair(btn) {{
      const card = btn.closest('.qa-card');
      const q = card.querySelector('.qa-question-text').innerText;
      const a = card.querySelector('.qa-answer-box').innerText;
      navigator.clipboard.writeText(`Question: ${{q}}\\n\\n${{a}}`).then(() => {{
        btn.textContent = '✅ Copied!';
        setTimeout(() => btn.textContent = '📋 Copy Question & Answer', 2000);
      }});
    }}

    function copyAllQA(title) {{
      let full = `# ${{title}}\\n\\n`;
      document.querySelectorAll('.qa-card').forEach(card => {{
        const q = card.querySelector('.qa-question-text').innerText;
        const a = card.querySelector('.qa-answer-box').innerText;
        full += `Q: ${{q}}\\n${{a}}\\n\\n---\\n\\n`;
      }});
      navigator.clipboard.writeText(full).then(() => alert('All questions and answers copied to clipboard!'));
    }}

    function copyCodeText(btn) {{
      const code = btn.closest('.doc-code-block, .cell-box').querySelector('code').innerText;
      navigator.clipboard.writeText(code).then(() => {{
        btn.textContent = '✅ Copied!';
        setTimeout(() => btn.textContent = '📋 Copy', 2000);
      }});
    }}

    function copyCurrentDocument() {{
      if (currentStandaloneText) {{
        navigator.clipboard.writeText(currentStandaloneText).then(() => alert('Document copied to clipboard!'));
      }}
    }}

    function closeStudioModal(e) {{
      if (!e || e.target.classList.contains('studio-modal') || e.target.classList.contains('studio-close-btn')) {{
        document.getElementById('studio-modal').style.display = 'none';
      }}
    }}

    function closeStandaloneModal(e) {{
      if (!e || e.target.classList.contains('studio-modal') || e.target.classList.contains('studio-close-btn')) {{
        document.getElementById('standalone-modal').style.display = 'none';
      }}
    }}

    function openDomainModal() {{
      document.getElementById('domain-modal').style.display = 'flex';
    }}

    function closeDomainModal(e) {{
      if (!e || e.target.classList.contains('domain-modal') || e.target.classList.contains('studio-close-btn') || e.target.tagName === 'BUTTON') {{
        document.getElementById('domain-modal').style.display = 'none';
      }}
    }}

    // Navigation & Section Filter
    function switchNavSection(sec, btn) {{
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');

      const coursesSec = document.getElementById('section-courses-group');
      const capstonesSec = document.getElementById('section-capstones-group');
      const v2Sec = document.getElementById('section-v2-group');
      const electivesSec = document.getElementById('section-electives-group');

      if (sec === 'all') {{
        coursesSec.style.display = '';
        capstonesSec.style.display = '';
        v2Sec.style.display = '';
        electivesSec.style.display = '';
      }} else if (sec === 'courses') {{
        coursesSec.style.display = '';
        capstonesSec.style.display = 'none';
        v2Sec.style.display = 'none';
        electivesSec.style.display = 'none';
      }} else if (sec === 'v2') {{
        coursesSec.style.display = 'none';
        capstonesSec.style.display = 'none';
        v2Sec.style.display = '';
        electivesSec.style.display = 'none';
      }} else if (sec === 'capstones') {{
        coursesSec.style.display = 'none';
        capstonesSec.style.display = '';
        v2Sec.style.display = 'none';
        electivesSec.style.display = 'none';
      }} else if (sec === 'interview') {{
        // Auto open studio on Course 6 Q&A as showcase
        coursesSec.style.display = '';
        capstonesSec.style.display = 'none';
        v2Sec.style.display = 'none';
        electivesSec.style.display = 'none';
        openStudio('06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/practice.ipynb', 'RAG Architectures & Retrieval Engineering', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/solutions.ipynb', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/basics.md', '06_IITK_AIML_Advanced_Generative_AI/01_rag_architectures/interview_qa.md', 'qa');
      }} else if (sec === 'theory') {{
        openStandaloneDoc('LEARNING_GUIDE.md', 'Master AI/ML Learning Guide');
      }}
    }}

    // Global Search
    function handleGlobalSearch(q) {{
      const query = q.toLowerCase().trim();
      const cards = document.querySelectorAll('.topic-card, .project-card');
      const courses = document.querySelectorAll('.course-section');

      if (!query) {{
        cards.forEach(c => c.style.display = '');
        courses.forEach(c => c.style.display = '');
        return;
      }}

      courses.forEach(c => {{
        const courseCards = c.querySelectorAll('.topic-card, .project-card');
        let matched = 0;
        courseCards.forEach(card => {{
          const title = card.getAttribute('data-title') || '';
          const cardText = card.textContent.toLowerCase();
          if (title.includes(query) || cardText.includes(query)) {{
            card.style.display = '';
            matched++;
          }} else {{
            card.style.display = 'none';
          }}
        }});

        if (matched > 0 || c.textContent.toLowerCase().includes(query)) {{
          c.style.display = '';
          c.classList.add('open');
        }} else {{
          c.style.display = 'none';
        }}
      }});
    }}

    function setCategoryFilter(cat, btn) {{
      document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');

      const courses = document.querySelectorAll('.course-section');
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

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') {{
        closeStudioModal();
        closeStandaloneModal();
        closeDomainModal();
      }}
      if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {{
        e.preventDefault();
        const searchInput = document.getElementById('global-search');
        if (searchInput) searchInput.focus();
      }}
    }});

    function escapeHtml(text) {{
      const div = document.createElement('div');
      div.textContent = text || '';
      return div.innerHTML;
    }}
  </script>
</body>
</html>
"""

def main():
    target = REPO_ROOT / "index.html"
    target.write_text(HTML_CONTENT, encoding="utf-8")
    print(f"🎉 Successfully built premier learning platform SPA at {target}!")

if __name__ == "__main__":
    main()
