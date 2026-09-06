# IITK AIML - Advanced Generative AI

**LMS name:** IITK AIML - Advanced Generative AI  
**Course number:** Course 6 (mandatory)  
**Note:** This is *not* the Capstone. Capstone is Course 7 (`07_IITK_AIML_Capstone/`).

Course-end projects built from your Downloads materials.

## Project locations

| Project | Notebook | Dataset |
|--------|----------|---------|
| **1. HR Assistant (Nestlé RAG)** | `project1_hr_assistant/hr_assistant.ipynb` | `project1_hr_assistant/Dataset/the_nestle_hr_policy_pdf_2012.pdf` |
| **2. Netflix Design Generator** | `project2_designs/netflix_design_generator.ipynb` | N/A (prompt-based) |

## Setup

```bash
cd 06_IITK_AIML_Advanced_Generative_AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notebooks auto-load your key from `~/Documents/opencareerai-secrets.json` (`ai_apis.openai_api_key`), or set `OPENAI_API_KEY`.

## Run

```bash
# Project 1
jupyter notebook project1_hr_assistant/hr_assistant.ipynb

# Project 2
jupyter notebook project2_designs/netflix_design_generator.ipynb

# Or both via helper
python run_both_projects.py
```

## LMS submission checklist

For **each** project upload:

1. **Source code:** the `.ipynb` notebook  
2. **Writeup:** convert `WRITEUP.md` to PDF/DOC  
3. **Screenshots:** UI + sample outputs  

## Notes

- Project 1 creates a local `chroma_hr_db/` folder (safe to delete and rebuild).  
- Image generation uses the account’s available image model (e.g. `gpt-image-1` if DALL·E is unavailable).  
