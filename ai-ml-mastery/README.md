# AI/ML Mastery Program

A deep-dive AI/ML curriculum (Python → Math → Classical ML → Deep Learning → Transformers → GenAI → Agents → MLOps → System Design), built as a static book site with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/), publishable to GitHub Pages for free.

## What's in here

- `mkdocs.yml` — site config: colorful Material theme, dark/light toggle, search, Mermaid diagrams, math (MathJax), code copy buttons.
- `docs/` — one folder per book (`01-python`, `02-mathematics`, ...). Each has an `index.md` table of contents, and topics get their own `.md` file as they're written.
- `docs/01-python/functions-deep-dive.md` — a fully written example chapter showing the target depth/format (What → Why → How → Math → Code → Internals → Memory → Debugging → Interview → Mastery ladder).
- `.github/workflows/deploy.yml` — automatically builds and publishes the site to GitHub Pages on every push to `main`.

## Run it locally

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve
```

Open `http://127.0.0.1:8000`. It live-reloads as you edit markdown.

## Publish to GitHub Pages

1. Create a new repo on GitHub (e.g. `ai-ml-mastery`) and push this folder:

   ```bash
   git init
   git add .
   git commit -m "Initial scaffold"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. In the repo, edit `mkdocs.yml` and replace `YOUR_USERNAME/YOUR_REPO` in `repo_url` / `repo_name`.
3. On GitHub: **Settings → Pages → Build and deployment → Source → "GitHub Actions"**. (Not "Deploy from a branch" — the workflow handles the build.)
4. Push again (or re-run the workflow from the **Actions** tab). After it finishes, your site is live at:

   ```
   https://YOUR_USERNAME.github.io/YOUR_REPO/
   ```

That's it — every future push to `main` rebuilds and redeploys automatically.

## Adding a new topic/chapter

1. Pick the right book folder, e.g. `docs/02-mathematics/`.
2. Create a new file, e.g. `eigenvalues-deep-dive.md`, following the structure in `docs/01-python/functions-deep-dive.md`.
3. Add it to `mkdocs.yml`'s `nav:` list (or link it from that book's `index.md` — both work; explicit `nav` entries give more control over ordering).
4. Commit + push. The site rebuilds automatically.

## Recommended way to keep building this with Claude

Ask for **one chapter at a time**, e.g. "write the deep-dive for Book 2 → Linear Algebra → Eigenvalues and Eigenvectors, same depth/format as the Python functions chapter." Thousands of pages can't be generated in one shot — but the scaffold, TOC, and template here are built so you (or Claude) can fill it in incrementally without ever losing consistency.
