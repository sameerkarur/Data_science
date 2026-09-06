"""
Add official Google Colab badges to all Jupyter notebooks in the repo.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPO = "sameerkarur/Data_science"
BRANCH = "main"

def add_colab_badge(nb_path: Path):
    try:
        rel_path = nb_path.relative_to(REPO_ROOT).as_posix()
        # skip virtual envs or checkpoints
        if ".venv" in rel_path or ".ipynb_checkpoints" in rel_path:
            return False

        content = nb_path.read_text(encoding="utf-8")
        nb = json.loads(content)
        
        cells = nb.get("cells", [])
        if not cells:
            return False

        colab_url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/{rel_path}"
        badge_html = f'<a href="{colab_url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>\n\n'

        # Check if already has colab badge
        first_cell = cells[0]
        if first_cell.get("cell_type") == "markdown":
            src = "".join(first_cell.get("source", []))
            if "colab.research.google.com" in src:
                return False  # Already present
            # Prepend badge
            new_source = [badge_html] + (first_cell.get("source", []) if isinstance(first_cell.get("source"), list) else [first_cell.get("source", "")])
            first_cell["source"] = new_source
        else:
            # Insert a new markdown cell at index 0
            new_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [badge_html]
            }
            cells.insert(0, new_cell)

        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error on {nb_path}: {e}")
        return False

def main():
    count = 0
    for p in sorted(REPO_ROOT.glob("**/*.ipynb")):
        if ".venv" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        if add_colab_badge(p):
            count += 1
    print(f"✅ Added Colab badges to {count} notebooks!")

if __name__ == "__main__":
    main()
