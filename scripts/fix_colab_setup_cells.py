"""
Fix Setup cells across all notebooks to make them 100% Google Colab and cloud-ready.
"""

import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]

UNIVERSAL_SETUP_HEADER = """from pathlib import Path
import os, sys, subprocess, urllib.request

# ⚡ Universal Colab & Workspace Setup
def setup_environment():
    if 'google.colab' in sys.modules or os.path.exists('/content'):
        colab_repo = Path('/content/Data_science')
        if not (colab_repo / 'datasets' / 'shared').exists():
            print("🚀 Google Colab detected: Cloning repository from GitHub to load shared datasets...")
            subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/sameerkarur/Data_science.git', str(colab_repo)], check=False)
        if colab_repo.exists():
            os.chdir(str(colab_repo))
            return colab_repo

    p = Path('.').resolve()
    for candidate in [p, *p.parents]:
        if (candidate / 'datasets' / 'shared').exists():
            return candidate
    return Path('.')

REPO_ROOT = setup_environment()
DATA_DIR = REPO_ROOT / 'datasets' / 'shared'

def ensure_dataset(filename):
    local_path = DATA_DIR / filename
    if not local_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_url = f"https://raw.githubusercontent.com/sameerkarur/Data_science/main/datasets/shared/{filename}"
        print(f"📥 Downloading {filename} from GitHub...")
        try:
            urllib.request.urlretrieve(raw_url, str(local_path))
        except Exception as e:
            print(f"Download failed: {e}")
    return local_path

print(f"Repo root : {REPO_ROOT}")
print(f"Datasets  : {DATA_DIR}")
if DATA_DIR.exists():
    print("Available CSVs:", sorted(p.name for p in DATA_DIR.glob('*.csv')))
"""

def update_notebook(nb_path: Path):
    try:
        content = nb_path.read_text(encoding="utf-8")
        nb = json.loads(content)
        modified = False

        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))

            # Case 1: Standard generated setup cell with find_repo_root
            if "def find_repo_root" in src:
                # Retain any extra dataset loading code in this cell
                after_setup = ""
                if "import pandas as pd" in src or "sales_path" in src or "mkt_path" in src or "hr_path" in src:
                    # extract anything after print("Available CSVs...
                    lines = src.splitlines()
                    extra_lines = []
                    recording = False
                    for line in lines:
                        if recording:
                            # Replace old path assignment with ensure_dataset
                            line = re.sub(r"sales_path\s*=\s*DATA_DIR\s*/\s*['\"]AusApparalSales4thQrt2020\.csv['\"]", "sales_path = ensure_dataset('AusApparalSales4thQrt2020.csv')", line)
                            line = re.sub(r"mkt_path\s*=\s*DATA_DIR\s*/\s*['\"]marketing_data\.csv['\"]", "mkt_path = ensure_dataset('marketing_data.csv')", line)
                            line = re.sub(r"hr_path\s*=\s*DATA_DIR\s*/\s*['\"]HR_comma_sep\.csv['\"]", "hr_path = ensure_dataset('HR_comma_sep.csv')", line)
                            extra_lines.append(line)
                        elif "Available CSVs" in line:
                            recording = True
                    after_setup = "\n".join(extra_lines).strip()

                new_src = UNIVERSAL_SETUP_HEADER.strip() + "\n\n" + (after_setup + "\n" if after_setup else "")
                cell["source"] = [s + "\n" for s in new_src.splitlines()]
                modified = True
                break

            # Case 2: NumPy / Pandas / Matplotlib / Seaborn bank setup cells
            elif "Setup - Run this cell first!" in src and "AusApparalSales4thQrt2020.csv" in src:
                colab_block = """# Setup - Run this cell first!
from pathlib import Path
import os, sys, subprocess, urllib.request

if 'google.colab' in sys.modules or os.path.exists('/content'):
    _repo = Path('/content/Data_science')
    if not (_repo / 'datasets' / 'shared').exists():
        print("🚀 Google Colab detected: Cloning repository from GitHub for datasets...")
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/sameerkarur/Data_science.git', str(_repo)], check=False)
    if _repo.exists():
        os.chdir(str(_repo))

def ensure_dataset(filename):
    for candidate in [Path(filename), Path(f'../{filename}'), Path(f'datasets/shared/{filename}'), Path(f'/content/Data_science/datasets/shared/{filename}')]:
        if candidate.exists():
            return candidate
    target = Path('datasets/shared') / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_url = f"https://raw.githubusercontent.com/sameerkarur/Data_science/main/datasets/shared/{filename}"
    print(f"📥 Fetching {filename} from GitHub...")
    urllib.request.urlretrieve(raw_url, str(target))
    return target

csv_path = ensure_dataset('AusApparalSales4thQrt2020.csv')
"""
                # Replace the old df = pd.read_csv('../AusApparalSales4thQrt2020.csv')
                updated_src = colab_block + "\n" + re.sub(r"df\s*=\s*pd\.read_csv\(['\"].*AusApparalSales4thQrt2020\.csv['\"]\)", "df = pd.read_csv(csv_path)", src)
                # Remove duplicate "# Setup - Run this cell first!" if present
                updated_src = re.sub(r"# Setup - Run this cell first!\s*\n# Setup - Run this cell first!", "# Setup - Run this cell first!", updated_src)
                cell["source"] = [s + "\n" for s in updated_src.splitlines()]
                modified = True
                break

        if modified:
            nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error updating {nb_path}: {e}")
        return False

def main():
    count = 0
    for p in sorted(REPO_ROOT.glob("**/*.ipynb")):
        if ".venv" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        if update_notebook(p):
            count += 1
    print(f"✅ Successfully updated {count} notebooks with Universal Colab & Workspace Setup!")

if __name__ == "__main__":
    main()
