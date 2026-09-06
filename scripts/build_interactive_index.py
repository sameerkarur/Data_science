"""
Rebuilds the complete index.html Single Page Application for AI/ML Practice Academy.
Delegates to scripts/build_spa.py which combines marked.umd.js, the Interview Hub,
and the OpenCareerAI design system.
"""

from pathlib import Path
import subprocess
import sys

def main():
    repo_root = Path(__file__).resolve().parents[1]
    builder = repo_root / "scripts" / "build_spa.py"
    res = subprocess.run([sys.executable, str(builder)], cwd=str(repo_root))
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
