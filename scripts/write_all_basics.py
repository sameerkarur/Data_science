"""
Orchestrates writing comprehensive, production-grade textbook basics.md files across all curriculum topics.
"""

from pathlib import Path
from basics_c01 import C01_BASICS
from basics_c02 import C02_BASICS_P1
from basics_c02_part2 import C02_BASICS_P2
from basics_c03 import C03_BASICS
from basics_c04 import C04_BASICS
from basics_c05 import C05_BASICS
from basics_c06 import C06_BASICS

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_BASICS = {}
ALL_BASICS.update(C01_BASICS)
ALL_BASICS.update(C02_BASICS_P1)
ALL_BASICS.update(C02_BASICS_P2)
ALL_BASICS.update(C03_BASICS)
ALL_BASICS.update(C04_BASICS)
ALL_BASICS.update(C05_BASICS)
ALL_BASICS.update(C06_BASICS)

written_count = 0
total_lines = 0

for rel_path, markdown_content in ALL_BASICS.items():
    target_dir = REPO_ROOT / rel_path
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_dir / "basics.md"
    cleaned = markdown_content.strip() + "\n"
    target_file.write_text(cleaned, encoding="utf-8")
    written_count += 1
    total_lines += len(cleaned.splitlines())

print(f"🎉 Successfully wrote {written_count} textbook-scale basics.md chapters ({total_lines} total lines) across the repository!")
