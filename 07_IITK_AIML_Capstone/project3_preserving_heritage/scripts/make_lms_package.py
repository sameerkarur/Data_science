#!/usr/bin/env python3
"""Build WRITEUP PDF + screenshots zip + lms_upload bundle."""
from __future__ import annotations
import json, zipfile, shutil
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LMS = ROOT / "lms_upload"
LMS.mkdir(exist_ok=True)

# Load metrics if present
metrics = {}
p1 = OUT / "part1_metrics.json"
p2 = OUT / "part2_summary.json"
if p1.exists():
    metrics.update(json.loads(p1.read_text()))
if p2.exists():
    metrics["part2"] = json.loads(p2.read_text())

writeup = (ROOT / "WRITEUP.md").read_text()
# Append result numbers
result_extra = []
if "final_best_val_accuracy" in metrics:
    result_extra.append(
        f"\n\n## Quantified results (from executed run)\n"
        f"- No-aug best val accuracy: {metrics.get('no_aug_best_val_accuracy'):.4f}\n"
        f"- With-aug best val accuracy: {metrics.get('aug_best_val_accuracy'):.4f}\n"
        f"- Final best val accuracy: {metrics.get('final_best_val_accuracy'):.4f}\n"
        f"- Best nature city: {metrics.get('part2', {}).get('best_nature_city')}\n"
        f"- Most liked category: {metrics.get('part2', {}).get('most_liked_category')}\n"
    )
    seed = metrics.get("part2", {}).get("sample_seed_place")
    recs = metrics.get("part2", {}).get("sample_recommendations", [])
    if seed and recs:
        result_extra.append(f"- Sample seed place: {seed}\n")
        for r in recs[:5]:
            result_extra.append(
                f"  - {r.get('Place_Name')} ({r.get('City')}, {r.get('Category')}) sim={r.get('similarity'):.3f}\n"
            )

full_text = writeup + "".join(result_extra)

# PDF
pdf_path = LMS / "WRITEUP_Preserving_Heritage.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=12))

doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER, leftMargin=0.75*inch, rightMargin=0.75*inch,
                        topMargin=0.7*inch, bottomMargin=0.7*inch)
story = []
for line in full_text.splitlines():
    s = line.strip()
    if not s:
        story.append(Spacer(1, 6))
        continue
    if s.startswith("# "):
        story.append(Paragraph(s[2:], styles["TitleCustom"]))
    elif s.startswith("## "):
        story.append(Paragraph(s[3:], styles["H"]))
    elif s.startswith("|") or s.startswith("---"):
        story.append(Preformatted(s, styles["Code"]))
    else:
        # escape XML specials for reportlab
        esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(esc, styles["Body"]))
doc.build(story)
print("PDF:", pdf_path)

# Screenshots zip
shot_names = [
    "class_counts.png",
    "curves_no_aug.png",
    "curves_with_aug.png",
    "eda_age_origins.png",
    "eda_categories_cities.png",
    "eda_liked_categories.png",
]
# include a few sample grids
for p in sorted(OUT.glob("samples_*.png"))[:4]:
    shot_names.append(p.name)

zip_path = LMS / "screenshots.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name in shot_names:
        fp = OUT / name
        if fp.exists():
            zf.write(fp, arcname=name)
    # also include recommendation csv
    for extra in ["sample_recommendations.csv", "part1_metrics.json", "part2_summary.json"]:
        fp = OUT / extra
        if fp.exists():
            zf.write(fp, arcname=extra)
print("ZIP:", zip_path)

# Copy notebook + writeup md
shutil.copy2(ROOT / "preserving_heritage.ipynb", LMS / "preserving_heritage.ipynb")
shutil.copy2(ROOT / "WRITEUP.md", LMS / "WRITEUP.md")
print("LMS contents:", sorted(p.name for p in LMS.iterdir()))
