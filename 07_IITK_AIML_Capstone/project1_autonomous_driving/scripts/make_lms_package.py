#!/usr/bin/env python3
"""Build WRITEUP PDF + screenshots zip + lms_upload bundle for Project 1."""
from __future__ import annotations
import json, zipfile, shutil
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LMS = ROOT / "lms_upload"
LMS.mkdir(exist_ok=True)

metrics = {}
p1 = OUT / "part1_metrics.json"
p2 = OUT / "part2_summary.json"
if p1.exists():
    metrics.update(json.loads(p1.read_text()))
if p2.exists():
    metrics["part2"] = json.loads(p2.read_text())

writeup = (ROOT / "WRITEUP.md").read_text()
extra = []
if "val_accuracy" in metrics:
    extra.append("\n\n## Quantified results (from executed run)\n")
    extra.append(f"- Crop classifier val accuracy: **{metrics['val_accuracy']*100:.2f}%**\n")
    if "coarse_val_accuracy" in metrics:
        extra.append(f"- Coarse-group val accuracy: **{metrics['coarse_val_accuracy']*100:.2f}%**\n")
    extra.append(f"- Train/val crops: {metrics.get('n_train_crops')}/{metrics.get('n_val_crops')} · classes: {metrics.get('num_classes')}\n")
p2m = metrics.get("part2", {})
if p2m:
    extra.append(f"- Tesla events analyzed: {p2m.get('n_events')}\n")
    extra.append(f"- Tesla driver death events: {p2m.get('tesla_driver_death_events')} ({p2m.get('tesla_driver_death_pct'):.1f}%)\n")
    extra.append(f"- Other-vehicle collisions: {p2m.get('other_vehicle_collision_events')} ({p2m.get('other_vehicle_collision_pct'):.1f}%)\n")
    extra.append(f"- Verified Autopilot death events: {p2m.get('verified_ap_death_events')}\n")

full_text = writeup + "".join(extra)

pdf_path = LMS / "WRITEUP_Autonomous_Driving.pdf"
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=12))

doc = SimpleDocTemplate(
    str(pdf_path), pagesize=LETTER,
    leftMargin=0.75*inch, rightMargin=0.75*inch,
    topMargin=0.7*inch, bottomMargin=0.7*inch,
)
story = []
for line in full_text.splitlines():
    s = line.strip()
    if not s:
        story.append(Spacer(1, 6))
        continue
    if s.startswith("# "):
        story.append(Paragraph(s[2:].replace("&", "&amp;"), styles["TitleCustom"]))
    elif s.startswith("## "):
        story.append(Paragraph(s[3:].replace("&", "&amp;"), styles["H"]))
    elif s.startswith("|") or s.startswith("---") or s.startswith("```"):
        story.append(Preformatted(s, styles["Code"]))
    else:
        esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # bold markdown
        while "**" in esc:
            esc = esc.replace("**", "<b>", 1).replace("**", "</b>", 1)
        story.append(Paragraph(esc, styles["Body"]))
doc.build(story)
print("PDF:", pdf_path)

shot_names = [
    "part1_class_counts.png",
    "part1_training_curves.png",
    "part1_confusion_matrix.png",
    "part1_inference_grid.png",
    "part2_events_date_geo.png",
    "part2_events_monthly.png",
    "part2_death_aspects.png",
    "part2_models.png",
    "part2_verified_autopilot.png",
]
for p in sorted(OUT.glob("inference_*.jpg"))[:6]:
    shot_names.append(p.name)

zip_path = LMS / "Screenshots.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name in shot_names:
        fp = OUT / name
        if fp.exists():
            zf.write(fp, arcname=name)
    for extra_f in ["part1_metrics.json", "part2_summary.json", "part1_classification_report.txt"]:
        fp = OUT / extra_f
        if fp.exists():
            zf.write(fp, arcname=extra_f)
print("ZIP:", zip_path)

shutil.copy2(ROOT / "autonomous_driving.ipynb", LMS / "autonomous_driving.ipynb")
shutil.copy2(ROOT / "WRITEUP.md", LMS / "WRITEUP.md")
print("LMS contents:", sorted(p.name for p in LMS.iterdir()))
