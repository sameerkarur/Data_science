#!/usr/bin/env python3
"""Embed execution outputs (metrics + key charts) into autonomous_driving.ipynb."""
from __future__ import annotations
import base64, json
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_output

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
nb_path = ROOT / "autonomous_driving.ipynb"
nb = nbf.read(nb_path, as_version=4)

metrics = json.loads((OUT / "part1_metrics.json").read_text())
p2 = json.loads((OUT / "part2_summary.json").read_text())
report = (OUT / "part1_classification_report.txt").read_text()

def png_b64(name: str) -> str | None:
    p = OUT / name
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")

def clear_outputs(nb):
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

clear_outputs(nb)

# Attach rich outputs to the last code cell (deliverables checklist summary)
summary_text = f"""
============================================================
CAPSTONE PROJECT 1 — COMPLETE (embedded results)
============================================================
Crop classifier val accuracy: {metrics['val_accuracy']*100:.2f}%
Coarse-group val accuracy: {metrics.get('coarse_val_accuracy', 0)*100:.2f}%
Classes: {metrics['num_classes']} {metrics['class_names']}
Train/Val crops: {metrics['n_train_crops']}/{metrics['n_val_crops']}

Tesla events: {p2['n_events']} | total deaths: {p2['total_deaths']:.0f} | mean/event: {p2['mean_deaths_per_event']:.2f}
Tesla driver death events: {p2['tesla_driver_death_events']} ({p2['tesla_driver_death_pct']:.1f}%)
Occupant/driver death events: {p2['occupant_or_driver_death_events']} ({p2['occupant_or_driver_death_pct']:.1f}%)
Cyclist/ped events: {p2['cyclist_ped_events']} ({p2['cyclist_ped_pct']:.1f}%)
Other-vehicle collisions: {p2['other_vehicle_collision_events']} ({p2['other_vehicle_collision_pct']:.1f}%)
Verified Autopilot death events: {p2['verified_ap_death_events']} (sum deaths={p2['verified_ap_deaths_total']:.0f})
Autopilot claimed events: {p2['autopilot_claimed_events']}

{report}
"""

# Find cells to annotate by source snippets
exec_count = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    exec_count += 1
    cell.execution_count = exec_count
    src = cell.source
    outs = []

    if "part1_metrics.json" in src and "val_acc" in src:
        outs.append(new_output(output_type="stream", name="stdout", text=report + f"\nCoarse: {metrics.get('coarse_val_accuracy',0):.4f}\n"))
        b64 = png_b64("part1_confusion_matrix.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<ConfusionMatrix>"}, metadata={}))
    elif "part1_training_curves" in src:
        outs.append(new_output(output_type="stream", name="stdout",
                               text=f"Phase A best: {metrics.get('phase_a_best_val_acc')}\nPhase B best: {metrics.get('phase_b_best_val_acc')}\nSaved models.\n"))
        b64 = png_b64("part1_training_curves.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<TrainingCurves>"}, metadata={}))
    elif "part1_class_counts" in src:
        b64 = png_b64("part1_class_counts.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<ClassCounts>"}, metadata={}))
    elif "part1_inference_grid" in src:
        outs.append(new_output(output_type="stream", name="stdout",
                               text=f"★ Crop classifier validation accuracy: {metrics['val_accuracy']*100:.2f}%\n"))
        b64 = png_b64("part1_inference_grid.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<InferenceGrid>"}, metadata={}))
    elif "part2_events_date_geo" in src:
        for name in ["part2_events_date_geo.png", "part2_events_monthly.png"]:
            b64 = png_b64(name)
            if b64:
                outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": name}, metadata={}))
    elif "part2_death_aspects" in src:
        b64 = png_b64("part2_death_aspects.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<DeathAspects>"}, metadata={}))
    elif "part2_models" in src:
        b64 = png_b64("part2_models.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<Models>"}, metadata={}))
    elif "part2_verified_autopilot" in src:
        outs.append(new_output(output_type="stream", name="stdout", text=json.dumps(p2, indent=2) + "\n"))
        b64 = png_b64("part2_verified_autopilot.png")
        if b64:
            outs.append(new_output(output_type="display_data", data={"image/png": b64, "text/plain": "<VerifiedAP>"}, metadata={}))
    elif "CAPSTONE PROJECT 1" in src or "Key outputs" in src:
        outs.append(new_output(output_type="stream", name="stdout", text=summary_text))
    elif "tesla_deaths_cleaned" in src and "Cleaned shape" in src:
        outs.append(new_output(output_type="stream", name="stdout",
                               text=f"Cleaned shape: ({p2['n_events']}, 15)\nTesla cleaning complete.\n"))

    cell.outputs = outs

nb.metadata["kernelspec"] = {
    "display_name": "Python (.venv_dl)",
    "language": "python",
    "name": "python3",
}
nbf.write(nb, nb_path)
print(f"Embedded outputs into {nb_path} ({exec_count} code cells)")
