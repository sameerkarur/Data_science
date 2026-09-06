# Autonomous Driving Capstone — STAR Write-up

**Project:** Course 7 Capstone Project 1 — Autonomous Driving  
**Client framing:** AV / ITS vehicle perception + Tesla Autopilot safety analytics  
**Stack:** Python, TensorFlow/Keras (MobileNetV2), OpenCV, pandas, scikit-learn, seaborn  
**Data:** Road-scene images with bounding boxes · Tesla Deaths public CSV

---

## Situation

Autonomous vehicles and intelligent transport systems need reliable **on-road object understanding** — not just “is something there,” but *what* it is and *where* it sits in the frame. Separately, Tesla’s Full Self-Driving / Autopilot narrative makes **fatal-incident patterns** a high-visibility public-safety and product-risk topic.

This capstone combined both:
1. **Part 1** — deep-learning vehicle-type recognition and localization visuals on annotated driving frames  
2. **Part 2** — structured EDA of Tesla fatality events (calendar, geography, victims, models, verified Autopilot deaths)

---

## Task

**Part 1 — Object detection / localization**
1. Create parent/child folders for custom model training  
2. Prepare the dataset (filter labels to images that exist on disk)  
3. Build a CNN of choice for detection / classification + localization presentation  
4. Evaluate with clear metrics  
5. Run inferences on sample images with annotated boxes  

**Part 2 — Tesla Deaths**
1. Preliminary inspection (dtypes, missing, duplicates) and drop irrelevant columns  
2. EDA covering events by date/year/day/state/country; death aspects; model mix; verified Autopilot deaths  

---

## Action

### Part 1
- Documented training layout under `data/part1/` (`Images/`, `labels.csv`, `crops/`) plus `models/` and `outputs/`.  
- Loaded headerless `labels.csv` (`img, cls, xmin, ymin, xmax, ymax`), padded IDs to 8 digits, and **filtered to 5,626 on-disk images (~17.9k boxes)**.  
- Dropped tiny boxes (<24px) and ambiguous catch-all labels (`motorized_vehicle`, `non-motorized_vehicle`) that collide with car/pickup, then built a **stratified ~3.6k-crop subset across 9 classes**.  
- Trained a **MobileNetV2 (ImageNet) transfer-learning crop classifier** (frozen head → light fine-tune) with mild class weights and augmentation.  
- Added a **multi-output MobileNet** (class + normalized bbox) on the largest object per frame as a localization demo (~71% val class accuracy).  
- Evaluated with accuracy, classification report, and confusion matrix; drew predicted labels on GT boxes and saved annotated inferences.

### Part 2
- Cleaned messy spaced column names; treated `-` as missing; coerced numerics.  
- Dropped URL/Unnamed columns, free-text `Description`, `Source`/`Note`, deceased-name PII, and **footer aggregate rows**.  
- Produced charts for calendar/geography, death aspects, model distribution, and verified Autopilot deaths.

---

## Result

### Part 1 — perception model
| Metric | Value |
|--------|-------|
| **Crop classifier val accuracy (9-class)** | **68.6%** |
| **Coarse-group accuracy** (passenger / large / VRU) | **89.9%** |
| Multi-output val class accuracy | ~71.2% |
| Train / val crops | 2,900 / 725 |

Ambiguous catch-all labels were the main noise source on the full 11-class taxonomy; after dropping them, fine-grain accuracy approached the 70% target and coarse grouping exceeded it. Annotated sample frames live in `outputs/inference_*.jpg`. Saved Keras models: `models/crop_classifier_final.keras`, `models/class_bbox_multioutput.keras`.

### Part 2 — Tesla safety EDA (highlights)
- **293** cleaned fatal events · **352** total deaths · mean **~1.20 deaths/event**  
- Tesla **driver** died in **~40%** of events; driver *or* occupant death in **~47%**  
- **Cyclist/pedestrian** involvement: **~15%** of events  
- Collision with **other vehicle(s):** **~38%**  
- Models mostly **unspecified**, then **S / 3 / X / Y**  
- **Verified Autopilot death events:** 16 (19 verified deaths total) vs **34** Autopilot-*claimed* events — claims exceed verified counts  

### Artifacts
- Executed notebook: `autonomous_driving.ipynb`  
- Charts & inferences: `outputs/`  
- Models: `models/`  
- Deliverables: Jupyter notebook, trained weights, inference visualizer, and executive writeup.

---

*Prepared for IITK AIML Capstone (Course 7) · Autonomous Driving*
