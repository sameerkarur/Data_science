# Preserving Heritage: Enhancing Tourism with AI — Capstone Writeup

## Situation
Heritage tourism faces two practical gaps: visitors struggle to identify architectural features in historic sites, and destination platforms need better ways to personalize place recommendations. Indonesia’s tourism data (users, places, ratings) plus a multi-class historical-structure image set make this a strong end-to-end AI opportunity—computer vision for structure recognition and collaborative filtering for trip planning.

## Task
Deliver a two-part Capstone solution:

1. **Structure classification** — Train a transfer-learning CNN (MobileNetV2) to recognize Gothic/heritage architectural elements (altar, apse, bell tower, column, domes, flying buttress, gargoyle, stained glass, vault), compare training with and without augmentation, and save the best model.
2. **Tourism intelligence** — Clean and explore Indonesian tourism datasets, answer business questions (origins, categories, nature-friendly cities, most-loved spots), and build a collaborative filtering recommender that suggests related places from a seed attraction.

## Action
**Part 1 — Vision**
- Organized train/test directories and documented class counts; used a **stratified subset (~350 images/class)** for practical CPU training while evaluating on the provided test set.
- Plotted 9 sample images per class with OpenCV + Matplotlib (`outputs/samples_*.png`).
- Built MobileNetV2 (ImageNet weights), **froze convolutional base**, added Dense(256) + Dropout + Softmax head.
- Compiled with Adam, sparse categorical cross-entropy, and accuracy; added a callback to stop when validation accuracy reaches **0.85**.
- Trained without augmentation, then with RandomFlip/Rotation/Zoom/Contrast; plotted train/val curves; saved the best checkpoint under `models/`.

**Part 2 — Tourism + recommender**
- Loaded `user.csv`, `tourism_rating.csv`, and `tourism_with_id.xlsx`; removed duplicate keys and unnamed columns.
- Analyzed age distribution and tourist origin locations; mapped spot categories and city coverage.
- Identified the strongest city for nature enthusiasts using **Cagar Alam** volume × rating.
- Merged ratings with place metadata to rank loved places, cities, and categories.
- Built an item–item collaborative filter with `sklearn.NearestNeighbors` (cosine) on the user–item matrix; generated recommendations from seed places such as *Monumen Nasional*.

## Result
- Delivered an executed notebook (`preserving_heritage.ipynb`), metrics JSON, accuracy curves, sample grids, and a saved Keras classifier in `models/`.
- **Final best validation accuracy: 92.53%** (no-augmentation MobileNetV2; exceeded 0.85 target by epoch 1 and continued to epoch 3 for stable curves).
- With-augmentation run: **89.6% → 91.4%** across 3 epochs (also above 0.85). On this strong ImageNet transfer baseline the non-augmented checkpoint edged out slightly; augmentation still showed a clear upward generalization trend.
- Tourism EDA: top origins include Bekasi / Semarang / Yogyakarta; **best city for nature enthusiasts = Bandung** (most `Cagar Alam` spots × rating); **most liked category = Taman Hiburan**.
- Sample CF recommendations for *Monumen Nasional*: Wisata Mangrove Tapak, Danau Rawa Pening, Museum Sonobudoyo Unit I, Dunia Fantasi, Situ Patenggang.

## Key artifacts
| Artifact | Path |
|---|---|
| Notebook | `preserving_heritage.ipynb` |
| Best model | `models/heritage_structure_classifier_best.keras` |
| Metrics | `outputs/part1_metrics.json`, `outputs/part2_summary.json` |
| Plots | `outputs/curves_*.png`, `outputs/samples_*.png`, `outputs/eda_*.png` |
| Sample recs | `outputs/sample_recommendations.csv` |

## Key Architecture Visualizations
1. Per-class sample image grids  
2. Class count bar chart  
3. Train/val accuracy curves (no-aug vs aug)  
4. Age/origins and category EDA plots  
5. Sample recommendation table output  
