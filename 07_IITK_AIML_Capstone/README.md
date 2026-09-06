# Course 07: Industrial AI/ML Capstone Portfolio
**Professional Certificate Program — E&ICT Academy, IIT Kanpur**

---

## 📌 Executive Overview

Course 07 represents the comprehensive culmination of the IIT Kanpur AI/ML curriculum. The Capstone Portfolio consists of **three distinct, industrial-scale artificial intelligence systems**, engineered to solve real-world problems spanning **Computer Vision Perception**, **Time-Series Retail Forecasting**, and **Multimodal Cultural Heritage AI**.

---

## 🏆 Capstone Projects Suite

| # | Project Title | Technical Focus | Core Algorithms & Models | Directory |
|---|---|---|---|---|
| **01** | **Autonomous Driving Perception & Localization** | Computer Vision & Object Localization | MobileNetV2, Multi-Output Bounding Box Regression, Grad-CAM, Transfer Learning | [`project1_autonomous_driving/`](project1_autonomous_driving/) |
| **02** | **Multi-Store Demand & Revenue Forecasting** | Time-Series Analytics & Forecasting | Stacking Regressors, XGBoost, Random Forest, Fourier Calendar Seasonality, Elasticity | [`project2_sales_forecasting/`](project2_sales_forecasting/) |
| **03** | **Cultural Heritage AI & Tourism Assistant** | Multimodal Vision & Content Recommendation | ResNet50 Classifier, Data Augmentation Invariance, Cosine Similarity Recommender | [`project3_preserving_heritage/`](project3_preserving_heritage/) |

---

## 🚗 Project 01: Autonomous Driving Perception & Localization

- **Problem Dilemma:** Real-time autonomous navigation requires simultaneous pixel-level semantic classification and spatial coordinate localization for surrounding traffic agents (vehicles, pedestrians, cyclists) under variable lighting conditions.
- **Architectural Implementation:**
  - Transfer learning perception backbone using pre-trained convolutional representations.
  - Multi-task head predicting class probabilities and normalized bounding box coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$.
  - Visual interpretability using **Gradient-weighted Class Activation Mapping (Grad-CAM)** to verify feature attention maps on critical pedestrian and vehicle contours.
  - Tesla crash safety statistical analysis examining autopilot incident telemetry.
- **Deliverables & Writeup:** [`project1_autonomous_driving/WRITEUP.md`](project1_autonomous_driving/WRITEUP.md)

---

## 📈 Project 02: Multi-Store Demand & Revenue Forecasting

- **Problem Dilemma:** Large-scale retail chains encounter substantial margin erosion from inventory stockouts during demand spikes and holding costs during sudden slumps.
- **Architectural Implementation:**
  - Multi-source integration uniting item sales histories, transactional ticket volumes, and catalog specifications.
  - Feature engineering pipeline deriving cyclical Fourier harmonics, weekend/holiday flags, rolling window statistics, and store-level revenue elasticities.
  - Comparative benchmark across **Linear Regression**, **Random Forest Regressors**, and **Extreme Gradient Boosting (XGBoost)** with holdout cross-validation.
  - 365-day forward horizon demand forecast with confidence intervals.
- **Deliverables & Writeup:** [`project2_sales_forecasting/WRITEUP.md`](project2_sales_forecasting/WRITEUP.md)

---

## 🏛️ Project 03: Cultural Heritage AI & Tourism Assistant

- **Problem Dilemma:** Preserving historical monuments requires rapid cataloging from crowd-sourced photography alongside intelligent discovery engines for cultural tourism.
- **Architectural Implementation:**
  - Deep convolutional feature extractor (**ResNet50**) trained with heavy spatial data augmentation (rotation, zoom, contrast jittering) to maintain viewpoint invariance.
  - Content-based architectural recommendation system matching historical sites using multidimensional feature similarity.
  - Exploratory historical origin analysis mapping chronological styles and architectural movements.
- **Deliverables & Writeup:** [`project3_preserving_heritage/WRITEUP.md`](project3_preserving_heritage/WRITEUP.md)

---

## 🛠️ Execution & Interactive Cloud Access

All three Capstone projects can be practiced and inspected either locally or directly in the cloud:
1. **Interactive Studio:** Open `index.html` and switch to the **Capstones** tab.
2. **Google Colab:** Click any Colab badge on the respective project cards to launch pre-configured GPU runtimes.
3. **Local Setup:**
   ```bash
   cd 07_IITK_AIML_Capstone/<project_folder>
   jupyter notebook
   ```
