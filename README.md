# 🔍 AI Explainability Dashboard for Vision & Sound Defect Analytics

This repository unifies **vision-based defect analysis** (casting classification + metal surface detection) with **sound anomaly detection** (two different approaches), and presents an **Explainable AI (XAI)** dashboard to interpret model predictions.

- **Vision:**  
  - **Casting classification** — YOLOv8 (classification head)  
  - **Metal surface defect detection** — Faster R-CNN (ResNet-50 FPN)  
  - **Explainability** — Grad-CAM, Integrated Gradients, Occlusion Sensitivity, Saliency Maps  

- **Sound:**  
  - **Approach 1 (`fan.ipynb`)** — Supervised CNN on Mel-spectrograms with a small UI (Gradio)  
  - **Approach 2 (`isolation_forest_sound+dataset.ipynb`)** — Unsupervised Isolation Forest with SHAP & LIME explanations  

The end product is a **Streamlit dashboard** (`combined.py`) that currently focuses on the **vision** pipelines + XAI, while the **sound** notebooks are exploratory and **complementary** (future candidates to integrate into the same dashboard).

---

## 📑 Table of Contents

1. [Project Goals & Design Rationale](#project-goals--design-rationale)  
2. [Repository Structure](#repository-structure)  
3. [File-by-File Technical Report](#file-by-file-technical-report)  
4. [Installation & Setup](#️-installation--setup)  
5. [Running the Dashboard](#-running-the-dashboard)  
6. [Models, Weights & Paths](#-models-weights--paths)  
7. [Explainability Methods](#-explainability-methods)  
8. [Results, Validation & What to Look For](#-results-validation--what-to-look-for)  
9. [Troubleshooting](#-troubleshooting)  
10. [Future Work](#-future-work)  
11. [License](#license)

---

## 🎯 Project Goals & Design Rationale

- **Trustworthy AI for manufacturing:** Predictions without explanations are hard to adopt on the shop floor. We therefore pair models with **XAI** to show *why* a decision was made.  
- **Two modalities, two roles:**  
  - **Vision** — for *spatial localization* and visual defect detection.  
  - **Sound** — for *machine health monitoring* and unsupervised anomaly detection.  
- **Model choices:**  
  - **YOLOv8-cls** — lightweight and fast for classification (casting defects).  
  - **Faster R-CNN ResNet-50 FPN** — robust multi-class detection (metal defects).  
  - **CNN on Mel-spectrograms** — effective when sound anomaly labels are available.  
  - **Isolation Forest** — unsupervised detection when labeled anomalies are scarce.

---

## 📂 Repository Structure

.
├── combined.py # Streamlit dashboard with XAI for vision models
├── Casting Defects (1).ipynb # YOLOv8 classification training/eval for casting
├── RCNN_Defect_Detection.ipynb # Faster R-CNN detection training/eval + XAI hooks
├── fan.ipynb # Supervised CNN on audio Mel-spectrograms (+ Gradio)
├── isolation_forest_sound+dataset.ipynb # Unsupervised Isolation Forest + SHAP/LIME on audio
├── best.pt # YOLOv8 classification weights (expected)
├── fasterrcnn_metal_defects_full50.pth # Faster R-CNN detection weights (expected)
└── README.md



---

## 📘 File-by-File Technical Report

### 1) `Casting Defects (1).ipynb` — YOLOv8 Classification
- **Purpose:** Train a YOLOv8 classification model for casting defect detection (binary classification).  
- **Approach:**  
  - Model: `yolov8n-cls.pt`.  
  - Training: `epochs=3`, `imgsz=300`, `batch=40`.  
  - Preprocessing matches dashboard (`Resize(300,300)`).  
- **Output:** `best.pt` (saved under `runs/classify/.../weights/best.pt`, then moved to repo root).  
- **Why YOLOv8?** Unified ecosystem, lightweight, efficient, easy integration with dashboard.

---

### 2) `RCNN_Defect_Detection.ipynb` — Faster R-CNN Detection
- **Purpose:** Detect and localize **10 classes** of metal surface defects.  
- **Classes:** welding_line, water_spot, waist_folding, silk_spot, punching_hole, rolled_pit, oil_spot, inclusion, crescent_gap, crease.  
- **Approach:**  
  - Data: VOC-style XML annotations parsed with `xml.etree.ElementTree`.  
  - Model: Faster R-CNN ResNet-50 FPN.  
  - Outputs: Bounding boxes, labels, confidence scores.  
  - Explainability: Grad-CAM from backbone layer4 + input saliency maps.  
- **Output:** `fasterrcnn_metal_defects_full50.pth`.  
- **Why Faster R-CNN?** Accurate bounding-box detection with smaller datasets, strong localization.

---

### 3) `fan.ipynb` — Supervised CNN for Fan Sound Anomalies
- **Purpose:** Supervised classification of fan sounds (normal vs abnormal).  
- **Approach:**  
  - Convert `.wav` → Mel-spectrogram (`128x128`).  
  - CNN: 3 conv layers + dense layers with dropout.  
  - Handles imbalance with class weights.  
  - Includes **Gradio UI** for demo.  
- **Why CNN on audio?** Effective supervised approach when anomaly labels are available.

---

### 4) `isolation_forest_sound+dataset.ipynb` — Unsupervised Isolation Forest for Sound Anomalies
- **Purpose:** Detect anomalies in fan/machine sounds without labels.  
- **Approach:**  
  - Extract features (MFCC + spectral features).  
  - Train **Isolation Forest** on *normal data only*.  
  - Score frames, flag anomalies with thresholds.  
  - Use **SHAP** and **LIME** for feature explanations.  
- **Why Isolation Forest?** Works well when anomalies are rare or unlabeled.

---

### 5) `combined.py` — Streamlit XAI Dashboard
- **Purpose:** Unified dashboard for vision defect classification and detection with XAI.  
- **Features:**  
  - **Tab 1 (YOLOv8 Classification):**  
    - Predicts casting defects.  
    - Explanations: Grad-CAM, Integrated Gradients, Occlusion, Saliency.  
  - **Tab 2 (Faster R-CNN Detection):**  
    - Predicts bounding boxes for metal surface defects.  
    - Displays ground truth vs predictions.  
    - Explanations: Grad-CAM overlays and Saliency maps.  
- **Why Streamlit?** Interactive, fast prototyping, accessible to non-technical users.

---

## ⚙️ Installation & Setup
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -U pip
pip install ultralytics torch torchvision torchaudio \
            streamlit opencv-python pillow matplotlib numpy \
            pytorch-grad-cam captum \
            librosa scikit-learn shap lime \
            gradio tqdm seaborn



## 📦 Models, Weights & Paths

- **YOLOv8 Classification**
  - Training produces weights under `runs/classify/.../best.pt`.
  - Copy to repo root as `best.pt`.

- **Faster R-CNN Detection**
  - Training produces `fasterrcnn_metal_defects_full50.pth`.
  - Place in repo root.

- **Ground Truth XMLs (for detection)**
  - Update path in `combined.py` (`LABELS_DIR`).
  - Ensure XML filenames match image filenames.

---

## 🔎 Explainability Methods

| Method                  | Where used              | What it shows                                    |
|-------------------------|-------------------------|--------------------------------------------------|
| **Grad-CAM**            | YOLOv8 & Faster R-CNN   | Highlights regions most responsible for prediction |
| **Integrated Gradients**| YOLOv8                  | Pixel-wise contributions vs baseline             |
| **Occlusion Sensitivity** | YOLOv8                | Regions that reduce confidence when hidden       |
| **Saliency Maps**       | YOLOv8 & Faster R-CNN   | Raw gradient-based pixel importance              |

---

## 📊 Results, Validation & What to Look For

- **Casting defects (YOLOv8):** Explanations highlight defective regions on castings.  
- **Metal defects (Faster R-CNN):** Predicted boxes align with ground truth, explanations validate focus.  
- **Sound anomalies:**  
  - CNN → produces classification metrics (accuracy, confusion matrix).  
  - Isolation Forest → anomaly scores over time, with SHAP/LIME explanations of features.  

---

## 🛠️ Troubleshooting

- **FRCNN won’t load:** Ensure `fasterrcnn_metal_defects_full50.pth` is in repo root. Match PyTorch/Torchvision versions used in training.  
- **No GT boxes:** Ensure `LABELS_DIR` points to correct XML folder and filenames match images.  
- **YOLO class mismatch:** Confirm `best.pt` includes correct class metadata.  
- **Slow explanations:** Integrated Gradients & Occlusion run slower on CPU. Use CUDA if available.  

---

## 🔮 Future Work

- Extend dashboard with sound anomaly detection tabs.  
- Add more XAI methods (e.g., SHAP for images, LIME superpixels).  
- Deploy with **Streamlit Cloud, Docker, or AWS** for easy sharing.  
- Version control for datasets & models (e.g., DVC, MLflow).  

