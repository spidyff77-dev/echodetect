# 🎧 EchoDetect: Intelligent Environmental Sound Classification Using Machine Learning

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: CC BY-NC 3.0](https://img.shields.io/badge/Dataset-UrbanSound8K-orange.svg)](http://serv.cusp.nyu.edu/projects/urbansounddataset)
[![Status: Production Ready](https://img.shields.io/badge/Status-Complete%20AIML%20Project-brightgreen.svg)]()
[![Web App: Streamlit](https://img.shields.io/badge/Frontend-Streamlit%20Dark%20Theme-red.svg)](http://localhost:8501)

An end-to-end Machine Learning and Audio Signal Processing system engineered to identify and classify environmental acoustic signatures into 10 distinct taxonomy classes using the **UrbanSound8K** benchmark. Built with **Librosa**, **Scikit-learn**, **Plotly**, and **Streamlit**.

---

## 📌 1. Project Overview & Problem Formulation

### Problem Statement
Urban agglomerations are flooded with heterogeneous acoustic noise. Traditional noise level meters measure simple sound pressure levels in decibels (dB), providing magnitude without semantic identity. They cannot differentiate between innocuous traffic hum, dangerous gunshots, industrial construction jackhammers, or emergency vehicle sirens.

### Objectives
1. **Automated Acoustic Recognition**: Reliably map raw audio waveforms into 10 urban sound classes.
2. **Fixed-Dimension Feature Engineering**: Standardize variable-length ($\le 4\text{s}$), variable-sample-rate (11 kHz – 96 kHz) audio clips into 272-dimensional cepstral and spectral vectors.
3. **Leak-Free Cross-Validation**: Adhere strictly to the pre-defined 10 folds of UrbanSound8K to guarantee that excerpt slices from the same original recording never cross train/validation/test partitions.
4. **Interpretable Classical ML**: Train and evaluate Support Vector Machines (RBF kernel with Platt probability calibration) and Random Forest ensembles (200 trees).
5. **Interactive Dark-Themed UI**: Provide a responsive Streamlit dashboard for real-time inference, audio analysis, and model benchmarking.

---

## 📁 2. Dataset Architecture (UrbanSound8K)

- **Total Clips**: 8,732 labeled sound recordings ($\le 4.0$ seconds each).
- **Taxonomy (10 Classes)**:
  - `0: air_conditioner` (1,000 samples)
  - `1: car_horn` (429 samples)
  - `2: children_playing` (1,000 samples)
  - `3: dog_bark` (1,000 samples)
  - `4: drilling` (1,000 samples)
  - `5: engine_idling` (1,000 samples)
  - `6: gun_shot` (374 samples)
  - `7: jackhammer` (1,000 samples)
  - `8: siren` (929 samples)
  - `9: street_music` (1,000 samples)
- **Directory Layout**:
  - `UrbanSound8K/audio/fold1/` through `fold10/`
  - `UrbanSound8K/metadata/UrbanSound8K.csv`
- **Zero Missing Data**: Verified 8,732 / 8,732 files present on disk with 0 null values.

---

## 🧠 3. Feature Extraction Architecture (272 Dimensions)

Every input audio file undergoes standardized extraction in `echodetect/features.py`:

```
Input Audio (.wav)
       │
       ▼
[Resample to 22,050 Hz] ───> [Downmix to Mono] ───> [Peak Normalization]
                                                          │
   ┌──────────────────────────────────────────────────────┴──────────────────────────────────────┐
   │                                                      │                                      │
   ▼                                                      ▼                                      ▼
[40 Static MFCCs]                               [Dynamic Derivatives]                  [Spectral Descriptors]
 • Mean (40)                                     • Delta Mean (40)                      • Spectral Centroid (Mean/Std = 2)
 • Standard Deviation (40)                       • Delta Std (40)                       • Spectral Bandwidth (Mean/Std = 2)
                                                 • Delta-Delta Mean (40)                • Spectral Rolloff (Mean/Std = 2)
                                                 • Delta-Delta Std (40)                 • Zero-Crossing Rate (Mean/Std = 2)
                                                                                        • Chroma STFT 12-Bands (Mean/Std = 24)
   │                                                      │                                      │
   └──────────────────────────────────────────────────────┼──────────────────────────────────────┘
                                                          │
                                                          ▼
                                            [272-Dimensional Feature Vector]
```

---

## 📊 4. Actual Model Performance Benchmark

Models were trained on **Folds 1–8** (7,057 samples), tuned on **Fold 9** (815 samples), and evaluated on the strictly held-out **Fold 10** (837 samples).

### Head-to-Head Classifier Comparison (Test Fold 10)

| Classifier | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **77.90%** | **0.8186** | **0.7889** | **0.7984** | **0.7817** | **11.90s** |
| **Random Forest (200 Trees)** | **74.79%** | **0.7895** | **0.7543** | **0.7648** | **0.7500** | **3.20s** |

### Per-Class Performance Breakdown (SVM on Test Fold 10)

| Sound Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Air Conditioner** | 0.93 | 0.78 | 0.85 | 100 |
| **Car Horn** | 0.93 | 0.82 | 0.87 | 33 |
| **Children Playing** | 0.67 | 0.82 | 0.74 | 100 |
| **Dog Bark** | 0.62 | 0.84 | 0.71 | 100 |
| **Drilling** | 0.74 | 0.79 | 0.76 | 100 |
| **Engine Idling** | 0.95 | 0.76 | 0.85 | 93 |
| **Gun Shot** | 1.00 | 0.94 | 0.97 | 32 |
| **Jackhammer** | 0.80 | 0.73 | 0.77 | 96 |
| **Siren** | 0.71 | 0.59 | 0.64 | 83 |
| **Street Music** | 0.84 | 0.82 | 0.83 | 100 |
| **Overall Macro Avg** | **0.82** | **0.79** | **0.80** | **837** |

---

## 💻 5. Project Structure

```
Echodetect/
├── app.py                          # Streamlit Interactive Web Application
├── train.py                        # Unified Model Training & Fold Caching Engine
├── evaluate.py                     # Checkpoint Evaluation Script
├── predict.py                      # CLI Audio Prediction Tool
├── dataset_inspector.py            # Dataset Verification & Audio Inspection
├── test_pipeline.py                # Pipeline Verification Script (Small Subset)
├── config.py                       # Central Configurations & Hyperparameters
├── requirements.txt                # Python Dependencies
├── README.md                       # Comprehensive Project Documentation
├── echodetect/                     # Modular Core Python Package
│   ├── __init__.py
│   ├── dataset.py                  # Metadata loader, subset generator, fold splitting
│   ├── features.py                 # 272D feature extractor & disk caching
│   ├── models.py                   # SVM & Random Forest pipelines, calibration & metrics
│   └── utils.py                    # Visualizations & metrics serialization
├── saved_models/                   # Persisted Model Artifacts
│   ├── svm_model.joblib            # Trained SVM Pipeline with Calibrated Probabilities
│   ├── rf_model.joblib             # Trained Random Forest Pipeline (200 Trees)
│   └── scaler.joblib               # Pre-fitted StandardScaler
├── features/                       # Cached Fold Features (fold_1 to fold_10 .npz)
└── reports/                        # Benchmarks & Viva Guide
    ├── evaluation_summary.json     # Machine-readable test fold 10 metrics
    ├── test_metrics.csv            # Comparative metrics table
    ├── svm_confusion_matrix.png    # SVM test fold 10 confusion matrix heatmap
    ├── rf_confusion_matrix.png     # Random Forest test fold 10 confusion matrix heatmap
    ├── rf_feature_importances.png  # Random Forest top 15 features bar plot
    └── VIVA_PREPARATION.md         # Comprehensive viva Q&A document (16 Questions)
```

---

## 🚀 6. Execution & Quickstart Guide

### Environment Setup
Activate the virtual environment located inside `Echodetect/`:
```bash
cd Echodetect
source venv/bin/activate
```

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser to access the 6 dashboard pages:
- **Overview Dashboard**: High-level KPIs, class distribution, and architecture flow.
- **Sound Detection**: Upload or choose test clips, listen, analyze, and inspect calibrated probabilities.
- **Audio Analysis**: Waveform, MFCC spectrogram heatmap, and feature dimension breakdowns.
- **Model Performance**: SVM vs RF benchmark tables, per-class bar charts, confusion matrices.
- **Dataset Explorer**: Filter and listen to original recordings across folds and classes.
- **About Project**: Problem statement, mathematical formulations, and viva defense notes.

### 2. Run Pipeline Test (Quick Verification)
```bash
python test_pipeline.py
```

### 3. Train Models
Train on the full dataset using pre-defined folds (Folds 1-8 Train, Fold 9 Val, Fold 10 Test):
```bash
python train.py --model both
```

Or train quickly on a subset (e.g., 200 samples) for rapid testing:
```bash
python train.py --subset 200
```

### 4. Evaluate Checkpoints on Test Fold 10
```bash
python evaluate.py --model saved_models/svm_model.joblib --test-fold 10
python evaluate.py --model saved_models/rf_model.joblib --test-fold 10
```

### 5. Classify Any Audio File via CLI
```bash
python predict.py --audio UrbanSound8K/audio/fold10/100648-1-0-0.wav
```

---

## 🎓 7. Academic Evaluation & Viva Defense

For comprehensive answers to 16 viva questions covering the mathematical formulation of MFCCs, Nyquist resampling, SVM reproducing kernel Hilbert spaces, Random Forest Gini impurity, data leakage prevention, and real-world applications, refer to:

👉 **[reports/VIVA_PREPARATION.md](file:///Users/subhojitroy/Downloads/hug-of-warmth-web-main/Echodetect/reports/VIVA_PREPARATION.md)**
