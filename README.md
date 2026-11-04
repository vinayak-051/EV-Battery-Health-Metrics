# EV Battery Health Prediction System

Predicts the **State of Health (SOH %)** and **Remaining Useful Life (RUL cycles)** of lithium-ion EV batteries using machine learning on the NASA Li-ion Battery Aging Dataset. A full-stack web app lets engineers enter five live discharge readings and receive an instant health assessment.

---

## Problem Statement

Lithium-ion batteries degrade with every charge-discharge cycle. Without accurate health monitoring, batteries either fail unexpectedly in the field or are retired too early — both outcomes are costly. Traditional capacity measurement requires a full controlled discharge, which takes hours and interrupts operation.

**The goal:** predict battery SOH and RUL from five routine discharge measurements that are available after any cycle — no controlled test required.

---

## Solution

An end-to-end ML pipeline trains on 2,737 discharge cycles across 34 NASA test batteries. Feature engineering derives 10 cycle-level signals from raw discharge data. Two separate models are trained and tuned:

| Target | Definition | Best Model | Test R² | Test RMSE |
|---|---|---|---:|---:|
| SOH (%) | `capacity / initial_capacity × 100` | XGBoost | **0.881** | 7.68 % |
| RUL (cycles) | Cycles remaining before SOH drops to 70% EOL | RandomForest | **0.820** | 4.55 cycles |

A React + FastAPI web interface wraps both models. The user provides five inputs; the backend derives all required features and returns SOH, RUL, and a health status label in real time.

---

## Dataset

**NASA Prognostics Center of Excellence — Li-ion Battery Aging Dataset**

| Property | Value |
|---|---|
| Batteries | 34 (B0005 – B0056) |
| Total discharge cycles | 2,737 (after cleaning) |
| Removed cycles | 13 (too few rows or voltage out of range) |
| Capacity range | 1.08 – 2.64 Ah |
| Discharge protocols | 1 A and 4 A, various cut-off voltages |

> Raw `.mat` files are converted to CSVs stored in `ML model/dataset/`. Do not modify them.

---

## Model Performance

### Before vs. After Tuning

| Task | Model | Test R² | Test RMSE |
|---|---|---:|---:|
| SOH | Linear Regression (baseline) | 0.846 | 8.73 % |
| SOH | Random Forest (baseline) | 0.737 | 11.42 % |
| SOH | XGBoost (baseline) | 0.741 | 11.33 % |
| **SOH** | **XGBoost (tuned — selected)** | **0.881** | **7.68 %** |
| RUL | Linear Regression (baseline) | 0.100 | 10.16 cycles |
| RUL | Random Forest (baseline) | 0.631 | 6.50 cycles |
| RUL | XGBoost (baseline) | 0.573 | 7.00 cycles |
| **RUL** | **Random Forest (tuned — selected)** | **0.820** | **4.55 cycles** |

Train/test split is at battery level (70/30, seed 42) so no battery appears in both sets.

---

## Features

**User inputs (5):**

| Input | Description |
|---|---|
| Cycle number | Discharge cycle count |
| Discharge energy (Wh) | Energy released in this cycle |
| Mean voltage (V) | Average discharge voltage |
| Mean current (A) | Average discharge current |
| Temperature (°C) | Average battery temperature |

**Derived by the backend (→ 10 SOH features, 14 RUL features):**

```
discharge_duration, voltage_min, voltage_std, temperature_max,
ambient_temp_C, soh (from SOH model), soh_prev, soh_delta, soh_slope_5
```

---

## Folder Structure

```
EV Project/
│
├── ML model/                          Machine learning pipeline
│   ├── dataset/
│   │   ├── data/                      NASA discharge CSVs (raw, do not modify)
│   │   └── extra_infos/               NASA dataset documentation
│   │
│   ├── cleaned_data/
│   │   ├── processed_dataset.csv      2,737 × 16  (all candidate features)
│   │   ├── featured_dataset.csv       2,737 × 12  (selected features)
│   │   └── final_dataset.csv          2,737 × 15  (features + SOH + RUL targets)
│   │
│   ├── notebooks/
│   │   ├── 01_dataset_inspection.ipynb
│   │   ├── 02_data_cleaning_and_preprocessing.ipynb
│   │   ├── 03_eda.ipynb
│   │   ├── 04_feature_engineering_final_dataset.ipynb
│   │   ├── 05_soh_target_calculation.ipynb
│   │   ├── 06_rul_target_calculation.ipynb
│   │   ├── 07_soh_model_training.ipynb
│   │   ├── 08_rul_model_training.ipynb
│   │   ├── 09_soh_validation_tuning.ipynb
│   │   ├── 10_rul_validation_tuning.ipynb
│   │   └── 11_final_analysis.ipynb
│   │
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── soh.py
│   │   ├── rul.py
│   │   ├── evaluation.py
│   │   ├── visualization.py
│   │   └── ev_model.py
│   │
│   ├── models/
│   │   ├── soh/
│   │   │   ├── linear_regression.pkl
│   │   │   ├── random_forest.pkl
│   │   │   ├── xgboost.pkl
│   │   │   └── best_soh_model.pkl      XGBoost, R² = 0.881
│   │   └── rul/
│   │       ├── linear_regression.pkl
│   │       ├── random_forest.pkl
│   │       ├── xgboost.pkl
│   │       ├── best_rul_model.pkl       RandomForest pipeline, R² = 0.820
│   │       └── best_rul_features_v2.json
│   │
│   ├── plots/
│   │   ├── eda/                        Capacity curves, feature distributions, heatmap
│   │   ├── features/                   Feature correlation plots
│   │   ├── soh/                        Actual vs predicted, residuals, model comparison
│   │   └── rul/                        Actual vs predicted, residuals, model comparison
│   │
│   └── results/
│       ├── metrics/                    CSV summaries of model performance
│       ├── predictions/                Test set predictions (SOH and RUL)
│       └── tables/                     Feature lists, battery splits, per-battery results
│
├── backend/                           FastAPI prediction server
│   ├── main.py                        REST API with /predict and /health endpoints
│   ├── predictor.py                   Loads models, derives features, returns predictions
│   └── requirements.txt
│
├── frontend/                          React web application
│   ├── src/
│   │   ├── App.jsx                    Root component and state management
│   │   ├── index.css                  Design tokens and all styles
│   │   └── components/
│   │       ├── InputForm.jsx          5-field input form
│   │       ├── SohGauge.jsx           SVG semicircle SOH gauge
│   │       └── ResultPanel.jsx        SOH + RUL results display
│   ├── index.html
│   ├── package.json
│   └── vite.config.js                 Dev proxy: /predict → localhost:8000
│
└── README.md
```

---

## How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+

---

### 1. Run the ML Notebooks (optional — models already trained)

```bash
cd "ML model"
pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost lightgbm joblib jupyter ipykernel

# Open Jupyter and run notebooks 01 through 11 in order
jupyter notebook
```

Or run the pipeline scripts directly:

```bash
python run_data_pipeline.py        # Stages 1–3: cleaning, features, targets
python run_training_pipeline.py    # Stage 4:   baseline LR / RF / XGBoost
```

---

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API starts at **http://localhost:8000**

- `POST /predict` — accepts 5 inputs, returns SOH %, RUL cycles, health status
- `GET  /health`  — server health check

---

### 3. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

### API Reference

**POST /predict**

```json
{
  "cycle": 150,
  "energy_Wh": 4.2,
  "voltage_mean": 3.85,
  "current_mean": 2.0,
  "temperature_mean": 32.0
}
```

**Response:**

```json
{
  "soh": 85.3,
  "rul": 47,
  "status": "Good",
  "discharge_duration_min": 31.5
}
```

**Status labels:** `Excellent` (≥90%) · `Good` (≥80%) · `Fair` (≥70%) · `Poor` (≥60%) · `Critical` (<60%)

---

## Limitations

- Only 34 batteries (10 in test set, 6 for RUL) — R² has high variance between splits.
- 11 batteries never reach EOL (70% SOH), so their late-life behaviour is unobserved and excluded from RUL training.
- Mixed discharge protocols (1 A / 4 A, different cut-off voltages) add variance the model cannot fully separate from aging.
- Single-cycle input: history features (`soh_prev`, `soh_delta`, `soh_slope_5`) default to neutral values when only one reading is provided, which reduces RUL accuracy compared to providing a full cycle history.
