---
title: AQI Prediction MLOps
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
python_version: "3.12"
app_file: app.py
pinned: false
---

# 🌍 AQI Prediction — Indian CPCB Standard

<div align="center">

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Real-time AQI forecasting for 29 Indian cities on the official CPCB 0–500 scale.**

<p align="center">
  <a href="https://bhautikvekariya21-aqi-prediction-mlops.hf.space/docs"><img src="https://img.shields.io/badge/Live%20API-Demo-0A66C2?style=for-the-badge&logo=fastapi&logoColor=white" alt="Live API"/></a>
  <a href="https://aqi-predictor.lovable.app"><img src="https://img.shields.io/badge/Website-Open-2EA44F?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Website"/></a>
  <a href="https://hub.docker.com/r/bhautikvekariya21/aqi-prediction-api"><img src="https://img.shields.io/badge/Docker%20Hub-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Hub"/></a>
  <a href="../../issues"><img src="https://img.shields.io/badge/Report-Bug-D73A49?style=for-the-badge&logo=github&logoColor=white" alt="Report Bug"/></a>
</p>

</div>

---

## Overview

This project trains an XGBoost model to predict Indian AQI following the **official CPCB methodology** (NAAQS 2014) and serves predictions through a FastAPI REST API.

The core design insight is **train/serve parity**: the feature vector the model sees at inference time is built the exact same way it was built during training — from CPCB time-averaged concentrations, not raw instantaneous readings.

---

## How the AQI is computed (CPCB methodology)

| Pollutant | Averaging window | Unit |
|---|---|---|
| PM2.5, PM10, NO₂, SO₂, NH₃ | 24 hours | µg/m³ |
| CO | 8 hours | mg/m³ (converted from µg/m³) |
| O₃ | 8 hours | µg/m³ |

Each pollutant's averaged concentration is mapped to a sub-index via piecewise-linear interpolation over the CPCB breakpoint table. The overall AQI = `max(sub-indices)`, valid only when ≥3 sub-indices are present and at least one PM sub-index is available.

Categories: **Good** (0–50) · **Satisfactory** (51–100) · **Moderate** (101–200) · **Poor** (201–300) · **Very Poor** (301–400) · **Severe** (401–500)

---

## Model performance

Evaluated on a held-out temporal test set (last 15% of data by date — no shuffling):

| Metric | Value |
|---|---|
| R² | **0.9738** |
| RMSE | **9.54 AQI points** |
| MAE | **2.51 AQI points** |
| Within ±10 AQI | 96.7% |
| Within ±25 AQI | 98.9% |

Per-category breakdown (test set):

| Category | Count | RMSE | R² |
|---|---|---|---|
| Good | 26,963 | 1.6 | 0.958 |
| Satisfactory | 64,801 | 2.2 | 0.976 |
| Moderate | 29,147 | 6.6 | 0.942 |
| Poor | 4,691 | 23.3 | 0.326 |
| Very Poor | 1,640 | 30.4 | — |
| Severe | 439 | 116 | — |

Very Poor and Severe are rare in the dataset (~1.6% combined); the model handles Good–Moderate well, which covers 95% of typical city-hours.

---

## Architecture

### Training pipeline (DVC, 8 stages)

```
Stage 1  data_ingestion        Raw parquet from Open-Meteo (851k rows, 29 cities)
Stage 2  data_preprocessing    Schema fix, winsorize, imputation
Stage 3  feature_engineering   CPCB rolling avgs + label, datetime features, encoders.json
Stage 4  feature_selection     45 features via correlation + decision tree + XGBoost importance
Stage 5  data_splitting        Temporal split (train 70% / val 15% / test 15%)
Stage 6  model_training        XGBoost (n_estimators=1200, early stopping on val)
Stage 7  model_optimization    Optuna (20 trials) → compressed model.json.gz (0.26 MB)
Stage 8  model_evaluation      Acceptance thresholds: R²≥0.90, RMSE≤15, MAE≤10
```

### Serving pipeline (app.py)

```
Request → fetch 48h Open-Meteo history (past_days=2)
        → compute 7 CPCB rolling averages per the same logic used in training
        → build 45-feature vector from features.txt
        → XGBoost inference → clip [0, 500]
        → CPCB category → JSON response
```

No post-hoc corrections, seasonal multipliers, or physics floors. The model output is the prediction.

### Deployable artifacts (repo root)

| File | Size | Purpose |
|---|---|---|
| `model.json.gz` | 0.26 MB | XGBoost model, gzip-compressed |
| `features.txt` | 1 KB | 45 feature names in training order |
| `encoders.json` | 1 KB | city/state label-encoding maps |

---

## API reference

**Base URL:** `https://bhautikvekariya21-aqi-prediction-mlops.hf.space`

### GET /predict/{city}

Real-time AQI and hourly forecast for one city.

```bash
curl "https://bhautikvekariya21-aqi-prediction-mlops.hf.space/predict/Delhi?days=2"
```

Parameters: `days` (1–5, default 2).

Response includes: `hourly[]` (AQI, category, pollutant readings per hour), `daily[]` (daily summary), `summary` (avg/max/min AQI).

### GET /predict/all/cities

Current AQI for all 29 cities in one request. Uses `ThreadPoolExecutor(max_workers=10)` internally.

### GET /cities

List all supported cities with coordinates and state.

### GET /health

Returns `{"status": "healthy", "features": 45, "cities": 29}` when artifacts are loaded.

---

## Supported cities (29)

| City | State | City | State | City | State |
|---|---|---|---|---|---|
| Agartala | Tripura | Hyderabad | Telangana | Raipur | Chhattisgarh |
| Ahmedabad | Gujarat | Imphal | Manipur | Ranchi | Jharkhand |
| Aizawl | Mizoram | Itanagar | Arunachal Pradesh | Shillong | Meghalaya |
| Bengaluru | Karnataka | Jaipur | Rajasthan | Shimla | Himachal Pradesh |
| Bhopal | Madhya Pradesh | Kohima | Nagaland | Thiruvananthapuram | Kerala |
| Bhubaneswar | Odisha | Kolkata | West Bengal | Visakhapatnam | Andhra Pradesh |
| Chandigarh | Punjab | Lucknow | Uttar Pradesh | Dehradun | Uttarakhand |
| Chennai | Tamil Nadu | Mumbai | Maharashtra | Gangtok | Sikkim |
| Delhi | Delhi | Panaji | Goa | Gurugram | Haryana |
| Guwahati | Assam | Patna | Bihar | | |

---

## Getting started

### Option A — Docker

```bash
docker pull bhautikvekariya21/aqi-prediction-api:latest
docker run -p 8000:7860 bhautikvekariya21/aqi-prediction-api:latest
# Open http://localhost:8000/docs
```

### Option B — Run from source

```bash
git clone https://github.com/BhautikVekariya21/aqi-prediction-mlops.git
cd aqi-prediction-mlops
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-api.txt
uvicorn app:app --reload
```

### Option C — Retrain the model

```bash
pip install -r requirements.txt
# Stage 1 (data ingestion) only needs to run once.
# Run stages 2–8:
python -m src.pipeline.stage_02_data_preprocessing
python -m src.pipeline.stage_03_feature_engineering
python -m src.pipeline.stage_04_feature_selection
python -m src.pipeline.stage_05_data_splitting
python -m src.pipeline.stage_06_model_training
python -m src.pipeline.stage_07_model_optimization
python -m src.pipeline.stage_08_model_evaluation
# Copy artifacts to root:
copy models\optimized\model.json.gz .\model.json.gz
copy models\optimized\features.txt .\features.txt
copy data\features\encoders.json .\encoders.json
```

Or use DVC: `dvc repro` (skips stages whose inputs haven't changed).

---

## Project structure

```
aqi-prediction-mlops/
├── src/
│   ├── components/          # Pipeline stage logic
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py   # CPCB label + rolling avgs
│   │   ├── feature_selection.py
│   │   ├── data_splitting.py
│   │   ├── model_trainer.py
│   │   ├── model_optimizer.py
│   │   └── model_evaluator.py
│   ├── pipeline/            # Stage entry points (stage_0N_*.py)
│   └── utils/
│       ├── aqi.py           # CPCB math — single source of truth for train & serve
│       ├── metrics.py       # CPCB category thresholds
│       └── ...
├── data/                    # DVC-tracked (not in Git)
├── models/                  # DVC-tracked intermediates; optimized/ kept in Git
├── configs/
│   └── cities.yaml
├── app.py                   # FastAPI serving (v6.0.0)
├── dvc.yaml                 # 8-stage pipeline definition
├── params.yaml              # All hyperparameters and paths
├── model.json.gz            # Deployable model artifact (0.26 MB)
├── features.txt             # 45 selected features in training order
├── encoders.json            # City/state label-encoding maps
├── Dockerfile
├── requirements.txt         # Training dependencies
└── requirements-api.txt     # Production API dependencies (minimal)
```

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

| Step | What it does |
|---|---|
| Verify | Checks `app.py`, `Dockerfile`, model artifact presence |
| Deploy → Hugging Face Spaces | Pushes via `huggingface_hub` SDK |
| Deploy → Docker Hub | Builds multi-platform image and pushes |

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

## License

MIT — see `LICENSE`.
