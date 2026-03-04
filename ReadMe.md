---
title: AQI Prediction MLOps
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🌍 AQI Prediction MLOps (Ultra-Lite & Physics-Aware)

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Deployed-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/bhautikvekariya21/aqi-prediction-mlops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**A production-grade, memory-optimized air quality forecasting engine for 29 Indian cities.**

<p align="center">
  <a href="https://bhautikvekariya21-aqi-prediction-mlops.hf.space/docs"><img src="https://img.shields.io/badge/Live%20API-Demo-0A66C2?style=for-the-badge&logo=fastapi&logoColor=white" alt="Live API Demo"/></a>
  <a href="https://aqi-predictor.lovable.app"><img src="https://img.shields.io/badge/Website-Open-2EA44F?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Live Website"/></a>
  <a href="https://www.youtube.com/"><img src="https://img.shields.io/badge/YouTube-Watch%20Demo-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube"/></a>
  <a href="https://hub.docker.com/r/bhautikvekariya21/aqi-prediction-api"><img src="https://img.shields.io/badge/Docker%20Hub-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Hub"/></a>
  <a href="../../issues"><img src="https://img.shields.io/badge/Report-Bug-D73A49?style=for-the-badge&logo=github&logoColor=white" alt="Report Bug"/></a>
</p>

</div>

---

## 📖 Overview

The **AQI Prediction MLOps** project is a high-performance REST API capable of forecasting Air Quality Index (AQI) values in real-time. Unlike standard ML deployments, this system features a custom **"Ultra-Lite" architecture** that reduces memory overhead by 70%, allowing it to run efficiently on resource-constrained cloud tiers (like Hugging Face Spaces free tier).

It integrates **real-time meteorological data** from Open-Meteo with a trained **XGBoost** model, enhanced by a **hybrid physics layer** that corrects predictions during extreme weather events (e.g., smog, winter inversion).

---

## ⚡ Key Innovations

### 🧠 Hybrid Physics-ML Engine
* **Physics Floor Protocol:** Pure ML models often under-predict extreme outliers. Our system enforces a physics-based floor: if `PM2.5 > 350` (Severe), the logic overrides the model to prevent dangerous false negatives.
* **Dynamic Winter Calibration:** Automatically detects stagnation periods (Oct–Feb) and applies meteorological multipliers to account for thermal inversion, which standard regression models miss.

### 🚀 Ultra-Lite Architecture
* **Zero-Heavy-Dependency:** We stripped out heavy libraries like `scikit-learn` and `scipy`. The model runs on **native XGBoost** and standard Python libraries.
* **Memory Optimization:** Reduced RAM usage from ~1.2GB to **<400MB**, eliminating OOM (Out of Memory) crashes on free-tier containers.
* **Parallel Computing:** Implements `ThreadPoolExecutor` to fetch and predict data for all 29 cities simultaneously, reducing bulk response time from **30s to <3s**.

### 🛡️ Robust Engineering
* **Connection Pooling:** Uses a global `requests.Session` with an optimized HTTP adapter to reuse SSL connections.
* **Smart Retries:** Implements exponential backoff strategies to handle external API rate limits gracefully.

---

## 🏗️ Architecture

### Request-to-Prediction Flow

1. **Client Request** → User or frontend calls `GET /predict/{city}` (or bulk endpoint).
2. **FastAPI Gateway** → Validates inputs and routes request to inference services.
3. **Weather Fetch Layer** → Pulls real-time meteorological features from Open-Meteo.
4. **Preprocessing Layer** → Aligns incoming weather data to the model feature schema.
5. **XGBoost Inference** → Generates baseline AQI prediction using the optimized model artifact.
6. **Physics Calibration Layer** → Applies safety floor + seasonal calibration during extreme conditions.
7. **Response Builder** → Returns final AQI, category, and forecast payload as JSON.

### Runtime Components

- **API Layer:** FastAPI app serving prediction endpoints.
- **Inference Layer:** Lightweight predictor + model loader for low-memory environments.
- **External Dependency:** Open-Meteo API for live weather inputs.
- **Concurrency:** ThreadPoolExecutor for multi-city parallel prediction.
- **Reliability Controls:** Shared HTTP session, retries, and backoff handling.

### Input and Output Contract (High-Level)

- **Input Sources:**
  - Path parameter: `city` (e.g., `Delhi`, `Mumbai`).
  - Query parameter: `days` forecast horizon (default: `2`).
  - Manual endpoint payload: complete model feature vector for simulation use-cases.
- **Output Fields:**
  - Current AQI value and category.
  - Time-series/hourly forecast for requested horizon.
  - Metadata required by dashboard clients for rendering status cards.

### Resilience and Fallback Strategy

- **External API instability:** Retries with exponential backoff reduce transient failures.
- **Connection overhead:** Shared HTTP session keeps latency low via connection reuse.
- **Extreme pollution edge-cases:** Physics floor prevents unsafe underestimation.
- **Multi-city burst traffic:** Parallel workers keep bulk endpoint response time stable.

### Deployment Notes

- Designed for low-memory cloud runtimes (free/shared tiers).
- Model artifact is compressed and loaded through a lightweight path.
- API service is container-ready and published for direct Docker deployment.

---

## 📡 API Reference

**Base URL:** `https://bhautikvekariya21-aqi-prediction-mlops.hf.space`

### 1. 🏙️ Real-Time Prediction

Get the current AQI and hourly forecast for a specific city.

- **Endpoint:** `GET /predict/{city}`
- **Parameters:** `days` (optional, default=2)
- **Example:**
  ```bash
  curl -X 'GET' \
    'https://bhautikvekariya21-aqi-prediction-mlops.hf.space/predict/Delhi?days=2' \
    -H 'accept: application/json'
  ```

### 2. 📊 National Dashboard (Bulk)

Get the current status of all 29 supported cities in a single request.

- **Endpoint:** `GET /predict/all/cities`
- **Performance:** Returns data for 29 cities in <3 seconds using threading.

### 3. 🧪 Simulation Mode (Manual)

Test "what-if" scenarios by sending raw environmental features manually.

- **Endpoint:** `POST /predict/manual`
- **Payload:** JSON object with 31 features (PM2.5, Wind Speed, etc.).

---

## 🚀 Getting Started

### Option A: Run with Docker (Recommended)

You don't need to install Python or libraries. Just run the container.

```bash
# Pull the pre-built image
docker pull bhautikvekariya21/aqi-prediction-api:latest

# Run on port 8000
docker run -p 8000:7860 bhautikvekariya21/aqi-prediction-api:latest
```

*Visit `http://localhost:8000/docs` to test.*

### Option B: Run from Source

1. **Clone the repository**

   ```bash
   git clone https://github.com/BhautikVekariya21/aqi-prediction-mlops.git
   cd aqi-prediction-mlops
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Server**

   ```bash
   uvicorn app:app --reload
   ```

---

## 🔄 CI/CD Pipeline

This project uses a sophisticated **Dual-Target CI/CD Pipeline** powered by GitHub Actions.

| Stage | Description | Target |
| :--- | :--- | :--- |
| **Verification** | Checks critical files (`app.py`, `Dockerfile`) and model integrity. | GitHub Actions Runner |
| **Deployment A** | Uses `huggingface_hub` Python SDK to bypass Git LFS limits and deploy the API. | **Hugging Face Spaces** |
| **Deployment B** | Builds a multi-platform Docker image and pushes it to the registry. | **Docker Hub** |

**Workflow File:** `.github/workflows/ci.yml`

---

## 📂 Project Structure

```bash
aqi-prediction-mlops/
├── .github/workflows/      # CI/CD pipelines
│   └── ci.yml              # Main deployment workflow
├── models/
│   └── optimized/          # Feature definitions
├── app.py                  # Main Application (FastAPI + Logic)
├── Dockerfile              # Production Docker configuration
├── model.json.gz           # Compressed XGBoost Model (~72MB)
├── requirements.txt        # Lite dependencies
└── README.md               # Documentation
```

---

## 📊 Supported Cities

The model is calibrated for 29 major Indian cities (including all state capitals and union territories):

| City | State/UT | City | State/UT | City | State/UT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Agartala | Tripura | Dehradun | Uttarakhand | Kohima | Nagaland |
| Ahmedabad | Gujarat | Delhi | Delhi | Kolkata | West Bengal |
| Aizawl | Mizoram | Gangtok | Sikkim | Lucknow | Uttar Pradesh |
| Bengaluru | Karnataka | Gurugram | Haryana | Mumbai | Maharashtra |
| Bhopal | Madhya Pradesh | Guwahati | Assam | Panaji | Goa |
| Bhubaneswar | Odisha | Hyderabad | Telangana | Patna | Bihar |
| Chandigarh | Punjab | Imphal | Manipur | Raipur | Chhattisgarh |
| Chennai | Tamil Nadu | Itanagar | Arunachal Pradesh | Ranchi | Jharkhand |
| | | Jaipur | Rajasthan | Shillong | Meghalaya |
| | | | | Shimla | Himachal Pradesh |
| | | | | Thiruvananthapuram | Kerala |
| | | | | Visakhapatnam | Andhra Pradesh |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for more information.
