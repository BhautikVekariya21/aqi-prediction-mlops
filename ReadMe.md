# 🌍 AQI Prediction MLOps (Ultra-Lite & Physics-Aware)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-red)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue)](https://www.docker.com/)

A production-grade, memory-optimized Air Quality Index (AQI) prediction API for **30 Indian cities**.  
Deployed on **Hugging Face Spaces** with an MLOps pipeline powered by **GitHub Actions**.

🔗 **Live API:**  
➡ https://bhautikvekariya21-aqi-prediction-mlops.hf.space/docs

---

## ⚡ Key Features & Engineering

### 🚀 Ultra-Lite Architecture
- Removed heavy ML stack (Scikit-learn, Joblib, SciPy)
- **RAM usage ↓ ~70%**
- Uses **JSON.GZ model** (~72MB instead of 200MB+ Pickle)

### ⚡ High-Performance Computing
- **Parallel bulk prediction** (`ThreadPoolExecutor`)
  - ⏱ ~30s → **<3s**
- **Connection pooling** → no timeout under load

### 🧠 Physics-Aware Logic
- **AQI safety floor** when pollution > Severe category
- **Winter calibration** (Oct–Feb) to account for thermal inversion

---

## 📡 API Endpoints

### 1️⃣ Predict City AQI (Auto-Fetch)
Fetches real-time weather + pollution.

| Method | Endpoint | Params |
|--------|----------|--------|
| GET | `/predict/{city}` | `days` → default=2, max=5 |

Example:
```
https://bhautikvekariya21-aqi-prediction-mlops.hf.space/predict/Delhi?days=3
```

---

### 2️⃣ Bulk Prediction (All Cities)
| Method | Endpoint | Performance |
|--------|----------|-------------|
| GET | `/predict/all/cities` | <3 seconds |

---

### 3️⃣ Manual Prediction (Simulation Mode)
| Method | Endpoint |
|--------|----------|
| POST | `/predict/manual` |

Example:
```json
{
  "pm2_5": 355.0,
  "pm10": 410.0,
  "nitrogen_dioxide": 65.0,
  "wind_speed_10m": 5.0,
  "is_weekend": 1,
  "month": 11
}
```

---

### 4️⃣ Health Check

| Method | Endpoint  |
| ------ | --------- |
| GET    | `/health` |

---

## 🛠️ Tech Stack

| Component   | Technology          | Purpose                    |
| ----------- | ------------------- | -------------------------- |
| Framework   | FastAPI             | Async API                  |
| Model       | XGBoost Booster     | Optimized AQI model        |
| Data Source | Open-Meteo          | Live air quality & weather |
| Container   | Docker Slim         | Lightweight deployment     |
| CI/CD       | GitHub Actions      | Auto testing + deploy      |
| Deployment  | Hugging Face Spaces | Free cloud GPU/CPU         |

---

## 📂 Project Structure

```bash
.
├── .github/workflows/
│   └── ci.yml
├── models/
│   └── optimized/
│       └── features.txt
├── app.py
├── Dockerfile
├── model.json.gz
├── requirements.txt
└── README.md
```

---

## 🚀 Local Setup

### 1️⃣ Clone Repo

```bash
git clone https://github.com/BhautikVekariya21/aqi-prediction-mlops.git
cd aqi-prediction-mlops
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate         # Linux/Mac
venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

### 3️⃣ Run API

```bash
uvicorn app:app --reload
```

Swagger UI → http://localhost:8000/docs

---

## 🔄 CI/CD Pipeline (GitHub Actions)

✔ Auto triggered on **main** branch  
✔ Validates required files  
✔ Automatically relocates model if nested  
✔ Uploads to Hugging Face using APIs (no git-LFS issues)

🔐 Required Secret:
- `HF_TOKEN` → Hugging Face Write Token

---

## 📊 Supported Cities

| State       | City                     | State          | City      |
| ----------- | ------------------------ | -------------- | --------- |
| Delhi       | Delhi                    | Maharashtra    | Mumbai    |
| Karnataka   | Bengaluru                | West Bengal    | Kolkata   |
| Tamil Nadu  | Chennai                  | Telangana      | Hyderabad |
| Gujarat     | Ahmedabad                | Uttar Pradesh  | Lucknow   |
| Rajasthan   | Jaipur                   | Bihar          | Patna     |
| Punjab      | Chandigarh               | Madhya Pradesh | Bhopal    |
| Kerala      | Thiruvananthapuram       | Assam          | Guwahati  |
| Odisha      | Bhubaneswar              | Uttarakhand    | Dehradun  |
| ...and more | (See `/cities` endpoint) |                |           |

---

## 📝 License

Released under the **MIT License**.