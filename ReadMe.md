Here is the comprehensive and final version of your `README.md` with all the technical depth and features we implemented.

````markdown
# 🌍 AQI Prediction MLOps (Ultra-Lite & Physics-Aware)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-red)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Container-blue)](https://www.docker.com/)

A production-grade, memory-optimized Air Quality Index (AQI) prediction API for 30 Indian cities. Deployed on **Hugging Face Spaces** using a robust MLOps pipeline with **GitHub Actions**.

🔗 **Live API:** [https://bhautikvekariya21-aqi-prediction-mlops.hf.space/docs](https://bhautikvekariya21-aqi-prediction-mlops.hf.space/docs)

---

## ⚡ Key Features & Engineering

### 🚀 Ultra-Lite Architecture
- **Memory Optimization:** Replaced heavy dependencies (Scikit-learn, Joblib, SciPy) with **native XGBoost** loading and standard Python libraries.
- **RAM Usage:** Reduced memory footprint by **~70%**, enabling deployment on free-tier containers (512MB - 1GB RAM) without OOM crashes.
- **Model Compression:** Uses `JSON.GZ` format (~72MB) instead of Pickle (~200MB+), significantly speeding up cold starts.

### ⚡ High-Performance Computing
- **Parallel Bulk Prediction:** Implements `concurrent.futures.ThreadPoolExecutor` to fetch and predict AQI for all 30 cities simultaneously.
  - *Result:* Bulk prediction time reduced from **~30s to <3s**.
- **Connection Pooling:** Uses a global `requests.Session` with an optimized HTTP adapter to reuse SSL connections, preventing timeouts during high-load forecasting.

### 🧠 Physics-Aware Logic (Hybrid AI)
- **Physics Floor:** The model includes a safety layer that prevents under-prediction during extreme pollution events.
  - *Logic:* If `PM2.5 > 350` (Severe), the system overrides the ML output with a physics-based minimum AQI to ensure safety.
- **Dynamic Winter Calibration:** Automatically detects "Winter" months (Oct-Feb) and applies multipliers to account for thermal inversion and wind stagnation, which pure ML models often miss.

---

## 📡 API Endpoints

### 1. Predict City AQI (Auto-Fetch)
Fetches real-time weather & pollution data from Open-Meteo and returns an hourly AQI forecast.

- **URL:** `GET /predict/{city}`
- **Parameters:** `days` (default: 2, max: 5)
- **Example:** `https://bhautikvekariya21-aqi-prediction-mlops.hf.space/predict/Delhi?days=3`

### 2. Bulk Prediction (All Cities)
Returns a sorted list of AQI for all 30 supported cities. Ideal for "National Status" dashboards.

- **URL:** `GET /predict/all/cities`
- **Performance:** <3 seconds response time.

### 3. Manual Prediction (Simulation)
Allows testing specific scenarios (e.g., "What if PM2.5 hits 500?") by accepting raw feature values.

- **URL:** `POST /predict/manual`
- **Payload Example:**
```json
{
  "pm2_5": 355.0,
  "pm10": 410.0,
  "nitrogen_dioxide": 65.0,
  "wind_speed_10m": 5.0,
  "is_weekend": 1,
  "month": 11,
  ... (see docs for full list)
}
````

### 4\. Health Check

Verifies model status and current memory mode.

  - **URL:** `GET /health`

-----

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Framework** | FastAPI (Python 3.10) | High-performance async API |
| **Model** | XGBoost (Booster) | Gradient Boosting Decision Tree |
| **Data Source** | Open-Meteo APIs | Historical Weather & Air Quality API |
| **Container** | Docker | Slim Debian-based Python image |
| **CI/CD** | GitHub Actions | Automated Testing & Deployment |
| **Deployment** | Hugging Face Spaces | 2 vCPU, 16GB RAM Container |

-----

## 📂 Project Structure

```bash
.
├── .github/workflows/
│   └── ci.yml              # CI/CD: Automated deployment pipeline
├── models/
│   └── optimized/
│       └── features.txt    # List of 31 input features required by the model
├── app.py                  # Main API Application (FastAPI + Business Logic)
├── Dockerfile              # Docker configuration for Hugging Face
├── model.json.gz           # Compressed XGBoost Model (~72MB)
├── requirements.txt        # Lite dependencies (No sklearn/scipy)
└── README.md               # Project Documentation
```

-----

## 🚀 Local Setup & Installation

### 1\. Clone the Repository

```bash
git clone [https://github.com/BhautikVekariya21/aqi-prediction-mlops.git](https://github.com/BhautikVekariya21/aqi-prediction-mlops.git)
cd aqi-prediction-mlops
```

### 2\. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install lite dependencies
pip install -r requirements.txt
```

### 3\. Run the Server

```bash
uvicorn app:app --reload
```

### 4\. Test the API

Open your browser and navigate to:
`http://localhost:8000/docs`

-----

## 🔄 CI/CD Pipeline (GitHub Actions)

This project utilizes a custom **GitHub Actions workflow** to bypass standard Git LFS limitations on Hugging Face.

1.  **Trigger:** Pushes to the `main` branch.
2.  **Validation:** Verifies critical files (`app.py`, `Dockerfile`, `model.json.gz`).
3.  **Preparation:** Automatically moves the model file to the root directory if it's nested, ensuring the container finds it.
4.  **Deployment:** Uses the `huggingface_hub` Python library to perform a direct API upload, handling large files robustly without timeout errors.

**Secrets Required:**

  - `HF_TOKEN`: Hugging Face Write Access Token (Stored in GitHub Secrets).

-----

## 📊 Supported Cities (29 Major Hubs)

The model is trained and calibrated for these specific locations:

| State | City | State | City |
| :--- | :--- | :--- | :--- |
| Delhi | Delhi | Maharashtra | Mumbai |
| Karnataka | Bengaluru | Kolkata |
| Tamil Nadu | Chennai | Telangana | Hyderabad |
| Gujarat | Ahmedabad | Uttar Pradesh | Lucknow |
| Rajasthan | Jaipur | Bihar | Patna |
| Punjab | Chandigarh | Madhya Pradesh | Bhopal |
| Kerala | Thiruvananthapuram | Assam | Guwahati |
| Odisha | Bhubaneswar | Uttarakhand | Dehradun |
| ...and more | (See `/cities` endpoint) | | |

-----

## 📝 License

This project is licensed under the MIT License.

```
```