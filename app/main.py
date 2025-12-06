# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from predictor import predict_aqi

app = FastAPI(
    title="India AQI Predictor API",
    description="6.6 MB XGBoost model – 2-day forecast ready",
    version="1.0.0"
)

class AQIInput(BaseModel):
    city_encoded: int
    state_encoded: int
    latitude: float
    longitude: float
    month: int
    is_weekend: int = 0
    pm2_5_ugm3: float = 50.0
    pm10_ugm3: float = 80.0
    co_ugm3: float = 500.0
    no2_ugm3: float = 30.0
    so2_ugm3: float = 10.0
    o3_ugm3: float = 50.0
    dust_ugm3: float = 10.0
    aod: float = 0.3
    humidity_percent: float = 60.0
    dew_point_c: float = 15.0
    wind_gusts_kmh: float = 20.0
    precipitation_mm: float = 0.0
    pressure_msl_hpa: float = 1013.0
    cloud_cover_percent: float = 30.0

@app.get("/")
def home():
    return {"message": "AQI Prediction API is LIVE", "status": "healthy"}

@app.post("/predict")
def predict(input: AQIInput):
    try:
        result = predict_aqi(input.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)