"""
AQI Prediction API - SIMPLIFIED
Just loads model and serves predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import joblib

# =============================================================================
# LOAD MODEL (SIMPLE)
# =============================================================================

MODEL_PATH = Path("models/optimized/model_final.pkl")
FEATURES_PATH = Path("models/optimized/features.txt")

print("Loading model...")
model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded: {type(model).__name__}")

print("Loading features...")
with open(FEATURES_PATH, 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]
print(f"✓ Features loaded: {len(feature_names)}")

# =============================================================================
# CITIES DATA
# =============================================================================

CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
}

CITY_ENC = {c: i for i, c in enumerate(sorted(CITIES.keys()))}
STATE_ENC = {s: i for i, s in enumerate(sorted(set(v['state'] for v in CITIES.values())))}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def aqi_category(aqi: float) -> Dict:
    """Get AQI category"""
    if aqi <= 50: return {"cat": "Good", "emoji": "🟢"}
    elif aqi <= 100: return {"cat": "Moderate", "emoji": "🟡"}
    elif aqi <= 150: return {"cat": "Unhealthy for Sensitive", "emoji": "🟠"}
    elif aqi <= 200: return {"cat": "Unhealthy", "emoji": "🔴"}
    elif aqi <= 300: return {"cat": "Very Unhealthy", "emoji": "🟣"}
    else: return {"cat": "Hazardous", "emoji": "🟤"}

def fetch_api_data(lat: float, lon: float, days: int = 2):
    """Fetch from Open-Meteo"""
    try:
        weather = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly": ["relative_humidity_2m", "dew_point_2m", "wind_speed_10m", 
                      "pressure_msl", "surface_pressure", "cloud_cover",
                      "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "is_day"],
            "timezone": "Asia/Kolkata", "forecast_days": days
        }, timeout=10).json()
        
        air = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": lat, "longitude": lon,
            "hourly": ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                      "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth"],
            "timezone": "Asia/Kolkata", "forecast_days": days
        }, timeout=10).json()
        
        return weather, air
    except:
        return None, None

def prepare_features(weather, air, city: str):
    """Prepare features for prediction"""
    city_info = CITIES[city]
    w_hourly = weather.get('hourly', {})
    a_hourly = air.get('hourly', {})
    n_hours = len(w_hourly.get('time', []))
    
    rows = []
    for i in range(n_hours):
        dt = pd.to_datetime(w_hourly['time'][i])
        
        row = {
            'year': dt.year, 'month': dt.month, 'day': dt.day, 'hour': dt.hour,
            'quarter': (dt.month - 1) // 3 + 1, 'week_of_year': dt.isocalendar()[1],
            'is_weekend': 1 if dt.dayofweek >= 5 else 0,
            'is_day': w_hourly.get('is_day', [1]*n_hours)[i] or 1,
            'latitude': city_info['lat'], 'longitude': city_info['lon'],
            'city_encoded': CITY_ENC.get(city, 0),
            'state_encoded': STATE_ENC.get(city_info['state'], 0),
            'pm2_5': a_hourly.get('pm2_5', [50]*n_hours)[i] or 50,
            'pm10': a_hourly.get('pm10', [80]*n_hours)[i] or 80,
            'ozone': a_hourly.get('ozone', [50]*n_hours)[i] or 50,
            'nitrogen_dioxide': a_hourly.get('nitrogen_dioxide', [30]*n_hours)[i] or 30,
            'sulphur_dioxide': a_hourly.get('sulphur_dioxide', [10]*n_hours)[i] or 10,
            'carbon_monoxide': a_hourly.get('carbon_monoxide', [500]*n_hours)[i] or 500,
            'dust': a_hourly.get('dust', [10]*n_hours)[i] or 10,
            'aerosol_optical_depth': a_hourly.get('aerosol_optical_depth', [0.3]*n_hours)[i] or 0.3,
            'relative_humidity_2m': w_hourly.get('relative_humidity_2m', [60]*n_hours)[i] or 60,
            'dew_point_2m': w_hourly.get('dew_point_2m', [15]*n_hours)[i] or 15,
            'wind_speed_10m': w_hourly.get('wind_speed_10m', [15]*n_hours)[i] or 15,
            'wind_gusts_10m': w_hourly.get('wind_speed_10m', [20]*n_hours)[i] or 20,
            'wind_direction_10m': 180,
            'pressure_msl': w_hourly.get('pressure_msl', [1013]*n_hours)[i] or 1013,
            'surface_pressure': w_hourly.get('surface_pressure', [1013]*n_hours)[i] or 1013,
            'cloud_cover': w_hourly.get('cloud_cover', [30]*n_hours)[i] or 30,
            'cloud_cover_low': w_hourly.get('cloud_cover_low', [0]*n_hours)[i] or 0,
            'cloud_cover_mid': w_hourly.get('cloud_cover_mid', [0]*n_hours)[i] or 0,
            'cloud_cover_high': w_hourly.get('cloud_cover_high', [0]*n_hours)[i] or 0,
            'datetime': dt,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="AQI Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    return {"status": "ok", "api": "AQI Prediction API", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "features": len(feature_names),
        "cities": len(CITIES)
    }

@app.get("/cities")
def get_cities():
    return {
        "total": len(CITIES),
        "cities": [{"name": c, **info} for c, info in CITIES.items()]
    }

@app.get("/predict/{city}")
def predict(city: str, days: int = 2):
    """Predict AQI for a city"""
    
    # Validate
    if city not in CITIES:
        raise HTTPException(400, f"City not found. Available: {list(CITIES.keys())}")
    
    days = max(1, min(3, days))
    city_info = CITIES[city]
    
    # Fetch data
    weather, air = fetch_api_data(city_info['lat'], city_info['lon'], days)
    if not weather or not air:
        raise HTTPException(503, "Failed to fetch data from Open-Meteo")
    
    # Prepare features
    df = prepare_features(weather, air, city)
    
    # Predict
    X = df[feature_names].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    predictions = model.predict(X)
    
    df['aqi'] = predictions
    
    # Format response
    hourly = []
    for _, row in df.iterrows():
        cat = aqi_category(row['aqi'])
        hourly.append({
            "datetime": row['datetime'].isoformat(),
            "hour": int(row['hour']),
            "aqi": round(float(row['aqi']), 1),
            "category": cat["cat"],
            "emoji": cat["emoji"],
            "pm2_5": round(float(row['pm2_5']), 1),
            "pm10": round(float(row['pm10']), 1),
        })
    
    return {
        "success": True,
        "city": city,
        "state": city_info['state'],
        "forecast_days": days,
        "generated_at": datetime.now().isoformat(),
        "hourly": hourly,
        "summary": {
            "avg_aqi": round(float(predictions.mean()), 1),
            "max_aqi": round(float(predictions.max()), 1),
            "min_aqi": round(float(predictions.min()), 1),
            "total_hours": len(hourly)
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))