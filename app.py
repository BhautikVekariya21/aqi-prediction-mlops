"""
AQI Prediction API - Ultra Lite (Robust + Production Ready)
Version: 1.9.0
"""

import os
import gc
import sys
import gzip
import tempfile
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Heavy imports - Loaded cautiously
import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Force garbage collection immediately
gc.collect()

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(".") 
API_VERSION = "1.9.0"

# Strict timeout: If weather API takes >3s, drop the request
EXTERNAL_API_TIMEOUT = 3.0 

REQUIRED_FEATURES = [
    'wind_gusts_10m', 'week_of_year', 'state_encoded', 'pm2_5', 'sulphur_dioxide',
    'longitude', 'surface_pressure', 'latitude', 'dust', 'pressure_msl',
    'pm10', 'cloud_cover', 'nitrogen_dioxide', 'year', 'cloud_cover_low',
    'relative_humidity_2m', 'month', 'carbon_monoxide', 'quarter',
    'wind_speed_10m', 'is_day', 'city_encoded', 'day', 'ozone',
    'cloud_cover_high', 'is_weekend', 'hour', 'aerosol_optical_depth',
    'dew_point_2m', 'cloud_cover_mid', 'wind_direction_10m'
]

print("🚀 Starting AQI Prediction API (Robust Mode)...")

# =============================================================================
# GLOBAL SESSION (Thread-Safe & Retries)
# =============================================================================
session = requests.Session()

# Retry logic: If connection fails, try 1 more time, then give up
retry_strategy = requests.adapters.Retry(
    total=1,
    backoff_factor=0.2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry_strategy)
session.mount('https://', adapter)

# Define Headers to prevent Rate Limiting (Identify as a valid app)
HEADERS = {
    "User-Agent": "AQI-Prediction-App/1.0 (bhautik.vekariya@example.com)"
}

# =============================================================================
# LOAD MODEL (JSON/GZIP ONLY)
# =============================================================================

print("📦 Loading model...")
model = None

try:
    gzip_path = MODEL_DIR / "model.json.gz"
    json_path = MODEL_DIR / "model.json"

    if gzip_path.exists():
        print(f"✓ Found GZIP model at {gzip_path}...")
        with gzip.open(gzip_path, 'rb') as f:
            model_bytes = f.read()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        
        model = xgb.Booster()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
        del model_bytes
        print("✓ Model loaded successfully")
        
    elif json_path.exists():
        print(f"✓ Found JSON model at {json_path}...")
        model = xgb.Booster()
        model.load_model(str(json_path))
        print("✓ Model loaded successfully")
        
    else:
        print(f"❌ CRITICAL: No model file found in {MODEL_DIR.absolute()}")

except Exception as e:
    print(f"❌ Error loading model: {e}")

gc.collect()

# =============================================================================
# DATA & HELPERS
# =============================================================================

CITIES = {
    "Agartala": {"lat": 23.8315, "lon": 91.2868, "state": "Tripura"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Aizawl": {"lat": 23.7271, "lon": 92.7176, "state": "Mizoram"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Gangtok": {"lat": 27.3389, "lon": 88.6065, "state": "Sikkim"},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Imphal": {"lat": 24.8170, "lon": 93.9368, "state": "Manipur"},
    "Itanagar": {"lat": 27.0844, "lon": 93.6053, "state": "Arunachal Pradesh"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Kohima": {"lat": 25.6751, "lon": 94.1086, "state": "Nagaland"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Panaji": {"lat": 15.4909, "lon": 73.8278, "state": "Goa"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "Shillong": {"lat": 25.5788, "lon": 91.8933, "state": "Meghalaya"},
    "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Noida": {"lat": 28.5355, "lon": 77.3910, "state": "Uttar Pradesh"},
}

CITY_ENC = {c: i for i, c in enumerate(sorted(CITIES.keys()))}
STATES = sorted(set(v['state'] for v in CITIES.values()))
STATE_ENC = {s: i for i, s in enumerate(STATES)}

def aqi_category(aqi: float) -> Dict:
    if aqi <= 50: return {"cat": "Good", "emoji": "🟢", "color": "#00e400"}
    elif aqi <= 100: return {"cat": "Moderate", "emoji": "🟡", "color": "#ffff00"}
    elif aqi <= 150: return {"cat": "Unhealthy for Sensitive", "emoji": "🟠", "color": "#ff7e00"}
    elif aqi <= 200: return {"cat": "Unhealthy", "emoji": "🔴", "color": "#ff0000"}
    elif aqi <= 300: return {"cat": "Very Unhealthy", "emoji": "🟣", "color": "#8f3f97"}
    else: return {"cat": "Hazardous", "emoji": "🟤", "color": "#7e0023"}

def calculate_physics_min_aqi(pm2_5: float, pm10: float) -> float:
    if pm2_5 > 500: return 500 + (pm2_5 - 500) * 0.8
    elif pm2_5 > 350: return 400 + (pm2_5 - 350) * 0.8
    elif pm2_5 > 250: return 300 + (pm2_5 - 250) * 0.9
    elif pm2_5 > 150: return 200 + (pm2_5 - 150) * 1.0
    elif pm2_5 > 55:  return 150 + (pm2_5 - 55) * 1.0
    if pm10 > 430: return 400
    if pm10 > 350: return 300
    return 0.0

def fetch_data(lat: float, lon: float, days: int):
    try:
        # Reduced timeout and specific params
        params = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "forecast_days": days}
        
        w_params = params.copy()
        w_params["hourly"] = ["relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
                              "wind_gusts_10m", "wind_direction_10m", "pressure_msl",
                              "surface_pressure", "cloud_cover", "cloud_cover_low",
                              "cloud_cover_mid", "cloud_cover_high", "is_day"]
        
        aq_params = params.copy()
        aq_params["hourly"] = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                               "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth"]

        # Uses Global Session + Headers + Strict Timeout
        weather = session.get("https://api.open-meteo.com/v1/forecast", params=w_params, headers=HEADERS, timeout=EXTERNAL_API_TIMEOUT).json()
        air_quality = session.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=aq_params, headers=HEADERS, timeout=EXTERNAL_API_TIMEOUT).json()
        
        return weather, air_quality
    except Exception as e:
        # Fail silently for bulk requests, log only
        # print(f"Error fetching data: {e}") 
        return None, None

def safe_get(data_dict, key, index, default, total_length):
    try:
        values = data_dict.get(key, [default] * total_length)
        return values[index] if index < len(values) and values[index] is not None else default
    except: return default

def prepare_features(weather: Dict, air_quality: Dict, city: str) -> Optional[pd.DataFrame]:
    city_info = CITIES[city]
    weather_hourly = weather.get('hourly', {})
    air_hourly = air_quality.get('hourly', {})
    n_hours = len(weather_hourly.get('time', []))
    
    if n_hours == 0: return None
    
    rows = []
    # Only process first 24 hours to save RAM/Time if doing bulk
    limit = n_hours if n_hours < 48 else 48 
    
    for i in range(limit):
        try:
            dt = pd.to_datetime(weather_hourly['time'][i])
            row = {
                'year': dt.year, 'month': dt.month, 'day': dt.day, 'hour': dt.hour,
                'quarter': (dt.month - 1) // 3 + 1, 'week_of_year': dt.isocalendar()[1],
                'is_weekend': 1 if dt.dayofweek >= 5 else 0,
                'is_day': safe_get(weather_hourly, 'is_day', i, 1, n_hours),
                'latitude': city_info['lat'], 'longitude': city_info['lon'],
                'city_encoded': CITY_ENC.get(city, 0), 'state_encoded': STATE_ENC.get(city_info['state'], 0),
                'pm2_5': safe_get(air_hourly, 'pm2_5', i, 50, n_hours),
                'pm10': safe_get(air_hourly, 'pm10', i, 80, n_hours),
                'ozone': safe_get(air_hourly, 'ozone', i, 50, n_hours),
                'nitrogen_dioxide': safe_get(air_hourly, 'nitrogen_dioxide', i, 30, n_hours),
                'sulphur_dioxide': safe_get(air_hourly, 'sulphur_dioxide', i, 10, n_hours),
                'carbon_monoxide': safe_get(air_hourly, 'carbon_monoxide', i, 500, n_hours),
                'dust': safe_get(air_hourly, 'dust', i, 10, n_hours),
                'aerosol_optical_depth': safe_get(air_hourly, 'aerosol_optical_depth', i, 0.3, n_hours),
                'relative_humidity_2m': safe_get(weather_hourly, 'relative_humidity_2m', i, 60, n_hours),
                'dew_point_2m': safe_get(weather_hourly, 'dew_point_2m', i, 15, n_hours),
                'wind_speed_10m': safe_get(weather_hourly, 'wind_speed_10m', i, 15, n_hours),
                'wind_gusts_10m': safe_get(weather_hourly, 'wind_gusts_10m', i, 20, n_hours),
                'wind_direction_10m': safe_get(weather_hourly, 'wind_direction_10m', i, 180, n_hours),
                'pressure_msl': safe_get(weather_hourly, 'pressure_msl', i, 1013, n_hours),
                'surface_pressure': safe_get(weather_hourly, 'surface_pressure', i, 1013, n_hours),
                'cloud_cover': safe_get(weather_hourly, 'cloud_cover', i, 30, n_hours),
                'cloud_cover_low': safe_get(weather_hourly, 'cloud_cover_low', i, 0, n_hours),
                'cloud_cover_mid': safe_get(weather_hourly, 'cloud_cover_mid', i, 0, n_hours),
                'cloud_cover_high': safe_get(weather_hourly, 'cloud_cover_high', i, 0, n_hours),
                'datetime': dt,
            }
            rows.append(row)
        except Exception: continue
    
    return pd.DataFrame(rows) if rows else None

# =============================================================================
# API MODELS
# =============================================================================

class HourlyForecast(BaseModel):
    datetime: str; hour: int; aqi: float; category: str; emoji: str; color: str
    pm2_5: float; pm10: float; ozone: float; nitrogen_dioxide: float
    sulphur_dioxide: float; carbon_monoxide: float; relative_humidity_2m: float; wind_speed_10m: float

class DailySummary(BaseModel):
    date: str; avg_aqi: float; max_aqi: float; min_aqi: float; category: str; emoji: str; color: str

class PredictionResponse(BaseModel):
    success: bool; city: str; state: str; coordinates: Dict; forecast_days: int
    generated_at: str; hourly: List[HourlyForecast]; daily: List[DailySummary]; summary: Dict

class ManualFeatures(BaseModel):
    pm2_5: float; pm10: float; nitrogen_dioxide: float; sulphur_dioxide: float
    ozone: float; carbon_monoxide: float; dust: float; aerosol_optical_depth: float
    wind_speed_10m: float; wind_gusts_10m: float; wind_direction_10m: float
    relative_humidity_2m: float; dew_point_2m: float; surface_pressure: float
    pressure_msl: float; cloud_cover: float; cloud_cover_low: float
    cloud_cover_mid: float; cloud_cover_high: float; is_day: int
    latitude: float; longitude: float; city_encoded: int; state_encoded: int
    year: int; month: int; day: int; hour: int; quarter: int
    week_of_year: int; is_weekend: int

class ManualResponse(BaseModel):
    aqi: float; category: str; emoji: str; color: str; physics_floor_applied: bool

class CitySummary(BaseModel):
    city: str; state: str; current_aqi: float; category: str; emoji: str

# =============================================================================
# API ENDPOINTS
# =============================================================================

app = FastAPI(title="AQI Prediction API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root(): return {"status": "ok", "api": "AQI Prediction API", "version": API_VERSION}

@app.get("/cities")
def get_cities(): 
    return {"total": len(CITIES), "cities": [{"name": c, "state": i["state"], "lat": i["lat"], "lon": i["lon"]} for c, i in sorted(CITIES.items())], "states": STATES}

@app.get("/health")
def health_check():
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "mode": "ultra_lite"}

@app.get("/predict/{city}", response_model=PredictionResponse)
def predict_city_aqi(city: str, days: int = 2):
    if city not in CITIES: raise HTTPException(status_code=400, detail="City not found")
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    days = max(1, min(5, days))
    city_info = CITIES[city]
    
    weather, air_quality = fetch_data(city_info['lat'], city_info['lon'], days)
    if not weather or not air_quality: raise HTTPException(status_code=503, detail="Open-Meteo API failed/timeout")
    
    df = prepare_features(weather, air_quality, city)
    if df is None or len(df) == 0: raise HTTPException(status_code=500, detail="Data processing failed")

    # Winter Calibration (Same logic as before)
    current_month = df['month'].iloc[0]
    is_winter = current_month in [10, 11, 12, 1, 2]
    CITY_TIERS = {"Delhi": 1.5, "Gurugram": 1.5, "Noida": 1.5, "Ghaziabad": 1.5, "Lucknow": 1.4, "Patna": 1.4, "Kanpur": 1.4, "Ahmedabad": 1.3, "Chandigarh": 1.3, "Jaipur": 1.3, "Kolkata": 1.2}

    if is_winter:
        base_factor = CITY_TIERS.get(city, 1.0)
        if base_factor > 1.0:
            for idx in df.index:
                raw_pm25 = df.at[idx, 'pm2_5']; raw_wind = df.at[idx, 'wind_speed_10m']
                if raw_wind > 12.0: active_factor = 1.2; wind_correction = 1.0
                else:
                    if raw_pm25 < 80.0: active_factor = base_factor
                    elif raw_pm25 < 150.0: active_factor = base_factor * 1.5
                    else: active_factor = base_factor * 2.2
                    wind_correction = 0.6
                df.at[idx, 'pm2_5'] = raw_pm25 * active_factor
                df.at[idx, 'pm10'] = df.at[idx, 'pm10'] * (active_factor * 0.9)
                df.at[idx, 'wind_speed_10m'] = raw_wind * wind_correction
                if active_factor > 1.5: df.at[idx, 'nitrogen_dioxide'] = df.at[idx, 'nitrogen_dioxide'] * 1.3

    # Prediction
    X = df[REQUIRED_FEATURES].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
    raw_predictions = model.predict(dmatrix)
    
    final_aqi = []
    for i, pred in enumerate(raw_predictions):
        min_aqi = calculate_physics_min_aqi(df.iloc[i]['pm2_5'], df.iloc[i]['pm10'])
        final_aqi.append(max(float(pred), min_aqi))
    df['aqi'] = final_aqi
    
    # Response
    hourly_forecast = []
    for _, row in df.iterrows():
        cat = aqi_category(row['aqi'])
        hourly_forecast.append({
            "datetime": row['datetime'].isoformat(), "hour": int(row['hour']),
            "aqi": round(row['aqi'], 1), "category": cat["cat"], "emoji": cat["emoji"], "color": cat["color"],
            "pm2_5": round(row['pm2_5'], 1), "pm10": round(row['pm10'], 1),
            "ozone": round(row['ozone'], 1), "nitrogen_dioxide": round(row['nitrogen_dioxide'], 1),
            "sulphur_dioxide": round(row['sulphur_dioxide'], 1), "carbon_monoxide": round(row['carbon_monoxide'], 1),
            "relative_humidity_2m": round(row['relative_humidity_2m'], 1), "wind_speed_10m": round(row['wind_speed_10m'], 1)
        })
    
    df['date'] = df['datetime'].dt.date
    daily_summary = []
    for date, group in df.groupby('date'):
        d_cat = aqi_category(group['aqi'].mean())
        daily_summary.append({
            "date": str(date), "avg_aqi": round(float(group['aqi'].mean()), 1),
            "max_aqi": round(float(group['aqi'].max()), 1), "min_aqi": round(float(group['aqi'].min()), 1),
            "category": d_cat["cat"], "emoji": d_cat["emoji"], "color": d_cat["color"]
        })
    
    avg_aqi = float(np.mean(final_aqi))
    o_cat = aqi_category(avg_aqi)
    del df, X, dmatrix, raw_predictions, final_aqi; gc.collect()

    return {
        "success": True, "city": city, "state": city_info['state'],
        "coordinates": {"lat": city_info['lat'], "lon": city_info['lon']},
        "forecast_days": days, "generated_at": datetime.now().isoformat(),
        "hourly": hourly_forecast, "daily": daily_summary,
        "summary": {
            "avg_aqi": round(avg_aqi, 1), "max_aqi": round(max([h['aqi'] for h in hourly_forecast]), 1),
            "min_aqi": round(min([h['aqi'] for h in hourly_forecast]), 1), "category": o_cat["cat"],
            "emoji": o_cat["emoji"], "color": o_cat["color"], "total_hours": len(hourly_forecast)
        }
    }

# =============================================================================
# BULK PREDICT (THREADED PARALLEL PROCESSING + ROBUSTNESS)
# =============================================================================
def process_single_city(city_name: str) -> Optional[Dict]:
    """Helper function to be run in parallel for each city"""
    try:
        city_info = CITIES[city_name]
        # Fetch minimal data (1 day) - timeout is handled inside fetch_data
        weather, air_quality = fetch_data(city_info['lat'], city_info['lon'], 1)
        
        if weather and air_quality:
            df = prepare_features(weather, air_quality, city_name)
            if df is not None and len(df) > 0:
                current_row = df.iloc[[0]].copy() 
                min_aqi = calculate_physics_min_aqi(current_row['pm2_5'].values[0], current_row['pm10'].values[0])
                X = current_row[REQUIRED_FEATURES].values.astype(np.float32)
                X = np.nan_to_num(X, nan=0.0)
                dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
                pred = float(model.predict(dmatrix)[0])
                final_aqi = max(pred, min_aqi)
                cat = aqi_category(final_aqi)
                
                return {
                    "city": city_name,
                    "state": city_info['state'],
                    "current_aqi": round(final_aqi, 1),
                    "category": cat['cat'],
                    "emoji": cat['emoji']
                }
    except Exception:
        # If one city fails, return None so the rest still work
        return None
    return None

@app.get("/predict/all/cities", response_model=List[CitySummary])
def predict_all_cities():
    """Fetches AQI for ALL supported cities concurrently."""
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    # 10 Workers is the sweet spot for Free Tier
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_city = {executor.submit(process_single_city, city): city for city in CITIES}
        
        for future in concurrent.futures.as_completed(future_to_city):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                continue # Skip failures gracefully

    # Memory cleanup after heavy operation
    gc.collect()
    
    return sorted(results, key=lambda x: x['current_aqi'], reverse=True)

@app.post("/predict/manual", response_model=ManualResponse)
def predict_manual(features: ManualFeatures):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = features.dict()
        df = pd.DataFrame([data])
        X = df[REQUIRED_FEATURES].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
        raw_pred = float(model.predict(dmatrix)[0])
        min_aqi = calculate_physics_min_aqi(features.pm2_5, features.pm10)
        final_aqi = max(raw_pred, min_aqi)
        cat = aqi_category(final_aqi)
        
        return {
            "aqi": round(final_aqi, 1),
            "category": cat["cat"],
            "emoji": cat["emoji"],
            "color": cat["color"],
            "physics_floor_applied": final_aqi > raw_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)