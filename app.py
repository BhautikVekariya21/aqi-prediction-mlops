"""
AQI Prediction API - Ultra Lite (OpenWeatherMap Calibrated)
Version: 3.2.0
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

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

gc.collect()

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(".") 
API_VERSION = "3.2.0 (OWM Calibration)"
EXTERNAL_API_TIMEOUT = 15.0 

# SECURE: Read OWM Token from Environment
OWM_TOKEN = os.getenv("OWM_TOKEN") 

REQUIRED_FEATURES = [
    'wind_gusts_10m', 'week_of_year', 'state_encoded', 'pm2_5', 'sulphur_dioxide',
    'longitude', 'surface_pressure', 'latitude', 'dust', 'pressure_msl',
    'pm10', 'cloud_cover', 'nitrogen_dioxide', 'year', 'cloud_cover_low',
    'relative_humidity_2m', 'month', 'carbon_monoxide', 'quarter',
    'wind_speed_10m', 'is_day', 'city_encoded', 'day', 'ozone',
    'cloud_cover_high', 'is_weekend', 'hour', 'aerosol_optical_depth',
    'dew_point_2m', 'cloud_cover_mid', 'wind_direction_10m'
]

print("🚀 Starting AQI Prediction API (OpenWeatherMap Mode)...")

# =============================================================================
# GLOBAL SESSION
# =============================================================================
session = requests.Session()
retry = requests.adapters.Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry)
session.mount('https://', adapter)
session.mount('http://', adapter)

HEADERS = {"User-Agent": "AQI-App/3.2"}

# =============================================================================
# LOAD MODEL & CITIES
# =============================================================================
print("📦 Loading model...")
model = None
try:
    if (MODEL_DIR / "model.json.gz").exists():
        with gzip.open(MODEL_DIR / "model.json.gz", 'rb') as f:
            model = xgb.Booster()
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            model.load_model(tmp_path); os.unlink(tmp_path)
    elif (MODEL_DIR / "model.json").exists():
        model = xgb.Booster(); model.load_model(str(MODEL_DIR / "model.json"))
except Exception as e: print(f"❌ Error: {e}")

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

# =============================================================================
# INDIAN AQI LOGIC (CPCB Standard)
# =============================================================================

def get_sub_index(conc, breakpoints):
    for (low_c, high_c, low_i, high_i) in breakpoints:
        if low_c <= conc <= high_c:
            return low_i + (high_i - low_i) * (conc - low_c) / (high_c - low_c)
    return 500 if conc > breakpoints[-1][1] else 0

def calculate_indian_aqi_floor(pm25: float, pm10: float) -> float:
    pm25_breakpoints = [(0,30,0,50), (30,60,51,100), (60,90,101,200), (90,120,201,300), (120,250,301,400), (250,5000,401,500)]
    pm10_breakpoints = [(0,50,0,50), (50,100,51,100), (100,250,101,200), (250,350,201,300), (350,430,301,400), (430,5000,401,500)]
    return max(get_sub_index(pm25, pm25_breakpoints), get_sub_index(pm10, pm10_breakpoints))

def aqi_category(aqi: float) -> Dict:
    if aqi <= 50: return {"cat": "Good", "emoji": "🟢", "color": "#00e400"}
    elif aqi <= 100: return {"cat": "Satisfactory", "emoji": "🟡", "color": "#ffff00"}
    elif aqi <= 200: return {"cat": "Moderate", "emoji": "🟠", "color": "#ff7e00"}
    elif aqi <= 300: return {"cat": "Poor", "emoji": "🔴", "color": "#ff0000"}
    elif aqi <= 400: return {"cat": "Very Poor", "emoji": "🟣", "color": "#8f3f97"}
    else: return {"cat": "Severe", "emoji": "🟤", "color": "#7e0023"}

# =============================================================================
# CALIBRATION LOGIC (OpenWeatherMap Only)
# =============================================================================

def get_owm_calibration_factor(lat: float, lon: float, model_pm25: float) -> float:
    """
    Fetches real-time PM2.5 from OpenWeatherMap to correct forecast bias.
    """
    if not OWM_TOKEN: 
        return 1.0
        
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_TOKEN}"
        resp = session.get(url, timeout=3.0).json()
        
        # OWM returns components in μg/m3
        if 'list' in resp and len(resp['list']) > 0:
            real_pm25 = float(resp['list'][0]['components']['pm2_5'])
            
            # Calculate Ratio
            model_val = max(model_pm25, 5.0) # Prevent zero division
            ratio = real_pm25 / model_val
            
            # Safety Caps: Min 0.5x, Max 6.0x (to catch extreme Delhi events)
            return max(0.5, min(ratio, 6.0))
            
    except Exception:
        pass # Fail silently to raw model if OWM fails
        
    return 1.0

# =============================================================================
# DATA PIPELINE
# =============================================================================

def fetch_data(lat: float, lon: float, days: int, city_name: str = None):
    try:
        # 1. Fetch Forecast (Open-Meteo)
        params = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "forecast_days": days}
        w_params = params.copy()
        w_params["hourly"] = ["relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
                              "wind_gusts_10m", "wind_direction_10m", "pressure_msl",
                              "surface_pressure", "cloud_cover", "cloud_cover_low",
                              "cloud_cover_mid", "cloud_cover_high", "is_day"]
        aq_params = params.copy()
        aq_params["hourly"] = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                               "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth"]

        weather = session.get("https://api.open-meteo.com/v1/forecast", params=w_params, headers=HEADERS, timeout=EXTERNAL_API_TIMEOUT).json()
        air_quality = session.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=aq_params, headers=HEADERS, timeout=EXTERNAL_API_TIMEOUT).json()

        # 2. Apply Calibration (OWM Ground Truth)
        if air_quality:
            try:
                forecast_pm25 = air_quality['hourly']['pm2_5'][0]
                
                # Get factor from OpenWeatherMap
                factor = get_owm_calibration_factor(lat, lon, forecast_pm25)
                
                if factor != 1.0:
                    air_quality['hourly']['pm2_5'] = [x * factor for x in air_quality['hourly']['pm2_5']]
                    air_quality['hourly']['pm10'] = [x * (factor * 0.95) for x in air_quality['hourly']['pm10']]
                    if factor > 2.0:
                        air_quality['hourly']['nitrogen_dioxide'] = [x * 1.15 for x in air_quality['hourly']['nitrogen_dioxide']]
            except Exception: pass

        return weather, air_quality
    except Exception: return None, None

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
# API MODELS & ENDPOINTS
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

app = FastAPI(title="AQI Prediction API", version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root(): return {"status": "ok", "api": "AQI Prediction API", "version": API_VERSION}

@app.get("/cities")
def get_cities(): 
    return {"total": len(CITIES), "cities": [{"name": c, "state": i["state"], "lat": i["lat"], "lon": i["lon"]} for c, i in sorted(CITIES.items())], "states": STATES}

@app.get("/health")
def health_check():
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "mode": "owm_calibrated"}

@app.get("/predict/{city}", response_model=PredictionResponse)
def predict_city_aqi(city: str, days: int = 2):
    if city not in CITIES: raise HTTPException(status_code=400, detail="City not found")
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    days = max(1, min(5, days)); city_info = CITIES[city]
    
    weather, air_quality = fetch_data(city_info['lat'], city_info['lon'], days, city_name=city)
    if not weather or not air_quality: raise HTTPException(status_code=503, detail="Open-Meteo API failed/timeout")
    
    df = prepare_features(weather, air_quality, city)
    if df is None or len(df) == 0: raise HTTPException(status_code=500, detail="Data processing failed")

    X = df[REQUIRED_FEATURES].values.astype(np.float32); X = np.nan_to_num(X, nan=0.0)
    dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
    raw_predictions = model.predict(dmatrix)
    final_aqi = []
    for i, pred in enumerate(raw_predictions):
        min_aqi = calculate_indian_aqi_floor(df.iloc[i]['pm2_5'], df.iloc[i]['pm10'])
        final_aqi.append(max(float(pred), min_aqi))
    df['aqi'] = final_aqi
    
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
    avg_aqi = float(np.mean(final_aqi)); o_cat = aqi_category(avg_aqi); del df, X, dmatrix, raw_predictions, final_aqi; gc.collect()
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

def process_single_city(city_name: str) -> Optional[Dict]:
    try:
        city_info = CITIES[city_name]
        weather, air_quality = fetch_data(city_info['lat'], city_info['lon'], 1, city_name=city_name)
        if weather and air_quality:
            df = prepare_features(weather, air_quality, city_name)
            if df is not None and len(df) > 0:
                current_row = df.iloc[[0]].copy() 
                min_aqi = calculate_indian_aqi_floor(current_row['pm2_5'].values[0], current_row['pm10'].values[0])
                X = current_row[REQUIRED_FEATURES].values.astype(np.float32); X = np.nan_to_num(X, nan=0.0)
                dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
                pred = float(model.predict(dmatrix)[0])
                final_aqi = max(pred, min_aqi)
                cat = aqi_category(final_aqi)
                return {
                    "city": city_name, "state": city_info['state'],
                    "current_aqi": round(final_aqi, 1), "category": cat['cat'], "emoji": cat['emoji']
                }
    except Exception: return None
    return None

@app.get("/predict/all/cities", response_model=List[CitySummary])
def predict_all_cities():
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_city = {executor.submit(process_single_city, city): city for city in CITIES}
        for future in concurrent.futures.as_completed(future_to_city):
            try:
                data = future.result()
                if data: results.append(data)
            except Exception: continue
    gc.collect()
    return sorted(results, key=lambda x: x['current_aqi'], reverse=True)

@app.post("/predict/manual", response_model=ManualResponse)
def predict_manual(features: ManualFeatures):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = features.dict(); df = pd.DataFrame([data])
        X = df[REQUIRED_FEATURES].values.astype(np.float32); X = np.nan_to_num(X, nan=0.0)
        dmatrix = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
        raw_pred = float(model.predict(dmatrix)[0])
        min_aqi = calculate_indian_aqi_floor(features.pm2_5, features.pm10)
        final_aqi = max(raw_pred, min_aqi)
        cat = aqi_category(final_aqi)
        return {
            "aqi": round(final_aqi, 1), "category": cat["cat"], "emoji": cat["emoji"], "color": cat["color"],
            "physics_floor_applied": final_aqi > raw_pred
        }
    except Exception as e: raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)