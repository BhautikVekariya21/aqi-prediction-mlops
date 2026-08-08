"""
AQI Prediction API — Indian CPCB Standard
Version: 6.0.0

Design contract:
  - Feature vector built identically to training (7 CPCB rolling avgs + instantaneous + datetime)
  - No seasonal multipliers, no physics floor, no post-hoc distortions
  - Encoders loaded from encoders.json (written by feature_engineering stage)
  - Feature list loaded from features.txt (written by feature_selection stage)
  - Output clamped to [0, 500] CPCB scale
"""

import gc
import gzip
import json
import os
import tempfile
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

gc.collect()

# =============================================================================
# STARTUP — load model artifacts
# =============================================================================

MODEL_DIR = Path(".")
API_VERSION = "6.0.0 (CPCB)"
EXTERNAL_API_TIMEOUT = 10.0

print("Starting AQI Prediction API (CPCB standard)...")

model: Optional[xgb.Booster] = None
try:
    gz = MODEL_DIR / "model.json.gz"
    plain = MODEL_DIR / "model.json"
    if gz.exists():
        with gzip.open(gz, "rb") as fh, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(fh.read())
            tmp_path = tmp.name
        model = xgb.Booster()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
    elif plain.exists():
        model = xgb.Booster()
        model.load_model(str(plain))
    print(f"Model loaded: {gz if gz.exists() else plain}")
except Exception as exc:
    print(f"Model load failed: {exc}")

REQUIRED_FEATURES: List[str] = []
try:
    fp = MODEL_DIR / "features.txt"
    if fp.exists():
        REQUIRED_FEATURES = [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]
        print(f"Features loaded: {len(REQUIRED_FEATURES)}")
    else:
        print("WARNING: features.txt not found — predictions disabled")
except Exception as exc:
    print(f"Feature list load failed: {exc}")

ENCODERS: Dict[str, Dict] = {}
try:
    ep = MODEL_DIR / "encoders.json"
    if ep.exists():
        ENCODERS = json.loads(ep.read_text())
        print(f"Encoders loaded: {list(ENCODERS.keys())}")
    else:
        print("WARNING: encoders.json not found")
except Exception as exc:
    print(f"Encoders load failed: {exc}")

CITY_ENC: Dict[str, int] = ENCODERS.get("city", {})
STATE_ENC: Dict[str, int] = ENCODERS.get("state", {})

# =============================================================================
# CITIES — 29 training cities (Noida excluded: not present in training data)
# =============================================================================

CITIES: Dict[str, Dict] = {
    "Agartala":          {"lat": 23.8315, "lon": 91.2868, "state": "Tripura"},
    "Ahmedabad":         {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "Aizawl":            {"lat": 23.7271, "lon": 92.7176, "state": "Mizoram"},
    "Bengaluru":         {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Bhopal":            {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "Bhubaneswar":       {"lat": 20.2961, "lon": 85.8245, "state": "Odisha"},
    "Chandigarh":        {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Chennai":           {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Dehradun":          {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Delhi":             {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "Gangtok":           {"lat": 27.3389, "lon": 88.6065, "state": "Sikkim"},
    "Gurugram":          {"lat": 28.4595, "lon": 77.0266, "state": "Haryana"},
    "Guwahati":          {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Hyderabad":         {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Imphal":            {"lat": 24.8170, "lon": 93.9368, "state": "Manipur"},
    "Itanagar":          {"lat": 27.0844, "lon": 93.6053, "state": "Arunachal Pradesh"},
    "Jaipur":            {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "Kohima":            {"lat": 25.6751, "lon": 94.1086, "state": "Nagaland"},
    "Kolkata":           {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Lucknow":           {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "Mumbai":            {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Panaji":            {"lat": 15.4909, "lon": 73.8278, "state": "Goa"},
    "Patna":             {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Raipur":            {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh"},
    "Ranchi":            {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand"},
    "Shillong":          {"lat": 25.5788, "lon": 91.8933, "state": "Meghalaya"},
    "Shimla":            {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "Thiruvananthapuram":{"lat": 8.5241,  "lon": 76.9366, "state": "Kerala"},
    "Visakhapatnam":     {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
}
STATES = sorted({v["state"] for v in CITIES.values()})

# =============================================================================
# CPCB CATEGORY HELPER
# =============================================================================

def cpcb_category(aqi: float) -> Dict:
    if aqi <= 50:    return {"cat": "Good",        "emoji": "🟢", "color": "#00e400"}
    if aqi <= 100:   return {"cat": "Satisfactory", "emoji": "🟡", "color": "#90ee90"}
    if aqi <= 200:   return {"cat": "Moderate",     "emoji": "🟠", "color": "#ff7e00"}
    if aqi <= 300:   return {"cat": "Poor",         "emoji": "🔴", "color": "#ff0000"}
    if aqi <= 400:   return {"cat": "Very Poor",    "emoji": "🟣", "color": "#8f3f97"}
    return             {"cat": "Severe",        "emoji": "🟤", "color": "#7e0023"}

# =============================================================================
# HTTP SESSION
# =============================================================================

_session = requests.Session()
_retry = requests.adapters.Retry(total=2, backoff_factor=0.5,
                                 status_forcelist=[429, 500, 502, 503, 504])
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20,
                                         max_retries=_retry)
_session.mount("https://", _adapter)
_HEADERS = {"User-Agent": "AQI-CPCB-App/6.0"}

# =============================================================================
# DATA FETCH  (past_days=2 gives 48 h of history for accurate 24 h averages)
# =============================================================================

def _fetch(lat: float, lon: float, forecast_days: int = 2):
    base = {"latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
            "past_days": 2, "forecast_days": forecast_days}
    w = {**base, "hourly": ["relative_humidity_2m", "dew_point_2m",
                             "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m",
                             "pressure_msl", "surface_pressure", "cloud_cover", "is_day"]}
    aq = {**base, "hourly": ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                              "sulphur_dioxide", "ozone", "ammonia",
                              "dust", "aerosol_optical_depth"]}
    try:
        weather = _session.get("https://api.open-meteo.com/v1/forecast",
                               params=w, headers=_HEADERS,
                               timeout=EXTERNAL_API_TIMEOUT).json()
        air_q   = _session.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                               params=aq, headers=_HEADERS,
                               timeout=EXTERNAL_API_TIMEOUT).json()
        return weather, air_q
    except Exception:
        return None, None


def _sv(d: dict, key: str, idx: int, default: float) -> float:
    """Safe value getter from Open-Meteo hourly dict."""
    vals = d.get(key, [])
    if idx < len(vals) and vals[idx] is not None:
        return float(vals[idx])
    return default


def _build_df(weather: dict, air_q: dict, city: str) -> Optional[pd.DataFrame]:
    """
    Build a full time-series DataFrame (history + forecast) for one city,
    including the 7 CPCB rolling-average features that mirror training exactly.
    """
    city_info = CITIES[city]
    wh = weather.get("hourly", {})
    ah = air_q.get("hourly", {})
    times = wh.get("time", [])
    if not times:
        return None

    rows = []
    for i, ts in enumerate(times):
        dt = pd.to_datetime(ts)
        rows.append({
            "datetime":           dt,
            "pm2_5":              _sv(ah, "pm2_5",              i, 50.0),
            "pm10":               _sv(ah, "pm10",               i, 80.0),
            "carbon_monoxide":    _sv(ah, "carbon_monoxide",    i, 500.0),
            "nitrogen_dioxide":   _sv(ah, "nitrogen_dioxide",   i, 30.0),
            "sulphur_dioxide":    _sv(ah, "sulphur_dioxide",    i, 10.0),
            "ozone":              _sv(ah, "ozone",              i, 50.0),
            "ammonia":            _sv(ah, "ammonia",            i, 10.0),
            "dust":               _sv(ah, "dust",               i, 10.0),
            "aerosol_optical_depth": _sv(ah, "aerosol_optical_depth", i, 0.3),
            "relative_humidity_2m":  _sv(wh, "relative_humidity_2m",  i, 60.0),
            "dew_point_2m":       _sv(wh, "dew_point_2m",       i, 15.0),
            "wind_speed_10m":     _sv(wh, "wind_speed_10m",     i, 15.0),
            "wind_gusts_10m":     _sv(wh, "wind_gusts_10m",     i, 20.0),
            "wind_direction_10m": _sv(wh, "wind_direction_10m", i, 180.0),
            "pressure_msl":       _sv(wh, "pressure_msl",       i, 1013.0),
            "surface_pressure":   _sv(wh, "surface_pressure",   i, 1013.0),
            "cloud_cover":        _sv(wh, "cloud_cover",        i, 30.0),
            "is_day":             _sv(wh, "is_day",             i, 1.0),
            "latitude":  city_info["lat"],
            "longitude": city_info["lon"],
        })

    df = pd.DataFrame(rows)

    # CPCB rolling averages — current-hour-inclusive, matching training logic
    df["pm2_5_avg24"]            = df["pm2_5"].rolling(24, min_periods=1).mean()
    df["pm10_avg24"]             = df["pm10"].rolling(24, min_periods=1).mean()
    df["nitrogen_dioxide_avg24"] = df["nitrogen_dioxide"].rolling(24, min_periods=1).mean()
    df["sulphur_dioxide_avg24"]  = df["sulphur_dioxide"].rolling(24, min_periods=1).mean()
    df["ammonia_avg24"]          = df["ammonia"].rolling(24, min_periods=1).mean()
    df["carbon_monoxide_avg8"]   = df["carbon_monoxide"].rolling(8, min_periods=1).mean()
    df["ozone_avg8"]             = df["ozone"].rolling(8, min_periods=1).mean()

    # Datetime features matching training
    df["year"]         = df["datetime"].dt.year
    df["month"]        = df["datetime"].dt.month
    df["day"]          = df["datetime"].dt.day
    df["hour"]         = df["datetime"].dt.hour
    df["day_of_week"]  = df["datetime"].dt.dayofweek
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["quarter"]      = df["datetime"].dt.quarter

    # Categorical encodings from training encoders.json
    df["city_encoded"]  = CITY_ENC.get(city, 0)
    df["state_encoded"] = STATE_ENC.get(city_info["state"], 0)

    return df


def _predict(df: pd.DataFrame) -> np.ndarray:
    X = df[REQUIRED_FEATURES].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    dmat = xgb.DMatrix(X, feature_names=REQUIRED_FEATURES)
    return np.clip(model.predict(dmat), 0.0, 500.0)

# =============================================================================
# FASTAPI APP
# =============================================================================

class HourlyForecast(BaseModel):
    datetime: str; hour: int; aqi: float; category: str; emoji: str; color: str
    pm2_5: float; pm10: float; ozone: float; nitrogen_dioxide: float
    sulphur_dioxide: float; carbon_monoxide: float
    relative_humidity_2m: float; wind_speed_10m: float

class DailySummary(BaseModel):
    date: str; avg_aqi: float; max_aqi: float; min_aqi: float
    category: str; emoji: str; color: str

class PredictionResponse(BaseModel):
    success: bool; city: str; state: str; coordinates: Dict
    forecast_days: int; generated_at: str
    hourly: List[HourlyForecast]; daily: List[DailySummary]; summary: Dict

class CitySummary(BaseModel):
    city: str; state: str; current_aqi: float; category: str; emoji: str

app = FastAPI(title="AQI Prediction API (CPCB)", version=API_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {"status": "ok", "api": "AQI Prediction API",
            "version": API_VERSION, "standard": "Indian CPCB (0-500)"}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    if not REQUIRED_FEATURES:
        raise HTTPException(503, "features.txt not loaded")
    return {"status": "healthy", "features": len(REQUIRED_FEATURES),
            "cities": len(CITIES)}


@app.get("/cities")
def get_cities():
    return {"total": len(CITIES), "states": STATES,
            "cities": [{"name": c, "state": d["state"],
                        "lat": d["lat"], "lon": d["lon"]}
                       for c, d in sorted(CITIES.items())]}


@app.get("/predict/{city}", response_model=PredictionResponse)
def predict_city(city: str, days: int = 2):
    if city not in CITIES:
        raise HTTPException(400, f"Unknown city. Available: {sorted(CITIES)}")
    if model is None or not REQUIRED_FEATURES:
        raise HTTPException(503, "Model or feature list not loaded")

    days = max(1, min(5, days))
    info = CITIES[city]

    weather, air_q = _fetch(info["lat"], info["lon"], forecast_days=days)
    if not weather or not air_q:
        raise HTTPException(503, "Open-Meteo API unavailable")

    df = _build_df(weather, air_q, city)
    if df is None or df.empty:
        raise HTTPException(500, "Data processing failed")

    # Keep only rows from current hour onward for the forecast display
    now_floor = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=5, minutes=30)
    now_floor = now_floor.floor("h")
    fdf = df[df["datetime"] >= now_floor].copy()
    if fdf.empty:
        fdf = df.tail(days * 24).copy()

    preds = _predict(fdf)
    fdf["aqi"] = preds

    hourly: List[Dict] = []
    for _, row in fdf.iterrows():
        cat = cpcb_category(float(row["aqi"]))
        hourly.append({
            "datetime": row["datetime"].isoformat(), "hour": int(row["hour"]),
            "aqi": round(float(row["aqi"]), 1),
            "category": cat["cat"], "emoji": cat["emoji"], "color": cat["color"],
            "pm2_5": round(float(row["pm2_5"]), 1),
            "pm10": round(float(row["pm10"]), 1),
            "ozone": round(float(row["ozone"]), 1),
            "nitrogen_dioxide": round(float(row["nitrogen_dioxide"]), 1),
            "sulphur_dioxide": round(float(row["sulphur_dioxide"]), 1),
            "carbon_monoxide": round(float(row["carbon_monoxide"]), 1),
            "relative_humidity_2m": round(float(row["relative_humidity_2m"]), 1),
            "wind_speed_10m": round(float(row["wind_speed_10m"]), 1),
        })

    fdf["date"] = fdf["datetime"].dt.date
    daily: List[Dict] = []
    for date, grp in fdf.groupby("date"):
        dc = cpcb_category(float(grp["aqi"].mean()))
        daily.append({
            "date": str(date),
            "avg_aqi": round(float(grp["aqi"].mean()), 1),
            "max_aqi": round(float(grp["aqi"].max()), 1),
            "min_aqi": round(float(grp["aqi"].min()), 1),
            "category": dc["cat"], "emoji": dc["emoji"], "color": dc["color"],
        })

    avg = float(np.mean(preds))
    oc = cpcb_category(avg)
    del df, fdf; gc.collect()

    return {
        "success": True, "city": city, "state": info["state"],
        "coordinates": {"lat": info["lat"], "lon": info["lon"]},
        "forecast_days": days, "generated_at": datetime.now().isoformat(),
        "hourly": hourly, "daily": daily,
        "summary": {
            "avg_aqi": round(avg, 1),
            "max_aqi": round(max(h["aqi"] for h in hourly), 1),
            "min_aqi": round(min(h["aqi"] for h in hourly), 1),
            "category": oc["cat"], "emoji": oc["emoji"], "color": oc["color"],
            "total_hours": len(hourly),
        },
    }


def _single_city(city_name: str) -> Optional[Dict]:
    try:
        info = CITIES[city_name]
        weather, air_q = _fetch(info["lat"], info["lon"], forecast_days=1)
        if not weather or not air_q:
            return None
        df = _build_df(weather, air_q, city_name)
        if df is None or df.empty:
            return None
        now_floor = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=5, minutes=30)
        now_floor = now_floor.floor("h")
        row_df = df[df["datetime"] >= now_floor]
        if row_df.empty:
            row_df = df.tail(1)
        pred = float(_predict(row_df)[0])
        cat = cpcb_category(pred)
        return {"city": city_name, "state": info["state"],
                "current_aqi": round(pred, 1),
                "category": cat["cat"], "emoji": cat["emoji"]}
    except Exception:
        return None


@app.get("/predict/all/cities", response_model=List[CitySummary])
def predict_all_cities():
    if model is None or not REQUIRED_FEATURES:
        raise HTTPException(503, "Model or feature list not loaded")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for data in ex.map(_single_city, CITIES):
            if data:
                results.append(data)
    gc.collect()
    return sorted(results, key=lambda x: x["current_aqi"], reverse=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
