# app/predictor.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path("/app/models/optimized/xgboost_improved_lzma.pkl")
FEATURES_PATH = Path("/app/models/optimized/feature_names.txt")

# Load model & features at startup (fast inference)
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    FEATURE_COLUMNS = [line.strip() for line in f]

def predict_aqi(input_data: dict) -> dict:
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Reorder and fill missing
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    
    # Predict
    pred = float(model.predict(df)[0])
    
    category = "Good" if pred <= 50 else \
               "Moderate" if pred <= 100 else \
               "Unhealthy for Sensitive Groups" if pred <= 150 else \
               "Unhealthy" if pred <= 200 else \
               "Very Unhealthy" if pred <= 300 else "Hazardous"
    
    emoji = "Green" if pred <= 50 else "Yellow" if pred <= 100 else "Orange" if pred <= 150 else "Red" if pred <= 200 else "Purple" if pred <= 300 else "Maroon"
    
    return {
        "predicted_aqi": round(pred, 1),
        "category": category,
        "emoji": emoji,
        "model_size_mb": round(MODEL_PATH.stat().st_size / (1024*1024), 2)
    }