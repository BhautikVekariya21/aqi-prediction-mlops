"""
FastAPI application for AQI prediction
Main entry point for the REST API
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.models import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ForecastRequest,
    ForecastResponse,
    ErrorResponse
)
from src.api.dependencies import get_model_loader, get_city_info
from src.inference.live_predictor import LivePredictor
from src.utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="AQI Prediction API",
    description="Production-grade Air Quality Index prediction for Indian cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AQI Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    Used by Railway and Docker health checks
    """
    try:
        model_loader = get_model_loader()
        model_loaded = model_loader.model is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            version="1.0.0",
            model_loaded=model_loaded,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_loaded=False,
            timestamp=datetime.now().isoformat()
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_manual(request: PredictionRequest):
    """
    Manual AQI prediction
    
    Provide pollutant and weather data to get AQI prediction.
    All parameters are optional - defaults will be used for missing values.
    """
    try:
        # Get model loader
        model_loader = get_model_loader()
        
        # Get city info
        city_info = get_city_info(request.city)
        
        # Create live predictor
        predictor = LivePredictor(model_loader)
        
        # Prepare input features (from notebook logic)
        features = predictor.prepare_manual_features(
            city_name=request.city,
            city_info=city_info,
            pm2_5=request.pm2_5,
            pm10=request.pm10,
            o3=request.o3,
            no2=request.no2,
            so2=request.so2,
            co=request.co,
            dust=request.dust,
            aod=request.aod,
            humidity=request.humidity,
            dew_point=request.dew_point,
            pressure=request.pressure,
            cloud_cover=request.cloud_cover,
            wind_gusts=request.wind_gusts,
            precipitation=request.precipitation,
            is_raining=request.is_raining,
            heavy_rain=request.heavy_rain,
            is_weekend=request.is_weekend,
            month=request.month
        )
        
        # Predict
        predicted_aqi, aqi_category, aqi_emoji = predictor.predict(features)
        
        # Determine confidence (based on input completeness)
        provided_params = sum([
            request.pm2_5 is not None,
            request.pm10 is not None,
            request.o3 is not None,
            request.no2 is not None,
            request.humidity is not None
        ])
        
        if provided_params >= 4:
            confidence = "high"
        elif provided_params >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            city=request.city,
            state=city_info['state'],
            predicted_aqi=predicted_aqi,
            aqi_category=aqi_category,
            aqi_emoji=aqi_emoji,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            input_features=features,
            model_version="1.0.0"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def forecast_api(request: ForecastRequest):
    """
    AQI forecast using Open-Meteo API
    
    Fetches weather and AQ forecast data from Open-Meteo and predicts AQI.
    """
    try:
        # Get model loader
        model_loader = get_model_loader()
        
        # Get city info
        city_info = get_city_info(request.city)
        
        # Create live predictor
        predictor = LivePredictor(model_loader)
        
        # Get forecast (from notebook logic)
        forecast_df = predictor.forecast_from_api(
            city_name=request.city,
            city_info=city_info,
            forecast_days=request.forecast_days
        )
        
        if forecast_df is None or len(forecast_df) == 0:
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch forecast data from Open-Meteo API"
            )
        
        # Create hourly forecasts
        hourly_forecasts = []
        for _, row in forecast_df.iterrows():
            hourly_forecasts.append({
                "datetime": row['datetime'].isoformat(),
                "hour": int(row['hour']),
                "predicted_aqi": float(row['predicted_aqi']),
                "aqi_category": row['aqi_category'],
                "aqi_emoji": row['aqi_emoji']
            })
        
        # Summary statistics
        summary = {
            "avg_aqi": float(forecast_df['predicted_aqi'].mean()),
            "max_aqi": float(forecast_df['predicted_aqi'].max()),
            "min_aqi": float(forecast_df['predicted_aqi'].min()),
            "max_hour": int(forecast_df.loc[forecast_df['predicted_aqi'].idxmax(), 'hour']),
            "min_hour": int(forecast_df.loc[forecast_df['predicted_aqi'].idxmin(), 'hour'])
        }
        
        return ForecastResponse(
            city=request.city,
            state=city_info['state'],
            forecast_days=request.forecast_days,
            summary=summary,
            hourly_forecasts=hourly_forecasts,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    except HTTPException:
        raise
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.get("/cities", tags=["Info"])
async def list_cities():
    """
    List all supported cities
    """
    from src.api.dependencies import get_cities_config
    
    cities = get_cities_config()
    
    return {
        "total_cities": len(cities),
        "cities": [
            {
                "name": name,
                "state": info['state'],
                "coordinates": {
                    "lat": info['lat'],
                    "lon": info['lon']
                }
            }
            for name, info in sorted(cities.items())
        ]
    }


# =============================================================================
# STARTUP & SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting AQI Prediction API...")
    
    try:
        model_loader = get_model_loader()
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"✓ Features: {len(model_loader.feature_names)}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AQI Prediction API...")


# =============================================================================
# MAIN (for local development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )