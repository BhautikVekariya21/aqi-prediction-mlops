"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool
    timestamp: str


class PredictionRequest(BaseModel):
    """Manual AQI prediction request"""
    
    city: str = Field(..., description="City name (must be one of 29 Indian cities)")
    
    # Pollutants
    pm2_5: Optional[float] = Field(None, ge=0, le=2000, description="PM2.5 concentration (µg/m³)")
    pm10: Optional[float] = Field(None, ge=0, le=4000, description="PM10 concentration (µg/m³)")
    o3: Optional[float] = Field(None, ge=0, le=600, description="Ozone concentration (µg/m³)")
    no2: Optional[float] = Field(None, ge=0, le=1000, description="NO2 concentration (µg/m³)")
    so2: Optional[float] = Field(None, ge=0, le=1000, description="SO2 concentration (µg/m³)")
    co: Optional[float] = Field(None, ge=0, le=15000, description="CO concentration (µg/m³)")
    dust: Optional[float] = Field(None, ge=0, le=5000, description="Dust concentration (µg/m³)")
    aod: Optional[float] = Field(None, ge=0, le=10, description="Aerosol Optical Depth")
    
    # Weather
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Relative humidity (%)")
    dew_point: Optional[float] = Field(None, ge=-40, le=50, description="Dew point (°C)")
    pressure: Optional[float] = Field(None, ge=900, le=1100, description="Pressure (hPa)")
    cloud_cover: Optional[float] = Field(None, ge=0, le=100, description="Cloud cover (%)")
    wind_gusts: Optional[float] = Field(None, ge=0, le=200, description="Wind gusts (km/h)")
    precipitation: Optional[float] = Field(0, ge=0, le=500, description="Precipitation (mm)")
    
    # Derived
    is_raining: Optional[int] = Field(0, ge=0, le=1, description="Is raining (0/1)")
    heavy_rain: Optional[int] = Field(0, ge=0, le=1, description="Heavy rain (0/1)")
    is_weekend: Optional[int] = Field(None, ge=0, le=1, description="Is weekend (0/1)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    
    @validator('city')
    def validate_city(cls, v):
        valid_cities = [
            'Visakhapatnam', 'Itanagar', 'Guwahati', 'Patna', 'Raipur',
            'Panaji', 'Ahmedabad', 'Gurugram', 'Shimla', 'Ranchi',
            'Bengaluru', 'Thiruvananthapuram', 'Bhopal', 'Mumbai', 'Imphal',
            'Shillong', 'Aizawl', 'Kohima', 'Bhubaneswar', 'Chandigarh',
            'Jaipur', 'Gangtok', 'Chennai', 'Hyderabad', 'Agartala',
            'Lucknow', 'Dehradun', 'Kolkata', 'Delhi'
        ]
        if v not in valid_cities:
            raise ValueError(f'City must be one of: {", ".join(valid_cities)}')
        return v


class PredictionResponse(BaseModel):
    """AQI prediction response"""
    
    city: str
    state: str
    predicted_aqi: float
    aqi_category: str
    aqi_emoji: str
    confidence: str
    timestamp: str
    
    input_features: Dict[str, float]
    model_version: str


class ForecastRequest(BaseModel):
    """Forecast request using Open-Meteo API"""
    
    city: str = Field(..., description="City name")
    forecast_days: int = Field(2, ge=1, le=3, description="Number of forecast days (1-3)")
    
    @validator('city')
    def validate_city(cls, v):
        valid_cities = [
            'Visakhapatnam', 'Itanagar', 'Guwahati', 'Patna', 'Raipur',
            'Panaji', 'Ahmedabad', 'Gurugram', 'Shimla', 'Ranchi',
            'Bengaluru', 'Thiruvananthapuram', 'Bhopal', 'Mumbai', 'Imphal',
            'Shillong', 'Aizawl', 'Kohima', 'Bhubaneswar', 'Chandigarh',
            'Jaipur', 'Gangtok', 'Chennai', 'Hyderabad', 'Agartala',
            'Lucknow', 'Dehradun', 'Kolkata', 'Delhi'
        ]
        if v not in valid_cities:
            raise ValueError(f'City must be one of: {", ".join(valid_cities)}')
        return v


class HourlyForecast(BaseModel):
    """Hourly AQI forecast"""
    
    datetime: str
    hour: int
    predicted_aqi: float
    aqi_category: str
    aqi_emoji: str


class ForecastResponse(BaseModel):
    """Forecast response"""
    
    city: str
    state: str
    forecast_days: int
    
    summary: Dict[str, float]
    hourly_forecasts: List[HourlyForecast]
    
    timestamp: str
    model_version: str


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str
    detail: str
    timestamp: str