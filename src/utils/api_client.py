"""
Open-Meteo API client (matching notebook API logic exactly)
"""

import requests
import time
import random
from typing import Dict, Optional, List
from datetime import datetime
from .logger import get_logger


logger = get_logger(__name__)


class OpenMeteoClient:
    """
    Client for Open-Meteo Weather and Air Quality APIs
    Matches notebook API calls exactly
    """
    
    # API URLs (from notebook)
    WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Parameters (from notebook)
    WEATHER_PARAMS = [
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "precipitation",
        "rain",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "is_day",
        "sunshine_duration",
    ]
    
    AQ_PARAMS = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust",
        "ammonia",
        "aerosol_optical_depth",
        "us_aqi",
    ]
    
    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 5,
        retry_delay: int = 5,
        rate_limit_min: float = 1.0,
        rate_limit_max: float = 2.0
    ):
        """
        Initialize API client
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            rate_limit_min: Minimum delay between requests (seconds)
            rate_limit_max: Maximum delay between requests (seconds)
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_min = rate_limit_min
        self.rate_limit_max = rate_limit_max
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
    
    def _safe_request(self, url: str, params: Dict) -> Optional[Dict]:
        """
        Make API request with retry logic (from notebook)
        
        Args:
            url: API endpoint URL
            params: Request parameters
        
        Returns:
            JSON response or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Rate limiting (from notebook)
                time.sleep(random.uniform(self.rate_limit_min, self.rate_limit_max))
                
                response = requests.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Waiting {wait}s...")
                    time.sleep(wait)
                
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}. Retrying...")
                    time.sleep(30)
                
                else:
                    logger.warning(f"HTTP {response.status_code}")
                    time.sleep(10)
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                time.sleep(20)
            
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(10)
        
        return None
    
    def fetch_historical_weather(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """
        Fetch historical weather data (archive API)
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Weather data or None
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.WEATHER_PARAMS),
            "timezone": "Asia/Kolkata"
        }
        
        logger.info(f"Fetching historical weather: {start_date} to {end_date}")
        return self._safe_request(self.WEATHER_ARCHIVE_URL, params)
    
    def fetch_weather_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 2
    ) -> Optional[Dict]:
        """
        Fetch weather forecast
        
        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days
        
        Returns:
            Weather forecast or None
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.WEATHER_PARAMS),
            "timezone": "Asia/Kolkata",
            "forecast_days": forecast_days
        }
        
        logger.info(f"Fetching {forecast_days}-day weather forecast")
        return self._safe_request(self.WEATHER_FORECAST_URL, params)
    
    def fetch_air_quality(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """
        Fetch air quality data
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Air quality data or None
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(self.AQ_PARAMS),
            "timezone": "Asia/Kolkata"
        }
        
        logger.info(f"Fetching air quality: {start_date} to {end_date}")
        return self._safe_request(self.AQ_URL, params)
    
    def fetch_air_quality_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 2
    ) -> Optional[Dict]:
        """
        Fetch air quality forecast
        
        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days
        
        Returns:
            Air quality forecast or None
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(self.AQ_PARAMS),
            "timezone": "Asia/Kolkata",
            "forecast_days": forecast_days
        }
        
        logger.info(f"Fetching {forecast_days}-day AQ forecast")
        return self._safe_request(self.AQ_URL, params)