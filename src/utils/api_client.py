"""
Open-Meteo API client for weather and air quality data
Handles missing fields gracefully
"""

import requests
from typing import Optional, Dict
from datetime import datetime, timedelta

from .logger import get_logger


logger = get_logger(__name__)


class OpenMeteoClient:
    """
    Client for Open-Meteo API
    Free weather and air quality forecasts
    """
    
    BASE_URL = "https://api.open-meteo.com/v1"
    
    def __init__(self):
        """Initialize API client"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AQI-Prediction-MLOps/1.0'
        })
    
    def fetch_weather_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 2
    ) -> Optional[Dict]:
        """
        Fetch weather forecast from Open-Meteo
        
        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days (1-3)
        
        Returns:
            Dictionary with weather data or None
        """
        try:
            # Prepare parameters (request only available fields)
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': [
                    'relative_humidity_2m',
                    'dew_point_2m',
                    'precipitation',
                    'pressure_msl',
                    'cloud_cover',
                    'wind_speed_10m'  # Use wind_speed instead of wind_gusts
                ],
                'forecast_days': min(forecast_days, 3),
                'timezone': 'Asia/Kolkata'
            }
            
            url = f"{self.BASE_URL}/forecast"
            
            logger.info(f"Fetching {forecast_days}-day weather forecast")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Add wind_gusts_10m as wind_speed_10m (fallback)
            if 'hourly' in data and 'wind_speed_10m' in data['hourly']:
                data['hourly']['wind_gusts_10m'] = data['hourly']['wind_speed_10m']
            
            logger.info(f"✓ Weather data fetched successfully")
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            return None
    
    def fetch_air_quality_forecast(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 2
    ) -> Optional[Dict]:
        """
        Fetch air quality forecast from Open-Meteo
        
        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days (1-3)
        
        Returns:
            Dictionary with AQ data or None
        """
        try:
            # Prepare parameters
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': [
                    'pm2_5',
                    'pm10',
                    'carbon_monoxide',
                    'nitrogen_dioxide',
                    'sulphur_dioxide',
                    'ozone',
                    'dust',
                    'aerosol_optical_depth'
                ],
                'forecast_days': min(forecast_days, 3),
                'timezone': 'Asia/Kolkata'
            }
            
            url = f"{self.BASE_URL}/air-quality"
            
            logger.info(f"Fetching {forecast_days}-day AQ forecast")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            logger.info(f"✓ AQ data fetched successfully")
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch AQ data: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error fetching AQ: {e}")
            return None
    
    def fetch_current_weather(
        self,
        lat: float,
        lon: float
    ) -> Optional[Dict]:
        """
        Fetch current weather conditions
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary with current weather or None
        """
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': [
                    'temperature_2m',
                    'relative_humidity_2m',
                    'precipitation',
                    'pressure_msl',
                    'cloud_cover',
                    'wind_speed_10m'
                ],
                'timezone': 'Asia/Kolkata'
            }
            
            url = f"{self.BASE_URL}/forecast"
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to fetch current weather: {e}")
            return None