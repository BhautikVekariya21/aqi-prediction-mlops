"""
Live AQI prediction
Combines model inference with API data fetching
Matches notebook prediction logic exactly
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional

from .model_loader import ModelLoader
from ..utils.logger import get_logger
from ..utils.api_client import OpenMeteoClient
from ..utils.metrics import AQIMetrics


logger = get_logger(__name__)


class LivePredictor:
    """
    Live AQI prediction using trained model
    Matches notebook prediction logic exactly
    """
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize live predictor
        
        Args:
            model_loader: ModelLoader instance
        """
        self.model_loader = model_loader
        self.api_client = OpenMeteoClient()
        
        # City and state encodings (from training)
        self.city_encoding = self._get_city_encoding()
        self.state_encoding = self._get_state_encoding()
    
    def _get_city_encoding(self) -> Dict[str, int]:
        """Get city encoding (alphabetical order, matching training)"""
        cities = [
            'Agartala', 'Ahmedabad', 'Aizawl', 'Bengaluru', 'Bhopal',
            'Bhubaneswar', 'Chandigarh', 'Chennai', 'Dehradun', 'Delhi',
            'Gangtok', 'Gurugram', 'Guwahati', 'Hyderabad', 'Imphal',
            'Itanagar', 'Jaipur', 'Kohima', 'Kolkata', 'Lucknow',
            'Mumbai', 'Panaji', 'Patna', 'Raipur', 'Ranchi',
            'Shillong', 'Shimla', 'Thiruvananthapuram', 'Visakhapatnam'
        ]
        return {city: idx for idx, city in enumerate(sorted(cities))}
    
    def _get_state_encoding(self) -> Dict[str, int]:
        """Get state encoding (alphabetical order, matching training)"""
        states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar',
            'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana',
            'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
            'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya',
            'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan',
            'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
            'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
        ]
        return {state: idx for idx, state in enumerate(sorted(states))}
    
    def prepare_manual_features(
        self,
        city_name: str,
        city_info: Dict,
        **kwargs
    ) -> Dict[str, float]:
        """
        Prepare features from manual input (from notebook)
        Matches notebook default value logic exactly
        """
        # Current datetime info
        now = datetime.now()
        month = kwargs.get('month') or now.month
        is_weekend = kwargs.get('is_weekend')
        if is_weekend is None:
            is_weekend = 1 if now.weekday() >= 5 else 0
        
        # Build feature dict (from notebook order)
        features = {
            'o3_ugm3': kwargs.get('o3') or 50,
            'pressure_msl_hpa': kwargs.get('pressure') or 1013,
            'heavy_rain': kwargs.get('heavy_rain') or 0,
            'co_ugm3': kwargs.get('co') or 500,
            'latitude': city_info['lat'],
            'humidity_percent': kwargs.get('humidity') or 60,
            'city_encoded': self.city_encoding.get(city_name, 0),
            'so2_ugm3': kwargs.get('so2') or 10,
            'precipitation_mm': kwargs.get('precipitation') or 0,
            'dust_ugm3': kwargs.get('dust') or 10,
            'pm2_5_ugm3': kwargs.get('pm2_5') or 50,
            'wind_gusts_kmh': kwargs.get('wind_gusts') or 20,
            'aod': kwargs.get('aod') or 0.3,
            'state_encoded': self.state_encoding.get(city_info['state'], 0),
            'pm10_ugm3': kwargs.get('pm10') or 80,
            'longitude': city_info['lon'],
            'is_raining': kwargs.get('is_raining') or 0,
            'cloud_cover_percent': kwargs.get('cloud_cover') or 30,
            'is_weekend': is_weekend,
            'dew_point_c': kwargs.get('dew_point') or 15,
            'no2_ugm3': kwargs.get('no2') or 30,
            'month': month
        }
        
        return features
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, str, str]:
        """
        Make AQI prediction
        
        Args:
            features: Feature dictionary
        
        Returns:
            Tuple of (predicted_aqi, category, emoji)
        """
        # Predict
        predicted_aqi = self.model_loader.predict(features)
        
        # Get category and emoji
        category, emoji = AQIMetrics.get_aqi_category(predicted_aqi)
        
        return predicted_aqi, category, emoji
    
    def forecast_from_api(
        self,
        city_name: str,
        city_info: Dict,
        forecast_days: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        Get AQI forecast using Open-Meteo API (from notebook)
        
        Args:
            city_name: City name
            city_info: Dict with lat, lon, state
            forecast_days: Number of forecast days
        
        Returns:
            DataFrame with hourly forecasts or None
        """
        lat = city_info['lat']
        lon = city_info['lon']
        
        logger.info(f"Fetching {forecast_days}-day forecast for {city_name}")
        
        # Fetch weather forecast
        weather_data = self.api_client.fetch_weather_forecast(lat, lon, forecast_days)
        
        if weather_data is None:
            logger.error("Failed to fetch weather data")
            return None
        
        # Fetch AQ forecast
        aq_data = self.api_client.fetch_air_quality_forecast(lat, lon, forecast_days)
        
        if aq_data is None:
            logger.error("Failed to fetch AQ data")
            return None
        
        # Prepare features (from notebook)
        weather_hourly = weather_data.get('hourly', {})
        aq_hourly = aq_data.get('hourly', {})
        
        n_hours = len(weather_hourly.get('time', []))
        
        if n_hours == 0:
            logger.error("No hourly data available")
            return None
        
        records = []
        
        for i in range(n_hours):
            try:
                dt = pd.to_datetime(weather_hourly['time'][i])
                
                def get_val(data, key, idx, default):
                    """Safely get value from data dict"""
                    try:
                        if key not in data:
                            return default
                        values = data[key]
                        if not values or idx >= len(values):
                            return default
                        val = values[idx]
                        return val if val is not None else default
                    except (IndexError, KeyError, TypeError):
                        return default
                
                # Weather features (with safe defaults)
                humidity = get_val(weather_hourly, 'relative_humidity_2m', i, 60)
                dew_point = get_val(weather_hourly, 'dew_point_2m', i, 15)
                wind_gusts = get_val(weather_hourly, 'wind_gusts_10m', i, 20)
                precipitation = get_val(weather_hourly, 'precipitation', i, 0)
                pressure = get_val(weather_hourly, 'pressure_msl', i, 1013)
                cloud_cover = get_val(weather_hourly, 'cloud_cover', i, 30)
                
                # AQ features (with safe defaults)
                pm25 = get_val(aq_hourly, 'pm2_5', i, 50)
                pm10 = get_val(aq_hourly, 'pm10', i, 80)
                co = get_val(aq_hourly, 'carbon_monoxide', i, 500)
                no2 = get_val(aq_hourly, 'nitrogen_dioxide', i, 30)
                so2 = get_val(aq_hourly, 'sulphur_dioxide', i, 10)
                o3 = get_val(aq_hourly, 'ozone', i, 50)
                dust = get_val(aq_hourly, 'dust', i, 10)
                aod = get_val(aq_hourly, 'aerosol_optical_depth', i, 0.3)
                
                # Derived features
                is_raining = 1 if precipitation > 0 else 0
                heavy_rain = 1 if precipitation > 7.5 else 0
                is_weekend = 1 if dt.dayofweek >= 5 else 0
                
                # Create feature dict
                features = {
                    'o3_ugm3': o3,
                    'pressure_msl_hpa': pressure,
                    'heavy_rain': heavy_rain,
                    'co_ugm3': co,
                    'latitude': city_info['lat'],
                    'humidity_percent': humidity,
                    'city_encoded': self.city_encoding.get(city_name, 0),
                    'so2_ugm3': so2,
                    'precipitation_mm': precipitation,
                    'dust_ugm3': dust,
                    'pm2_5_ugm3': pm25,
                    'wind_gusts_kmh': wind_gusts,
                    'aod': aod,
                    'state_encoded': self.state_encoding.get(city_info['state'], 0),
                    'pm10_ugm3': pm10,
                    'longitude': city_info['lon'],
                    'is_raining': is_raining,
                    'cloud_cover_percent': cloud_cover,
                    'is_weekend': is_weekend,
                    'dew_point_c': dew_point,
                    'no2_ugm3': no2,
                    'month': dt.month
                }
                
                # Predict
                predicted_aqi, aqi_category, aqi_emoji = self.predict(features)
                
                # Add to records
                record = {
                    'datetime': dt,
                    'hour': dt.hour,
                    'predicted_aqi': predicted_aqi,
                    'aqi_category': aqi_category,
                    'aqi_emoji': aqi_emoji
                }
                
                records.append(record)
            
            except Exception as e:
                logger.warning(f"Skipping hour {i} due to error: {e}")
                continue
        
        if not records:
            logger.error("No valid forecast records generated")
            return None
        
        df = pd.DataFrame(records)
        
        logger.info(f"Generated {len(df)} hourly forecasts")
        
        return df