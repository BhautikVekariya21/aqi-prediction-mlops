"""
Stage 3: Feature Engineering
Create datetime features, derived features, and encodings
Exact logic from notebook: 03_feature_engineering.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import json

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader


logger = get_logger(__name__)


class FeatureEngineering:
    """
    Engineer features from processed data
    Matches notebook feature engineering logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize feature engineering
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get feature engineering parameters
        fe_config = config.get_section("feature_engineering")
        
        self.input_path = Path(fe_config.get("input_path", "data/processed/aqi_india_processed.parquet"))
        self.output_dir = Path(fe_config.get("output_dir", "data/features"))
        self.target = fe_config.get("target", "us_aqi")
        
        self.categorical_columns = fe_config.get("categorical_columns", ["city", "state"])
        self.encoding_method = fe_config.get("encoding_method", "label")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Feature Engineering initialized")
    
    def run(self) -> str:
        """
        Run feature engineering pipeline
        
        Returns:
            Path to output parquet file
        """
        logger.info("="*90)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*90)
        
        # Load processed data
        logger.info(f"\n1. Loading processed data from: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        initial_columns = df.shape[1]
        
        # Step 1: Datetime features (from notebook)
        logger.info(f"\n2. Creating datetime features")
        df = self._create_datetime_features(df)
        
        # Step 2: Derived weather features (from notebook)
        logger.info(f"\n3. Creating derived features")
        df = self._create_derived_features(df)
        
        # Step 3: Categorical encoding (from notebook: label encoding)
        logger.info(f"\n4. Encoding categorical features")
        df = self._encode_categorical(df)
        
        # Step 4: Ensure correct data types
        logger.info(f"\n5. Setting data types")
        df = self._set_data_types(df)
        
        # Save feature-engineered data
        output_file = self.output_dir / "aqi_features.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"\nOK Feature-engineered data saved: {output_file}")
        
        # Generate metrics
        metrics = self._generate_metrics(df, initial_columns)
        
        # Save metrics
        metrics_file = self.output_dir / "feature_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"OK Metrics saved: {metrics_file}")
        
        # Print summary
        self._print_summary(df, initial_columns)
        
        return str(output_file)
    
    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create datetime-based features (from notebook)
        """
        # Convert datetime if needed
        if 'datetime' in df.columns and df['datetime'].dtype != 'datetime64[ns]':
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract datetime components (from notebook)
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_name'] = df['datetime'].dt.day_name()
        df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['datetime'].dt.quarter
        
        # Season (Indian seasons - from notebook)
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Monsoon"
            else:
                return "Post_Monsoon"
        
        df['season'] = df['month'].apply(get_season)
        
        # Time of day (from notebook)
        def get_time_of_day(hour):
            if 5 <= hour < 9:
                return "Early_Morning"
            elif 9 <= hour < 12:
                return "Morning"
            elif 12 <= hour < 15:
                return "Afternoon"
            elif 15 <= hour < 18:
                return "Evening"
            elif 18 <= hour < 21:
                return "Night"
            else:
                return "Late_Night"
        
        df['time_of_day'] = df['hour'].apply(get_time_of_day)
        
        logger.info(f"   OK Created datetime features")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features (from notebook)
        """
        # Humidity category (from notebook)
        if 'humidity_percent' in df.columns:
            def get_humidity_category(h):
                if pd.isna(h):
                    return None
                if h < 30:
                    return "Dry"
                elif h < 60:
                    return "Comfortable"
                elif h < 80:
                    return "Humid"
                else:
                    return "Very_Humid"
            
            df['humidity_category'] = df['humidity_percent'].apply(get_humidity_category)
        
        # Wind category (from notebook)
        if 'wind_gusts_kmh' in df.columns:
            def get_wind_category(w):
                if pd.isna(w):
                    return None
                if w < 5:
                    return "Calm"
                elif w < 15:
                    return "Light"
                elif w < 30:
                    return "Moderate"
                elif w < 50:
                    return "Strong"
                else:
                    return "Very_Strong"
            
            df['wind_category'] = df['wind_gusts_kmh'].apply(get_wind_category)
        
        # Precipitation features (from notebook)
        if 'precipitation_mm' in df.columns:
            df['is_raining'] = (df['precipitation_mm'] > 0).astype(int)
            df['heavy_rain'] = (df['precipitation_mm'] > 7.5).astype(int)
        
        # AQI category (from notebook)
        if 'us_aqi' in df.columns:
            def get_aqi_category(aqi):
                if pd.isna(aqi):
                    return None
                if aqi <= 50:
                    return "Good"
                elif aqi <= 100:
                    return "Moderate"
                elif aqi <= 150:
                    return "Unhealthy for Sensitive Groups"
                elif aqi <= 200:
                    return "Unhealthy"
                elif aqi <= 300:
                    return "Very Unhealthy"
                else:
                    return "Hazardous"
            
            df['aqi_category'] = df['us_aqi'].apply(get_aqi_category)
        
        # PM2.5 category India (from notebook)
        if 'pm2_5_ugm3' in df.columns:
            def get_pm25_category_india(pm25):
                if pd.isna(pm25):
                    return None
                if pm25 <= 30:
                    return "Good"
                elif pm25 <= 60:
                    return "Satisfactory"
                elif pm25 <= 90:
                    return "Moderate"
                elif pm25 <= 120:
                    return "Poor"
                elif pm25 <= 250:
                    return "Very_Poor"
                else:
                    return "Severe"
            
            df['pm25_category_india'] = df['pm2_5_ugm3'].apply(get_pm25_category_india)
        
        # Festival period (Diwali: Oct 15 - Nov 15, from notebook)
        df['festival_period'] = (
            ((df['month'] == 10) & (df['day'] >= 15)) |
            ((df['month'] == 11) & (df['day'] <= 15))
        ).astype(int)
        
        # Crop burning season (Oct-Nov for North India, from notebook)
        north_states = ["Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Bihar"]
        if 'state' in df.columns:
            df['crop_burning_season'] = (
                (df['state'].isin(north_states)) &
                ((df['month'] == 10) | (df['month'] == 11))
            ).astype(int)
        else:
            df['crop_burning_season'] = 0
        
        logger.info(f"   OK Created derived features")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables (from notebook: label encoding)
        """
        # Label encoding (from notebook)
        for col in self.categorical_columns:
            if col in df.columns:
                # Create sorted encoding (alphabetical)
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                logger.info(f"   OK Encoded {col} to {col}_encoded ({df[col].nunique()} categories)")
        
        return df
    
    def _set_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set appropriate data types for efficiency"""
        # Integer columns
        int_cols = [
            'year', 'month', 'day', 'hour', 'day_of_week', 'week_of_year',
            'is_weekend', 'quarter', 'is_raining', 'heavy_rain',
            'festival_period', 'crop_burning_season', 'city_encoded', 'state_encoded'
        ]
        
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype('int32')
        
        # Float columns (use float32 for memory efficiency)
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        logger.info(f"   OK Data types optimized")
        
        return df
    
    def _generate_metrics(self, df: pd.DataFrame, initial_columns: int) -> Dict:
        """Generate feature engineering metrics"""
        metrics = {
            "initial_columns": int(initial_columns),
            "final_columns": int(df.shape[1]),
            "new_features_created": int(df.shape[1] - initial_columns),
            "total_rows": int(df.shape[0]),
            "datetime_features": [
                'year', 'month', 'day', 'hour', 'day_of_week', 'day_name',
                'week_of_year', 'is_weekend', 'quarter', 'season', 'time_of_day'
            ],
            "derived_features": [
                'humidity_category', 'wind_category', 'is_raining', 'heavy_rain',
                'aqi_category', 'pm25_category_india', 'festival_period', 'crop_burning_season'
            ],
            "encoded_features": [f"{col}_encoded" for col in self.categorical_columns if col in df.columns],
        }
        
        return metrics
    
    def _print_summary(self, df: pd.DataFrame, initial_columns: int):
        """Print feature engineering summary"""
        print("\n" + "="*90)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*90)
        print(f"Initial Columns:      {initial_columns}")
        print(f"Final Columns:        {df.shape[1]}")
        print(f"New Features:         {df.shape[1] - initial_columns}")
        print(f"Total Rows:           {df.shape[0]:,}")
        
        print(f"\nFeature Categories:")
        print(f"  Datetime:           11 features")
        print(f"  Derived:            8 features")
        print(f"  Encoded:            {len([col for col in df.columns if col.endswith('_encoded')])} features")
        
        print("="*90)