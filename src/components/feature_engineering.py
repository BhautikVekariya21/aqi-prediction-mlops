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
from ..utils.aqi import cpcb_aqi_array_from_ugm3


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
        self.create_cyclical_features = fe_config.get("create_cyclical_features", False)

        # Temporal features (lag / rolling) -- key drivers of AQI autocorrelation.
        # These are computed per-city in time order, so they are leakage-safe
        # under a temporal train/val/test split.
        self.create_lag_features = fe_config.get("create_lag_features", False)
        self.create_rolling_features = fe_config.get("create_rolling_features", False)
        self.create_interaction_features = fe_config.get("create_interaction_features", False)

        # Exogenous drivers to lag/roll (raw Open-Meteo names, matching schema).
        self.temporal_driver_columns = fe_config.get("temporal_driver_columns", [
            "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
            "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth",
            "wind_speed_10m", "relative_humidity_2m",
        ])
        self.lag_hours = fe_config.get("lag_hours", [1, 3, 24])
        self.rolling_windows = fe_config.get("rolling_windows", [6, 24])
        
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

        # Step 0: Compute CPCB AQI label + 7 CPCB rolling-average features.
        # Must run first so the avg columns exist before any downstream step.
        logger.info(f"\n2. Computing CPCB AQI label and rolling-average features")
        df = self._compute_cpcb_label_and_avg_features(df)

        # Step 1: Datetime features (from notebook)
        logger.info(f"\n3. Creating datetime features")
        df = self._create_datetime_features(df)
        
        # Step 2: Derived weather features (from notebook)
        logger.info(f"\n4. Creating derived features")
        df = self._create_derived_features(df)

        # Step 2b: Temporal features (lag / rolling) -- the accuracy lever.
        logger.info(f"\n3b. Creating temporal (lag/rolling) features")
        df = self._create_temporal_features(df)

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
    
    def _compute_cpcb_label_and_avg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the CPCB AQI label and the 7 CPCB rolling-average features.

        Per-city, sort by datetime, then compute current-hour-inclusive trailing
        means (rolling with min_periods=1 so early rows get partial means):
          - 24 h: pm2_5, pm10, nitrogen_dioxide, sulphur_dioxide, ammonia
          -  8 h: carbon_monoxide (µg/m³, converted to mg/m³ inside
                  cpcb_aqi_array_from_ugm3), ozone

        The 7 avg columns are also kept as model features so the serving code
        can reconstruct the identical vector from a 48 h history window.

        Rows where aqi_cpcb is NaN (< 3 sub-indices valid, or no PM sub-index)
        are dropped — they carry no ground-truth label.
        """
        required = {"datetime", "city", "pm2_5", "pm10", "carbon_monoxide",
                    "nitrogen_dioxide", "sulphur_dioxide", "ozone"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"_compute_cpcb_label_and_avg_features: missing columns {missing}")

        df = df.sort_values(["city", "datetime"]).reset_index(drop=True)

        # --- rolling averages per city ---
        avg_specs = {
            "pm2_5_avg24":           ("pm2_5",            24),
            "pm10_avg24":            ("pm10",             24),
            "nitrogen_dioxide_avg24":("nitrogen_dioxide", 24),
            "sulphur_dioxide_avg24": ("sulphur_dioxide",  24),
            "ammonia_avg24":         ("ammonia",          24),
            "carbon_monoxide_avg8":  ("carbon_monoxide",   8),
            "ozone_avg8":            ("ozone",             8),
        }

        new_avg: Dict[str, pd.Series] = {}
        grp = df.groupby("city", sort=False)
        for feat_name, (src_col, window) in avg_specs.items():
            if src_col in df.columns:
                new_avg[feat_name] = (
                    grp[src_col]
                    .transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
                    .astype("float32")
                )
            else:
                logger.warning(f"   Column '{src_col}' not found — {feat_name} set to NaN")
                new_avg[feat_name] = pd.Series(np.nan, index=df.index, dtype="float32")

        df = pd.concat([df, pd.DataFrame(new_avg, index=df.index)], axis=1)

        # --- CPCB AQI label from the 7 averages (dict API, CO µg/m³ → mg/m³ done inside) ---
        df["aqi_cpcb"] = cpcb_aqi_array_from_ugm3({
            "pm2_5":            df["pm2_5_avg24"].to_numpy(dtype=float),
            "pm10":             df["pm10_avg24"].to_numpy(dtype=float),
            "nitrogen_dioxide": df["nitrogen_dioxide_avg24"].to_numpy(dtype=float),
            "sulphur_dioxide":  df["sulphur_dioxide_avg24"].to_numpy(dtype=float),
            "ammonia":          df["ammonia_avg24"].to_numpy(dtype=float),
            "carbon_monoxide":  df["carbon_monoxide_avg8"].to_numpy(dtype=float),
            "ozone":            df["ozone_avg8"].to_numpy(dtype=float),
        }).astype("float32")

        before = len(df)
        df = df.dropna(subset=["aqi_cpcb"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(f"   Dropped {dropped:,} rows where aqi_cpcb is NaN "
                        f"(< 3 sub-indices or no PM sub-index)")

        logger.info(
            f"   OK aqi_cpcb computed — range [{df['aqi_cpcb'].min():.0f}, "
            f"{df['aqi_cpcb'].max():.0f}], mean {df['aqi_cpcb'].mean():.1f}"
        )
        logger.info(f"   OK 7 CPCB avg features added: {list(new_avg.keys())}")
        return df

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

        # Cyclical encoding for seasonal/time patterns
        if self.create_cyclical_features:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
            df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
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
        # Humidity category — real column name is relative_humidity_2m
        humidity_col = next((c for c in ("relative_humidity_2m", "humidity_percent") if c in df.columns), None)
        if humidity_col:
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
            df['humidity_category'] = df[humidity_col].apply(get_humidity_category)

        # Wind category — real column name is wind_gusts_10m (m/s); convert to km/h for thresholds
        wind_col = next((c for c in ("wind_gusts_10m", "wind_gusts_kmh") if c in df.columns), None)
        if wind_col:
            # wind_gusts_10m is in m/s; multiply by 3.6 to get km/h for the same thresholds
            scale = 3.6 if wind_col == "wind_gusts_10m" else 1.0
            def get_wind_category(w):
                if pd.isna(w):
                    return None
                v = w * scale
                if v < 5:
                    return "Calm"
                elif v < 15:
                    return "Light"
                elif v < 30:
                    return "Moderate"
                elif v < 50:
                    return "Strong"
                else:
                    return "Very_Strong"
            df['wind_category'] = df[wind_col].apply(get_wind_category)

        # Precipitation features — real column name is precipitation (mm)
        precip_col = next((c for c in ("precipitation", "precipitation_mm") if c in df.columns), None)
        if precip_col:
            df['is_raining'] = (df[precip_col] > 0).astype(int)
            df['heavy_rain'] = (df[precip_col] > 7.5).astype(int)

        # CPCB AQI category (replaces old US-EPA aqi_category)
        if 'aqi_cpcb' in df.columns:
            from ..utils.aqi import cpcb_category
            df['aqi_category'] = df['aqi_cpcb'].apply(
                lambda v: cpcb_category(v) if not pd.isna(v) else None
            )

        # PM2.5 India category — real column name is pm2_5
        pm25_col = next((c for c in ("pm2_5", "pm2_5_ugm3") if c in df.columns), None)
        if pm25_col:
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
            df['pm25_category_india'] = df[pm25_col].apply(get_pm25_category_india)

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

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag and rolling-window features for exogenous drivers.

        Computed per city in chronological order so no future information leaks
        into any row (safe under a temporal split). Only exogenous drivers
        (pollutants + weather) are lagged -- never the target -- so the features
        are reconstructable at serving time from the forecast API.

        NaNs introduced at the start of each city's series are filled per city
        (ffill -> bfill -> median) so tree models that don't accept NaN still run.
        """
        if not (self.create_lag_features or self.create_rolling_features):
            logger.info("   Skipped (lag/rolling disabled in params)")
            return df

        if 'datetime' not in df.columns or 'city' not in df.columns:
            logger.warning("   Skipped: 'datetime'/'city' required for temporal features")
            return df

        # Restrict to drivers actually present in the data
        drivers = [c for c in self.temporal_driver_columns if c in df.columns]
        if not drivers:
            logger.warning("   Skipped: none of the configured driver columns are present")
            return df

        # Ensure chronological order within each city
        df = df.sort_values(['city', 'datetime']).reset_index(drop=True)
        grouped = df.groupby('city', sort=False)

        new_cols = {}

        # Lag features
        if self.create_lag_features:
            for col in drivers:
                for lag in self.lag_hours:
                    new_cols[f'{col}_lag_{lag}h'] = grouped[col].shift(lag)
            logger.info(f"   OK Lags {self.lag_hours}h for {len(drivers)} drivers "
                        f"(+{len(drivers) * len(self.lag_hours)} cols)")

        # Rolling-window features (shifted by 1 so the current row is excluded)
        if self.create_rolling_features:
            for col in drivers:
                shifted = grouped[col].shift(1)
                for win in self.rolling_windows:
                    roll = shifted.groupby(df['city'], sort=False).rolling(
                        window=win, min_periods=1
                    )
                    new_cols[f'{col}_rollmean_{win}h'] = roll.mean().reset_index(level=0, drop=True)
                    # Volatility only for the primary PM drivers (keeps width sane)
                    if col in ('pm2_5', 'pm10'):
                        new_cols[f'{col}_rollstd_{win}h'] = roll.std().reset_index(level=0, drop=True)
            logger.info(f"   OK Rolling means {self.rolling_windows}h for {len(drivers)} drivers")

        # Attach all new columns at once (avoids fragmented inserts)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Fill NaNs from the lag/roll warm-up, per city
        temporal_cols = list(new_cols.keys())
        df[temporal_cols] = (
            df.groupby('city', sort=False)[temporal_cols]
              .transform(lambda s: s.ffill().bfill())
        )
        # Any city still fully-NaN for a column -> global median (last resort)
        for c in temporal_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())

        logger.info(f"   OK Temporal features added: {len(temporal_cols)} columns")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables (label encoding, from notebook).
        Also writes encoders.json so the serving layer can replicate the mapping.
        """
        encoders: Dict[str, Dict] = {}

        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            # Deterministic: sort categories alphabetically, assign 0-based codes
            categories = sorted(df[col].dropna().unique().tolist())
            cat_to_code = {cat: idx for idx, cat in enumerate(categories)}
            df[f'{col}_encoded'] = df[col].map(cat_to_code).astype("int32")
            encoders[col] = cat_to_code
            logger.info(
                f"   OK Encoded '{col}' → '{col}_encoded' "
                f"({len(categories)} categories)"
            )

        # Persist so app.py / serving can reconstruct the same integer mapping
        encoders_path = self.output_dir / "encoders.json"
        with open(encoders_path, "w") as f:
            json.dump(encoders, f, indent=2)
        logger.info(f"   OK Encoders saved: {encoders_path}")

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
            "cyclical_features": [
                'month_sin', 'month_cos', 'week_of_year_sin',
                'week_of_year_cos', 'hour_sin', 'hour_cos'
            ] if self.create_cyclical_features else [],
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
