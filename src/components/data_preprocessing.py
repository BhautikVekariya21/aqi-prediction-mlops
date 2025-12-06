# src/components/data_preprocessing.py
import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats.mstats import winsorize

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataPreprocessingConfig:
    input_path: Path
    output_dir: Path
    drop_columns: list
    winsorize_limits: tuple = (0.005, 0.003)
    knn_neighbors: int = 5

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from {self.config.input_path}")
        return pd.read_parquet(self.config.input_path)

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Dropping high-correlation/redundant columns...")
        drop_cols = [c for c in self.config.drop_columns if c in df.columns]
        return df.drop(columns=drop_cols)

    def fix_impossible_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fixing impossible values (PM2.5 > PM10, negatives)...")
        if 'pm2_5_ugm3' in df.columns and 'pm10_ugm3' in df.columns:
            mask = df['pm2_5_ugm3'] > df['pm10_ugm3']
            df.loc[mask, ['pm2_5_ugm3', 'pm10_ugm3']] = df.loc[mask, ['pm10_ugm3', 'pm2_5_ugm3']].values

        pollutant_cols = ['pm2_5_ugm3','pm10_ugm3','co_ugm3','no2_ugm3','so2_ugm3','o3_ugm3','dust_ugm3']
        for col in pollutant_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        return df

    def winsorize_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Applying winsorization for extreme outliers...")
        cols = ['pm2_5_ugm3','pm10_ugm3','co_ugm3','no2_ugm3','so2_ugm3','o3_ugm3','dust_ugm3','aod','us_aqi']
        cols = [c for c in cols if c in df.columns]
        for col in cols:
            df[col] = winsorize(df[col], limits=self.config.winsorize_limits)
        return df

    def knn_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Running KNN Imputation (n_neighbors={self.config.knn_neighbors})...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=self.config.knn_neighbors, weights='distance')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding derived features...")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
        df['pm_ratio'] = (df['pm2_5_ugm3'] / df['pm10_ugm3']).clip(0, 1)
        return df

    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        final_cols = [
            'city', 'state', 'latitude', 'longitude', 'datetime',
            'month', 'is_weekend',
            'humidity_percent', 'dew_point_c', 'wind_gusts_kmh',
            'precipitation_mm', 'pressure_msl_hpa', 'cloud_cover_percent',
            'pm2_5_ugm3', 'pm10_ugm3', 'co_ugm3', 'no2_ugm3', 'so2_ugm3', 'o3_ugm3',
            'dust_ugm3', 'aod', 'us_aqi', 'pm_ratio'
        ]
        final_cols = [c for c in final_cols if c in df.columns]
        df = df[final_cols].copy()
        df = df.astype({'city': 'category', 'state': 'category'})
        return df

    def run(self) -> Path:
        df = self.load_data()
        df = self.clean_columns(df)
        df = self.fix_impossible_values(df)
        df = self.winsorize_outliers(df)
        df = self.knn_imputation(df)
        df = self.add_derived_features(df)
        df = self.finalize(df)

        output_path = self.config.output_dir / "aqi_india_processed.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Preprocessing complete! Saved to {output_path}")
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Missing values: {df.isna().sum().sum()}")

        return output_path