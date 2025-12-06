# src/components/feature_engineering.py
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FeatureEngineeringConfig:
    input_path: Path
    output_dir: Path
    target: str = "us_aqi"
    categorical_columns: list = None

class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.categorical_columns is None:
            self.config.categorical_columns = ["city", "state"]

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading processed data from {self.config.input_path}")
        return pd.read_parquet(self.config.input_path)

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Label encoding categorical columns...")
        for col in self.config.categorical_columns:
            if col in df.columns:
                df[f"{col}_encoded"] = pd.Categorical(df[col]).codes.astype("int32")
        return df

    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating cyclical month features...")
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop original categorical strings (keep encoded)
        cols_to_drop = ['city', 'state', 'datetime']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        return df

    def run(self) -> Path:
        df = self.load_data()
        df = self.encode_categorical(df)
        df = self.create_cyclical_features(df)
        df = self.finalize_features(df)

        output_path = self.config.output_dir / "aqi_features.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Feature engineering complete! Saved to {output_path}")
        logger.info(f"Final features: {len(df.columns) - 1} + target")

        return output_path