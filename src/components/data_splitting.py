# src/components/data_splitting.py
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class DataSplittingConfig:
    input_path: Path
    output_dir: Path
    test_size: float = 0.15
    val_size: float = 0.15
    stratify: bool = True
    random_state: int = 42

class DataSplitting:
    def __init__(self, config: DataSplittingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading selected features from {self.config.input_path}")
        return pd.read_parquet(self.config.input_path)

    def create_stratification_bins(self, y: pd.Series) -> pd.Series:
        """Create bins to preserve extreme AQI events (Diwali 2025: 700-800 AQI)"""
        bins = [0, 50, 100, 150, 200, 300, 500, 1000]
        return pd.cut(y, bins=bins, labels=False, include_lowest=True)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        y = df["us_aqi"]
        X = df.drop(columns=["us_aqi"])

        if self.config.stratify:
            logger.info("Using stratified split to preserve extreme AQI distribution...")
            y_bins = self.create_stratification_bins(y)
            X_temp, X_test, y_temp, y_test, _, _ = train_test_split(
                X, y, y_bins, test_size=self.config.test_size,
                random_state=self.config.random_state, stratify=y_bins
            )
            val_ratio = self.config.val_size / (1 - self.config.test_size)
            y_bins_temp = self.create_stratification_bins(y_temp)
            X_train, X_val, y_train, y_val, _, _ = train_test_split(
                X_temp, y_temp, y_bins_temp, test_size=val_ratio,
                random_state=self.config.random_state, stratify=y_bins_temp
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            val_ratio = self.config.val_size / (1 - self.config.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.config.random_state
            )

        # Reconstruct DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df):
        train_path = self.config.output_dir / "train.parquet"
        val_path = self.config.output_dir / "val.parquet"
        test_path = self.config.output_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        logger.info(f"Train: {len(train_df):,} samples → {train_path}")
        logger.info(f"Val:   {len(val_df):,} samples → {val_path}")
        logger.info(f"Test:  {len(test_df):,} samples → {test_path}")

        # Save feature names
        feature_names_path = self.config.output_dir / "feature_names.txt"
        with open(feature_names_path, "w") as f:
            for col in train_df.columns:
                if col != "us_aqi":
                    f.write(f"{col}\n")
        logger.info(f"Feature names saved: {feature_names_path}")

        return train_path, val_path, test_path, feature_names_path

    def run(self):
        df = self.load_data()
        logger.info(f"Original dataset: {len(df):,} rows, {len(df.columns)} columns")

        train_df, val_df, test_df = self.split_data(df)
        paths = self.save_splits(train_df, val_df, test_df)

        logger.info("Extreme AQI distribution check:")
        for name, data in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            extreme = (data["us_aqi"] > 300).sum()
            logger.info(f"  {name}: {extreme:,} samples with AQI > 300")

        return paths