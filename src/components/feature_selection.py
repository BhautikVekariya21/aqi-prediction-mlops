# src/components/feature_selection.py
from pathlib import Path
from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class FeatureSelectionConfig:
    input_path: Path
    output_dir: Path
    target: str = "us_aqi"
    top_n: int = 30
    must_have: List[str] = None

class FeatureSelection:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.must_have is None:
            self.config.must_have = []

    def load_data(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.input_path)

    def get_features(self, df: pd.DataFrame) -> tuple:
        X = df.drop(columns=[self.config.target])
        y = df[self.config.target]
        feature_names = X.columns.tolist()
        return X.values.astype(np.float32), y.values.astype(np.float32), feature_names

    def rank_features(self, X, y, feature_names):
        # 1. XGBoost Gain (best for AQI)
        logger.info("Computing XGBoost feature importance...")
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4
        )
        model.fit(X, y)
        xgb_importance = model.feature_importances_

        # 2. Mutual Information
        logger.info("Computing Mutual Information...")
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Combine
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'xgboost_gain': xgb_importance,
            'mutual_info': mi_scores,
            'combined_score': xgb_importance * 0.7 + mi_scores * 0.3
        }).sort_values('combined_score', ascending=False)

        return importance_df

    def select_features(self, importance_df: pd.DataFrame) -> List[str]:
        top_features = importance_df.head(self.config.top_n)['feature'].tolist()
        final_features = list(set(top_features + self.config.must_have))
        logger.info(f"Selected {len(final_features)} features (top {self.config.top_n} + must-have)")
        return final_features

    def run(self) -> tuple[Path, Path]:
        df = self.load_data()
        X, y, feature_names = self.get_features(df)
        importance_df = self.rank_features(X, y, feature_names)
        selected_features = self.select_features(importance_df)

        # Save selected dataset
        selected_df = df[selected_features + [self.config.target]]
        output_path = self.config.output_dir / "aqi_selected_features.parquet"
        selected_df.to_parquet(output_path, index=False)

        # Save feature list
        features_txt = self.config.output_dir / "selected_features.txt"
        with open(features_txt, "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")

        logger.info(f"Feature selection complete! Dataset: {output_path}")
        return output_path, features_txt