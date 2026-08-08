"""
Stage 2: Data Preprocessing
Clean, impute, and handle outliers
Exact logic from notebook: 02_data_preprocessing.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import json
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader


logger = get_logger(__name__)


class DataPreprocessing:
    """
    Preprocess raw data with cleaning, imputation, and outlier handling
    Matches notebook preprocessing logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize data preprocessing
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get preprocessing parameters
        preprocess_config = config.get_section("data_preprocessing")
        
        self.input_path = Path(preprocess_config.get("input_path", "data/raw/aqi_india_raw.parquet"))
        self.output_dir = Path(preprocess_config.get("output_dir", "data/processed"))
        
        self.drop_columns = preprocess_config.get("drop_columns", [])
        
        # Imputation settings (from notebook: KNN with n_neighbors=5)
        imputation_config = preprocess_config.get("imputation", {})
        self.imputation_method = imputation_config.get("method", "knn")
        self.knn_neighbors = imputation_config.get("n_neighbors", 5)
        self.knn_weights = imputation_config.get("weights", "distance")
        
        # Outlier settings (from notebook: winsorize 0.5% to 99.7%)
        outlier_config = preprocess_config.get("outliers", {})
        self.outlier_method = outlier_config.get("method", "winsorize")
        self.lower_percentile = outlier_config.get("lower_percentile", 0.005)
        self.upper_percentile = outlier_config.get("upper_percentile", 0.997)
        self.outlier_columns = outlier_config.get("columns", [])
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data Preprocessing initialized")
    
    def run(self) -> str:
        """
        Run data preprocessing pipeline
        
        Returns:
            Path to output parquet file
        """
        logger.info("="*90)
        logger.info("STARTING DATA PREPROCESSING")
        logger.info("="*90)
        
        # Load raw data
        logger.info(f"\n1. Loading raw data from: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        initial_shape = df.shape
        
        # Step 1: Drop highly correlated columns (from notebook)
        logger.info(f"\n2. Dropping highly correlated columns")
        df = self._drop_columns(df)
        logger.info(f"   Remaining columns: {df.shape[1]}")
        
        # Step 2: Handle out-of-range values (from notebook)
        logger.info(f"\n3. Handling out-of-range values")
        df = self._handle_out_of_range(df)
        
        # Step 3: Fix logical issues (PM2.5 > PM10)
        logger.info(f"\n4. Fixing logical issues")
        df = self._fix_logical_issues(df)
        
        # Step 4: Handle negative values (from notebook)
        logger.info(f"\n5. Handling negative values")
        df = self._handle_negative_values(df)
        
        # Step 5: Winsorization (from notebook)
        logger.info(f"\n6. Applying winsorization ({self.lower_percentile*100:.2f}% to {self.upper_percentile*100:.2f}%)")
        df = self._winsorize_outliers(df)
        
        # Step 6: KNN Imputation (from notebook)
        logger.info(f"\n7. Applying KNN imputation (n_neighbors={self.knn_neighbors})")
        df = self._knn_imputation(df)
        
        # Step 7: Remove duplicates (from notebook)
        logger.info(f"\n8. Removing duplicates")
        df = self._remove_duplicates(df)
        
        # Save processed data
        output_file = self.output_dir / "aqi_india_processed.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"\nOK Processed data saved: {output_file}")
        
        # Generate metrics
        metrics = self._generate_metrics(df, initial_shape)
        
        # Save metrics
        metrics_file = self.output_dir / "preprocessing_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"OK Metrics saved: {metrics_file}")
        
        # Print summary
        self._print_summary(df, initial_shape)
        
        return str(output_file)
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop highly correlated and redundant columns (from notebook)"""
        existing_drop_cols = [col for col in self.drop_columns if col in df.columns]
        
        if existing_drop_cols:
            logger.info(f"   Dropping {len(existing_drop_cols)} columns:")
            for col in existing_drop_cols:
                logger.info(f"     - {col}")
            
            df = df.drop(columns=existing_drop_cols)
        else:
            logger.info("   No columns to drop")
        
        return df
    
    def _handle_out_of_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle out-of-range values (from notebook)
        Set to NaN for later imputation
        """
        # Valid ranges keyed on the real Open-Meteo schema (Delhi-proof).
        valid_ranges = {
            'relative_humidity_2m': (0, 100),
            'dew_point_2m': (-40, 50),
            'wind_speed_10m': (0, 200),
            'wind_gusts_10m': (0, 250),
            'precipitation': (0, 500),
            'pressure_msl': (900, 1100),
            'cloud_cover': (0, 100),
            'pm2_5': (0, 2000),
            'pm10': (0, 4000),
            'carbon_monoxide': (0, 15000),
            'nitrogen_dioxide': (0, 1000),
            'sulphur_dioxide': (0, 1000),
            'ozone': (0, 600),
            'ammonia': (0, 2000),
            'dust': (0, 5000),
            'aerosol_optical_depth': (0, 10),
            'us_aqi': (0, 1000),
        }
        
        # Only check columns that exist
        for col, (low, high) in valid_ranges.items():
            if col in df.columns:
                out_of_range = ((df[col] < low) | (df[col] > high)).sum()
                if out_of_range > 0:
                    logger.info(f"   {col}: {out_of_range} out-of-range values to NaN")
                    df.loc[(df[col] < low) | (df[col] > high), col] = np.nan
        
        return df
    
    def _fix_logical_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix PM2.5 > PM10 (physically impossible)
        From notebook: swap values
        """
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            mask = df['pm2_5'] > df['pm10']
            n_swaps = mask.sum()

            if n_swaps > 0:
                logger.info(f"   Fixing {n_swaps} cases where PM2.5 > PM10 to swapping values")
                df.loc[mask, ['pm2_5', 'pm10']] = \
                    df.loc[mask, ['pm10', 'pm2_5']].values
        
        return df
    
    def _handle_negative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle negative values in pollutant columns (from notebook)
        Clip to 0
        """
        pollutant_cols = [
            'pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide',
            'sulphur_dioxide', 'ozone', 'dust', 'ammonia',
            'aerosol_optical_depth', 'precipitation', 'us_aqi'
        ]
        
        for col in pollutant_cols:
            if col in df.columns:
                n_negative = (df[col] < 0).sum()
                if n_negative > 0:
                    logger.info(f"   {col}: {n_negative} negative values to clipped to 0")
                    df[col] = df[col].clip(lower=0)
        
        return df
    
    def _winsorize_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize outliers (from notebook: 0.5% to 99.7%)
        Preserves extreme values while capping statistical outliers
        """
        # Only winsorize columns that exist
        existing_outlier_cols = [col for col in self.outlier_columns if col in df.columns]
        
        if not existing_outlier_cols:
            logger.info("   No columns to winsorize")
            return df
        
        logger.info(f"   Winsorizing {len(existing_outlier_cols)} columns:")
        
        for col in existing_outlier_cols:
            # Convert to float so Python None → np.nan before winsorizing
            s = pd.to_numeric(df[col], errors="coerce")
            arr = s.values.astype(float)

            # Winsorize only the finite (non-NaN) positions; preserve NaN locations
            nan_mask = np.isnan(arr)
            if nan_mask.all():
                logger.info(f"     SKIP {col} (all NaN)")
                continue

            limits = (self.lower_percentile, 1 - self.upper_percentile)
            arr_winsorized = np.where(nan_mask, np.nan,
                                      winsorize(np.where(nan_mask, 0.0, arr),
                                                limits=limits))
            df[col] = pd.Series(arr_winsorized, index=df.index, dtype="float32")
            logger.info(f"     OK {col}")
        
        return df
    
    def _knn_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        KNN imputation for missing values (from notebook)
        Best method for AQI data (preserves spatial-temporal patterns)
        """
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check missing values before imputation
        missing_before = df[numeric_cols].isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("   No missing values to impute")
            return df
        
        logger.info(f"   Missing values before imputation: {missing_before:,}")
        
        # KNN Imputation (from notebook: n_neighbors=5, weights='distance')
        imputer = KNNImputer(
            n_neighbors=self.knn_neighbors,
            weights=self.knn_weights
        )
        
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Verify
        missing_after = df[numeric_cols].isnull().sum().sum()
        logger.info(f"   Missing values after imputation: {missing_after:,}")
        logger.info(f"   OK KNN imputation complete")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records (from notebook)"""
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        final_count = len(df)
        
        duplicates_removed = initial_count - final_count
        
        if duplicates_removed > 0:
            logger.info(f"   Removed {duplicates_removed:,} duplicate rows")
        else:
            logger.info(f"   No duplicates found")
        
        return df
    
    def _generate_metrics(self, df: pd.DataFrame, initial_shape: tuple) -> Dict:
        """Generate preprocessing metrics"""
        metrics = {
            "initial_rows": int(initial_shape[0]),
            "initial_columns": int(initial_shape[1]),
            "final_rows": int(df.shape[0]),
            "final_columns": int(df.shape[1]),
            "rows_removed": int(initial_shape[0] - df.shape[0]),
            "columns_removed": int(initial_shape[1] - df.shape[1]),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": 0,  # Already removed
            "pm25_mean": float(df.get('pm2_5', pd.Series([np.nan])).mean()),
            "us_aqi_mean": float(df.get('us_aqi', pd.Series([np.nan])).mean()),
        }
        
        return metrics
    
    def _print_summary(self, df: pd.DataFrame, initial_shape: tuple):
        """Print preprocessing summary"""
        print("\n" + "="*90)
        print("DATA PREPROCESSING SUMMARY")
        print("="*90)
        print(f"Initial Shape:        {initial_shape[0]:,} rows × {initial_shape[1]} columns")
        print(f"Final Shape:          {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Rows Removed:         {initial_shape[0] - df.shape[0]:,}")
        print(f"Columns Removed:      {initial_shape[1] - df.shape[1]}")
        print(f"\nData Quality:")
        print(f"  Missing Values:     {df.isnull().sum().sum():,}")
        print(f"  Duplicate Rows:     0 (removed)")
        
        if 'pm2_5' in df.columns:
            print(f"\nPM2.5 Statistics:")
            print(f"  Mean:               {df['pm2_5'].mean():.2f}")
            print(f"  Max:                {df['pm2_5'].max():.2f}")
        
        print("="*90)