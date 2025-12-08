"""
Stage 5: Data Splitting
Stratified random split (Train/Val/Test)
Exact logic from notebook: 05_data_splitting.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import json
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader


logger = get_logger(__name__)


class DataSplitting:
    """
    Split data into train/validation/test sets
    Matches notebook splitting logic exactly (stratified random split)
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize data splitting
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get splitting parameters
        split_config = config.get_section("data_splitting")
        
        self.input_path = Path(split_config.get("input_path", "data/features/selection/aqi_selected_features.parquet"))
        self.output_dir = Path(split_config.get("output_dir", "data/splits"))
        
        self.test_size = split_config.get("test_size", 0.15)
        self.validation_size = split_config.get("validation_size", 0.15)
        
        # Stratification settings (from notebook)
        self.stratify = split_config.get("stratify", True)
        self.stratify_bins = split_config.get("stratify_bins", [0, 50, 100, 150, 200, 300, 500, 1000])
        
        # Random state
        self.random_state = config.get("project.random_state", 42)
        
        # Target
        self.target = "us_aqi"
        
        # Reference columns
        self.reference_cols = ['datetime', 'city', 'state']
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data Splitting initialized")
        logger.info(f"Test size: {self.test_size*100:.0f}%")
        logger.info(f"Validation size: {self.validation_size*100:.0f}%")
        logger.info(f"Stratified: {self.stratify}")
    
    def run(self) -> Tuple[str, str, str]:
        """
        Run data splitting pipeline
        
        Returns:
            Tuple of (train_path, val_path, test_path)
        """
        logger.info("="*90)
        logger.info("STARTING DATA SPLITTING")
        logger.info("="*90)
        
        # Load selected features data
        logger.info(f"\n1. Loading selected features from: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Convert datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Step 1: Identify features
        logger.info(f"\n2. Identifying features")
        exclude_cols = [self.target] + self.reference_cols
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Target: {self.target}")
        
        # Step 2: Data overview (from notebook)
        logger.info(f"\n3. Data overview")
        self._print_data_overview(df)
        
        # Step 3: Prepare data for splitting
        X = df[feature_cols].copy()
        y = df[self.target].copy()
        datetime_col = df['datetime'].copy() if 'datetime' in df.columns else None
        city_col = df['city'].copy() if 'city' in df.columns else None
        state_col = df['state'].copy() if 'state' in df.columns else None
        
        # Step 4: Create stratification bins (from notebook)
        if self.stratify:
            logger.info(f"\n4. Creating stratification bins")
            y_bins = pd.cut(y, bins=self.stratify_bins, labels=False)
            logger.info(f"   Bins: {self.stratify_bins}")
            logger.info(f"   Distribution:")
            for i, (low, high) in enumerate(zip(self.stratify_bins[:-1], self.stratify_bins[1:])):
                count = (y_bins == i).sum()
                pct = count / len(y_bins) * 100
                logger.info(f"     {low:>4} - {high:>4}: {count:>7,} ({pct:>5.2f}%)")
        else:
            y_bins = None
        
        # Step 5: First split - separate test set (from notebook)
        logger.info(f"\n5. Splitting data (stratified random)")
        
        split_data = [X, y]
        if datetime_col is not None:
            split_data.append(datetime_col)
        if city_col is not None:
            split_data.append(city_col)
        if state_col is not None:
            split_data.append(state_col)
        if y_bins is not None:
            split_data.append(y_bins)
        
        # First split: train+val vs test
        temp_data, test_data = train_test_split(
            *split_data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_bins if self.stratify else None,
            shuffle=True
        )
        
        # Extract temp data
        X_temp = temp_data[0]
        y_temp = temp_data[1]
        idx = 2
        dt_temp = temp_data[idx] if datetime_col is not None else None
        idx += 1 if datetime_col is not None else 0
        city_temp = temp_data[idx] if city_col is not None else None
        idx += 1 if city_col is not None else 0
        state_temp = temp_data[idx] if state_col is not None else None
        idx += 1 if state_col is not None else 0
        bins_temp = temp_data[idx] if y_bins is not None else None
        
        # Extract test data
        X_test = test_data[0]
        y_test = test_data[1]
        idx = 2
        dt_test = test_data[idx] if datetime_col is not None else None
        idx += 1 if datetime_col is not None else 0
        city_test = test_data[idx] if city_col is not None else None
        idx += 1 if city_col is not None else 0
        state_test = test_data[idx] if state_col is not None else None
        
        # Second split: train vs validation (from notebook)
        val_size_adjusted = self.validation_size / (1 - self.test_size)
        
        temp_data2 = [X_temp, y_temp]
        if dt_temp is not None:
            temp_data2.append(dt_temp)
        if city_temp is not None:
            temp_data2.append(city_temp)
        if state_temp is not None:
            temp_data2.append(state_temp)
        
        train_data, val_data = train_test_split(
            *temp_data2,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=bins_temp if self.stratify else None,
            shuffle=True
        )
        
        # Extract train data
        X_train = train_data[0]
        y_train = train_data[1]
        idx = 2
        dt_train = train_data[idx] if dt_temp is not None else None
        idx += 1 if dt_temp is not None else 0
        city_train = train_data[idx] if city_temp is not None else None
        idx += 1 if city_temp is not None else 0
        state_train = train_data[idx] if state_temp is not None else None
        
        # Extract val data
        X_val = val_data[0]
        y_val = val_data[1]
        idx = 2
        dt_val = val_data[idx] if dt_temp is not None else None
        idx += 1 if dt_temp is not None else 0
        city_val = val_data[idx] if city_temp is not None else None
        idx += 1 if city_temp is not None else 0
        state_val = val_data[idx] if state_temp is not None else None
        
        logger.info(f"   ✓ Split complete:")
        logger.info(f"     Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"     Val:   {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        logger.info(f"     Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
        
        # Step 6: Verify stratification (from notebook)
        if self.stratify:
            logger.info(f"\n6. Verifying stratification")
            self._verify_stratification(y_train, y_val, y_test)
        
        # Step 7: Create DataFrames (from notebook)
        logger.info(f"\n7. Creating final DataFrames")
        
        train_df = self._create_dataframe(X_train, y_train, feature_cols, dt_train, city_train, state_train)
        val_df = self._create_dataframe(X_val, y_val, feature_cols, dt_val, city_val, state_val)
        test_df = self._create_dataframe(X_test, y_test, feature_cols, dt_test, city_test, state_test)
        
        # Step 8: Save splits
        logger.info(f"\n8. Saving splits")
        
        train_path = self.output_dir / "train.parquet"
        val_path = self.output_dir / "validation.parquet"
        test_path = self.output_dir / "test.parquet"
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        
        logger.info(f"   ✓ Train saved: {train_path.name}")
        logger.info(f"   ✓ Validation saved: {val_path.name}")
        logger.info(f"   ✓ Test saved: {test_path.name}")
        
        # Save feature names
        features_file = self.output_dir / "feature_names.txt"
        with open(features_file, 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        logger.info(f"   ✓ Feature names saved: {features_file.name}")
        
        # Generate metrics
        metrics = self._generate_metrics(df, train_df, val_df, test_df, feature_cols)
        
        # Save metrics
        metrics_file = self.output_dir / "split_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"   ✓ Metrics saved: {metrics_file.name}")
        
        # Print summary
        self._print_summary(train_df, val_df, test_df)
        
        return str(train_path), str(val_path), str(test_path)
    
    def _print_data_overview(self, df: pd.DataFrame):
        """Print data overview (from notebook)"""
        logger.info(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"\n   Target ({self.target}) statistics:")
        logger.info(f"     Mean:   {df[self.target].mean():.2f}")
        logger.info(f"     Median: {df[self.target].median():.2f}")
        logger.info(f"     Std:    {df[self.target].std():.2f}")
        logger.info(f"     Min:    {df[self.target].min():.2f}")
        logger.info(f"     Max:    {df[self.target].max():.2f}")
        
        # Extreme values check (from notebook)
        extreme_threshold = 300
        n_extreme = (df[self.target] > extreme_threshold).sum()
        pct_extreme = n_extreme / len(df) * 100
        
        logger.info(f"\n   Extreme AQI (>{extreme_threshold}):")
        logger.info(f"     Count: {n_extreme:,} ({pct_extreme:.2f}%)")
    
    def _verify_stratification(self, y_train, y_val, y_test):
        """Verify stratification worked (from notebook)"""
        extreme_threshold = 300
        
        n_extreme_train = (y_train > extreme_threshold).sum()
        n_extreme_val = (y_val > extreme_threshold).sum()
        n_extreme_test = (y_test > extreme_threshold).sum()
        
        pct_train = n_extreme_train / len(y_train) * 100
        pct_val = n_extreme_val / len(y_val) * 100
        pct_test = n_extreme_test / len(y_test) * 100
        
        logger.info(f"   Extreme AQI (>{extreme_threshold}) distribution:")
        logger.info(f"     Train: {n_extreme_train:,} ({pct_train:.2f}%)")
        logger.info(f"     Val:   {n_extreme_val:,} ({pct_val:.2f}%)")
        logger.info(f"     Test:  {n_extreme_test:,} ({pct_test:.2f}%)")
        logger.info(f"   ✓ Stratification successful - extreme values in all splits")
    
    def _create_dataframe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list,
        datetime_col: pd.Series = None,
        city_col: pd.Series = None,
        state_col: pd.Series = None
    ) -> pd.DataFrame:
        """Create final DataFrame with all columns (from notebook)"""
        df = pd.DataFrame(X, columns=feature_cols)
        df[self.target] = y.values
        
        if datetime_col is not None:
            df['datetime'] = datetime_col.values
        if city_col is not None:
            df['city'] = city_col.values
        if state_col is not None:
            df['state'] = state_col.values
        
        return df
    
    def _generate_metrics(
        self,
        df_original: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list
    ) -> dict:
        """Generate splitting metrics (from notebook)"""
        metrics = {
            "total_samples": int(len(df_original)),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "train_pct": float(len(train_df) / len(df_original) * 100),
            "val_pct": float(len(val_df) / len(df_original) * 100),
            "test_pct": float(len(test_df) / len(df_original) * 100),
            "n_features": int(len(feature_cols)),
            "split_method": "stratified_random",
            "test_size": float(self.test_size),
            "validation_size": float(self.validation_size),
            "random_state": int(self.random_state),
            "target_stats": {
                "train": {
                    "mean": float(train_df[self.target].mean()),
                    "std": float(train_df[self.target].std()),
                    "min": float(train_df[self.target].min()),
                    "max": float(train_df[self.target].max())
                },
                "val": {
                    "mean": float(val_df[self.target].mean()),
                    "std": float(val_df[self.target].std()),
                    "min": float(val_df[self.target].min()),
                    "max": float(val_df[self.target].max())
                },
                "test": {
                    "mean": float(test_df[self.target].mean()),
                    "std": float(test_df[self.target].std()),
                    "min": float(test_df[self.target].min()),
                    "max": float(test_df[self.target].max())
                }
            }
        }
        
        return metrics
    
    def _print_summary(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Print splitting summary (from notebook)"""
        total = len(train_df) + len(val_df) + len(test_df)
        
        print("\n" + "="*90)
        print("DATA SPLITTING SUMMARY")
        print("="*90)
        print(f"Total Samples:        {total:,}")
        print(f"\nSplit Distribution:")
        print(f"  Train:              {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
        print(f"  Validation:         {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
        print(f"  Test:               {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
        
        print(f"\nTarget Statistics:")
        print(f"{'Set':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 54)
        print(f"{'Train':<12} {train_df[self.target].mean():>10.2f} {train_df[self.target].std():>10.2f} "
              f"{train_df[self.target].min():>10.2f} {train_df[self.target].max():>10.2f}")
        print(f"{'Validation':<12} {val_df[self.target].mean():>10.2f} {val_df[self.target].std():>10.2f} "
              f"{val_df[self.target].min():>10.2f} {val_df[self.target].max():>10.2f}")
        print(f"{'Test':<12} {test_df[self.target].mean():>10.2f} {test_df[self.target].std():>10.2f} "
              f"{test_df[self.target].min():>10.2f} {test_df[self.target].max():>10.2f}")
        
        print("\n" + "="*90)