"""
Stage 4: Feature Selection
Multi-method feature selection (Correlation, MI, Decision Tree, LightGBM, XGBoost)
Exact logic from notebook: 04_feature_selection.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import lightgbm as lgb
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader


logger = get_logger(__name__)


class FeatureSelection:
    """
    Select best features using multiple methods
    Matches notebook feature selection logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize feature selection
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get feature selection parameters
        fs_config = config.get_section("feature_selection")
        
        self.input_path = Path(fs_config.get("input_path", "data/features/aqi_features.parquet"))
        self.output_dir = Path(fs_config.get("output_dir", "data/features/selection"))
        self.target = fs_config.get("target", "us_aqi")
        self.top_n_features = fs_config.get("top_n_features", 30)
        
        # Columns to exclude (prevent data leakage - from notebook)
        self.exclude_columns = fs_config.get("exclude_columns", [])
        
        # Must-have features (from notebook)
        self.must_have_features = fs_config.get("must_have_features", [])
        
        # Selection methods (from notebook)
        self.methods = fs_config.get("methods", {})
        
        # Random state
        self.random_state = config.get("project.random_state", 42)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Feature Selection initialized")
        logger.info(f"Target: {self.target}")
        logger.info(f"Top N features: {self.top_n_features}")
    
    def run(self) -> str:
        """
        Run feature selection pipeline
        
        Returns:
            Path to output parquet file
        """
        logger.info("="*90)
        logger.info("STARTING FEATURE SELECTION")
        logger.info("="*90)
        
        # Load feature-engineered data
        logger.info(f"\n1. Loading feature data from: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Step 1: Prepare features
        logger.info(f"\n2. Preparing features")
        feature_cols, categorical_cols = self._prepare_features(df)
        logger.info(f"   Candidate features: {len(feature_cols)}")
        logger.info(f"   Categorical features: {len(categorical_cols)}")
        
        # Step 2: Encode categorical features (from notebook)
        logger.info(f"\n3. Encoding categorical features")
        df_encoded = self._encode_categorical(df, categorical_cols, feature_cols)
        
        # Step 3: Prepare X and y
        X = df_encoded[feature_cols].values.astype(np.float32)
        y = df_encoded[self.target].values.astype(np.float32)
        
        logger.info(f"   X shape: {X.shape}")
        logger.info(f"   y shape: {y.shape}")
        
        # Step 4: Run all selection methods
        all_importances = {}
        
        if self.methods.get('correlation', True):
            logger.info(f"\n4. Method 1: Correlation with target")
            corr_importance = self._correlation_selection(df_encoded, feature_cols)
            all_importances['correlation'] = corr_importance
        
        if self.methods.get('mutual_info', True):
            logger.info(f"\n5. Method 2: Mutual Information")
            mi_importance = self._mutual_info_selection(X, y, feature_cols)
            all_importances['mutual_info'] = mi_importance
        
        if self.methods.get('decision_tree', True):
            logger.info(f"\n6. Method 3: Decision Tree Importance")
            dt_importance = self._decision_tree_selection(X, y, feature_cols)
            all_importances['decision_tree'] = dt_importance
        
        if self.methods.get('lightgbm', True):
            logger.info(f"\n7. Method 4: LightGBM Importance")
            lgb_importance = self._lightgbm_selection(X, y, feature_cols)
            all_importances['lightgbm'] = lgb_importance
        
        if self.methods.get('xgboost', True):
            logger.info(f"\n8. Method 5: XGBoost Importance")
            xgb_importance = self._xgboost_selection(X, y, feature_cols)
            all_importances['xgboost'] = xgb_importance
        
        # Step 5: Combine rankings (from notebook)
        logger.info(f"\n9. Combining feature rankings")
        combined_df, final_features = self._combine_rankings(
            all_importances,
            feature_cols,
            categorical_cols
        )
        
        # Step 6: Create final dataset with selected features
        logger.info(f"\n10. Creating final dataset with selected features")
        final_columns = final_features + [self.target, 'datetime', 'city', 'state']
        final_columns = [c for c in final_columns if c in df_encoded.columns]
        
        df_selected = df_encoded[final_columns].copy()
        
        # Save selected features dataset
        output_file = self.output_dir / "aqi_selected_features.parquet"
        df_selected.to_parquet(output_file, index=False)
        logger.info(f"   OK Selected features dataset saved: {output_file}")
        
        # Save feature names
        features_file = self.output_dir / "selected_features.txt"
        with open(features_file, 'w') as f:
            for feat in final_features:
                f.write(f"{feat}\n")
        logger.info(f"   OK Feature names saved: {features_file}")
        
        # Save all importances
        self._save_importances(all_importances, combined_df)
        
        # Generate metrics
        metrics = self._generate_metrics(df, df_selected, final_features, all_importances)
        
        # Save metrics
        metrics_file = self.output_dir / "selection_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"   OK Metrics saved: {metrics_file}")
        
        # Print summary
        self._print_summary(df, df_selected, final_features, combined_df)
        
        return str(output_file)
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Prepare feature columns (from notebook)
        Returns: (numeric_features, categorical_features)
        """
        feature_cols = []
        categorical_cols = []
        
        for col in df.columns:
            # Skip excluded columns (prevent data leakage)
            if col in self.exclude_columns:
                continue
            
            # Check data type
            dtype = df[col].dtype
            
            # Numeric features
            if dtype in ['float32', 'float64', 'int32', 'int64', 'uint8', 'bool']:
                feature_cols.append(col)
            
            # Categorical features (city, state)
            elif col in ['city', 'state']:
                categorical_cols.append(col)
        
        logger.info(f"\n   Excluded columns (prevent data leakage):")
        for col in self.exclude_columns:
            if col in df.columns:
                logger.info(f"     FAILED {col}")
        
        return feature_cols, categorical_cols
    
    def _encode_categorical(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Encode categorical features (from notebook: label encoding)
        """
        df_encoded = df.copy()
        
        for cat_col in categorical_cols:
            encoded_col = f"{cat_col}_encoded"
            
            # Check if already encoded
            if encoded_col not in df_encoded.columns:
                df_encoded[encoded_col] = pd.Categorical(df_encoded[cat_col]).codes
                logger.info(f"   OK Encoded {cat_col} to {encoded_col}")
            
            # Add to feature list
            if encoded_col not in feature_cols:
                feature_cols.append(encoded_col)
        
        return df_encoded
    
    def _correlation_selection(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Method 1: Correlation with target (from notebook)
        """
        correlations = df[feature_cols].corrwith(df[self.target]).abs()
        
        corr_df = pd.DataFrame({
            'feature': feature_cols,
            'correlation': correlations.values
        }).sort_values('correlation', ascending=False)
        
        logger.info(f"   OK Correlation computed for {len(feature_cols)} features")
        logger.info(f"   Top 5: {corr_df.head(5)['feature'].tolist()}")
        
        return corr_df
    
    def _mutual_info_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Method 2: Mutual Information (from notebook)
        Uses sampling for large datasets to avoid memory issues
        """
        # Sample if dataset is too large (> 500k samples)
        max_samples = 500000
        
        if X.shape[0] > max_samples:
            logger.info(f"   Dataset too large ({X.shape[0]:,} samples)")
            logger.info(f"   Using stratified sample of {max_samples:,} for MI computation...")
            
            # Stratified sampling based on target bins
            from sklearn.model_selection import train_test_split
            
            # Create bins for stratification
            y_bins = pd.cut(y, bins=[0, 50, 100, 150, 200, 300, 500, 1000], labels=False)
            
            # Sample
            try:
                X_sample, _, y_sample, _ = train_test_split(
                    X, y,
                    train_size=max_samples,
                    stratify=y_bins,
                    random_state=self.random_state
                )
                logger.info(f"   [OK] Sampled {len(X_sample):,} records (stratified)")
            except:
                # Fallback: random sampling without stratification
                indices = np.random.RandomState(self.random_state).choice(
                    X.shape[0], max_samples, replace=False
                )
                X_sample = X[indices]
                y_sample = y[indices]
                logger.info(f"   [OK] Sampled {len(X_sample):,} records (random)")
        else:
            X_sample = X
            y_sample = y
            logger.info(f"   Computing MI for {X.shape[0]:,} samples...")
        
        # Compute MI
        try:
            mi_scores = mutual_info_regression(
                X_sample, y_sample,
                random_state=self.random_state
            )
        except MemoryError as e:
            logger.error(f"   ✗ MI computation failed: {e}")
            logger.warning(f"   Using correlation as fallback...")
            # Fallback: use correlation instead
            mi_scores = np.abs([np.corrcoef(X_sample[:, i], y_sample)[0, 1] 
                            for i in range(X_sample.shape[1])])
        
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        logger.info(f"   [OK] Mutual Information computed")
        logger.info(f"   Top 5: {mi_df.head(5)['feature'].tolist()}")
        
        return mi_df
    
    def _decision_tree_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Method 3: Decision Tree importance (from notebook)
        """
        dt = DecisionTreeRegressor(
            max_depth=15,
            min_samples_split=100,
            min_samples_leaf=50,
            random_state=self.random_state
        )
        
        logger.info(f"   Training Decision Tree...")
        dt.fit(X, y)
        
        dt_df = pd.DataFrame({
            'feature': feature_cols,
            'dt_importance': dt.feature_importances_
        }).sort_values('dt_importance', ascending=False)
        
        logger.info(f"   OK Decision Tree trained")
        logger.info(f"   Top 5: {dt_df.head(5)['feature'].tolist()}")
        
        return dt_df
    
    def _lightgbm_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Method 4: LightGBM importance (from notebook)
        """
        lgb_train = lgb.Dataset(X, label=y, feature_name=feature_cols, free_raw_data=False)
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': self.random_state
        }
        
        logger.info(f"   Training LightGBM...")
        lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=300)
        
        lgb_df = pd.DataFrame({
            'feature': feature_cols,
            'lgb_importance': lgb_model.feature_importance(importance_type='gain')
        }).sort_values('lgb_importance', ascending=False)
        
        logger.info(f"   OK LightGBM trained")
        logger.info(f"   Top 5: {lgb_df.head(5)['feature'].tolist()}")
        
        return lgb_df
    
    def _xgboost_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Method 5: XGBoost importance (from notebook)
        """
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist',
            verbosity=0
        )
        
        logger.info(f"   Training XGBoost...")
        xgb_model.fit(X, y)
        
        xgb_df = pd.DataFrame({
            'feature': feature_cols,
            'xgb_importance': xgb_model.feature_importances_
        }).sort_values('xgb_importance', ascending=False)
        
        logger.info(f"   OK XGBoost trained")
        logger.info(f"   Top 5: {xgb_df.head(5)['feature'].tolist()}")
        
        return xgb_df
    
    def _combine_rankings(
        self,
        all_importances: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        categorical_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Combine all rankings and select final features (from notebook)
        """
        # Start with all features
        combined_df = pd.DataFrame({'feature': feature_cols})
        
        # Merge all importance scores
        for method, importance_df in all_importances.items():
            combined_df = combined_df.merge(importance_df, on='feature', how='left')
        
        # Create rankings for each method
        for method in all_importances.keys():
            score_col = list(all_importances[method].columns)[1]  # Get score column name
            combined_df[f'{method}_rank'] = combined_df[score_col].rank(ascending=False)
        
        # Calculate average rank (from notebook)
        rank_cols = [col for col in combined_df.columns if col.endswith('_rank')]
        combined_df['avg_rank'] = combined_df[rank_cols].mean(axis=1)
        
        # Sort by average rank
        combined_df = combined_df.sort_values('avg_rank')
        
        # Select top N features
        selected_features = combined_df.head(self.top_n_features)['feature'].tolist()
        
        # Add must-have features (from notebook)
        for feat in self.must_have_features:
            # Check if encoded version exists
            if feat in feature_cols and feat not in selected_features:
                selected_features.append(feat)
        
        # Ensure categorical encodings are included
        for cat_col in categorical_cols:
            encoded_col = f"{cat_col}_encoded"
            if encoded_col not in selected_features and encoded_col in feature_cols:
                selected_features.append(encoded_col)
        
        # Remove duplicates
        final_features = list(set(selected_features))
        
        logger.info(f"   OK Combined rankings from {len(all_importances)} methods")
        logger.info(f"   OK Selected {len(final_features)} final features")
        
        return combined_df, final_features
    
    def _save_importances(self, all_importances: Dict[str, pd.DataFrame], combined_df: pd.DataFrame):
        """Save all importance scores"""
        # Save individual importance scores
        for method, importance_df in all_importances.items():
            output_file = self.output_dir / f"importance_{method}.csv"
            importance_df.to_csv(output_file, index=False)
            logger.info(f"   OK Saved {method} importance: {output_file.name}")
        
        # Save combined rankings
        combined_file = self.output_dir / "all_feature_importances.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"   OK Saved combined importances: {combined_file.name}")
    
    def _generate_metrics(
        self,
        df_original: pd.DataFrame,
        df_selected: pd.DataFrame,
        final_features: List[str],
        all_importances: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Generate feature selection metrics"""
        metrics = {
            "total_features_before": int(df_original.shape[1]),
            "total_features_after": int(len(final_features)),
            "features_removed": int(df_original.shape[1] - len(final_features)),
            "selection_methods_used": list(all_importances.keys()),
            "top_n_requested": int(self.top_n_features),
            "final_features_count": int(len(final_features)),
            "must_have_features": self.must_have_features,
            "selected_features": final_features,
        }
        
        return metrics
    
    def _print_summary(
        self,
        df_original: pd.DataFrame,
        df_selected: pd.DataFrame,
        final_features: List[str],
        combined_df: pd.DataFrame
    ):
        """Print feature selection summary"""
        print("\n" + "="*90)
        print("FEATURE SELECTION SUMMARY")
        print("="*90)
        print(f"Original Features:    {df_original.shape[1]}")
        print(f"Selected Features:    {len(final_features)}")
        print(f"Features Removed:     {df_original.shape[1] - len(final_features)}")
        
        print(f"\nTop 15 Features by Combined Ranking:")
        print("-" * 90)
        print(f"{'Rank':<6} {'Feature':<30} {'Avg Rank':<12}")
        print("-" * 90)
        
        for idx, row in combined_df.head(15).iterrows():
            print(f"{idx+1:<6} {row['feature']:<30} {row['avg_rank']:<12.2f}")
        
        print("-" * 90)
        print("="*90)