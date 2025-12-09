"""
Stage 6: Model Training
Train 5 regression models (Decision Tree, Random Forest, Extra Trees, XGBoost, CatBoost)
Exact logic from notebook: 06_model_training.ipynb
NO Gradient Boosting (removed as requested)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import json
import time
import joblib
import gc

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader
from ..utils.metrics import AQIMetrics


logger = get_logger(__name__)


class ModelTrainer:
    """
    Train multiple regression models for AQI prediction
    Matches notebook training logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize model trainer
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get training parameters
        train_config = config.get_section("model_training")
        
        self.splits_dir = Path(train_config.get("splits_dir", "data/splits"))
        self.output_dir = Path(train_config.get("output_dir", "models"))
        self.target = train_config.get("target", "us_aqi")
        
        # Models to train (from notebook, NO gradient_boosting)
        self.models_to_train = train_config.get("models", [
            "decision_tree",
            "random_forest",
            "extra_trees",
            "xgboost",
            "catboost"
        ])
        
        self.primary_metric = train_config.get("primary_metric", "r2_score")
        self.early_stopping_rounds = train_config.get("early_stopping_rounds", 50)
        self.use_float32 = train_config.get("use_float32", True)
        self.n_jobs = train_config.get("n_jobs", 4)
        
        # Random state
        self.random_state = config.get("project.random_state", 42)
        
        # Get model hyperparameters
        self.model_params = config.get_section("model_params")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model Trainer initialized")
        logger.info(f"Models to train: {self.models_to_train}")
        logger.info(f"Primary metric: {self.primary_metric}")
    
    def run(self) -> str:
        """
        Run model training pipeline
        
        Returns:
            Path to best model file
        """
        logger.info("="*90)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*90)
        
        # Step 1: Load data
        logger.info(f"\n1. Loading train/val/test data")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = self._load_data()
        
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Val:   {X_val.shape}")
        logger.info(f"   Test:  {X_test.shape}")
        logger.info(f"   Features: {len(feature_names)}")
        
        # Step 2: Initialize models (from notebook)
        logger.info(f"\n2. Initializing {len(self.models_to_train)} models")
        models = self._initialize_models()
        
        # Step 3: Train all models
        logger.info(f"\n3. Training models")
        results = {}
        training_times = {}
        
        for idx, (model_name, model) in enumerate(models.items(), 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"[{idx}/{len(models)}] Training: {model_name}")
            logger.info(f"{'='*70}")
            
            # Force garbage collection before each model
            gc.collect()
            
            try:
                # Train model
                trained_model, train_time = self._train_model(
                    model_name, model, X_train, y_train, X_val, y_val
                )
                
                # Evaluate on all sets
                train_metrics = self._evaluate(trained_model, X_train, y_train, "Train")
                val_metrics = self._evaluate(trained_model, X_val, y_val, "Validation")
                test_metrics = self._evaluate(trained_model, X_test, y_test, "Test")
                
                # Store results
                results[model_name] = {
                    'model': trained_model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'training_time': train_time
                }
                
                training_times[model_name] = train_time
                
                # Print metrics
                self._print_model_metrics(model_name, train_metrics, val_metrics, test_metrics, train_time)
                
                # Save model
                self._save_model(model_name, trained_model, feature_names)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
            
            # Cleanup
            gc.collect()
        
        # Step 4: Compare models and select best
        logger.info(f"\n4. Comparing models")
        best_model_name, comparison_df = self._compare_models(results)
        
        # Step 5: Save comparison results
        logger.info(f"\n5. Saving results")
        self._save_comparison_results(comparison_df, results)
        
        # Step 6: Generate final metrics
        metrics = self._generate_metrics(results, best_model_name, feature_names)
        
        # Save metrics
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"   OK Metrics saved: {metrics_file}")
        
        # Print final summary
        self._print_final_summary(comparison_df, best_model_name, results)
        
        # Return path to best model
        best_model_path = self.output_dir / f"{best_model_name}.json"
        
        logger.info(f"\nOK Model training complete!")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Path: {best_model_path}")
        
        return str(best_model_path)
    
    def _load_data(self) -> Tuple:
        """Load train/val/test data (from notebook)"""
        # Load splits
        train_df = pd.read_parquet(self.splits_dir / "train.parquet")
        val_df = pd.read_parquet(self.splits_dir / "validation.parquet")
        test_df = pd.read_parquet(self.splits_dir / "test.parquet")
        
        # Load feature names
        with open(self.splits_dir / "feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Separate X and y
        X_train = train_df[feature_names].values
        y_train = train_df[self.target].values
        
        X_val = val_df[feature_names].values
        y_val = val_df[self.target].values
        
        X_test = test_df[feature_names].values
        y_test = test_df[self.target].values
        
        # Convert to float32 if needed (from notebook: memory optimization)
        if self.use_float32:
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with hyperparameters (from notebook)
        """
        models = {}
        
        for model_name in self.models_to_train:
            if model_name == "decision_tree":
                params = self.model_params.get("decision_tree", {})
                models["decision_tree"] = DecisionTreeRegressor(
                    max_depth=params.get("max_depth", 15),
                    min_samples_split=params.get("min_samples_split", 20),
                    min_samples_leaf=params.get("min_samples_leaf", 10),
                    random_state=self.random_state
                )
                logger.info(f"   OK Decision Tree initialized")
            
            elif model_name == "random_forest":
                params = self.model_params.get("random_forest", {})
                models["random_forest"] = RandomForestRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 15),
                    min_samples_split=params.get("min_samples_split", 10),
                    min_samples_leaf=params.get("min_samples_leaf", 5),
                    max_features=params.get("max_features", "sqrt"),
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
                logger.info(f"   OK Random Forest initialized")
            
            elif model_name == "extra_trees":
                params = self.model_params.get("extra_trees", {})
                models["extra_trees"] = ExtraTreesRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 15),
                    min_samples_split=params.get("min_samples_split", 10),
                    min_samples_leaf=params.get("min_samples_leaf", 5),
                    max_features=params.get("max_features", "sqrt"),
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
                logger.info(f"   OK Extra Trees initialized")
            
            elif model_name == "xgboost":
                params = self.model_params.get("xgboost", {})
                models["xgboost"] = xgb.XGBRegressor(
                    n_estimators=params.get("n_estimators", 500),
                    max_depth=params.get("max_depth", 8),
                    learning_rate=params.get("learning_rate", 0.05),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 0.1),
                    tree_method=params.get("tree_method", "hist"),
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbosity=0
                )
                logger.info(f"   OK XGBoost initialized")
            
            elif model_name == "catboost":
                params = self.model_params.get("catboost", {})
                models["catboost"] = CatBoostRegressor(
                    iterations=params.get("iterations", 500),
                    depth=params.get("depth", 8),
                    learning_rate=params.get("learning_rate", 0.05),
                    l2_leaf_reg=params.get("l2_leaf_reg", 3),
                    random_seed=self.random_state,
                    verbose=0,
                    thread_count=self.n_jobs
                )
                logger.info(f"   OK CatBoost initialized")
        
        return models
    
    def _train_model(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, float]:
        """
        Train a single model (from notebook)
        """
        start_time = time.time()
        
        # Models with early stopping (from notebook)
        if model_name in ["xgboost", "catboost"]:
            if model_name == "xgboost":
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            elif model_name == "catboost":
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False
                )
        else:
            # Sklearn models (no early stopping)
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        logger.info(f"   Training time: {train_time:.2f} seconds")
        
        return model, train_time
    
    def _evaluate(self, model: Any, X: np.ndarray, y: np.ndarray, set_name: str) -> Dict[str, float]:
        """
        Evaluate model on a dataset (from notebook)
        """
        y_pred = model.predict(X)
        metrics = AQIMetrics.calculate_all_metrics(y, y_pred)
        return metrics
    
    def _print_model_metrics(
        self,
        model_name: str,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict,
        train_time: float
    ):
        """Print metrics for a model (from notebook format)"""
        logger.info(f"\n   {'Metric':<20} {'Train':>12} {'Validation':>12} {'Test':>12}")
        logger.info(f"   {'-'*60}")
        logger.info(f"   {'RMSE':<20} {train_metrics['rmse']:>12.2f} {val_metrics['rmse']:>12.2f} {test_metrics['rmse']:>12.2f}")
        logger.info(f"   {'MAE':<20} {train_metrics['mae']:>12.2f} {val_metrics['mae']:>12.2f} {test_metrics['mae']:>12.2f}")
        logger.info(f"   {'R²':<20} {train_metrics['r2_score']:>12.4f} {val_metrics['r2_score']:>12.4f} {test_metrics['r2_score']:>12.4f}")
        logger.info(f"   {'MAPE %':<20} {train_metrics['mape']:>12.2f} {val_metrics['mape']:>12.2f} {test_metrics['mape']:>12.2f}")
        logger.info(f"   {'Within ±10':<20} {train_metrics['within_10_pct']:>11.1f}% {val_metrics['within_10_pct']:>11.1f}% {test_metrics['within_10_pct']:>11.1f}%")
        logger.info(f"   {'Within ±25':<20} {train_metrics['within_25_pct']:>11.1f}% {val_metrics['within_25_pct']:>11.1f}% {test_metrics['within_25_pct']:>11.1f}%")
        logger.info(f"   {'-'*60}")
    
    def _save_model(self, model_name: str, model: Any, feature_names: list):
        """Save trained model (from notebook)"""
        # Save based on model type
        if model_name == "xgboost":
            model_file = self.output_dir / "xgboost.json"
            model.save_model(str(model_file))
            logger.info(f"   OK Saved: xgboost.json")
        
        elif model_name == "catboost":
            model_file = self.output_dir / "catboost.cbm"
            model.save_model(str(model_file))
            logger.info(f"   OK Saved: catboost.cbm")
        
        else:
            # Sklearn models - use joblib
            model_file = self.output_dir / f"{model_name}.pkl"
            joblib.dump(model, model_file)
            logger.info(f"   OK Saved: {model_name}.pkl")
        
        # Save feature names (once)
        features_file = self.output_dir / "features.txt"
        if not features_file.exists():
            with open(features_file, 'w') as f:
                for feat in feature_names:
                    f.write(f"{feat}\n")
            logger.info(f"   OK Saved: features.txt")
    
    def _compare_models(self, results: Dict) -> Tuple[str, pd.DataFrame]:
        """
        Compare all models and select best (from notebook)
        """
        comparison_data = []
        
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'RMSE': test_metrics['rmse'],
                'MAE': test_metrics['mae'],
                'R²': test_metrics['r2_score'],
                'MAPE (%)': test_metrics['mape'],
                'Within ±10 (%)': test_metrics['within_10_pct'],
                'Within ±25 (%)': test_metrics['within_25_pct'],
                'Within ±50 (%)': test_metrics['within_50_pct'],
                'Train Time (s)': result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (from notebook)
        if self.primary_metric == "r2_score":
            comparison_df = comparison_df.sort_values('R²', ascending=False)
        elif self.primary_metric == "rmse":
            comparison_df = comparison_df.sort_values('RMSE', ascending=True)
        else:
            comparison_df = comparison_df.sort_values('MAE', ascending=True)
        
        comparison_df = comparison_df.reset_index(drop=True)
        comparison_df.index = comparison_df.index + 1
        
        # Best model
        best_model_name = comparison_df.iloc[0]['Model']
        
        logger.info(f"\n   Best model: {best_model_name}")
        logger.info(f"   R² Score: {comparison_df.iloc[0]['R²']:.4f}")
        logger.info(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.2f}")
        
        return best_model_name, comparison_df
    
    def _save_comparison_results(self, comparison_df: pd.DataFrame, results: Dict):
        """Save model comparison results"""
        # Save comparison table
        comparison_file = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=True)
        logger.info(f"   OK Saved: model_comparison.csv")
        
        # Save detailed results
        detailed_results = {}
        for model_name, result in results.items():
            detailed_results[model_name] = {
                'train_metrics': result['train_metrics'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics'],
                'training_time': result['training_time']
            }
        
        detailed_file = self.output_dir / "detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"   OK Saved: detailed_results.json")
    
    def _generate_metrics(self, results: Dict, best_model_name: str, feature_names: list) -> Dict:
        """Generate training metrics"""
        best_result = results[best_model_name]
        
        metrics = {
            "models_trained": list(results.keys()),
            "best_model": best_model_name,
            "best_model_metrics": {
                "train": best_result['train_metrics'],
                "validation": best_result['val_metrics'],
                "test": best_result['test_metrics']
            },
            "best_model_training_time": best_result['training_time'],
            "n_features": len(feature_names),
            "primary_metric": self.primary_metric,
            "random_state": self.random_state
        }
        
        return metrics
    
    def _print_final_summary(self, comparison_df: pd.DataFrame, best_model_name: str, results: Dict):
        """Print final training summary (from notebook)"""
        print("\n" + "="*90)
        print("MODEL TRAINING SUMMARY")
        print("="*90)
        
        print("\nModel Comparison (Test Set Performance):")
        print(comparison_df.to_string())
        
        print("\n" + "="*90)
        print(f"🏆 BEST MODEL: {best_model_name}")
        
        best_metrics = results[best_model_name]['test_metrics']
        print(f"   R² Score:         {best_metrics['r2_score']:.4f}")
        print(f"   RMSE:             {best_metrics['rmse']:.2f}")
        print(f"   MAE:              {best_metrics['mae']:.2f}")
        print(f"   Within ±25 AQI:   {best_metrics['within_25_pct']:.1f}%")
        print("="*90)