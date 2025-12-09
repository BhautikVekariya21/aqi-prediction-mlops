"""
Stage 7: Model Optimization
Hyperparameter tuning with Optuna + Model compression for Railway deployment
Exact logic from notebook: 07_model_optimization.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
import time
import joblib
import gzip
import lzma

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader
from ..utils.metrics import AQIMetrics


logger = get_logger(__name__)


class ModelOptimizer:
    """
    Optimize XGBoost model with Optuna and compress for deployment
    Matches notebook optimization logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize model optimizer
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get optimization parameters
        opt_config = config.get_section("model_optimization")
        
        self.models_dir = Path(opt_config.get("models_dir", "models"))
        self.output_dir = Path(opt_config.get("output_dir", "models/optimized"))
        self.splits_dir = Path(opt_config.get("splits_dir", "data/splits"))
        
        self.target_size_mb = opt_config.get("target_size_mb", 25.0)
        self.compression_method = opt_config.get("compression_method", "gzip")
        self.compression_level = opt_config.get("compression_level", 9)
        
        self.enable_precision_reduction = opt_config.get("enable_precision_reduction", True)
        self.precision = opt_config.get("precision", 3)
        self.enable_retraining = opt_config.get("enable_retraining", True)
        
        self.output_formats = opt_config.get("output_formats", ["pkl", "gzip"])
        
        # Optuna settings
        self.optuna_config = opt_config.get("optuna", {})
        self.n_trials = self.optuna_config.get("n_trials", 20)
        self.timeout = self.optuna_config.get("timeout", 3600)
        self.study_name = self.optuna_config.get("study_name", "xgboost_optimization")
        
        # XGBoost optimized params (fallback if Optuna fails)
        self.xgboost_optimized = opt_config.get("xgboost_optimized", {})
        self.xgboost_training = opt_config.get("xgboost_training", {})
        
        self.max_performance_degradation = opt_config.get("max_performance_degradation", 0.02)
        
        # Random state
        self.random_state = config.get("project.random_state", 42)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model Optimizer initialized")
        logger.info(f"Target size: {self.target_size_mb} MB")
        logger.info(f"Compression: {self.compression_method}")
        logger.info(f"Optuna trials: {self.n_trials}")
    
    def run(self) -> str:
        """
        Run model optimization pipeline
        
        Returns:
            Path to optimized model file
        """
        logger.info("="*90)
        logger.info("STARTING MODEL OPTIMIZATION")
        logger.info("="*90)
        
        # Step 1: Load data
        logger.info(f"\n1. Loading train/val/test data")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = self._load_data()
        
        # Step 2: Load baseline model performance
        logger.info(f"\n2. Loading baseline model performance")
        baseline_metrics = self._load_baseline_metrics()
        
        logger.info(f"   Baseline R²: {baseline_metrics['r2_score']:.4f}")
        logger.info(f"   Baseline RMSE: {baseline_metrics['rmse']:.2f}")
        
        # Step 3: Hyperparameter tuning with Optuna
        logger.info(f"\n3. Hyperparameter tuning with Optuna")
        best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Step 4: Train optimized model
        logger.info(f"\n4. Training optimized model")
        optimized_model = self._train_optimized_model(
            best_params, X_train, y_train, X_val, y_val, feature_names
        )
        
        # Step 5: Evaluate optimized model
        logger.info(f"\n5. Evaluating optimized model")
        optimized_metrics = self._evaluate_model(optimized_model, X_test, y_test)
        
        # Step 6: Verify performance
        logger.info(f"\n6. Verifying performance")
        self._verify_performance(baseline_metrics, optimized_metrics)
        
        # Step 7: Compress model
        logger.info(f"\n7. Compressing model for deployment")
        model_files = self._compress_model(optimized_model, feature_names)
        
        # Step 8: Save metadata
        logger.info(f"\n8. Saving metadata")
        metadata = self._save_metadata(
            best_params, baseline_metrics, optimized_metrics, model_files
        )
        
        # Step 9: Generate metrics
        metrics = self._generate_metrics(
            baseline_metrics, optimized_metrics, best_params, model_files
        )
        
        # Save metrics
        metrics_file = self.output_dir / "optimization_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"   OK Metrics saved: {metrics_file}")
        
        # Print summary
        self._print_summary(baseline_metrics, optimized_metrics, model_files)
        
        logger.info(f"\nOK Model optimization complete!")
        
        # Return primary model file (PKL for Railway)
        return str(model_files['pkl'])
    
    def _load_data(self) -> Tuple:
        """Load train/val/test data"""
        train_df = pd.read_parquet(self.splits_dir / "train.parquet")
        val_df = pd.read_parquet(self.splits_dir / "validation.parquet")
        test_df = pd.read_parquet(self.splits_dir / "test.parquet")
        
        with open(self.splits_dir / "feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        X_train = train_df[feature_names].values.astype(np.float32)
        y_train = train_df['us_aqi'].values.astype(np.float32)
        
        X_val = val_df[feature_names].values.astype(np.float32)
        y_val = val_df['us_aqi'].values.astype(np.float32)
        
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df['us_aqi'].values.astype(np.float32)
        
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Val:   {X_val.shape}")
        logger.info(f"   Test:  {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline XGBoost metrics from training"""
        try:
            with open(self.models_dir / "training_metrics.json", 'r') as f:
                training_metrics = json.load(f)
            
            # Get XGBoost test metrics
            if training_metrics['best_model'] == 'xgboost':
                return training_metrics['best_model_metrics']['test']
            else:
                # If XGBoost is not best, find it in detailed results
                with open(self.models_dir / "detailed_results.json", 'r') as f:
                    detailed = json.load(f)
                
                if 'xgboost' in detailed:
                    return detailed['xgboost']['test_metrics']
        
        except Exception as e:
            logger.warning(f"Could not load baseline metrics: {e}")
        
        # Return dummy baseline if not found
        return {
            'r2_score': 0.95,
            'rmse': 10.0,
            'mae': 7.0
        }
    
    def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Optimize hyperparameters with Optuna (from notebook concept)
        """
        logger.info(f"   Starting Optuna study: {self.study_name}")
        logger.info(f"   Trials: {self.n_trials}, Timeout: {self.timeout}s")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',  # Maximize R²
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Define objective function
        def objective(trial):
            # Sample hyperparameters (from notebook search space)
            search_space = self.optuna_config.get('search_space', {})
            
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': self.random_state,
                'n_jobs': 4,
                
                # Tunable parameters
                'max_depth': trial.suggest_int('max_depth', *search_space.get('max_depth', [6, 15])),
                'learning_rate': trial.suggest_float('learning_rate', *search_space.get('learning_rate', [0.01, 0.2]), log=True),
                'subsample': trial.suggest_float('subsample', *search_space.get('subsample', [0.7, 0.95])),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space.get('colsample_bytree', [0.7, 0.95])),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', *search_space.get('colsample_bylevel', [0.7, 0.95])),
                'reg_alpha': trial.suggest_float('reg_alpha', *search_space.get('reg_alpha', [0.00001, 0.1]), log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', *search_space.get('reg_lambda', [0.00001, 0.1]), log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', *search_space.get('min_child_weight', [1, 10])),
                'gamma': trial.suggest_float('gamma', *search_space.get('gamma', [0, 1.0])),
                'max_bin': trial.suggest_categorical('max_bin', search_space.get('max_bin', [128, 256]))
            }
            
            # Train model
            model = xgb.XGBRegressor(**params, n_estimators=300, verbosity=0)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate on validation
            y_pred = model.predict(X_val)
            metrics = AQIMetrics.calculate_all_metrics(y_val, y_pred)
            
            return metrics['r2_score']
        
        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,  # Sequential to avoid memory issues
                show_progress_bar=False
            )
            
            logger.info(f"   OK Optuna completed")
            logger.info(f"   Best R²: {study.best_value:.4f}")
            logger.info(f"   Best params: {study.best_params}")
            
            # Combine with fixed params
            best_params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'random_state': self.random_state,
                **study.best_params
            }
            
        except Exception as e:
            logger.warning(f"   Optuna failed: {e}")
            logger.warning(f"   Using fallback parameters")
            best_params = self.xgboost_optimized.copy()
        
        return best_params
    
    def _train_optimized_model(
        self,
        params: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list
    ) -> xgb.XGBRegressor:
        """Train final optimized model"""
        training_params = self.xgboost_training
        
        model = xgb.XGBRegressor(
            **params,
            n_estimators=training_params.get('num_boost_round', 600),
            verbosity=0,
            n_jobs=4
        )
        
        logger.info(f"   Training with {training_params.get('num_boost_round', 600)} rounds...")
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info(f"   OK Training complete")
        
        return model
    
    def _evaluate_model(self, model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate optimized model"""
        y_pred = model.predict(X_test)
        metrics = AQIMetrics.calculate_all_metrics(y_test, y_pred)
        
        logger.info(f"   Optimized R²: {metrics['r2_score']:.4f}")
        logger.info(f"   Optimized RMSE: {metrics['rmse']:.2f}")
        logger.info(f"   Optimized MAE: {metrics['mae']:.2f}")
        
        return metrics
    
    def _verify_performance(self, baseline_metrics: Dict, optimized_metrics: Dict):
        """Verify performance didn't degrade"""
        r2_degradation = baseline_metrics['r2_score'] - optimized_metrics['r2_score']
        
        if r2_degradation > self.max_performance_degradation:
            logger.warning(f"   Warning  Performance degraded by {r2_degradation:.4f}")
            logger.warning(f"   Max allowed: {self.max_performance_degradation}")
        else:
            logger.info(f"   OK Performance verified (degradation: {r2_degradation:.4f})")
    
    def _compress_model(self, model: xgb.XGBRegressor, feature_names: list) -> Dict[str, Path]:
        """
        Compress model for Railway deployment (from notebook)
        """
        model_files = {}
        
        # 1. Save as PKL (full functionality, for Railway)
        if 'pkl' in self.output_formats:
            pkl_file = self.output_dir / "model_final.pkl"
            joblib.dump(model, pkl_file, compress=('gzip', self.compression_level))
            pkl_size = pkl_file.stat().st_size / (1024 * 1024)
            
            logger.info(f"   OK PKL (gzip): {pkl_size:.2f} MB")
            model_files['pkl'] = pkl_file
        
        # 2. Save as JSON.GZ (ultra-compressed, backup)
        if 'gzip' in self.output_formats:
            json_file = self.output_dir / "model.json"
            model.save_model(str(json_file))
            
            # Compress with gzip
            gzip_file = self.output_dir / "model.json.gz"
            with open(json_file, 'rb') as f_in:
                with gzip.open(gzip_file, 'wb', compresslevel=self.compression_level) as f_out:
                    f_out.write(f_in.read())
            
            # Remove uncompressed JSON
            json_file.unlink()
            
            gzip_size = gzip_file.stat().st_size / (1024 * 1024)
            logger.info(f"   OK JSON.GZ: {gzip_size:.2f} MB")
            model_files['gzip'] = gzip_file
        
        # 3. Save features
        features_file = self.output_dir / "features.txt"
        with open(features_file, 'w') as f:
            for feat in feature_names:
                f.write(f"{feat}\n")
        
        logger.info(f"   OK Features saved")
        model_files['features'] = features_file
        
        return model_files
    
    def _save_metadata(
        self,
        best_params: Dict,
        baseline_metrics: Dict,
        optimized_metrics: Dict,
        model_files: Dict
    ) -> Dict:
        """Save model metadata"""
        metadata = {
            "model_type": "XGBoost",
            "optimization_method": "Optuna",
            "n_trials": self.n_trials,
            "best_params": best_params,
            "baseline_metrics": baseline_metrics,
            "optimized_metrics": optimized_metrics,
            "performance_degradation": baseline_metrics['r2_score'] - optimized_metrics['r2_score'],
            "compression_method": self.compression_method,
            "compression_level": self.compression_level,
            "target_size_mb": self.target_size_mb,
            "model_files": {k: str(v) for k, v in model_files.items()},
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_file = self.output_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   OK Metadata saved: {metadata_file}")
        
        return metadata
    
    def _generate_metrics(
        self,
        baseline_metrics: Dict,
        optimized_metrics: Dict,
        best_params: Dict,
        model_files: Dict
    ) -> Dict:
        """Generate optimization metrics"""
        metrics = {
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "improvement": {
                "r2_score": optimized_metrics['r2_score'] - baseline_metrics['r2_score'],
                "rmse": baseline_metrics['rmse'] - optimized_metrics['rmse'],
                "mae": baseline_metrics['mae'] - optimized_metrics['mae']
            },
            "best_params": best_params,
            "model_sizes_mb": {
                k: v.stat().st_size / (1024 * 1024) 
                for k, v in model_files.items() 
                if k != 'features'
            }
        }
        
        return metrics
    
    def _print_summary(self, baseline_metrics: Dict, optimized_metrics: Dict, model_files: Dict):
        """Print optimization summary"""
        print("\n" + "="*90)
        print("MODEL OPTIMIZATION SUMMARY")
        print("="*90)
        
        print("\nPerformance Comparison:")
        print(f"{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Improvement':>12}")
        print("-" * 60)
        print(f"{'R² Score':<20} {baseline_metrics['r2_score']:>12.4f} {optimized_metrics['r2_score']:>12.4f} {optimized_metrics['r2_score']-baseline_metrics['r2_score']:>+12.4f}")
        print(f"{'RMSE':<20} {baseline_metrics['rmse']:>12.2f} {optimized_metrics['rmse']:>12.2f} {baseline_metrics['rmse']-optimized_metrics['rmse']:>+12.2f}")
        print(f"{'MAE':<20} {baseline_metrics['mae']:>12.2f} {optimized_metrics['mae']:>12.2f} {baseline_metrics['mae']-optimized_metrics['mae']:>+12.2f}")
        
        print("\nModel Files:")
        for name, file_path in model_files.items():
            if name != 'features':
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name:<30} {size_mb:>8.2f} MB")
        
        print("="*90)