"""
Stage 8: Model Evaluation
Comprehensive evaluation of the optimized model on test set
Exact logic from notebook: 08_model_evaluation.ipynb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json
import joblib
import gzip

import xgboost as xgb

from ..utils.logger import get_logger
from ..utils.config_reader import ConfigReader
from ..utils.metrics import AQIMetrics


logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation on test set
    Matches notebook evaluation logic exactly
    """
    
    def __init__(self, config: ConfigReader):
        """
        Initialize model evaluator
        
        Args:
            config: ConfigReader instance with params.yaml
        """
        self.config = config
        
        # Get evaluation parameters
        eval_config = config.get_section("model_evaluation")
        
        self.model_path = Path(eval_config.get("model_path", "models/optimized/model_final.pkl"))
        self.model_gzip_path = Path(eval_config.get("model_gzip_path", "models/optimized/model.json.gz"))
        self.test_data_path = Path(eval_config.get("test_data_path", "data/splits/test.parquet"))
        self.output_dir = Path(eval_config.get("output_dir", "models/evaluation"))
        
        # Metrics to calculate
        self.metrics_list = eval_config.get("metrics", [
            "rmse", "mae", "r2_score", "mape",
            "within_5_pct", "within_10_pct", "within_25_pct", "within_50_pct"
        ])
        
        # Acceptance thresholds
        self.acceptance_thresholds = eval_config.get("acceptance_thresholds", {})
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model Evaluator initialized")
    
    def run(self) -> str:
        """
        Run model evaluation pipeline
        
        Returns:
            Path to evaluation report
        """
        logger.info("="*90)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("="*90)
        
        # Step 1: Load optimized model
        logger.info(f"\n1. Loading optimized model")
        model, feature_names = self._load_model()
        
        # Step 2: Load test data
        logger.info(f"\n2. Loading test data")
        X_test, y_test, test_df = self._load_test_data(feature_names)
        
        logger.info(f"   Test samples: {len(X_test):,}")
        logger.info(f"   Features: {len(feature_names)}")
        
        # Step 3: Make predictions
        logger.info(f"\n3. Making predictions on test set")
        y_pred = model.predict(X_test)
        
        logger.info(f"   ✓ Predictions generated")
        
        # Step 4: Calculate all metrics
        logger.info(f"\n4. Calculating metrics")
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Step 5: Analyze predictions by category
        logger.info(f"\n5. Analyzing predictions by AQI category")
        category_analysis = self._analyze_by_category(y_test, y_pred)
        
        # Step 6: Analyze predictions by city
        logger.info(f"\n6. Analyzing predictions by city")
        city_analysis = self._analyze_by_city(test_df, y_test, y_pred, feature_names)
        
        # Step 7: Error analysis
        logger.info(f"\n7. Performing error analysis")
        error_analysis = self._error_analysis(y_test, y_pred)
        
        # Step 8: Feature importance analysis
        logger.info(f"\n8. Analyzing feature importance")
        feature_importance = self._analyze_feature_importance(model, feature_names)
        
        # Step 9: Check acceptance criteria
        logger.info(f"\n9. Checking acceptance criteria")
        acceptance_status = self._check_acceptance_criteria(metrics)
        
        # Step 10: Generate comprehensive report
        logger.info(f"\n10. Generating evaluation report")
        evaluation_report = {
            "overall_metrics": metrics,
            "category_analysis": category_analysis,
            "city_analysis": city_analysis,
            "error_analysis": error_analysis,
            "feature_importance": feature_importance,
            "acceptance_status": acceptance_status,
            "test_samples": int(len(X_test)),
            "n_features": int(len(feature_names))
        }
        
        # Save evaluation report
        report_file = self.output_dir / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        logger.info(f"   ✓ Evaluation report saved: {report_file}")
        
        # Save final metrics (for DVC)
        final_metrics = {
            "test_rmse": float(metrics['rmse']),
            "test_mae": float(metrics['mae']),
            "test_r2_score": float(metrics['r2_score']),
            "test_mape": float(metrics['mape']),
            "test_within_10_pct": float(metrics['within_10_pct']),
            "test_within_25_pct": float(metrics['within_25_pct']),
            "test_within_50_pct": float(metrics['within_50_pct']),
            "acceptance_status": acceptance_status['overall']
        }
        
        metrics_file = self.output_dir / "final_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"   ✓ Final metrics saved: {metrics_file}")
        
        # Print summary
        self._print_summary(metrics, acceptance_status, category_analysis)
        
        logger.info(f"\n✓ Model evaluation complete!")
        
        return str(report_file)
    
    def _load_model(self) -> Tuple[xgb.XGBRegressor, list]:
        """Load optimized model and feature names"""
        # Try loading PKL first (preferred for Railway)
        if self.model_path.exists():
            logger.info(f"   Loading model from: {self.model_path}")
            model = joblib.load(self.model_path)
            logger.info(f"   ✓ Model loaded (PKL)")
        
        # Fallback to GZIP
        elif self.model_gzip_path.exists():
            logger.info(f"   Loading model from: {self.model_gzip_path}")
            
            # Decompress
            temp_json = self.model_gzip_path.parent / "temp_model.json"
            with gzip.open(self.model_gzip_path, 'rb') as f_in:
                with open(temp_json, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Load model
            model = xgb.XGBRegressor()
            model.load_model(str(temp_json))
            
            # Cleanup
            temp_json.unlink()
            
            logger.info(f"   ✓ Model loaded (GZIP)")
        
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path} or {self.model_gzip_path}")
        
        # Load feature names
        features_file = self.model_path.parent / "features.txt"
        with open(features_file, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"   ✓ Features loaded: {len(feature_names)}")
        
        return model, feature_names
    
    def _load_test_data(self, feature_names: list) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load test data"""
        test_df = pd.read_parquet(self.test_data_path)
        
        X_test = test_df[feature_names].values.astype(np.float32)
        y_test = test_df['us_aqi'].values.astype(np.float32)
        
        return X_test, y_test, test_df
    
    def _calculate_metrics(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all metrics (from notebook)"""
        metrics = AQIMetrics.calculate_all_metrics(y_test, y_pred)
        
        logger.info(f"   RMSE:             {metrics['rmse']:.2f}")
        logger.info(f"   MAE:              {metrics['mae']:.2f}")
        logger.info(f"   R² Score:         {metrics['r2_score']:.4f}")
        logger.info(f"   MAPE:             {metrics['mape']:.2f}%")
        logger.info(f"   Within ±10 AQI:   {metrics['within_10_pct']:.1f}%")
        logger.info(f"   Within ±25 AQI:   {metrics['within_25_pct']:.1f}%")
        
        return metrics
    
    def _analyze_by_category(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze predictions by AQI category (from notebook)
        """
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'error': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        
        # Add actual category
        analysis_df['actual_category'] = analysis_df['actual'].apply(
            lambda x: AQIMetrics.get_aqi_category(x)[0]
        )
        
        # Calculate metrics per category
        category_metrics = {}
        
        for category in ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']:
            
            cat_data = analysis_df[analysis_df['actual_category'] == category]
            
            if len(cat_data) > 0:
                cat_metrics = AQIMetrics.calculate_all_metrics(
                    cat_data['actual'].values,
                    cat_data['predicted'].values
                )
                
                category_metrics[category] = {
                    'count': int(len(cat_data)),
                    'percentage': float(len(cat_data) / len(analysis_df) * 100),
                    'rmse': cat_metrics['rmse'],
                    'mae': cat_metrics['mae'],
                    'r2_score': cat_metrics['r2_score']
                }
                
                logger.info(f"   {category:<35} Count: {len(cat_data):>6,} | RMSE: {cat_metrics['rmse']:>6.2f} | R²: {cat_metrics['r2_score']:>6.4f}")
        
        return category_metrics
    
    def _analyze_by_city(
        self,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        feature_names: list
    ) -> Dict:
        """
        Analyze predictions by city (from notebook)
        """
        # Add predictions to test_df
        test_df_copy = test_df.copy()
        test_df_copy['predicted_aqi'] = y_pred
        test_df_copy['error'] = y_test - y_pred
        test_df_copy['abs_error'] = np.abs(y_test - y_pred)
        
        # Calculate metrics per city
        city_metrics = {}
        
        if 'city' in test_df_copy.columns:
            for city in test_df_copy['city'].unique():
                city_data = test_df_copy[test_df_copy['city'] == city]
                
                if len(city_data) > 0:
                    city_y_test = city_data['us_aqi'].values
                    city_y_pred = city_data['predicted_aqi'].values
                    
                    city_met = AQIMetrics.calculate_all_metrics(city_y_test, city_y_pred)
                    
                    city_metrics[city] = {
                        'count': int(len(city_data)),
                        'rmse': city_met['rmse'],
                        'mae': city_met['mae'],
                        'r2_score': city_met['r2_score'],
                        'avg_actual_aqi': float(city_y_test.mean()),
                        'avg_predicted_aqi': float(city_y_pred.mean())
                    }
            
            # Log top 5 cities by sample count
            sorted_cities = sorted(city_metrics.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
            logger.info(f"   Top 5 cities by sample count:")
            for city, metrics in sorted_cities:
                logger.info(f"     {city:<20} Samples: {metrics['count']:>6,} | RMSE: {metrics['rmse']:>6.2f} | R²: {metrics['r2_score']:>6.4f}")
        
        return city_metrics
    
    def _error_analysis(self, y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze prediction errors (from notebook)
        """
        errors = y_test - y_pred
        abs_errors = np.abs(errors)
        
        error_stats = {
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'median_error': float(np.median(errors)),
            'mean_abs_error': float(abs_errors.mean()),
            'median_abs_error': float(np.median(abs_errors)),
            'max_overestimate': float(errors.max()),
            'max_underestimate': float(errors.min()),
            'percentile_25': float(np.percentile(errors, 25)),
            'percentile_75': float(np.percentile(errors, 75)),
            'percentile_95': float(np.percentile(abs_errors, 95)),
            'percentile_99': float(np.percentile(abs_errors, 99))
        }
        
        logger.info(f"   Mean error:           {error_stats['mean_error']:>8.2f}")
        logger.info(f"   Std error:            {error_stats['std_error']:>8.2f}")
        logger.info(f"   Max overestimate:     {error_stats['max_overestimate']:>8.2f}")
        logger.info(f"   Max underestimate:    {error_stats['max_underestimate']:>8.2f}")
        logger.info(f"   95th percentile |e|:  {error_stats['percentile_95']:>8.2f}")
        
        return error_stats
    
    def _analyze_feature_importance(self, model: xgb.XGBRegressor, feature_names: list) -> Dict:
        """
        Analyze feature importance (from notebook)
        """
        # Get feature importance
        importance = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Convert to dict (top 20)
        top_20 = importance_df.head(20).to_dict('records')
        
        logger.info(f"   Top 10 features:")
        for idx, row in enumerate(importance_df.head(10).itertuples(), 1):
            logger.info(f"     {idx:>2}. {row.feature:<30} {row.importance:>8.4f}")
        
        return {
            'top_20_features': top_20,
            'total_features': int(len(feature_names))
        }
    
    def _check_acceptance_criteria(self, metrics: Dict) -> Dict:
        """
        Check if model meets acceptance criteria (from notebook)
        """
        acceptance_status = {}
        
        for criterion, threshold in self.acceptance_thresholds.items():
            if criterion == 'r2_score_min':
                passed = metrics['r2_score'] >= threshold
                acceptance_status['r2_score'] = {
                    'passed': passed,
                    'threshold': threshold,
                    'actual': metrics['r2_score']
                }
            
            elif criterion == 'rmse_max':
                passed = metrics['rmse'] <= threshold
                acceptance_status['rmse'] = {
                    'passed': passed,
                    'threshold': threshold,
                    'actual': metrics['rmse']
                }
            
            elif criterion == 'mae_max':
                passed = metrics['mae'] <= threshold
                acceptance_status['mae'] = {
                    'passed': passed,
                    'threshold': threshold,
                    'actual': metrics['mae']
                }
            
            elif criterion == 'within_25_pct_min':
                passed = metrics['within_25_pct'] >= threshold
                acceptance_status['within_25_pct'] = {
                    'passed': passed,
                    'threshold': threshold,
                    'actual': metrics['within_25_pct']
                }
        
        # Overall acceptance
        all_passed = all(status['passed'] for status in acceptance_status.values())
        acceptance_status['overall'] = all_passed
        
        logger.info(f"\n   Acceptance Criteria:")
        for criterion, status in acceptance_status.items():
            if criterion != 'overall':
                status_symbol = "✓" if status['passed'] else "✗"
                logger.info(f"     {status_symbol} {criterion:<20} Threshold: {status['threshold']:>8.2f} | Actual: {status['actual']:>8.2f}")
        
        logger.info(f"\n   Overall Status: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
        
        return acceptance_status
    
    def _print_summary(self, metrics: Dict, acceptance_status: Dict, category_analysis: Dict):
        """Print evaluation summary (from notebook)"""
        print("\n" + "="*90)
        print("MODEL EVALUATION SUMMARY")
        print("="*90)
        
        print("\nOverall Test Metrics:")
        print(f"  RMSE:                {metrics['rmse']:>10.2f}")
        print(f"  MAE:                 {metrics['mae']:>10.2f}")
        print(f"  R² Score:            {metrics['r2_score']:>10.4f}")
        print(f"  MAPE:                {metrics['mape']:>10.2f}%")
        print(f"  Within ±5 AQI:       {metrics['within_5_pct']:>10.1f}%")
        print(f"  Within ±10 AQI:      {metrics['within_10_pct']:>10.1f}%")
        print(f"  Within ±25 AQI:      {metrics['within_25_pct']:>10.1f}%")
        print(f"  Within ±50 AQI:      {metrics['within_50_pct']:>10.1f}%")
        
        print("\nPerformance by AQI Category:")
        print(f"{'Category':<35} {'Count':>8} {'%':>7} {'RMSE':>8} {'R²':>8}")
        print("-" * 75)
        
        for category, cat_metrics in category_analysis.items():
            print(f"{category:<35} {cat_metrics['count']:>8,} {cat_metrics['percentage']:>6.1f}% "
                  f"{cat_metrics['rmse']:>8.2f} {cat_metrics['r2_score']:>8.4f}")
        
        print("\nAcceptance Status:")
        for criterion, status in acceptance_status.items():
            if criterion != 'overall':
                status_symbol = "✓" if status['passed'] else "✗"
                print(f"  {status_symbol} {criterion:<20} Required: {status['threshold']:>8.2f} | Actual: {status['actual']:>8.2f}")
        
        print("\n" + "="*90)
        print(f"OVERALL STATUS: {'PASSED ✓' if acceptance_status['overall'] else 'FAILED ✗'}")
        print("="*90)