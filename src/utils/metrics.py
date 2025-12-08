"""
AQI prediction metrics (matching notebook logic exactly)
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class AQIMetrics:
    """
    Calculate AQI prediction metrics
    Matches notebook evaluation logic exactly
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all regression metrics for AQI prediction
        
        Args:
            y_true: True AQI values
            y_pred: Predicted AQI values
        
        Returns:
            Dictionary of metrics
        """
        
        # Standard regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Custom AQI accuracy metrics (from notebook)
        within_5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100
        within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        within_25 = np.mean(np.abs(y_true - y_pred) <= 25) * 100
        within_50 = np.mean(np.abs(y_true - y_pred) <= 50) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape),
            'within_5_pct': float(within_5),
            'within_10_pct': float(within_10),
            'within_25_pct': float(within_25),
            'within_50_pct': float(within_50)
        }
    
    @staticmethod
    def get_aqi_category(aqi_value: float) -> Tuple[str, str]:
        """
        Convert AQI value to category and emoji
        Matches notebook categorization exactly
        
        Args:
            aqi_value: AQI value
        
        Returns:
            Tuple of (category_name, emoji)
        """
        if aqi_value <= 50:
            return "Good", "🟢"
        elif aqi_value <= 100:
            return "Moderate", "🟡"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups", "🟠"
        elif aqi_value <= 200:
            return "Unhealthy", "🔴"
        elif aqi_value <= 300:
            return "Very Unhealthy", "🟣"
        else:
            return "Hazardous", "🟤"
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float], dataset_name: str = "Test") -> None:
        """
        Print formatted metrics summary
        
        Args:
            metrics: Metrics dictionary
            dataset_name: Name of dataset (Train/Val/Test)
        """
        print(f"\n{'='*70}")
        print(f"{dataset_name} Metrics Summary")
        print(f"{'='*70}")
        print(f"  RMSE:              {metrics['rmse']:>10.2f}")
        print(f"  MAE:               {metrics['mae']:>10.2f}")
        print(f"  R² Score:          {metrics['r2_score']:>10.4f}")
        print(f"  MAPE:              {metrics['mape']:>10.2f}%")
        print(f"  Within ±5 AQI:     {metrics['within_5_pct']:>10.1f}%")
        print(f"  Within ±10 AQI:    {metrics['within_10_pct']:>10.1f}%")
        print(f"  Within ±25 AQI:    {metrics['within_25_pct']:>10.1f}%")
        print(f"  Within ±50 AQI:    {metrics['within_50_pct']:>10.1f}%")
        print(f"{'='*70}")
    
    @staticmethod
    def compare_metrics(
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> None:
        """
        Print side-by-side comparison of train/val/test metrics
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
        """
        print(f"\n{'='*90}")
        print(f"{'Metric':<20} {'Train':>15} {'Validation':>15} {'Test':>15}")
        print(f"{'='*90}")
        
        metric_names = {
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2_score': 'R² Score',
            'mape': 'MAPE (%)',
            'within_10_pct': 'Within ±10 (%)',
            'within_25_pct': 'Within ±25 (%)',
            'within_50_pct': 'Within ±50 (%)'
        }
        
        for key, name in metric_names.items():
            train_val = train_metrics.get(key, 0)
            val_val = val_metrics.get(key, 0)
            test_val = test_metrics.get(key, 0)
            
            if key == 'r2_score':
                print(f"{name:<20} {train_val:>15.4f} {val_val:>15.4f} {test_val:>15.4f}")
            else:
                print(f"{name:<20} {train_val:>15.2f} {val_val:>15.2f} {test_val:>15.2f}")
        
        print(f"{'='*90}")