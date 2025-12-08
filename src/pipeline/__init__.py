"""
DVC pipeline stages
Each stage wraps a component and handles MLflow logging
"""

__all__ = [
    "stage_01_data_ingestion",
    "stage_02_data_preprocessing",
    "stage_03_feature_engineering",
    "stage_04_feature_selection",
    "stage_05_data_splitting",
    "stage_06_model_training",
    "stage_07_model_optimization",
    "stage_08_model_evaluation",
    "stage_09_prediction_comparison",
]