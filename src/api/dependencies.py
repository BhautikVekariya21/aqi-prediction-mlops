"""
Shared dependencies for FastAPI endpoints
"""

from pathlib import Path
from functools import lru_cache
import os

from ..inference.model_loader import ModelLoader
from ..utils.logger import get_logger
from ..utils.config_reader import load_cities_config


logger = get_logger(__name__)


# Singleton model loader
_model_loader = None


def get_model_loader() -> ModelLoader:
    """
    Get or create ModelLoader instance (singleton)
    """
    global _model_loader
    
    if _model_loader is None:
        model_path = os.getenv(
            "MODEL_PATH",
            "models/optimized/model_final.pkl"
        )
        features_path = os.getenv(
            "FEATURES_PATH",
            "models/optimized/features.txt"
        )
        
        logger.info(f"Loading model from: {model_path}")
        _model_loader = ModelLoader(model_path, features_path)
    
    return _model_loader


@lru_cache()
def get_cities_config() -> dict:
    """
    Get cities configuration (cached)
    """
    return load_cities_config("configs/cities.yaml")


def get_city_info(city_name: str) -> dict:
    """
    Get city information
    
    Args:
        city_name: City name
    
    Returns:
        Dictionary with lat, lon, state
    
    Raises:
        ValueError: If city not found
    """
    cities = get_cities_config()
    
    if city_name not in cities:
        raise ValueError(f"City '{city_name}' not found in configuration")
    
    return cities[city_name]