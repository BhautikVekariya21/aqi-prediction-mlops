"""
Model loader for compressed models
Handles both PKL and GZIP formats
"""

import joblib
import gzip
import tempfile
from pathlib import Path
from typing import Tuple
import xgboost as xgb

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ModelLoader:
    """
    Load and manage trained model
    Supports both PKL and GZIP compressed formats
    """
    
    def __init__(self, model_path: str, features_path: str):
        """
        Initialize model loader
        
        Args:
            model_path: Path to model file (.pkl or .json.gz)
            features_path: Path to features.txt
        """
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        
        # Load model
        self.model = self._load_model()
        
        # Load feature names
        self.feature_names = self._load_feature_names()
        
        logger.info(f"Model loaded from: {self.model_path}")
        logger.info(f"Features loaded: {len(self.feature_names)}")
    
    def _load_model(self):
        """Load model from file"""
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Try loading as PKL first (preferred)
        if self.model_path.suffix == '.pkl':
            logger.info("Loading model from PKL...")
            return joblib.load(self.model_path)
        
        # Try loading as GZIP
        elif self.model_path.suffix == '.gz':
            logger.info("Loading model from GZIP...")
            
            # Decompress to temporary file
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name
                
                with gzip.open(self.model_path, 'rb') as f_in:
                    temp_file.write(f_in.read())
            
            # Load model
            model = xgb.XGBRegressor()
            model.load_model(temp_path)
            
            # Cleanup
            Path(temp_path).unlink()
            
            return model
        
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
    
    def _load_feature_names(self) -> list:
        """Load feature names from file"""
        
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        with open(self.features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        return features
    
    def predict(self, X) -> float:
        """
        Make prediction
        
        Args:
            X: Feature array or dict
        
        Returns:
            Predicted AQI value
        """
        import numpy as np
        
        # Convert dict to array if needed
        if isinstance(X, dict):
            X = np.array([[X[feat] for feat in self.feature_names]], dtype=np.float32)
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(X)
        
        return float(prediction[0])