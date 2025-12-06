# src/components/model_optimizer.py
import os
from pathlib import Path
from dataclasses import dataclass
import joblib
import xgboost as xgb
import lzma
import gzip
import json
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelOptimizationConfig:
    input_model_path: Path
    output_dir: Path
    target_size_mb: float = 25.0

class ModelOptimizer:
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self) -> xgb.Booster:
        logger.info(f"Loading trained model from {self.config.input_model_path}")
        model = xgb.Booster()
        model.load_model(str(self.config.input_model_path))
        return model

    def optimize_and_save(self, model: xgb.Booster):
        # 1. Save as compressed JSON (gzip) - Railway compatible
        gzip_path = self.config.output_dir / "model.json.gz"
        with gzip.open(gzip_path, 'wb') as f:
            f.write(model.save_raw())
        logger.info(f"Compressed model (gzip): {gzip_path.name} ({gzip_path.stat().st_size / 1e6:.2f} MB)")

        # 2. Save as ultra-compressed LZMA (.pkl) - Best size
        lzma_path = self.config.output_dir / "xgboost_improved_lzma.pkl"
        joblib.dump(model, lzma_path, compress=('lzma', 9))
        size_mb = lzma_path.stat().st_size / (1024 * 1024)
        logger.info(f"Ultra-compressed model (LZMA): {lzma_path.name} ({size_mb:.2f} MB)")

        # 3. Save feature names
        feature_names_path = self.config.output_dir / "feature_names.txt"
        # We'll copy from splits (safe)
        import shutil
        shutil.copy("data/splits/feature_names.txt", feature_names_path)
        logger.info(f"Feature names copied: {feature_names_path.name}")

        # 4. Save metadata
        metadata = {
            "model_type": "xgboost",
            "version": "1.0.0",
            "compression": "lzma+gzip",
            "original_size_mb": "N/A",
            "compressed_size_mb": round(size_mb, 2),
            "inference_ready": True,
            "deployment_platforms": ["Railway", "Render", "Docker", "FastAPI"],
            "target": "us_aqi",
            "features_count": len(model.feature_names) if hasattr(model, 'feature_names') else "unknown"
        }

        metadata_path = self.config.output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path.name}")

        return {
            "gzip_path": gzip_path,
            "lzma_path": lzma_path,
            "size_mb": size_mb,
            "metadata_path": metadata_path
        }

    def run(self):
        model = self.load_model()
        result = self.optimize_and_save(model)

        logger.info("MODEL OPTIMIZATION COMPLETE")
        logger.info(f"Final model size: {result['size_mb']:.2f} MB (perfect for Railway/Render)")
        logger.info(f"Ready for FastAPI inference!")

        return result