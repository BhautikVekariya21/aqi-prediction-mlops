"""
DVC Stage 5: Data Splitting
Stratified random split (Train/Val/Test)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_splitting import DataSplitting
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run data splitting stage"""
    
    with LoggerContext("Stage 5: Data Splitting") as stage_logger:
        try:
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_05_data_splitting",
                tags={"stage": "05_data_splitting"}
            )
            
            # Log parameters
            split_params = config.get_section("data_splitting")
            dagshub_manager.log_params({
                "test_size": split_params.get("test_size"),
                "validation_size": split_params.get("validation_size"),
                "stratify": split_params.get("stratify"),
                "random_state": config.get("project.random_state")
            })
            
            # Run data splitting
            stage_logger.info("Starting data splitting component")
            data_splitting = DataSplitting(config)
            train_path, val_path, test_path = data_splitting.run()
            
            # Log output artifacts
            dagshub_manager.log_artifact(train_path)
            dagshub_manager.log_artifact(val_path)
            dagshub_manager.log_artifact(test_path)
            
            # Log metrics
            import json
            metrics_file = Path(train_path).parent / "split_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                dagshub_manager.log_metrics({
                    "total_samples": metrics.get("total_samples", 0),
                    "train_samples": metrics.get("train_samples", 0),
                    "val_samples": metrics.get("val_samples", 0),
                    "test_samples": metrics.get("test_samples", 0),
                    "train_pct": metrics.get("train_pct", 0),
                    "val_pct": metrics.get("val_pct", 0),
                    "test_pct": metrics.get("test_pct", 0)
                })
                
                dagshub_manager.log_artifact(str(metrics_file))
                
                # Log feature names
                features_file = Path(train_path).parent / "feature_names.txt"
                if features_file.exists():
                    dagshub_manager.log_artifact(str(features_file))
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Data splitting completed successfully")
            stage_logger.info(f"Train: {train_path}")
            stage_logger.info(f"Val: {val_path}")
            stage_logger.info(f"Test: {test_path}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                dagshub_manager.end_run()
            except:
                pass
            
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)