"""
DVC Stage 2: Data Preprocessing
Clean, impute, and handle outliers
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_preprocessing import DataPreprocessing
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run data preprocessing stage"""
    
    with LoggerContext("Stage 2: Data Preprocessing") as stage_logger:
        try:
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_02_data_preprocessing",
                tags={"stage": "02_data_preprocessing"}
            )
            
            # Log parameters
            preprocess_params = config.get_section("data_preprocessing")
            dagshub_manager.log_params({
                "imputation_method": preprocess_params.get("imputation", {}).get("method"),
                "knn_neighbors": preprocess_params.get("imputation", {}).get("n_neighbors"),
                "outlier_method": preprocess_params.get("outliers", {}).get("method"),
                "lower_percentile": preprocess_params.get("outliers", {}).get("lower_percentile"),
                "upper_percentile": preprocess_params.get("outliers", {}).get("upper_percentile")
            })
            
            # Run preprocessing
            stage_logger.info("Starting data preprocessing component")
            preprocessing = DataPreprocessing(config)
            output_file = preprocessing.run()
            
            # Log output artifact
            dagshub_manager.log_artifact(output_file)
            
            # Log metrics
            import json
            metrics_file = Path(output_file).parent / "preprocessing_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                dagshub_manager.log_metrics({
                    "final_rows": metrics.get("final_rows", 0),
                    "final_columns": metrics.get("final_columns", 0),
                    "rows_removed": metrics.get("rows_removed", 0),
                    "columns_removed": metrics.get("columns_removed", 0),
                    "missing_values": metrics.get("missing_values", 0)
                })
                
                dagshub_manager.log_artifact(str(metrics_file))
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Data preprocessing completed successfully")
            stage_logger.info(f"Output: {output_file}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
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