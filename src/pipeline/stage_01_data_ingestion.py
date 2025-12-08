"""
DVC Stage 1: Data Ingestion
Download historical weather and AQ data from Open-Meteo API
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.data_ingestion import DataIngestion
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run data ingestion stage"""
    
    with LoggerContext("Stage 1: Data Ingestion") as stage_logger:
        try:
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_01_data_ingestion",
                tags={"stage": "01_data_ingestion"}
            )
            
            # Log parameters
            ingestion_params = config.get_section("data_ingestion")
            dagshub_manager.log_params({
                "start_date": ingestion_params.get("start_date"),
                "end_date": ingestion_params.get("end_date") or "current",
                "api_timeout": ingestion_params.get("api", {}).get("timeout"),
                "cities_per_batch": ingestion_params.get("output", {}).get("cities_per_batch")
            })
            
            # Run data ingestion
            stage_logger.info("Starting data ingestion component")
            ingestion = DataIngestion(config)
            output_file = ingestion.run()
            
            # Log output artifact
            dagshub_manager.log_artifact(output_file)
            
            # Log metrics
            import json
            metrics_file = Path(output_file).parent / "ingestion_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                dagshub_manager.log_metrics({
                    "total_records": metrics.get("total_records", 0),
                    "successful_cities": metrics.get("successful_cities", 0),
                    "pm25_coverage_pct": metrics.get("pm25_coverage_pct", 0)
                })
                
                dagshub_manager.log_artifact(str(metrics_file))
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Data ingestion completed successfully")
            stage_logger.info(f"Output: {output_file}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            
            # End MLflow run with failure
            try:
                dagshub_manager.end_run()
            except:
                pass
            
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)