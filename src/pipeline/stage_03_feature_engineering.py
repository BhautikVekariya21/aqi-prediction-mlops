"""
DVC Stage 3: Feature Engineering
Create datetime features, derived features, and encodings
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager
from src.utils.memory_manager import MemoryManager


logger = get_logger(__name__)


def main():
    """Run feature engineering stage"""
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    with LoggerContext("Stage 3: Feature Engineering") as stage_logger:
        try:
            # Cleanup memory before starting
            stage_logger.info("=" * 80)
            memory_manager.cleanup_memory("Stage 3: Feature Engineering")
            memory_manager.start_monitoring("Stage 3")
            stage_logger.info("=" * 80)
            
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_03_feature_engineering",
                tags={"stage": "03_feature_engineering"}
            )
            
            # Log parameters
            fe_params = config.get_section("feature_engineering")
            dagshub_manager.log_params({
                "target": fe_params.get("target"),
                "encoding_method": fe_params.get("encoding_method"),
                "categorical_columns": str(fe_params.get("categorical_columns"))
            })
            
            # Run feature engineering
            stage_logger.info("Starting feature engineering component")
            feature_engineering = FeatureEngineering(config)
            output_file = feature_engineering.run()
            
            # Log output artifact
            dagshub_manager.log_artifact(output_file)
            
            # Log metrics
            import json
            metrics_file = Path(output_file).parent / "feature_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                dagshub_manager.log_metrics({
                    "initial_columns": metrics.get("initial_columns", 0),
                    "final_columns": metrics.get("final_columns", 0),
                    "new_features_created": metrics.get("new_features_created", 0),
                    "total_rows": metrics.get("total_rows", 0)
                })
                
                dagshub_manager.log_artifact(str(metrics_file))
            
            # End memory monitoring
            stage_logger.info("=" * 80)
            mem_stats = memory_manager.end_monitoring("Stage 3")
            
            if mem_stats:
                dagshub_manager.log_metrics({
                    "memory_initial_mb": mem_stats['initial_mb'],
                    "memory_final_mb": mem_stats['final_mb'],
                    "memory_increase_mb": mem_stats['increase_mb']
                })
            stage_logger.info("=" * 80)
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Feature engineering completed successfully")
            stage_logger.info(f"Output: {output_file}")
            
            # Final cleanup
            memory_manager.cleanup_memory("Stage 3: Post-execution")
            
            return 0
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                dagshub_manager.end_run()
            except:
                pass
            
            memory_manager.cleanup_memory("Stage 3: Error cleanup")
            
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)