"""
DVC Stage 4: Feature Selection
Multi-method feature selection (Correlation, MI, Decision Tree, LightGBM, XGBoost)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.feature_selection import FeatureSelection
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run feature selection stage"""
    
    with LoggerContext("Stage 4: Feature Selection") as stage_logger:
        try:
            # Load config
            config = ConfigReader("params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_04_feature_selection",
                tags={"stage": "04_feature_selection"}
            )
            
            # Log parameters
            fs_params = config.get_section("feature_selection")
            dagshub_manager.log_params({
                "target": fs_params.get("target"),
                "top_n_features": fs_params.get("top_n_features"),
                "methods": str(fs_params.get("methods"))
            })
            
            # Run feature selection
            stage_logger.info("Starting feature selection component")
            feature_selection = FeatureSelection(config)
            output_file = feature_selection.run()
            
            # Log output artifact
            dagshub_manager.log_artifact(output_file)
            
            # Log metrics
            import json
            metrics_file = Path(output_file).parent / "selection_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                dagshub_manager.log_metrics({
                    "total_features_before": metrics.get("total_features_before", 0),
                    "total_features_after": metrics.get("total_features_after", 0),
                    "features_removed": metrics.get("features_removed", 0)
                })
                
                dagshub_manager.log_artifact(str(metrics_file))
                
                # Log selected features as artifact
                features_file = Path(output_file).parent / "selected_features.txt"
                if features_file.exists():
                    dagshub_manager.log_artifact(str(features_file))
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Feature selection completed successfully")
            stage_logger.info(f"Output: {output_file}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
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