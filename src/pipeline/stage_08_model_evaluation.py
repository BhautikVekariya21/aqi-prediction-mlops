"""
DVC Stage 8: Model Evaluation
Comprehensive evaluation of the optimized model on test set
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.model_evaluator import ModelEvaluator
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run model evaluation stage"""
    
    with LoggerContext("Stage 8: Model Evaluation") as stage_logger:
        try:
            # Load config
            config = ConfigReader("params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_08_model_evaluation",
                tags={"stage": "08_model_evaluation"}
            )
            
            # Log parameters
            eval_params = config.get_section("model_evaluation")
            dagshub_manager.log_params({
                "metrics": str(eval_params.get("metrics")),
                "acceptance_thresholds": str(eval_params.get("acceptance_thresholds"))
            })
            
            # Run model evaluation
            stage_logger.info("Starting model evaluation component")
            model_evaluator = ModelEvaluator(config)
            evaluation_report_path = model_evaluator.run()
            
            # Log evaluation artifacts
            eval_dir = Path(evaluation_report_path).parent
            
            for eval_file in eval_dir.glob("*"):
                if eval_file.is_file():
                    dagshub_manager.log_artifact(str(eval_file))
            
            # Log final metrics
            import json
            metrics_file = eval_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Log all test metrics
                dagshub_manager.log_metrics(metrics)
            
            # Log evaluation report details
            with open(evaluation_report_path, 'r') as f:
                report = json.load(f)
            
            # Log category-wise metrics
            category_analysis = report.get("category_analysis", {})
            for category, cat_metrics in category_analysis.items():
                dagshub_manager.log_metrics({
                    f"category_{category}_rmse": cat_metrics.get("rmse", 0),
                    f"category_{category}_r2": cat_metrics.get("r2_score", 0)
                })
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Model evaluation completed successfully")
            stage_logger.info(f"Evaluation report: {evaluation_report_path}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
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