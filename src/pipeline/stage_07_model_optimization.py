"""
DVC Stage 7: Model Optimization
Hyperparameter tuning with Optuna + Model compression for Railway
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.model_optimizer import ModelOptimizer
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager


logger = get_logger(__name__)


def main():
    """Run model optimization stage"""
    
    with LoggerContext("Stage 7: Model Optimization") as stage_logger:
        try:
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_07_model_optimization",
                tags={"stage": "07_model_optimization"}
            )
            
            # Log parameters
            opt_params = config.get_section("model_optimization")
            dagshub_manager.log_params({
                "target_size_mb": opt_params.get("target_size_mb"),
                "compression_method": opt_params.get("compression_method"),
                "compression_level": opt_params.get("compression_level"),
                "optuna_n_trials": opt_params.get("optuna", {}).get("n_trials"),
                "optuna_timeout": opt_params.get("optuna", {}).get("timeout"),
                "random_state": config.get("project.random_state")
            })
            
            # Run model optimization
            stage_logger.info("Starting model optimization component")
            model_optimizer = ModelOptimizer(config)
            optimized_model_path = model_optimizer.run()
            
            # Log optimized model artifacts
            optimized_dir = Path(optimized_model_path).parent
            
            # Log all optimized model files
            for model_file in optimized_dir.glob("*"):
                if model_file.is_file():
                    dagshub_manager.log_artifact(str(model_file))
            
            # Log metrics
            import json
            metrics_file = optimized_dir / "optimization_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Log baseline vs optimized metrics
                baseline = metrics.get("baseline", {})
                optimized = metrics.get("optimized", {})
                improvement = metrics.get("improvement", {})
                
                for metric_name in ['rmse', 'mae', 'r2_score']:
                    if metric_name in baseline:
                        dagshub_manager.log_metrics({
                            f"baseline_{metric_name}": baseline[metric_name],
                            f"optimized_{metric_name}": optimized.get(metric_name, 0),
                            f"improvement_{metric_name}": improvement.get(metric_name, 0)
                        })
                
                # Log model sizes
                model_sizes = metrics.get("model_sizes_mb", {})
                for format_name, size_mb in model_sizes.items():
                    dagshub_manager.log_metrics({
                        f"model_size_{format_name}_mb": size_mb
                    })
                
                dagshub_manager.log_artifact(str(metrics_file))
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Model optimization completed successfully")
            stage_logger.info(f"Optimized model: {optimized_model_path}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
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