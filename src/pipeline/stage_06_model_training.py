"""
DVC Stage 6: Model Training
Train 5 regression models (Decision Tree, Random Forest, Extra Trees, XGBoost, CatBoost)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.components.model_trainer import ModelTrainer
from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader
from src.utils.dagshub_utils import DagsHubManager
from src.utils.memory_manager import MemoryManager


logger = get_logger(__name__)


def main():
    """Run model training stage"""
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    with LoggerContext("Stage 6: Model Training") as stage_logger:
        try:
            # Cleanup memory before starting
            stage_logger.info("=" * 80)
            memory_manager.cleanup_memory("Stage 6: Model Training")
            memory_manager.start_monitoring("Stage 6")
            stage_logger.info("=" * 80)
            
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Initialize DagsHub manager
            dagshub_manager = DagsHubManager(config)
            
            # Start MLflow run
            mlflow_run = dagshub_manager.start_mlflow_run(
                run_name="stage_06_model_training",
                tags={"stage": "06_model_training"}
            )
            
            # Log parameters
            train_params = config.get_section("model_training")
            dagshub_manager.log_params({
                "target": train_params.get("target"),
                "models": str(train_params.get("models")),
                "primary_metric": train_params.get("primary_metric"),
                "early_stopping_rounds": train_params.get("early_stopping_rounds"),
                "n_jobs": train_params.get("n_jobs"),
                "random_state": config.get("project.random_state")
            })
            
            # Log model hyperparameters
            model_params = config.get_section("model_params")
            for model_name, params in model_params.items():
                for param_name, param_value in params.items():
                    dagshub_manager.log_params({
                        f"{model_name}_{param_name}": param_value
                    })
            
            # Run model training
            stage_logger.info("Starting model training component")
            model_trainer = ModelTrainer(config)
            best_model_path = model_trainer.run()
            
            # Log model artifacts
            models_dir = Path(best_model_path).parent
            
            # Log all model files
            for model_file in models_dir.glob("*"):
                if model_file.is_file() and model_file.suffix in ['.json', '.pkl', '.cbm', '.txt', '.csv']:
                    dagshub_manager.log_artifact(str(model_file))
            
            # Log metrics
            import json
            metrics_file = models_dir / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Log best model metrics
                best_metrics = metrics.get("best_model_metrics", {})
                for split in ['train', 'validation', 'test']:
                    split_metrics = best_metrics.get(split, {})
                    for metric_name, metric_value in split_metrics.items():
                        dagshub_manager.log_metrics({
                            f"{split}_{metric_name}": metric_value
                        })
                
                dagshub_manager.log_artifact(str(metrics_file))
            
            # End memory monitoring
            stage_logger.info("=" * 80)
            mem_stats = memory_manager.end_monitoring("Stage 6")
            
            if mem_stats:
                dagshub_manager.log_metrics({
                    "memory_initial_mb": mem_stats['initial_mb'],
                    "memory_final_mb": mem_stats['final_mb'],
                    "memory_increase_mb": mem_stats['increase_mb']
                })
            stage_logger.info("=" * 80)
            
            # End MLflow run
            dagshub_manager.end_run()
            
            stage_logger.info(f"Model training completed successfully")
            stage_logger.info(f"Best model: {best_model_path}")
            
            # Final cleanup
            memory_manager.cleanup_memory("Stage 6: Post-execution")
            
            return 0
        
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                dagshub_manager.end_run()
            except:
                pass
            
            memory_manager.cleanup_memory("Stage 6: Error cleanup")
            
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)