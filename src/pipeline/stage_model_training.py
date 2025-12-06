# src/pipeline/stage_model_training.py
import yaml
from pathlib import Path
from src.components.model_trainer import XGBoostTrainer, ModelTrainingConfig

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    config = ModelTrainingConfig(
        train_path=Path("data/splits/train.parquet"),
        val_path=Path("data/splits/val.parquet"),
        output_dir=Path("models"),
        n_trials=50
    )

    trainer = XGBoostTrainer(config)
    model, study = trainer.run()

    print("XGBoost TRAINING COMPLETE")
    print(f"   Model: models/xgboost.json")
    print(f"   Best RMSE: {study.best_value:.3f}")
    print(f"   MLflow run logged")

if __name__ == "__main__":
    main()