# src/components/model_trainer.py
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
import mlflow
import mlflow.xgboost

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ModelTrainingConfig:
    train_path: Path
    val_path: Path
    output_dir: Path
    target: str = "us_aqi"
    n_trials: int = 50
    random_state: int = 42

class XGBoostTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_params = None
        self.best_model = None

    def load_data(self):
        logger.info("Loading train/val splits...")
        train_df = pd.read_parquet(self.config.train_path)
        val_df = pd.read_parquet(self.config.val_path)

        X_train = train_df.drop(columns=[self.config.target])
        y_train = train_df[self.config.target]
        X_val = val_df.drop(columns=[self.config.target])
        y_val = val_df[self.config.target]

        return X_train, y_train, X_val, y_val

    def objective(self, trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
            "tree_method": "hist",
            "random_state": self.config.random_state,
            "n_jobs": 4,
            "verbosity": 0
        }

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        pruning_callback = XGBoostPruningCallback(trial, "rmse")

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )

        preds = bst.predict(dval)
        rmse = np.sqrt(((preds - y_val) ** 2).mean())
        return rmse

    def train_with_optuna(self, X_train, y_train, X_val, y_val):
        logger.info(f"Starting Optuna tuning with {self.config.n_trials} trials...")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.config.n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(f"Best RMSE: {study.best_value:.3f}")
        logger.info(f"Best params: {self.best_params}")

        return study

    def train_final_model(self, X_train, y_train, X_val, y_val):
        logger.info("Training final model with best parameters...")
        final_params = self.best_params.copy()
        final_params.update({
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": self.config.random_state,
            "verbosity": 0
        })

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.best_model = xgb.train(
            final_params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=100
        )

        # Save model
        model_path = self.config.output_dir / "xgboost.json"
        self.best_model.save_model(model_path)
        logger.info(f"Final model saved to {model_path}")

        # Log to MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/yourname/aqi-prediction-mlops.mlflow"))
        mlflow.set_experiment("aqi-prediction")

        with mlflow.start_run(run_name="xgboost_optuna_final"):
            mlflow.log_params(final_params)
            mlflow.log_metric("best_rmse", self.best_model.best_score)
            mlflow.log_metric("best_iteration", self.best_model.best_iteration)
            mlflow.xgboost.log_model(self.best_model, "model")
            mlflow.log_artifact(str(model_path), "model")

        return self.best_model

    def run(self):
        X_train, y_train, X_val, y_val = self.load_data()
        study = self.train_with_optuna(X_train, y_train, X_val, y_val)
        model = self.train_final_model(X_train, y_train, X_val, y_val)

        # Save best params
        params_path = self.config.output_dir / "best_params.json"
        import json
        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=2)

        logger.info("XGBoost training + Optuna + MLflow COMPLETE")
        return model, study