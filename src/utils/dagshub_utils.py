"""
DagsHub utilities for DVC and MLflow integration
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
from .logger import get_logger
from .config_reader import ConfigReader


logger = get_logger(__name__)


class DagsHubManager:
    """
    Manage DagsHub integration for DVC and MLflow
    """
    
    def __init__(self, params_config: Optional[ConfigReader] = None):
        """
        Initialize DagsHub manager
        
        Args:
            params_config: ConfigReader instance with params.yaml
        """
        if params_config is None:
            params_config = ConfigReader("configs/params.yaml")
        
        self.config = params_config
        self._setup_credentials()
        self._setup_mlflow()
    
    def _setup_credentials(self):
        """Setup DagsHub credentials from environment"""
        self.repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        self.repo_name = os.getenv("DAGSHUB_REPO_NAME")
        self.token = os.getenv("DAGSHUB_TOKEN")
        
        if not all([self.repo_owner, self.repo_name, self.token]):
            logger.warning("DagsHub credentials not fully configured. Some features may not work.")
            logger.warning("Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN in .env")
    
    def _setup_mlflow(self):
        """Configure MLflow to use DagsHub"""
        if not all([self.repo_owner, self.repo_name, self.token]):
            logger.warning("Skipping MLflow setup - missing credentials")
            return
        
        # Set MLflow tracking URI
        tracking_uri = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set credentials
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.token
        
        logger.info(f"MLflow tracking URI set: {tracking_uri}")
    
    def start_mlflow_run(
        self,
        run_name: str,
        experiment_name: str = "aqi-prediction",
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start MLflow run
        
        Args:
            run_name: Name for this run
            experiment_name: MLflow experiment name
            tags: Additional tags
        
        Returns:
            Active MLflow run
        """
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=tags)
        
        logger.info(f"Started MLflow run: {run_name}")
        logger.info(f"Run ID: {run.info.run_id}")
        
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params: Parameters dictionary
        """
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact to MLflow
        
        Args:
            local_path: Local file path
            artifact_path: Path within artifact store
        """
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
    
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow
        
        Args:
            model: Model object
            artifact_path: Path in artifact store
            registered_model_name: Name for model registry
        """
        try:
            mlflow.xgboost.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
            logger.info(f"Logged model to MLflow: {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
    
    def end_run(self):
        """End current MLflow run"""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")
    
    @staticmethod
    def setup_dvc_remote():
        """
        Setup DVC remote for DagsHub
        Run this once during project setup
        """
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        
        if not all([repo_owner, repo_name]):
            logger.error("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME")
            return False
        
        dvc_remote_url = f"https://dagshub.com/{repo_owner}/{repo_name}.dvc"
        
        # These commands should be run manually or in setup script
        commands = [
            f"dvc remote add -d dagshub {dvc_remote_url}",
            f"dvc remote modify dagshub --local auth basic",
            f"dvc remote modify dagshub --local user {repo_owner}",
            f"dvc remote modify dagshub --local password $DAGSHUB_TOKEN"
        ]
        
        logger.info("Run these commands to setup DVC remote:")
        for cmd in commands:
            logger.info(f"  {cmd}")
        
        return True


def get_dagshub_manager() -> DagsHubManager:
    """
    Get singleton DagsHub manager instance
    
    Returns:
        DagsHubManager instance
    """
    return DagsHubManager()