"""
Configuration reader for YAML files with environment variable support
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
import re


class ConfigReader:
    """
    Read and parse YAML configuration files
    Supports environment variable substitution: ${VAR_NAME}
    """
    
    def __init__(self, config_path: str):
        """
        Initialize config reader
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML config with environment variable substitution"""
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Replace ${VAR_NAME} with environment variable values
        content = self._substitute_env_vars(content)
        
        config = yaml.safe_load(content)
        return config
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Replace ${VAR_NAME} with environment variable values
        
        Args:
            content: YAML content as string
        
        Returns:
            Content with substituted environment variables
        """
        pattern = re.compile(r'\$\{([^}]+)\}')
        
        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")  # Keep original if not found
        
        return pattern.sub(replacer, content)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., "data_ingestion.start_date")
            default: Default value if key not found
        
        Returns:
            Config value
        
        Example:
            config.get("model_params.xgboost.learning_rate")
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire config section
        
        Args:
            section: Section name (e.g., "data_ingestion")
        
        Returns:
            Section dictionary
        """
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full config dictionary"""
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access: config['key']"""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator"""
        return key in self._config


def load_cities_config(config_path: str = "configs/cities.yaml") -> Dict[str, Dict[str, Any]]:
    """
    Load cities configuration
    
    Args:
        config_path: Path to cities.yaml
    
    Returns:
        Dictionary of cities with lat, lon, state
    """
    config = ConfigReader(config_path)
    return config.get("cities", {})


def load_params_config(config_path: str = "configs/params.yaml") -> ConfigReader:
    """
    Load pipeline parameters
    
    Args:
        config_path: Path to params.yaml
    
    Returns:
        ConfigReader instance
    """
    return ConfigReader(config_path)