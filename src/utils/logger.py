"""
Custom logging utility with file and console handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Get configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Enable file logging
        log_to_console: Enable console logging
    
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"{datetime.now().strftime('%Y%m%d')}_pipeline.log"
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerContext:
    """
    Context manager for scoped logging
    
    Usage:
        with LoggerContext("Stage 1: Data Ingestion"):
            # Your code here
            pass
    """
    
    def __init__(self, stage_name: str, logger_name: str = "pipeline"):
        self.stage_name = stage_name
        self.logger = get_logger(logger_name)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info("="*80)
        self.logger.info(f"STARTED: {self.stage_name}")
        self.logger.info("="*80)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(f"FAILED: {self.stage_name}")
            self.logger.error(f"Error: {exc_val}")
            self.logger.error("="*80)
            return False
        
        self.logger.info(f"COMPLETED: {self.stage_name}")
        self.logger.info(f"Duration: {elapsed:.2f} seconds")
        self.logger.info("="*80)
        return True