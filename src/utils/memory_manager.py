"""
Memory management utilities for pipeline stages
Clear memory before/after each stage to prevent OOM issues
"""

import gc
import sys
import psutil
import os
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger


logger = get_logger(__name__)


class MemoryManager:
    """Manage memory for pipeline stages"""
    
    def __init__(self):
        """Initialize memory manager"""
        self.process = psutil.Process(os.getpid())
        self.initial_memory_mb = 0
        self.peak_memory_mb = 0
    
    def get_memory_usage(self) -> dict:
        """
        Get current memory usage
        
        Returns:
            dict: Memory usage statistics
        """
        try:
            mem_info = self.process.memory_info()
            
            return {
                "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": self.process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "total_mb": psutil.virtual_memory().total / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {}
    
    def log_memory_usage(self, stage_name: str = ""):
        """
        Log current memory usage
        
        Args:
            stage_name: Name of the stage
        """
        mem_usage = self.get_memory_usage()
        
        if mem_usage:
            logger.info(f"{'[' + stage_name + '] ' if stage_name else ''}Memory Usage:")
            logger.info(f"  RSS: {mem_usage['rss_mb']:.2f} MB")
            logger.info(f"  VMS: {mem_usage['vms_mb']:.2f} MB")
            logger.info(f"  Process: {mem_usage['percent']:.2f}%")
            logger.info(f"  Available: {mem_usage['available_mb']:.2f} MB / {mem_usage['total_mb']:.2f} MB")
    
    def cleanup_memory(self, stage_name: str = "") -> dict:
        """
        Cleanup memory before stage execution
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            dict: Memory statistics before and after cleanup
        """
        logger.info(f"{'[' + stage_name + '] ' if stage_name else ''}Starting memory cleanup...")
        
        # Get memory before cleanup
        mem_before = self.get_memory_usage()
        
        # Clear matplotlib figures if imported
        try:
            if 'matplotlib.pyplot' in sys.modules:
                import matplotlib.pyplot as plt
                plt.close('all')
                logger.debug("Cleared matplotlib figures")
        except Exception as e:
            logger.debug(f"Could not clear matplotlib: {e}")
        
        # Clear any pandas dataframes in globals (if running in notebook/interactive)
        try:
            if 'pandas' in sys.modules:
                import pandas as pd
                # Force garbage collection of pandas objects
                for obj in gc.get_objects():
                    if isinstance(obj, pd.DataFrame):
                        del obj
                logger.debug("Cleared pandas DataFrames from memory")
        except Exception as e:
            logger.debug(f"Could not clear pandas objects: {e}")
        
        # Clear Python garbage collector
        collected = gc.collect()
        logger.debug(f"Garbage collector: {collected} objects collected")
        
        # Force another round of garbage collection
        gc.collect()
        gc.collect()
        
        # Get memory after cleanup
        mem_after = self.get_memory_usage()
        
        # Calculate freed memory
        if mem_before and mem_after:
            freed_mb = mem_before['rss_mb'] - mem_after['rss_mb']
            logger.info(f"{'[' + stage_name + '] ' if stage_name else ''}Memory cleanup completed:")
            logger.info(f"  Before: {mem_before['rss_mb']:.2f} MB")
            logger.info(f"  After: {mem_after['rss_mb']:.2f} MB")
            logger.info(f"  Freed: {freed_mb:.2f} MB")
            
            return {
                "before_mb": mem_before['rss_mb'],
                "after_mb": mem_after['rss_mb'],
                "freed_mb": freed_mb,
                "available_mb": mem_after['available_mb']
            }
        
        return {}
    
    def check_memory_available(self, required_mb: float = 500) -> bool:
        """
        Check if enough memory is available
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            bool: True if enough memory available
        """
        mem_usage = self.get_memory_usage()
        
        if mem_usage:
            available_mb = mem_usage['available_mb']
            
            if available_mb < required_mb:
                logger.warning(f"Low memory: {available_mb:.2f} MB available, {required_mb:.2f} MB required")
                return False
            
            return True
        
        return True
    
    def start_monitoring(self, stage_name: str = ""):
        """
        Start memory monitoring for a stage
        
        Args:
            stage_name: Name of the stage
        """
        mem_usage = self.get_memory_usage()
        if mem_usage:
            self.initial_memory_mb = mem_usage['rss_mb']
            self.peak_memory_mb = mem_usage['rss_mb']
            
            logger.info(f"{'[' + stage_name + '] ' if stage_name else ''}Started memory monitoring")
            logger.info(f"  Initial memory: {self.initial_memory_mb:.2f} MB")
    
    def end_monitoring(self, stage_name: str = "") -> dict:
        """
        End memory monitoring and report statistics
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            dict: Memory statistics
        """
        mem_usage = self.get_memory_usage()
        
        if mem_usage:
            final_memory_mb = mem_usage['rss_mb']
            memory_increase_mb = final_memory_mb - self.initial_memory_mb
            
            logger.info(f"{'[' + stage_name + '] ' if stage_name else ''}Memory monitoring results:")
            logger.info(f"  Initial: {self.initial_memory_mb:.2f} MB")
            logger.info(f"  Final: {final_memory_mb:.2f} MB")
            logger.info(f"  Increase: {memory_increase_mb:.2f} MB")
            
            return {
                "initial_mb": self.initial_memory_mb,
                "final_mb": final_memory_mb,
                "increase_mb": memory_increase_mb,
                "peak_mb": self.peak_memory_mb
            }
        
        return {}


def cleanup_stage_memory(stage_name: str = ""):
    """
    Convenience function to cleanup memory for a stage
    
    Args:
        stage_name: Name of the stage
        
    Returns:
        dict: Memory cleanup statistics
    """
    manager = MemoryManager()
    return manager.cleanup_memory(stage_name)


def log_stage_memory(stage_name: str = ""):
    """
    Convenience function to log memory for a stage
    
    Args:
        stage_name: Name of the stage
    """
    manager = MemoryManager()
    manager.log_memory_usage(stage_name)


if __name__ == "__main__":
    # Test memory manager
    manager = MemoryManager()
    
    print("\n=== Initial Memory ===")
    manager.log_memory_usage("Test")
    
    print("\n=== Creating dummy data ===")
    import numpy as np
    dummy_data = [np.random.rand(1000, 1000) for _ in range(10)]
    manager.log_memory_usage("After allocation")
    
    print("\n=== Cleanup ===")
    stats = manager.cleanup_memory("Test")
    print(f"Cleanup stats: {stats}")
    
    print("\n=== Final Memory ===")
    manager.log_memory_usage("Test")