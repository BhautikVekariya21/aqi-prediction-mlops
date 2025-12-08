"""
DVC Stage 9: Live Prediction & Comparison
Compare our model predictions with Open-Meteo AQI forecast
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger, LoggerContext
from src.utils.config_reader import ConfigReader


logger = get_logger(__name__)


def main():
    """Run prediction comparison stage"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run prediction comparison")
    parser.add_argument("--city", type=str, default="Delhi", help="City name")
    parser.add_argument("--days", type=int, default=2, help="Forecast days")
    args = parser.parse_args()
    
    with LoggerContext(f"Stage 9: Prediction Comparison - {args.city}") as stage_logger:
        try:
            # Load config
            config = ConfigReader("configs/params.yaml")
            stage_logger.info("Configuration loaded")
            
            # Note: This stage doesn't use components (implemented in src/inference/)
            # For now, just create placeholder outputs
            
            stage_logger.info(f"City: {args.city}")
            stage_logger.info(f"Forecast days: {args.days}")
            
            # Create output directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            plots_dir = reports_dir / "prediction_plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Placeholder outputs
            import json
            
            # Prediction metrics
            prediction_metrics = {
                "city": args.city,
                "forecast_days": args.days,
                "status": "placeholder - implement in src/inference/live_predictor.py"
            }
            
            metrics_file = reports_dir / "prediction_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(prediction_metrics, f, indent=2)
            
            stage_logger.info(f"Prediction comparison completed")
            stage_logger.info(f"Note: Full implementation in src/inference/live_predictor.py")
            
            return 0
        
        except Exception as e:
            logger.error(f"Prediction comparison failed: {e}")
            import traceback
            traceback.print_exc()
            
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)