# src/pipeline/stage_feature_engineering.py
from pathlib import Path
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig
import yaml

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["feature_engineering"]

    config = FeatureEngineeringConfig(
        input_path=Path(params["input_path"]),
        output_dir=Path(params["output_dir"])
    )
    fe = FeatureEngineering(config)
    output_path = fe.run()
    print(f"Feature engineering complete: {output_path}")

if __name__ == "__main__":
    main()