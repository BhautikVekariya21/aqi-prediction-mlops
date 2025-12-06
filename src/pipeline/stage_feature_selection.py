# src/pipeline/stage_feature_selection.py
from pathlib import Path
from src.components.feature_selection import FeatureSelection, FeatureSelectionConfig
import yaml

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["feature_selection"]

    config = FeatureSelectionConfig(
        input_path=Path(params["input_path"]),
        output_dir=Path(params["output_dir"]),
        top_n=params["top_n"],
        must_have=params["must_have"]
    )
    selector = FeatureSelection(config)
    data_path, features_path = selector.run()
    print(f"Feature selection complete!")
    print(f"   → Dataset: {data_path}")
    print(f"   → Features: {features_path}")

if __name__ == "__main__":
    main()