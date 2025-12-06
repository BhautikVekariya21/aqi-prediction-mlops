# src/pipeline/stage_data_splitting.py
from pathlib import Path
from src.components.data_splitting import DataSplitting, DataSplittingConfig
import yaml

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data_splitting"]

    config = DataSplittingConfig(
        input_path=Path(params["input_path"]),
        output_dir=Path(params["output_dir"]),
        test_size=params["test_size"],
        val_size=params["val_size"],
        stratify=params["stratify"]
    )

    splitter = DataSplitting(config)
    train_path, val_path, test_path, features_path = splitter.run()

    print("DATA SPLITTING COMPLETE")
    print(f"   Train → {train_path}")
    print(f"   Val   → {val_path}")
    print(f"   Test  → {test_path}")
    print(f"   Features → {features_path}")

if __name__ == "__main__":
    main()