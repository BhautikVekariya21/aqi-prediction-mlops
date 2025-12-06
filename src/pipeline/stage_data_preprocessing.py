# src/pipeline/stage_data_preprocessing.py
from pathlib import Path
from src.components.data_preprocessing import DataPreprocessing, DataPreprocessingConfig
import yaml

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    config = DataPreprocessingConfig(
        input_path=Path(params["data_preprocessing"]["input_path"]),
        output_dir=Path(params["data_preprocessing"]["output_dir"]),
        drop_columns=params["data_preprocessing"]["drop_columns"],
        winsorize_limits=tuple(params["data_preprocessing"]["winsorize_limits"]),
        knn_neighbors=params["data_preprocessing"]["knn_neighbors"]
    )

    preprocessor = DataPreprocessing(config)
    output_path = preprocessor.run()
    print(f"Preprocessing completed: {output_path}")

if __name__ == "__main__":
    main()