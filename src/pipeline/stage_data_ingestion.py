# src/pipeline/stage_data_ingestion.py
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from pathlib import Path

def main():
    config = DataIngestionConfig(
        raw_data_dir=Path("data/raw"),
        cities_config_path=Path("configs/cities.yaml")
    )
    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion(resume=True)
    print(f"Ingestion completed: {artifact.message}")

if __name__ == "__main__":
    main()