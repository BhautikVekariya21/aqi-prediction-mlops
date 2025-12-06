# src/pipeline/stage_model_optimization.py
from pathlib import Path
from src.components.model_optimizer import ModelOptimizer, ModelOptimizationConfig

def main():
    config = ModelOptimizationConfig(
        input_model_path=Path("models/xgboost.json"),
        output_dir=Path("models/optimized")
    )

    optimizer = ModelOptimizer(config)
    result = optimizer.run()

    print("MODEL OPTIMIZATION COMPLETE")
    print(f"   GZIP Model → {result['gzip_path']}")
    print(f"   LZMA Model → {result['lzma_path']} ({result['size_mb']:.2f} MB)")
    print(f"   Metadata   → models/optimized/model_metadata.json")
    print("   READY FOR DEPLOYMENT")

if __name__ == "__main__":
    main()