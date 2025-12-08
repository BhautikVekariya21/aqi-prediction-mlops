#!/bin/bash
# =============================================================================
# RUN FULL DVC PIPELINE
# Executes all stages from data ingestion to model evaluation
# =============================================================================

set -e

echo "========================================"
echo "Running Full DVC Pipeline"
echo "========================================"

# Check if DVC is initialized
if [ ! -d ".dvc" ]; then
    echo "ERROR: DVC not initialized!"
    echo "Run: ./scripts/setup_dagshub.sh first"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Load environment variables
source .env

echo ""
echo "Pipeline Configuration:"
echo "  Start Date: $(grep 'start_date' configs/params.yaml | awk '{print $2}')"
echo "  Target: us_aqi"
echo "  Models: Decision Tree, Random Forest, Extra Trees, XGBoost, CatBoost"
echo ""

# Pull latest data from DVC (if exists)
echo "Pulling latest data from DVC remote..."
dvc pull 2>/dev/null || echo "No remote data to pull (first run)"

# Run DVC pipeline
echo ""
echo "Running DVC pipeline..."
echo "This may take several hours for first run..."
echo ""

dvc repro

# Check if pipeline completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Pipeline completed successfully!"
    echo "========================================"
    echo ""
    echo "Results:"
    echo "  - Data: data/splits/"
    echo "  - Models: models/optimized/"
    echo "  - Evaluation: models/evaluation/"
    echo ""
    echo "Push results to DagsHub:"
    echo "  dvc push"
    echo "  git add dvc.lock"
    echo "  git commit -m 'Update pipeline results'"
    echo "  git push"
    echo ""
else
    echo ""
    echo "ERROR: Pipeline failed!"
    echo "Check logs above for details"
    exit 1
fi