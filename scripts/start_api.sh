#!/bin/bash
# =============================================================================
# START FASTAPI SERVER LOCALLY (NO DOCKER)
# For local development and testing
# =============================================================================

set -e

echo "========================================"
echo "Starting AQI Prediction API (Local)"
echo "========================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found!"
    echo "API will use default configurations"
fi

# Check if optimized model exists
if [ ! -f "models/optimized/model_final.pkl" ] && [ ! -f "models/optimized/model.json.gz" ]; then
    echo "ERROR: Optimized model not found!"
    echo "Run the pipeline first: dvc repro"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set defaults
export PORT=${PORT:-8000}
export MODEL_PATH=${MODEL_PATH:-models/optimized/model_final.pkl}
export FEATURES_PATH=${FEATURES_PATH:-models/optimized/features.txt}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Model: $MODEL_PATH"
echo "  Features: $FEATURES_PATH"
echo "  Log Level: $LOG_LEVEL"
echo ""

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "ERROR: uvicorn not installed!"
    echo "Install dependencies: pip install -r requirements.txt"
    exit 1
fi

echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:$PORT"
echo "API documentation: http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn with hot reload for development
uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level ${LOG_LEVEL,,}