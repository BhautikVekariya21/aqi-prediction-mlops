#!/bin/bash
# =============================================================================
# QUICKSTART DEPLOYMENT SCRIPT
# One-command deployment for lightweight Docker image
# =============================================================================

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   AQI PREDICTION API - LIGHTWEIGHT DEPLOYMENT            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
echo "📋 Step 1/5: Checking prerequisites..."
echo ""

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Please install Docker first."
    exit 1
fi
echo "✅ Docker installed: $(docker --version)"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi
echo "✅ Python3 installed: $(python3 --version)"
echo ""

# Step 2: Check and compress model
echo "🗜️  Step 2/5: Checking model files..."
echo ""

if [ ! -f "models/optimized/model_final.pkl" ]; then
    echo "❌ ERROR: Trained model not found!"
    echo "   Expected: models/optimized/model_final.pkl"
    echo ""
    echo "   Please run the training pipeline first:"
    echo "   dvc repro"
    exit 1
fi

if [ ! -f "models/optimized/features.txt" ]; then
    echo "❌ ERROR: Features file not found!"
    echo "   Expected: models/optimized/features.txt"
    exit 1
fi

echo "✅ Model files found"
echo ""

# Compress model if needed
if [ ! -f "models/optimized/model.json.gz" ]; then
    echo "Compressing model (this may take a minute)..."
    python3 compress_model.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Model compression failed!"
        exit 1
    fi
    echo ""
else
    echo "✅ Compressed model already exists"
    echo ""
fi

# Step 3: Build Docker image
echo "🐳 Step 3/5: Building Docker image..."
echo ""

docker build \
    --progress=plain \
    -t aqi-prediction-api:latest \
    -f Dockerfile \
    . 2>&1 | grep -E "(Step|DONE|ERROR|✅|❌)" || true

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "✅ Docker image built successfully"
echo ""

# Get image size
IMAGE_SIZE=$(docker images aqi-prediction-api:latest --format "{{.Size}}")
echo "📦 Image size: $IMAGE_SIZE"
echo ""

# Step 4: Test image
echo "🧪 Step 4/5: Testing Docker image..."
echo ""

docker run --rm aqi-prediction-api:latest python -c "
import sys
sys.path.insert(0, '/app')
from src.inference.model_loader import ModelLoader
loader = ModelLoader('/app/models/model.json.gz', '/app/models/features.txt')
print('✅ Model loaded successfully')
print(f'✅ Features: {len(loader.feature_names)}')
" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Image test failed!"
    exit 1
fi

echo ""

# Step 5: Start container
echo "🚀 Step 5/5: Starting API container..."
echo ""

# Stop any existing container
docker-compose down 2>/dev/null || true

# Start new container
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo "✅ Container started"
echo ""

# Wait for health check
echo "⏳ Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is healthy!"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Test the API
echo ""
echo "🎯 Testing API endpoint..."
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              ✅ DEPLOYMENT SUCCESSFUL! ✅                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 API is running at: http://localhost:8000"
echo ""
echo "📚 Available endpoints:"
echo "   • GET  /          - API info"
echo "   • GET  /health    - Health check"
echo "   • GET  /docs      - Interactive docs"
echo "   • POST /predict   - AQI prediction"
echo "   • POST /forecast  - AQI forecast"
echo "   • GET  /cities    - List cities"
echo ""
echo "🧪 Quick test:"
echo "   curl http://localhost:8000/health"
echo ""
echo "📖 Full docs:"
echo "   Open http://localhost:8000/docs in browser"
echo ""
echo "🛑 Stop API:"
echo "   docker-compose down"
echo ""
echo "📊 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "☁️  Deploy to Railway:"
echo "   1. docker tag aqi-prediction-api:latest YOUR_USERNAME/aqi-prediction-api"
echo "   2. docker push YOUR_USERNAME/aqi-prediction-api"
echo "   3. Connect Railway to Docker Hub"
echo ""