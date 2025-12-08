#!/bin/bash
# =============================================================================
# BUILD & PUSH DOCKER IMAGE
# Builds optimized Docker image and pushes to Docker Hub
# =============================================================================

set -e

# Configuration
IMAGE_NAME="aqi-prediction-api"
VERSION=${1:-latest}

echo "========================================"
echo "Building Docker Image"
echo "========================================"
echo "Image: $IMAGE_NAME:$VERSION"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

# Load Docker Hub credentials
source .env

if [ -z "$DOCKER_USERNAME" ]; then
    echo "ERROR: DOCKER_USERNAME not set in .env"
    exit 1
fi

# Verify optimized model exists
if [ ! -f "models/optimized/model_final.pkl" ]; then
    echo "ERROR: Optimized model not found!"
    echo "Run: dvc repro first to generate the model"
    exit 1
fi

echo "✓ Optimized model found"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t $DOCKER_USERNAME/$IMAGE_NAME:$VERSION .

# Also tag as latest if version is specified
if [ "$VERSION" != "latest" ]; then
    docker tag $DOCKER_USERNAME/$IMAGE_NAME:$VERSION $DOCKER_USERNAME/$IMAGE_NAME:latest
fi

echo ""
echo "✓ Docker image built successfully"
echo ""

# Test the image
echo "Testing Docker image..."
docker run --rm $DOCKER_USERNAME/$IMAGE_NAME:$VERSION python -c "import src; print('✓ Import successful')"

echo ""
echo "Image size:"
docker images $DOCKER_USERNAME/$IMAGE_NAME:$VERSION --format "{{.Size}}"

echo ""
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Logging in to Docker Hub..."
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
    
    echo ""
    echo "Pushing image to Docker Hub..."
    docker push $DOCKER_USERNAME/$IMAGE_NAME:$VERSION
    
    if [ "$VERSION" != "latest" ]; then
        docker push $DOCKER_USERNAME/$IMAGE_NAME:latest
    fi
    
    echo ""
    echo "========================================"
    echo "✓ Docker image pushed successfully!"
    echo "========================================"
    echo ""
    echo "Image: $DOCKER_USERNAME/$IMAGE_NAME:$VERSION"
    echo "URL: https://hub.docker.com/r/$DOCKER_USERNAME/$IMAGE_NAME"
    echo ""
    echo "Deploy to Railway:"
    echo "  1. Connect Railway to your Docker Hub"
    echo "  2. Railway will auto-deploy on new pushes"
    echo ""
else
    echo ""
    echo "Skipping push to Docker Hub"
fi