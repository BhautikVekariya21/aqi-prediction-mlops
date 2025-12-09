#!/bin/bash
# =============================================================================
# DEPLOY TO RAILWAY (DIRECT - NO DOCKER)
# Deploys FastAPI app directly using Railway CLI
# =============================================================================

set -e

echo "========================================"
echo "Deploy to Railway (Direct)"
echo "========================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "ERROR: Railway CLI not installed!"
    echo ""
    echo "Install with:"
    echo "  npm i -g @railway/cli"
    echo ""
    echo "Or use:"
    echo "  brew install railway"
    echo ""
    exit 1
fi

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "Not logged in to Railway"
    echo ""
    read -p "Login now? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        railway login
    else
        echo "Please login first: railway login"
        exit 1
    fi
fi

echo ""
echo "Railway User: $(railway whoami)"
echo ""

# Verify optimized model exists
if [ ! -f "models/optimized/model_final.pkl" ] && [ ! -f "models/optimized/model.json.gz" ]; then
    echo "ERROR: Optimized model not found!"
    echo "Run: dvc repro to generate the model"
    exit 1
fi

echo "✓ Optimized model found"
echo ""

# Link to project (if not already linked)
if [ ! -f "railway.json" ] && [ ! -f ".railway" ]; then
    echo "Project not linked to Railway"
    echo ""
    read -p "Link to existing project? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        railway link
    else
        echo "Creating new project..."
        railway init
    fi
fi

# Set environment variables
echo ""
echo "Setting environment variables..."
railway variables set MODEL_PATH=models/optimized/model_final.pkl
railway variables set FEATURES_PATH=models/optimized/features.txt
railway variables set LOG_LEVEL=INFO

echo "✓ Environment variables set"
echo ""

# Deploy
echo "Deploying to Railway..."
echo "This may take a few minutes..."
echo ""

railway up

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Deployment successful!"
    echo "========================================"
    echo ""
    echo "Your API is now live!"
    echo ""
    echo "View deployment:"
    echo "  railway open"
    echo ""
    echo "View logs:"
    echo "  railway logs"
    echo ""
    echo "Get domain:"
    echo "  railway domain"
    echo ""
else
    echo ""
    echo "ERROR: Deployment failed!"
    echo "Check logs with: railway logs"
    exit 1
fi