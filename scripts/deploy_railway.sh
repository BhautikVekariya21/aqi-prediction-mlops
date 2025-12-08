#!/bin/bash
# =============================================================================
# DEPLOY TO RAILWAY
# Helper script for Railway deployment via Railway CLI
# =============================================================================

set -e

echo "========================================"
echo "Deploy to Railway"
echo "========================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "ERROR: Railway CLI not installed!"
    echo ""
    echo "Install with:"
    echo "  npm i -g @railway/cli"
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

# Link to project (if not already linked)
if [ ! -f "railway.json" ]; then
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

echo ""
echo "Deploying to Railway..."
echo ""

# Deploy
railway up

echo ""
echo "========================================"
echo "✓ Deployment initiated!"
echo "========================================"
echo ""
echo "Monitor deployment:"
echo "  railway logs"
echo ""
echo "Open in browser:"
echo "  railway open"
echo ""