#!/bin/bash
# =============================================================================
# SETUP DAGSHUB INTEGRATION
# Initialize DVC remote with DagsHub
# =============================================================================

set -e

echo "========================================"
echo "Setting up DagsHub Integration"
echo "========================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Load environment variables
source .env

# Check required variables
if [ -z "$DAGSHUB_REPO_OWNER" ] || [ -z "$DAGSHUB_REPO_NAME" ] || [ -z "$DAGSHUB_TOKEN" ]; then
    echo "ERROR: Missing DagsHub credentials in .env"
    echo "Required: DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN"
    exit 1
fi

echo ""
echo "DagsHub Configuration:"
echo "  Owner: $DAGSHUB_REPO_OWNER"
echo "  Repo:  $DAGSHUB_REPO_NAME"
echo ""

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    git add .dvc
fi

# Configure DVC remote
echo "Configuring DVC remote..."
DVC_REMOTE_URL="https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}.dvc"

dvc remote add -d dagshub "$DVC_REMOTE_URL" 2>/dev/null || dvc remote modify dagshub url "$DVC_REMOTE_URL"

# Set authentication
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user "$DAGSHUB_REPO_OWNER"
dvc remote modify dagshub --local password "$DAGSHUB_TOKEN"

echo ""
echo "✓ DVC remote configured: $DVC_REMOTE_URL"

# Configure MLflow tracking
echo ""
echo "Configuring MLflow tracking..."
MLFLOW_TRACKING_URI="https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}.mlflow"

export MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"
export MLFLOW_TRACKING_USERNAME="$DAGSHUB_REPO_OWNER"
export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"

echo "✓ MLflow tracking configured: $MLFLOW_TRACKING_URI"

# Test connection
echo ""
echo "Testing DVC connection..."
dvc remote list

echo ""
echo "========================================"
echo "✓ DagsHub setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run pipeline: dvc repro"
echo "  2. Push data: dvc push"
echo "  3. View experiments: https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}"
echo ""