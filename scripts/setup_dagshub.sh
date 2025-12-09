#!/bin/bash
# =============================================================================
# SETUP DAGSHUB INTEGRATION
# Initialize DVC remote with DagsHub
# =============================================================================

set -e

echo "========================================"
echo "========================================"
echo "Setting up DagsHub Integration"
echo "========================================"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Load environment variables (only DagsHub related)
export DAGSHUB_REPO_OWNER=$(grep "^DAGSHUB_REPO_OWNER=" .env | cut -d '=' -f2)
export DAGSHUB_REPO_NAME=$(grep "^DAGSHUB_REPO_NAME=" .env | cut -d '=' -f2)
export DAGSHUB_TOKEN=$(grep "^DAGSHUB_TOKEN=" .env | cut -d '=' -f2)

# Check required variables
if [ -z "$DAGSHUB_REPO_OWNER" ] || [ -z "$DAGSHUB_REPO_NAME" ] || [ -z "$DAGSHUB_TOKEN" ]; then
    echo "ERROR: Missing DagsHub credentials in .env"
    echo "Required: DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN"
    exit 1
fi

echo "DagsHub Configuration:"
echo "  Owner: $DAGSHUB_REPO_OWNER"
echo "  Repo:  $DAGSHUB_REPO_NAME"
echo ""

# Unset PYTHONPATH to avoid conflicts
unset PYTHONPATH

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init
    git add .dvc .dvcignore
fi

# Configure DVC remote
echo "Configuring DVC remote..."
DVC_REMOTE_URL="https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}.dvc"

# Remove existing remote if present
dvc remote remove dagshub 2>/dev/null || true

# Add new remote
dvc remote add -d dagshub "$DVC_REMOTE_URL"

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

# Update .env with MLflow settings (don't export globally)
grep -v "MLFLOW_TRACKING_URI" .env > .env.tmp || true
grep -v "MLFLOW_TRACKING_USERNAME" .env.tmp > .env.tmp2 || true
grep -v "MLFLOW_TRACKING_PASSWORD" .env.tmp2 > .env.tmp3 || true

echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI" >> .env.tmp3
echo "MLFLOW_TRACKING_USERNAME=$DAGSHUB_REPO_OWNER" >> .env.tmp3
echo "MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN" >> .env.tmp3

mv .env.tmp3 .env
rm -f .env.tmp .env.tmp2

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
echo "  1. Run pipeline: ./scripts/run_pipeline.sh"
echo "  2. Or manual: dvc repro"
echo "  3. Push data: dvc push"
echo "  4. View experiments: https://dagshub.com/${DAGSHUB_REPO_OWNER}/${DAGSHUB_REPO_NAME}"
echo ""