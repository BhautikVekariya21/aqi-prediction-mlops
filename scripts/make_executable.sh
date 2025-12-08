#!/bin/bash
# Make all scripts executable

chmod +x scripts/setup_dagshub.sh
chmod +x scripts/train_pipeline.sh
chmod +x scripts/build_docker.sh
chmod +x scripts/deploy_railway.sh

echo "✓ All scripts are now executable"