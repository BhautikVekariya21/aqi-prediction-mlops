#!/bin/bash

echo "Updating params.yaml path references..."
echo "========================================"

# Update pipeline stages
echo "✓ Updating pipeline stages..."
find src/pipeline -name "stage_*.py" -type f -exec sed -i.bak 's/ConfigReader("configs\/params.yaml")/ConfigReader("params.yaml")/g' {} \;

# Update dagshub_utils.py
echo "✓ Updating dagshub_utils.py..."
sed -i.bak 's/ConfigReader("configs\/params.yaml")/ConfigReader("params.yaml")/g' src/utils/dagshub_utils.py

# Update config_reader.py
echo "✓ Updating config_reader.py..."
sed -i.bak 's/config_path: str = "configs\/params.yaml"/config_path: str = "params.yaml"/g' src/utils/config_reader.py

# Remove backup files
echo "✓ Cleaning up backup files..."
find . -name "*.bak" -type f -delete

echo ""
echo "=========================================="
echo "✓ All references updated successfully!"
echo ""
echo "Moved file location:"
echo "  configs/params.yaml → params.yaml"
echo ""
echo "Files updated:"
echo "  - All src/pipeline/stage_*.py files"
echo "  - src/utils/dagshub_utils.py"
echo "  - src/utils/config_reader.py"
echo ""

# Verify the changes
echo "Verifying changes..."
if grep -r "configs/params.yaml" src/ 2>/dev/null; then
    echo "⚠️  Warning: Some references to configs/params.yaml still exist"
else
    echo "✓ No references to configs/params.yaml found"
fi

echo ""
echo "Next steps:"
echo "  1. Move the file: mv configs/params.yaml params.yaml"
echo "  2. Verify: cat params.yaml"
echo "  3. Test: dvc params diff"
echo "  4. Run pipeline: ./scripts/train_pipeline.sh"
