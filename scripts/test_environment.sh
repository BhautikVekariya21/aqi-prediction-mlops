#!/bin/bash
# =============================================================================
# TEST ENVIRONMENT
# Verify Python environment and DVC setup
# =============================================================================

set -e

echo "========================================"
echo "Testing Environment"
echo "========================================"

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo ""
echo "1. Project root: $PROJECT_ROOT"
echo ""

echo "2. Python environment:"
which python
python --version
echo ""

echo "3. DVC installation:"
which dvc
dvc version
echo ""

echo "4. Virtual environment:"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo ""

echo "5. Python path:"
echo "PYTHONPATH: ${PYTHONPATH:-'(not set)'}"
echo ""

echo "6. Checking for conflicting files:"
echo "Looking for traceback.py in project..."
find . -name "traceback.py" -not -path "./.venv/*" 2>/dev/null || echo "None found ✓"
echo ""

echo "7. Testing Python imports:"
python -c "import sys; print('Python path:'); [print(p) for p in sys.path]"
echo ""

echo "8. Testing DVC import (in clean environment):"
unset PYTHONPATH
python -c "import dvc; print('DVC imported successfully ✓')"
echo ""

echo "========================================"
echo "✓ Environment test complete!"
echo "========================================"