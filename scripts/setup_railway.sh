#!/bin/bash
# =============================================================================
# SETUP RAILWAY DEPLOYMENT FILES
# =============================================================================

echo "🚀 Setting up Railway deployment files..."
echo ""

# Check if in project root
if [ ! -d "src" ]; then
    echo "❌ Error: Run this from project root"
    exit 1
fi

# 1. Create Procfile
echo "📝 Creating Procfile..."
cat > Procfile << 'EOF'
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 2
EOF

# 2. Create railway.json
echo "📝 Creating railway.json..."
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install --no-cache-dir -r requirements.txt"
  },
  "deploy": {
    "startCommand": "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 2",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# 3. Create runtime.txt
echo "📝 Creating runtime.txt..."
cat > runtime.txt << 'EOF'
python-3.10.12
EOF

# 4. Create .railwayignore
echo "📝 Creating .railwayignore..."
cat > .railwayignore << 'EOF'
data/
notebooks/
tests/
.vscode/
.idea/
*.pyc
__pycache__/
.pytest_cache/
.git/
.dvc/
.github/
*.md
!README.md
logs/
*.log
.env
.cache/
*.egg-info/
dist/
build/
.DS_Store
Thumbs.db
EOF

# 5. Create __init__.py files if missing
echo "📝 Creating __init__.py files..."
touch src/__init__.py
touch src/api/__init__.py

# 6. Verify model files
echo ""
echo "🔍 Verifying model files..."

if [ -f "models/optimized/model_final.pkl" ] || [ -f "models/optimized/model.json.gz" ]; then
    echo "✅ Model file found"
else
    echo "❌ Model file NOT found!"
    echo "   Expected: models/optimized/model_final.pkl"
fi

if [ -f "models/optimized/features.txt" ]; then
    FEATURE_COUNT=$(wc -l < models/optimized/features.txt)
    echo "✅ Features file found ($FEATURE_COUNT features)"
else
    echo "❌ Features file NOT found!"
    echo "   Expected: models/optimized/features.txt"
fi

# 7. Verify requirements.txt
echo ""
echo "🔍 Verifying requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
else
    echo "❌ requirements.txt NOT found!"
fi

# 8. Summary
echo ""
echo "=" * 70
echo "✅ Railway deployment files created!"
echo "=" * 70
echo ""
echo "Files created:"
echo "  ✅ Procfile"
echo "  ✅ railway.json"
echo "  ✅ runtime.txt"
echo "  ✅ .railwayignore"
echo "  ✅ src/__init__.py"
echo "  ✅ src/api/__init__.py"
echo ""
echo "Next steps:"
echo "  1. Verify model files exist"
echo "  2. Test locally: uvicorn src.api.main:app --reload"
echo "  3. Push to GitHub"
echo "  4. Deploy on Railway: https://railway.app/new"
echo ""