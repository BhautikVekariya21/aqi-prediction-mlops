#!/bin/bash
# =============================================================================
# TEST API ENDPOINTS LOCALLY
# Quick tests for all endpoints
# =============================================================================

set -e

API_URL=${1:-http://localhost:8000}

echo "========================================"
echo "Testing AQI Prediction API"
echo "========================================"
echo "API URL: $API_URL"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    echo -n "Testing $method $endpoint ... "
    
    if [ -z "$data" ]; then
        response=$(curl -s -X $method "$API_URL$endpoint")
    else
        response=$(curl -s -X $method "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
        # echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        echo -e "${RED}FAILED${NC}"
    fi
}

# 1. Test root endpoint
test_endpoint GET "/"

# 2. Test health check
test_endpoint GET "/health"

# 3. Test cities list
test_endpoint GET "/cities"

# 4. Test manual prediction
echo ""
echo "Testing manual prediction (Delhi)..."
test_endpoint POST "/predict" '{
  "city": "Delhi",
  "pm2_5": 150,
  "pm10": 250,
  "o3": 80,
  "no2": 60,
  "humidity": 65
}'

# 5. Test forecast
echo ""
echo "Testing forecast (Mumbai)..."
test_endpoint POST "/forecast" '{
  "city": "Mumbai",
  "forecast_days": 2
}'

echo ""
echo "========================================"
echo "API Testing Complete"
echo "========================================"
echo ""
echo "For detailed responses, use:"
echo "  curl -X GET $API_URL/health | jq"
echo "  curl -X POST $API_URL/predict -H 'Content-Type: application/json' -d '{...}' | jq"
echo ""