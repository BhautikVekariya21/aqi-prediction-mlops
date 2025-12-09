# =============================================================================
# ULTRA-LIGHTWEIGHT DOCKERFILE FOR AQI PREDICTION API
# Target: <200MB final image size
# Only serves model - no training dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BUILDER - Install only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /build

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create requirements-runtime.txt with ONLY serving dependencies
RUN echo "fastapi==0.109.0" > requirements-runtime.txt && \
    echo "uvicorn[standard]==0.27.0" >> requirements-runtime.txt && \
    echo "pydantic==2.5.3" >> requirements-runtime.txt && \
    echo "xgboost==2.0.3" >> requirements-runtime.txt && \
    echo "numpy==1.24.3" >> requirements-runtime.txt && \
    echo "pandas==2.0.3" >> requirements-runtime.txt && \
    echo "joblib==1.3.2" >> requirements-runtime.txt && \
    echo "requests==2.31.0" >> requirements-runtime.txt && \
    echo "python-dotenv==1.0.0" >> requirements-runtime.txt && \
    echo "pyyaml==6.0.1" >> requirements-runtime.txt

# Install Python packages to a target directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install -r requirements-runtime.txt

# -----------------------------------------------------------------------------
# STAGE 2: RUNTIME - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    MODEL_PATH=/app/models/model.json.gz \
    FEATURES_PATH=/app/models/features.txt

WORKDIR /app

# Install ONLY runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages

# Create directory structure
RUN mkdir -p /app/models /app/src/api /app/src/inference /app/src/utils /app/configs

# Copy ONLY necessary application code (no training code)
COPY src/api/*.py /app/src/api/
COPY src/inference/*.py /app/src/inference/
COPY src/utils/__init__.py /app/src/utils/
COPY src/utils/logger.py /app/src/utils/
COPY src/utils/config_reader.py /app/src/utils/
COPY src/utils/api_client.py /app/src/utils/
COPY src/utils/metrics.py /app/src/utils/

# Copy configs
COPY configs/cities.yaml /app/configs/

# Copy ONLY compressed model and features (CRITICAL: Use .gz model, NOT .pkl)
COPY models/optimized/model.json.gz /app/models/
COPY models/optimized/features.txt /app/models/

# Create __init__.py files
RUN touch /app/src/__init__.py && \
    touch /app/src/api/__init__.py && \
    touch /app/src/inference/__init__.py && \
    touch /app/src/utils/__init__.py

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run with single worker (lightweight)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]