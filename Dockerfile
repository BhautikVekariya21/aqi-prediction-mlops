# =============================================================================
# MULTI-STAGE DOCKERFILE FOR AQI PREDICTION API
# Optimized for Railway deployment (<500MB target)
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BUILDER - Install dependencies
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 2: RUNTIME - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Copy optimized model (must be present in models/optimized/)
COPY models/optimized/model_final.pkl ./models/optimized/
COPY models/optimized/features.txt ./models/optimized/
COPY models/optimized/model_metadata.json ./models/optimized/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (Railway will override this)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run FastAPI app with Uvicorn
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 2