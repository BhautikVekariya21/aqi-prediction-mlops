# Dockerfile
FROM python:3.11-slim as base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app & model
COPY app/ ./app/
COPY models/optimized/ ./models/optimized/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]