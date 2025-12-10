# Use a lightweight Python Linux image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from generating .pyc files (saves space)
# and force stdout flushing (better logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
# Install dependencies without caching to save image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 8080 (Standard for Fly.io)
EXPOSE 8080

# Command to run the application
# We use port 8080 here to match Fly.io defaults
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]