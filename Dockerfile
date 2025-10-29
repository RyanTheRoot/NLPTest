# Sentiment & Toxicity Analysis API - Offline Docker Image
FROM python:3.11-slim

# Build argument for git SHA
ARG GIT_SHA=unknown

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    MODEL_BACKEND=transformer \
    GIT_SHA=${GIT_SHA} \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py inference.py ./
COPY models/ ./models/
COPY data/ ./data/
COPY tests/ ./tests/

# Download models and train artifacts during build
RUN python models/bootstrap_models.py

# Clean up build dependencies to reduce image size
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /root/.cache

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app /opt/models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Set offline mode for runtime (models already downloaded)
ENV TRANSFORMERS_OFFLINE=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]

