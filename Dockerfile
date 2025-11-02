# Use Python 3.11 slim (smallest base)
FROM python:3.11-slim

WORKDIR /app

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model (caches it in image)
# This prevents downloading on first request and makes startup faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY *.py ./
COPY models/ ./models/
COPY *.json ./

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway, Render, Fly.io use PORT env variable)
EXPOSE 8000

# Run uvicorn (use PORT env variable for platform compatibility)
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
