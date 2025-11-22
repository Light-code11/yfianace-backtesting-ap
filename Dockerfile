# Dockerfile for AI Trading Platform API Server
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-trading-platform.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-trading-platform.txt

# Copy application files
COPY *.py ./
COPY yfinance/ ./yfinance/

# Copy and set executable permission for entrypoint script
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Expose port (Railway sets PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use entrypoint script to handle PORT variable
ENTRYPOINT ["./docker-entrypoint.sh"]

