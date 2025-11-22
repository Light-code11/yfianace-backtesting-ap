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

# Expose port (Railway sets PORT environment variable)
EXPOSE 8000

# Run the application using Python startup script
CMD ["python", "run.py"]

