#!/bin/bash
set -e

# Use PORT environment variable or default to 8000
PORT=${PORT:-8000}

echo "Starting server on port $PORT..."

# Start uvicorn
exec uvicorn trading_platform_api:app --host 0.0.0.0 --port "$PORT"
