#!/bin/bash
# Startup script for Railway deployment
PORT=${PORT:-8000}
uvicorn trading_platform_api:app --host 0.0.0.0 --port $PORT
