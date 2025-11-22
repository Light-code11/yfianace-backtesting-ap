#!/bin/bash
# Universal start script for the YFinance API server
# Works on macOS, Linux, and other Unix-like systems

set -e  # Exit on error

echo "🚀 Starting YFinance API Server..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo "❌ Error: Python is not installed!"
    echo "Please install Python 3.6 or higher"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not found. Installing dependencies..."
    echo "Installing: fastapi, uvicorn, python-multipart..."
    $PIP_CMD install fastapi 'uvicorn[standard]' python-multipart
    if [ $? -ne 0 ]; then
        echo "⚠️  Installation with extras failed. Trying basic install..."
        $PIP_CMD install fastapi uvicorn python-multipart
    fi
else
    echo "✅ Dependencies are installed"
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Warning: Port 8000 is already in use!"
    echo "Would you like to kill the existing process? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        lsof -ti:8000 | xargs kill -9
        echo "✅ Killed existing process on port 8000"
    else
        echo "❌ Cannot start server. Port 8000 is in use."
        exit 1
    fi
fi

# Start the server
echo ""
echo "✨ Starting server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Alternative docs: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

$PYTHON_CMD api_server.py

