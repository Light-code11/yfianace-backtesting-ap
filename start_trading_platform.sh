#!/bin/bash

# AI Trading Platform Startup Script
# This script starts both the API server and Streamlit frontend

echo "🚀 Starting AI Trading Platform..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠ Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠ Dependencies not installed. Installing..."
    pip install -r requirements-trading-platform.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠ .env file not found!"
    echo "Please create a .env file with your OpenAI API key"
    echo "You can copy .env.example and fill in your values:"
    echo "  cp .env.example .env"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check if port 8000 is in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠ Port 8000 is already in use"
    read -p "Kill the process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8000 | xargs kill -9
        echo "✓ Process killed"
    else
        echo "❌ Exiting. Please free port 8000 manually."
        exit 1
    fi
fi

# Check if port 8501 is in use
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠ Port 8501 is already in use"
    read -p "Kill the process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8501 | xargs kill -9
        echo "✓ Process killed"
    else
        echo "❌ Exiting. Please free port 8501 manually."
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Starting Backend API Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start API server in background
python3 trading_platform_api.py > api_server.log 2>&1 &
API_PID=$!

# Wait for API to start
echo "Waiting for API server to start..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ API Server is running!"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo "❌ API Server failed to start. Check api_server.log for errors."
        kill $API_PID 2>/dev/null
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Starting Streamlit Frontend..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start Streamlit in background
streamlit run streamlit_app.py > streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "Waiting for Streamlit to start..."
for i in {1..10}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "✓ Streamlit is running!"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo "❌ Streamlit failed to start. Check streamlit.log for errors."
        kill $API_PID 2>/dev/null
        kill $STREAMLIT_PID 2>/dev/null
        exit 1
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ AI Trading Platform is Running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Web Interface:  http://localhost:8501"
echo "🔌 API Server:     http://localhost:8000"
echo "📚 API Docs:       http://localhost:8000/docs"
echo ""
echo "Process IDs:"
echo "  API Server:     $API_PID"
echo "  Streamlit:      $STREAMLIT_PID"
echo ""
echo "Logs:"
echo "  API Server:     api_server.log"
echo "  Streamlit:      streamlit.log"
echo ""
echo "To stop the platform:"
echo "  kill $API_PID $STREAMLIT_PID"
echo "  or press Ctrl+C"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping AI Trading Platform...'; kill $API_PID $STREAMLIT_PID 2>/dev/null; echo '✓ Platform stopped'; exit 0" INT TERM

# Keep script running
wait
