# Installation Instructions for macOS

## 📦 What You Need to Install

You need to install FastAPI and Uvicorn to run the API server. Everything else is already installed.

## 🚀 Installation Steps

### Copy and paste these commands into your terminal:

```bash
# Step 1: Navigate to the project directory
cd /Users/alanphilip/yfinance

# Step 2: Install only the API dependencies (not the extras)
# Note: Quote uvicorn[standard] for zsh shell on macOS
pip3 install fastapi 'uvicorn[standard]' python-multipart

# Step 3: Run the server
python3 api_server.py
```

That's it! The server should start on http://localhost:8000

## 🧪 Verify Installation

Open a new terminal and test:

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test getting a stock price
curl http://localhost:8000/ticker/AAPL/price
```

## 🌐 Expose with Ngrok (for n8n)

If you want to access from n8n or externally:

```bash
# Install ngrok if you don't have it
brew install ngrok

# Expose your local server
ngrok http 8000
```

Copy the https URL from ngrok (e.g., `https://abc123.ngrok.io`) and use it in n8n!

## ❓ Troubleshooting

### If pip3 gives permission errors:

```bash
pip3 install --user fastapi 'uvicorn[standard]' python-multipart
```

### If you get "command not found: pip3":

```bash
# Install pip for Python 3
python3 -m ensurepip --upgrade

# Then try again
python3 -m pip install fastapi 'uvicorn[standard]' python-multipart
```

### If uvicorn fails with SSL errors:

Just install the basic version:
```bash
pip3 install fastapi uvicorn python-multipart
```

Then run with:
```bash
python3 api_server.py
```

## 📚 Next Steps

Once the server is running:

1. **Test it**: Open http://localhost:8000/docs in your browser
2. **Try an endpoint**: http://localhost:8000/ticker/AAPL/price
3. **Connect n8n**: Use HTTP Request node pointing to your endpoints

See [QUICKSTART.md](QUICKSTART.md) for more details!

