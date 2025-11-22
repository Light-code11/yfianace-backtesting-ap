# YFinance API Server - Deployment Guide

This guide explains how to deploy the YFinance API server and integrate it with n8n.

## 🚀 Quick Start

### Local Development

> **Note for macOS users:** Use `python3` and `pip3` instead of `python` and `pip`

1. **Navigate to the project directory:**
```bash
cd /Users/alanphilip/yfinance
# Or wherever you cloned the repository
```

2. **Install dependencies:**
```bash
# macOS/Linux
pip3 install -r requirements-api.txt

# Or if you have pip linked to Python 3
pip install -r requirements-api.txt
```

3. **Run the server:**
```bash
# macOS/Linux
python3 api_server.py

# Or if you have python linked to Python 3
python api_server.py
```

The API will be available at `http://localhost:8000`

4. **View API documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Using Docker

1. **Build and run with Docker Compose:**
```bash
docker-compose up -d
```

2. **Or build and run manually:**
```bash
docker build -t yfinance-api .
docker run -d -p 8000:8000 --name yfinance-api yfinance-api
```

3. **Check logs:**
```bash
docker logs -f yfinance-api
```

4. **Stop the service:**
```bash
docker-compose down
```

## 🌐 Deployment Options

### Option 1: Deploy to a VPS (DigitalOcean, AWS EC2, etc.)

1. **SSH into your server:**
```bash
ssh user@your-server-ip
```

2. **Clone your repository:**
```bash
git clone https://github.com/yourusername/yfinance.git
cd yfinance
```

3. **Install Docker (if not installed):**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install docker-compose
```

4. **Run with Docker Compose:**
```bash
docker-compose up -d
```

5. **Configure firewall to allow port 8000:**
```bash
sudo ufw allow 8000
```

### Option 2: Deploy to Heroku

1. **Create a `Procfile`:**
```
web: uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

2. **Deploy:**
```bash
heroku create yfinance-api
git push heroku main
```

### Option 3: Deploy to Railway.app

1. Connect your GitHub repository to Railway
2. Railway will auto-detect the Dockerfile
3. Set environment variables if needed
4. Deploy!

### Option 4: Deploy to Render.com

1. Create a new Web Service
2. Connect your repository
3. Use Docker deployment
4. Set port to 8000
5. Deploy

### Option 5: Use Ngrok (for testing/development)

If you want to quickly expose your local server:

```bash
# First, navigate to the project directory
cd /Users/alanphilip/yfinance

# Run the API locally (use python3 on macOS)
python3 api_server.py

# In another terminal, expose it
ngrok http 8000
```

Ngrok will give you a public URL like `https://abc123.ngrok.io`

> **macOS Note:** If you get "command not found: python", use `python3` instead

## 📡 API Endpoints

### Health Check
```
GET /health
```

### Get Ticker Information
```
GET /ticker/{symbol}
GET /ticker/AAPL
```

### Get Historical Data
```
GET /ticker/{symbol}/history?period=1mo&interval=1d
GET /ticker/AAPL/history?period=1y&interval=1d
```

### Get Current Price
```
GET /ticker/{symbol}/price
GET /ticker/AAPL/price
```

### Get Financial Statements
```
GET /ticker/{symbol}/financials?statement=income
GET /ticker/AAPL/financials?statement=balance
```

### Get Recommendations
```
GET /ticker/{symbol}/recommendations
GET /ticker/AAPL/recommendations
```

### Get Actions (Dividends & Splits)
```
GET /ticker/{symbol}/actions
GET /ticker/AAPL/actions
```

### Search Tickers
```
GET /search?query=apple&max_results=5
```

### Download Multiple Tickers
```
POST /download?tickers=AAPL&tickers=MSFT&period=1mo
```

### Get Market Info
```
GET /market/{country_code}
GET /market/US
```

## 🔗 Integrating with n8n

### Method 1: Using HTTP Request Node

1. **Add an HTTP Request node in n8n**

2. **Configure the node:**
   - **Method**: GET (or POST for `/download`)
   - **URL**: `http://your-server:8000/ticker/AAPL/price`
   - **Authentication**: None (add if you implement auth)
   - **Response Format**: JSON

3. **Example configurations:**

#### Get Current Price
```
Method: GET
URL: http://your-server:8000/ticker/{{ $json.symbol }}/price
```

#### Get Historical Data
```
Method: GET
URL: http://your-server:8000/ticker/AAPL/history
Query Parameters:
  - period: 1mo
  - interval: 1d
```

#### Search for Stocks
```
Method: GET
URL: http://your-server:8000/search
Query Parameters:
  - query: {{ $json.searchTerm }}
  - max_results: 10
```

### Method 2: Using Webhook (Trigger-based)

If you want n8n to receive data pushed from your API:

1. **In n8n, add a Webhook node**
2. **Copy the webhook URL** (e.g., `https://your-n8n.com/webhook/abc123`)
3. **Create a custom endpoint in your API** that pushes to n8n when certain conditions are met

### Example n8n Workflow

```
1. [Schedule Trigger] (Every 1 hour)
   ↓
2. [HTTP Request] GET /ticker/AAPL/price
   ↓
3. [Function] Process price data
   ↓
4. [Slack/Email] Send notification if price changed
```

## 🔒 Security Recommendations

### Add API Key Authentication

Update `api_server.py` to add authentication:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to endpoints:
@app.get("/ticker/{symbol}", dependencies=[Security(verify_api_key)])
```

Then in n8n, add a header:
```
X-API-Key: your-secret-api-key
```

### Use HTTPS

- Use a reverse proxy like Nginx with Let's Encrypt SSL
- Or use a platform that provides HTTPS automatically (Heroku, Railway, etc.)

### Rate Limiting

Consider adding rate limiting to prevent abuse:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/ticker/{symbol}")
@limiter.limit("10/minute")
async def get_ticker_info(symbol: str):
    # ... existing code
```

## 🧪 Testing the API

### Using curl
```bash
curl http://localhost:8000/ticker/AAPL/price
```

### Using Python
```python
import requests

response = requests.get("http://localhost:8000/ticker/AAPL/price")
print(response.json())
```

### Using n8n
1. Add HTTP Request node
2. Set URL to your endpoint
3. Execute the workflow
4. Check the response

## 📊 Monitoring

### Check Server Status
```bash
curl http://your-server:8000/health
```

### Docker Logs
```bash
docker logs -f yfinance-api
```

### Add Logging
The API server logs are printed to stdout. You can redirect them to a file:

```bash
docker-compose logs -f > api.log
```

## 🐛 Troubleshooting

### "command not found: python" on macOS

**Problem:** Running `python api_server.py` gives "command not found: python"

**Solution:** macOS uses `python3` by default. Use these commands instead:

```bash
# Navigate to project directory first
cd /Users/alanphilip/yfinance

# Install dependencies
pip3 install -r requirements-api.txt

# Run the server
python3 api_server.py
```

**Alternative:** Create an alias in your `~/.zshrc`:
```bash
echo "alias python=python3" >> ~/.zshrc
echo "alias pip=pip3" >> ~/.zshrc
source ~/.zshrc
```

**Or use the start script:**
```bash
./start_server.sh
```

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Docker Issues
```bash
# Rebuild without cache
docker-compose build --no-cache
docker-compose up -d

# Clean up
docker system prune -a
```

### Module Not Found
Make sure you're in the correct directory and all dependencies are installed:
```bash
cd /Users/alanphilip/yfinance
pip3 install -r requirements-api.txt
```

### Wrong Directory
If you get file not found errors, make sure you're in the yfinance directory:
```bash
pwd  # Should show /Users/alanphilip/yfinance
cd /Users/alanphilip/yfinance
```

## 📝 Notes

- **Yahoo Finance API**: Remember that yfinance uses Yahoo's public APIs, which are intended for personal use only. Check Yahoo's terms of service.
- **Rate Limits**: Yahoo may rate-limit requests. Consider caching responses if you're making frequent requests.
- **Data Delays**: Market data may be delayed by 15-20 minutes depending on the exchange.

## 🔄 Updates

To update the server:

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

## 💡 Tips for n8n Integration

1. **Use n8n's Schedule Trigger** to fetch data at regular intervals
2. **Store results in a database** using n8n's database nodes
3. **Set up alerts** based on price changes using conditional logic
4. **Create a dashboard** by combining with services like Notion or Google Sheets
5. **Use n8n's error handling** to retry failed requests

## 🎯 Example Use Cases

1. **Price Alerts**: Trigger notifications when a stock hits a target price
2. **Portfolio Tracking**: Regularly fetch prices for your portfolio
3. **Market Analysis**: Collect historical data for analysis
4. **Trading Signals**: Combine with technical indicators in n8n
5. **News Integration**: Combine stock data with news APIs

## 📚 Additional Resources

- [YFinance Documentation](https://ranaroussi.github.io/yfinance)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [n8n Documentation](https://docs.n8n.io/)
- [Docker Documentation](https://docs.docker.com/)

