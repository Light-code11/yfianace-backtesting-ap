# AI Trading Platform - Complete Setup Guide

## 🚀 Quick Start

This guide will help you set up and run the complete AI Trading Platform with strategy generation, backtesting, paper trading, and portfolio optimization.

---

## 📋 Prerequisites

1. **Python 3.9 or higher**
2. **OpenAI API Key** (for AI strategy generation)
3. **Terminal/Command Line access**
4. **8GB RAM minimum** (16GB recommended)

---

## 🔧 Installation Steps

### Step 1: Navigate to Directory

```bash
cd /Users/alanphilip/Downloads/yfinance
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements-trading-platform.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project directory:

```bash
# Create .env file
cat > .env << EOF
# OpenAI API Key (required for AI strategy generation)
OPENAI_API_KEY=your_openai_api_key_here

# Database URL (optional, defaults to SQLite)
DATABASE_URL=sqlite:///./trading_platform.db

# API Configuration (optional)
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

**Important:** Replace `your_openai_api_key_here` with your actual OpenAI API key.

To get an OpenAI API key:
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy and paste it into the `.env` file

### Step 5: Initialize Database

```bash
# The database will be created automatically on first run
# But you can initialize it manually:
python3 -c "from database import init_db; init_db()"
```

---

## 🎯 Running the Platform

### Method 1: Run Both Backend and Frontend (Recommended)

**Terminal 1 - Start the API Server:**

```bash
cd /Users/alanphilip/Downloads/yfinance
python3 trading_platform_api.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Start the Streamlit Web Interface:**

```bash
cd /Users/alanphilip/Downloads/yfinance
streamlit run streamlit_app.py
```

Your browser should automatically open to http://localhost:8501

If not, navigate to: http://localhost:8501

### Method 2: API Only (for programmatic access)

```bash
python3 trading_platform_api.py
```

Access API documentation at: http://localhost:8000/docs

---

## 📖 Using the Platform

### 1. Dashboard

- View overall platform statistics
- See top-performing strategies
- Monitor recent backtests
- Track paper trading performance

### 2. Generate AI Strategies

1. Navigate to "Generate Strategies" page
2. Enter stock tickers (e.g., AAPL, MSFT, GOOGL)
3. Select historical data period
4. Choose number of strategies to generate (1-10)
5. Enable "Use Past Performance" for AI learning
6. Click "Generate Strategies"

**What happens:**
- AI analyzes market data
- Reviews historical strategy performance
- Generates unique trading strategies with:
  - Entry/exit conditions
  - Technical indicators
  - Risk management rules
  - Market analysis

### 3. Backtest Strategies

1. Navigate to "Backtest" page
2. Select a strategy from the dropdown
3. Set initial capital
4. Click "Run Backtest"

**Results include:**
- Total return %
- Sharpe ratio
- Win rate
- Maximum drawdown
- Equity curve chart
- Drawdown analysis
- Trade distribution
- Complete trade history

### 4. Paper Trading

1. Navigate to "Paper Trading" page
2. Select a strategy
3. Click "Execute Strategy"

**Features:**
- Tests strategies with live market data
- No real money at risk
- Tracks positions and P&L
- Real-time performance metrics

### 5. Portfolio Optimization

1. Navigate to "Portfolio Optimizer" page
2. Select multiple strategies
3. Set total capital
4. Choose optimization method:
   - **Maximize Sharpe Ratio** (best risk-adjusted returns)
   - **Minimize Variance** (lowest volatility)
   - **Maximize Return** (highest returns)
   - **Risk Parity** (equal risk distribution)
5. Click "Optimize Portfolio"

**Results include:**
- Optimal allocation percentages
- Expected return and volatility
- Capital allocations per strategy
- Portfolio allocation chart

### 6. AI Learning

1. Navigate to "AI Learning" page
2. Click "Analyze & Learn from Recent Results"

**AI extracts:**
- Success patterns from winning strategies
- Failure patterns to avoid
- Parameter insights
- Market observations
- Recommendations for future strategies

---

## 🔌 API Endpoints

The platform exposes a comprehensive REST API at http://localhost:8000

### Key Endpoints:

#### Strategy Generation
- `POST /strategies/generate` - Generate AI strategies
- `GET /strategies` - List all strategies
- `GET /strategies/{id}` - Get strategy details

#### Backtesting
- `POST /backtest` - Run backtest
- `GET /backtest/results` - List backtest results
- `GET /backtest/results/{id}` - Get detailed results

#### Paper Trading
- `POST /paper-trading/execute` - Execute paper trade
- `GET /paper-trading/positions` - View current positions
- `GET /paper-trading/performance` - Get performance summary

#### Portfolio Optimization
- `POST /portfolio/optimize` - Optimize portfolio

#### AI Learning
- `POST /ai/learn` - Extract learning insights
- `GET /ai/learning` - View historical insights

#### Analytics
- `GET /analytics/dashboard` - Dashboard analytics

**Interactive API Documentation:** http://localhost:8000/docs

---

## 💡 Example Workflow

### Complete Trading Strategy Workflow:

1. **Generate Strategies**
   - Go to "Generate Strategies"
   - Enter: `AAPL, MSFT, NVDA, AMD`
   - Period: `6mo`
   - Generate 3 strategies

2. **Backtest All Strategies**
   - Go to "Backtest"
   - Test each generated strategy
   - Compare performance metrics
   - Identify best-performing strategy

3. **Paper Trade Best Strategy**
   - Go to "Paper Trading"
   - Select top-performing strategy
   - Execute paper trades
   - Monitor real-time performance

4. **Optimize Portfolio**
   - Go to "Portfolio Optimizer"
   - Select top 3 strategies
   - Optimize with Sharpe ratio method
   - Get optimal allocation

5. **Learn & Improve**
   - Go to "AI Learning"
   - Run analysis
   - Review insights
   - Generate new strategies based on learnings

---

## 🧪 Testing the Setup

### Test API Server:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-22T..."
}
```

### Test Strategy Generation (via API):

```bash
curl -X POST http://localhost:8000/strategies/generate \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "period": "3mo",
    "num_strategies": 2,
    "use_past_performance": false
  }'
```

---

## 🛠️ Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements-trading-platform.txt
```

### Issue: "OpenAI API Error 401"

**Solution:**
- Check your `.env` file has the correct API key
- Verify the API key is valid at https://platform.openai.com/api-keys
- Ensure you have GPT-4 access

### Issue: "Database locked"

**Solution:**
```bash
# Stop all running instances
pkill -f trading_platform_api
pkill -f streamlit

# Delete database and restart
rm trading_platform.db
python3 trading_platform_api.py
```

### Issue: "Port 8000 already in use"

**Solution:**
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or change the port in trading_platform_api.py:
# uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Issue: "No market data available"

**Solution:**
- Check internet connection
- Verify ticker symbols are valid
- Try a different time period
- Yahoo Finance may have temporary issues

---

## 📊 Understanding the Metrics

### Sharpe Ratio
- Measures risk-adjusted returns
- **> 1.0:** Excellent
- **0.5 - 1.0:** Good
- **< 0.5:** Poor

### Win Rate
- Percentage of profitable trades
- **> 60%:** Strong
- **40-60%:** Average
- **< 40%:** Weak

### Maximum Drawdown
- Largest peak-to-trough decline
- **< 10%:** Conservative
- **10-20%:** Moderate
- **> 20%:** Aggressive

### Quality Score (0-100)
- Composite score based on:
  - Sharpe ratio (25%)
  - Win rate (25%)
  - Drawdown (25%)
  - Total return (25%)
- **> 70:** Excellent
- **50-70:** Good
- **< 50:** Needs improvement

---

## 🔒 Security Best Practices

1. **Never commit `.env` file to git**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Keep API key secure**
   - Don't share your OpenAI API key
   - Rotate keys periodically
   - Set usage limits in OpenAI dashboard

3. **Database backups**
   ```bash
   # Backup database
   cp trading_platform.db trading_platform_backup_$(date +%Y%m%d).db
   ```

4. **For production deployment:**
   - Use PostgreSQL instead of SQLite
   - Add authentication/authorization
   - Use HTTPS
   - Set up rate limiting

---

## 💰 Cost Estimation

### OpenAI API Costs (GPT-4):

- **Strategy Generation:** ~$0.05-0.10 per run (3 strategies)
- **Learning Analysis:** ~$0.03-0.06 per analysis
- **Recommendations:** ~$0.02-0.04 per request

**Typical Monthly Cost (light usage):**
- 4 strategy generations/month: $0.40
- 4 learning analyses/month: $0.24
- **Total: ~$0.50-1.00/month**

**Typical Monthly Cost (heavy usage):**
- 20 strategy generations/month: $2.00
- 20 learning analyses/month: $1.20
- **Total: ~$3-5/month**

### Infrastructure:
- **Compute:** Free (runs locally)
- **Data:** Free (yfinance)
- **Database:** Free (SQLite)

---

## 🚀 Advanced Configuration

### Use PostgreSQL Database:

1. Install PostgreSQL
2. Create database:
   ```sql
   CREATE DATABASE trading_platform;
   ```
3. Update `.env`:
   ```
   DATABASE_URL=postgresql://username:password@localhost/trading_platform
   ```

### Run with Docker:

```bash
# Build image
docker build -t trading-platform .

# Run container
docker run -p 8000:8000 -p 8501:8501 \
  -e OPENAI_API_KEY=your_key_here \
  trading-platform
```

### Schedule Automatic Strategy Generation:

Use cron (macOS/Linux):
```bash
# Edit crontab
crontab -e

# Add line to generate strategies every Monday at 9 AM:
0 9 * * 1 cd /Users/alanphilip/Downloads/yfinance && python3 -c "import requests; requests.post('http://localhost:8000/strategies/generate', json={'tickers': ['AAPL','MSFT','GOOGL'], 'period': '6mo', 'num_strategies': 3})"
```

---

## 📚 Additional Resources

- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **Streamlit Documentation:** https://docs.streamlit.io
- **yfinance Documentation:** https://pypi.org/project/yfinance/
- **OpenAI API Documentation:** https://platform.openai.com/docs

---

## 🐛 Getting Help

If you encounter issues:

1. Check this documentation first
2. Review the troubleshooting section
3. Check API logs in terminal
4. Verify all dependencies are installed
5. Ensure OpenAI API key is valid

---

## ✅ Checklist

Before running the platform, ensure:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] `.env` file created with OpenAI API key
- [ ] Database initialized
- [ ] Port 8000 and 8501 available
- [ ] Internet connection active

---

## 🎉 You're Ready!

Start the platform:

```bash
# Terminal 1
python3 trading_platform_api.py

# Terminal 2
streamlit run streamlit_app.py
```

Open your browser to: **http://localhost:8501**

**Happy Trading! 📈**

---

**Version:** 2.0.0
**Last Updated:** November 2025
**Platform:** macOS, Linux, Windows
