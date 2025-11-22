# 🤖 AI Trading Strategy Platform

> **AI-powered trading strategy generator with real-time backtesting, paper trading, and portfolio optimization**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 🌟 Features

### ✨ AI Strategy Generation
- **GPT-4 Powered:** Generates sophisticated trading strategies based on market analysis
- **Learning Mechanism:** AI improves over time by analyzing past performance
- **Multiple Strategy Types:** Momentum, mean reversion, breakout, trend following
- **Custom Indicators:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX

### 📊 Advanced Backtesting
- **Comprehensive Metrics:** Total return, Sharpe ratio, Sortino ratio, win rate, max drawdown
- **Technical Indicators:** 8+ built-in technical indicators
- **Risk Management:** Automatic stop-loss and take-profit execution
- **Trade Analysis:** Complete trade history with entry/exit details

### 📝 Paper Trading
- **Live Market Data:** Test strategies with real-time prices without risk
- **Position Tracking:** Monitor open positions and P&L
- **Automatic Execution:** Signal detection and order execution
- **Performance Monitoring:** Real-time portfolio valuation

### 💼 Portfolio Optimization
- **4 Optimization Methods:**
  - Maximize Sharpe Ratio (best risk-adjusted returns)
  - Minimize Variance (lowest volatility)
  - Maximize Return (highest returns)
  - Risk Parity (equal risk distribution)
- **Capital Allocation:** Optimal allocation across multiple strategies
- **Expected Metrics:** Return, volatility, and Sharpe ratio projections

### 🧠 AI Learning System
- **Pattern Recognition:** Identifies successful strategy patterns
- **Failure Analysis:** Learns from unsuccessful strategies
- **Continuous Improvement:** Recommendations for future strategy generation
- **Insight Tracking:** Historical learning database

### 🎨 Interactive Web Interface
- **Modern Dashboard:** Real-time analytics and performance metrics
- **Visual Charts:** Equity curves, drawdown analysis, allocation charts
- **Easy Navigation:** Intuitive interface for all features
- **Responsive Design:** Works on desktop and mobile

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd /Users/alanphilip/Downloads/yfinance
pip install -r requirements-trading-platform.txt
```

### 2. Configure OpenAI API Key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Start Platform
```bash
./start_trading_platform.sh
```

**That's it!** Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
yfinance/
├── database.py                    # Database models and configuration
├── ai_strategy_generator.py      # AI strategy generation engine
├── backtesting_engine.py          # Backtesting with technical indicators
├── paper_trading.py               # Live paper trading simulator
├── portfolio_optimizer.py         # Portfolio optimization algorithms
├── trading_platform_api.py        # FastAPI REST API server
├── streamlit_app.py               # Web interface
├── requirements-trading-platform.txt
├── TRADING_PLATFORM_SETUP.md      # Detailed setup guide
├── start_trading_platform.sh      # Startup script
└── .env.example                   # Environment variables template
```

---

## 🎯 Usage Examples

### Generate Strategies via Web Interface

1. Open http://localhost:8501
2. Navigate to "Generate Strategies"
3. Enter tickers: `AAPL, MSFT, NVDA, AMD`
4. Select period: `6mo`
5. Click "Generate Strategies"

### Generate Strategies via API

```bash
curl -X POST http://localhost:8000/strategies/generate \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "period": "6mo",
    "num_strategies": 3,
    "use_past_performance": true
  }'
```

### Backtest a Strategy

```python
import requests

response = requests.post('http://localhost:8000/backtest', json={
    "strategy_id": 1,
    "initial_capital": 100000
})

results = response.json()
print(f"Total Return: {results['metrics']['total_return_pct']}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']}")
```

### Optimize Portfolio

```python
import requests

response = requests.post('http://localhost:8000/portfolio/optimize', json={
    "strategy_ids": [1, 2, 3],
    "total_capital": 100000,
    "method": "sharpe"
})

allocation = response.json()
print(f"Expected Return: {allocation['expected_return']}%")
print(f"Allocations: {allocation['allocations']}")
```

---

## 📊 Performance Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Total Return** | Overall percentage gain/loss | > 15% annually |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Win Rate** | Percentage of winning trades | > 55% |
| **Max Drawdown** | Largest peak-to-trough decline | < 15% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Quality Score** | Composite score (0-100) | > 70 |

---

## 🔌 API Endpoints

### Strategies
- `POST /strategies/generate` - Generate AI strategies
- `GET /strategies` - List all strategies
- `GET /strategies/{id}` - Get strategy details

### Backtesting
- `POST /backtest` - Run backtest
- `GET /backtest/results` - List results
- `GET /backtest/results/{id}` - Get detailed results

### Paper Trading
- `POST /paper-trading/execute` - Execute paper trade
- `GET /paper-trading/positions` - View positions
- `GET /paper-trading/performance` - Performance summary

### Portfolio
- `POST /portfolio/optimize` - Optimize portfolio

### AI Learning
- `POST /ai/learn` - Extract insights
- `GET /ai/learning` - View insights

**Full API Documentation:** http://localhost:8000/docs

---

## 💰 Cost Breakdown

### OpenAI API (GPT-4)
- Strategy Generation: ~$0.05-0.10 per run
- Learning Analysis: ~$0.03-0.06 per run
- **Monthly (light use):** ~$1-2
- **Monthly (heavy use):** ~$3-5

### Infrastructure
- **Data:** Free (yfinance)
- **Database:** Free (SQLite)
- **Compute:** Free (runs locally)

**Total Monthly Cost:** $1-5 depending on usage

---

## 🔒 Security & Best Practices

### Environment Variables
- Never commit `.env` file
- Rotate API keys regularly
- Set OpenAI usage limits

### Database Backups
```bash
# Backup database
cp trading_platform.db backups/trading_platform_$(date +%Y%m%d).db
```

### Production Deployment
- Use PostgreSQL instead of SQLite
- Add authentication/authorization
- Enable HTTPS
- Implement rate limiting
- Use environment-specific configs

---

## 🛠️ Troubleshooting

### Common Issues

**OpenAI API Error:**
```
Check .env file has valid OPENAI_API_KEY
```

**Port Already in Use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Module Not Found:**
```bash
pip install -r requirements-trading-platform.txt
```

**No Market Data:**
```
Verify ticker symbols and internet connection
Try different time period
```

See [TRADING_PLATFORM_SETUP.md](TRADING_PLATFORM_SETUP.md) for detailed troubleshooting.

---

## 📈 Example Workflow

1. **Generate 3 AI strategies** for tech stocks (AAPL, MSFT, NVDA)
2. **Backtest all strategies** over 1 year period
3. **Select top 2 performers** (Sharpe > 1.0)
4. **Optimize portfolio** with $100k capital
5. **Paper trade** the optimized portfolio
6. **Monitor and adjust** based on performance
7. **AI learns** from results for next generation

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional technical indicators
- More optimization algorithms
- Real-time alerts and notifications
- Integration with broker APIs
- Machine learning models
- Sentiment analysis
- Multi-asset support

---

## ⚠️ Disclaimer

**IMPORTANT:** This platform is for educational and research purposes only.

- Not financial advice
- Past performance doesn't guarantee future results
- All trading involves risk of loss
- Backtest results may not reflect real-world execution
- Always do your own research
- Consult a financial advisor before investing

---

## 📚 Tech Stack

- **Backend:** FastAPI, SQLAlchemy, Python 3.9+
- **AI:** OpenAI GPT-4
- **Data:** yfinance (Yahoo Finance API)
- **Frontend:** Streamlit
- **Visualization:** Plotly
- **Optimization:** SciPy
- **Database:** SQLite (PostgreSQL supported)

---

## 📞 Support

- **Documentation:** [TRADING_PLATFORM_SETUP.md](TRADING_PLATFORM_SETUP.md)
- **API Docs:** http://localhost:8000/docs
- **Issues:** Check troubleshooting section

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎉 Acknowledgments

- **yfinance** for market data
- **OpenAI** for GPT-4
- **FastAPI** for the web framework
- **Streamlit** for the UI framework
- **Plotly** for visualizations

---

## 🚀 Get Started Now!

```bash
# Clone or download the project
cd /Users/alanphilip/Downloads/yfinance

# Install dependencies
pip install -r requirements-trading-platform.txt

# Set up OpenAI API key
cp .env.example .env
# Edit .env and add your key

# Start the platform
./start_trading_platform.sh

# Open in browser
open http://localhost:8501
```

**Happy Trading! 📈🤖**

---

**Version:** 2.0.0
**Last Updated:** November 2025
**Author:** AI Trading Platform Team
