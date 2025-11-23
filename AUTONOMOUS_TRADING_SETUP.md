# 🤖 Autonomous Trading System Setup Guide

This guide will help you set up the fully autonomous paper trading system that makes decisions and learns without human input.

## Phase 1: Get Alpaca Paper Trading Account (5 minutes)

### Step 1: Create Account
1. Go to: https://app.alpaca.markets/signup
2. Sign up for a FREE account
3. Select **Paper Trading** (not real money)
4. Complete verification

### Step 2: Generate API Keys
1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Click "**View**" next to API Keys
3. Click "**Generate New Keys**"
4. **IMPORTANT**: Copy both keys immediately:
   - API Key (starts with `PK...`)
   - Secret Key (long alphanumeric)

### Step 3: Add Keys to .env File
Open `/Users/alanphilip/Downloads/yfinance/.env` and update:

```bash
# Replace these with your actual keys
ALPACA_API_KEY=PK... # Your API key here
ALPACA_SECRET_KEY=... # Your secret key here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Enable autonomous trading
AUTO_TRADING_ENABLED=false  # Keep false until you're ready!
```

## Phase 2: Test the System (10 minutes)

### Test 1: Test Alpaca Connection
```bash
cd /Users/alanphilip/Downloads/yfinance
python3 alpaca_client.py
```

You should see:
```
✅ Portfolio Value: $100,000.00
Cash: $100,000.00
Market Open: true/false
```

### Test 2: Initialize Database
```bash
python3 -c "from database import init_db; init_db(); print('✅ Database initialized')"
```

### Test 3: Test Autonomous Engine (DRY RUN)
```bash
python3 autonomous_trading_engine.py
```

This will run the full cycle but **NOT execute trades** (because AUTO_TRADING_ENABLED=false).

You should see:
```
🤖 Autonomous Trading Engine initialized
   Auto-trading: DISABLED
   ...
🔍 Scanning market for trading signals...
   Found X potential signals
```

## Phase 3: Understanding the System

### How It Works

**Daily Cycle** (runs once per day):
```
1. Check if enabled ✅
   ↓
2. Check circuit breakers (max loss, market hours) 🛑
   ↓
3. Sync positions from Alpaca 📊
   ↓
4. Scan market with all active strategies 🔍
   ↓
5. Evaluate signals (filter by confidence, performance) 🧠
   ↓
6. Calculate position sizes (based on strategy performance) 💰
   ↓
7. Execute trades via Alpaca API 🚀
   ↓
8. Update performance tracking 📈
   ↓
9. Learn from results (deprecate bad strategies) 🤖
```

### Safety Features

**Circuit Breakers** (auto-stop trading):
- Daily loss > 5% → Stop trading for the day
- Market closed → No trading
- Max 10 positions → Prevent over-diversification

**Position Sizing**:
- Default: 25% of portfolio per position
- Adjusted by strategy live performance
- Capped at 20% max

**Risk Management**:
- Every trade has stop loss
- Every trade has take profit
- Only HIGH confidence signals (unless changed)

## Phase 4: Enable Auto-Trading (When Ready!)

### ⚠️ IMPORTANT: Only enable after testing!

1. Update `.env`:
```bash
AUTO_TRADING_ENABLED=true
```

2. Run manually first:
```bash
python3 autonomous_trading_engine.py
```

3. Watch it execute trades in paper account

4. Check Alpaca dashboard:
   https://app.alpaca.markets/paper/dashboard/overview

## Phase 5: Automation (Next Step)

Currently the system runs manually. To make it truly autonomous:

**Option 1: cron job (Linux/Mac)**
```bash
# Run every day at 10:00 AM ET (after market opens)
0 10 * * 1-5 cd /Users/alanphilip/Downloads/yfinance && python3 autonomous_trading_engine.py
```

**Option 2: systemd timer (Linux)**
**Option 3: Heroku Scheduler (Cloud)**
**Option 4: APScheduler (Python-based)**

We'll set this up in the next phase!

## Configuration Options

Edit `.env` to customize:

```bash
# Trading Rules
MAX_POSITION_SIZE_PCT=20        # Max % of portfolio per position
MAX_DAILY_LOSS_PCT=5            # Stop trading if lose > 5% in a day
MAX_PORTFOLIO_POSITIONS=10      # Max number of concurrent positions
MIN_SIGNAL_CONFIDENCE=HIGH      # Only take HIGH confidence signals

# Learning Settings
STRATEGY_REOPTIMIZE_DAYS=7      # Re-optimize parameters every 7 days
MIN_TRADES_FOR_EVALUATION=10    # Need 10 trades to evaluate strategy
PERFORMANCE_THRESHOLD_SHARPE=0.8 # Deprecate if Sharpe < 0.8
AUTO_DEPRECATE_STRATEGIES=true  # Auto-disable bad strategies
```

## Monitoring the System

### View Active Positions
```python
from alpaca_client import AlpacaClient
client = AlpacaClient()
positions = client.get_positions()
print(positions)
```

### View Trade History
```python
from database import SessionLocal, TradeExecution
db = SessionLocal()
trades = db.query(TradeExecution).all()
for trade in trades:
    print(f"{trade.ticker}: {trade.signal_type} @ ${trade.signal_price}")
```

### View Strategy Performance
```python
from database import SessionLocal, StrategyPerformance
db = SessionLocal()
perf = db.query(StrategyPerformance).all()
for p in perf:
    print(f"{p.strategy_name}: Live Sharpe={p.live_sharpe}, Weight={p.allocation_weight}")
```

## Next Steps (Advanced Features)

1. **Task Scheduler** - Auto-run daily without manual intervention
2. **Performance Analyzer** - Compare live vs backtest, auto-adjust weights
3. **Strategy Re-optimizer** - Re-optimize parameters weekly
4. **Reinforcement Learning** - Learn which signals to take
5. **Auto Strategy Generator** - Generate new strategies automatically

## Troubleshooting

### Error: "Alpaca API key not configured"
- Check `.env` file has your actual API keys
- Make sure keys start with `PK` for paper trading

### Error: "Market is closed"
- System only trades when market is open (9:30 AM - 4:00 PM ET, Mon-Fri)
- This is a safety feature

### Error: "No active strategies found"
- Create strategies using the "Generate Strategies" page in Streamlit
- Make sure strategies have backtest results

### No trades executing
- Check `AUTO_TRADING_ENABLED=true` in .env
- Check `MIN_SIGNAL_CONFIDENCE` - might be too restrictive
- Check if strategies are generating signals (run Market Scanner manually)

## Questions?

This is a complex system! It's normal to have questions. Start with Phase 1 and Phase 2, then test extensively before enabling auto-trading.
