# 🧬 Auto Strategy Generator Guide

## What This Does

The **Auto Strategy Generator** is an evolutionary learning system that automatically:

1. **Generates** random trading strategy variations
2. **Tests** them on 2 years of historical data
3. **Selects** only the best performers (high Sharpe, good win rate)
4. **Deploys** winners to your live trading system
5. **Monitors** their performance
6. **Repeats** weekly to keep finding better strategies

**You literally never have to create strategies manually again!**

---

## How It Works

### Step 1: Generate Random Strategies

The system creates strategies by randomly combining:

- **Strategy Types**: momentum, mean_reversion, breakout, trend_following
- **Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR (2-4 per strategy)
- **Tickers**: Random selection of 2-4 stocks from universe
- **Risk Management**: Stop loss 3-15%, take profit 1.5-3x stop loss
- **Position Size**: 10-30% of portfolio
- **Holding Period**: 5-60 days

**Example Generated Strategy:**
```
Name: Breakout - AutoGen_1
Type: breakout
Tickers: ['NVDA', 'AMD', 'GOOGL']
Indicators:
  - SMA(50)
  - RSI(14)
  - ATR(14)
Risk: Stop 8%, Target 20%, Size 25%
```

### Step 2: Backtest Everything

Each strategy is backtested on:
- **Period**: Last 2 years
- **Capital**: $100,000 initial
- **Metrics tracked**: Sharpe ratio, win rate, drawdown, profit factor

### Step 3: Select Winners

Only strategies that meet ALL criteria are deployed:
- ✅ **Sharpe Ratio > 1.5** (risk-adjusted returns)
- ✅ **Total Trades > 30** (statistical significance)
- ✅ **Max Drawdown < 20%** (risk control)
- ✅ **Win Rate > 45%** (consistency)

### Step 4: Deploy to Live Trading

Winners are automatically:
- Saved to database
- Marked as **active**
- Added to Market Scanner
- Used by Autonomous Trading Engine
- Tracked for performance vs backtest

### Step 5: Monitor & Evolve

The system tracks:
- Live Sharpe vs backtest Sharpe
- If live performance drops → strategy is deprecated
- Top performers get higher allocation weights
- New variations are generated from best strategies

---

## Configuration

Edit `.env` to customize:

```bash
# Enable/disable auto-generation
AUTO_GENERATE_STRATEGIES=false  # Set true to enable

# How often to run (days)
AUTO_GENERATION_FREQUENCY_DAYS=7  # Weekly

# How many strategies to test each time
STRATEGIES_PER_BATCH=20  # More = better chance of finding winners

# Quality threshold
MIN_SHARPE_FOR_DEPLOYMENT=1.5  # Higher = stricter (fewer but better strategies)

# How many to deploy
TOP_N_STRATEGIES_TO_DEPLOY=3  # Deploy top 3 winners
```

---

## Running Manually

### Test with Small Batch (Quick)
```bash
python3 -c "
from auto_strategy_generator import AutoStrategyGenerator
gen = AutoStrategyGenerator()
results = gen.run_full_cycle(
    num_strategies=5,    # Only 5 for speed
    min_sharpe=1.0,      # Lower threshold
    top_n=2              # Deploy top 2
)
print(results)
"
```

### Full Production Run
```bash
python3 -c "
from auto_strategy_generator import AutoStrategyGenerator
gen = AutoStrategyGenerator()
results = gen.run_full_cycle(
    num_strategies=20,   # Full batch
    min_sharpe=1.5,      # Production threshold
    top_n=3              # Deploy top 3
)
print(results)
"
```

---

## What Gets Generated

Each winning strategy includes:

```python
{
    "name": "Momentum - AutoGen_5",
    "strategy_type": "momentum",
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "indicators": [
        {"name": "SMA", "period": 20},
        {"name": "RSI", "period": 14},
        {"name": "MACD"}
    ],
    "risk_management": {
        "stop_loss_pct": 7.5,
        "take_profit_pct": 18.0,
        "position_size_pct": 25.0
    },
    "backtest_results": {
        "sharpe_ratio": 2.1,
        "win_rate": 58.5,
        "total_trades": 45,
        "max_drawdown_pct": -12.3
    }
}
```

---

## Integration with Autonomous Trading

Once deployed, auto-generated strategies:

1. **Appear in Market Scanner** - Used to find signals
2. **Appear in Live Signals** - Generate BUY/SELL recommendations
3. **Used by Autonomous Engine** - Actually execute trades
4. **Tracked for Performance** - Compared live vs backtest

**The full autonomous loop:**
```
Generate Strategies (Weekly)
    ↓
Backtest & Select Winners
    ↓
Deploy to Active Pool
    ↓
Market Scanner Uses Them (Daily)
    ↓
Autonomous Engine Executes (Daily)
    ↓
Track Performance (Daily)
    ↓
Deprecate Bad Strategies (Weekly)
    ↓
Generate New Variations (Weekly)
    ↓
[Repeat Forever]
```

---

## Expected Results

### Realistic Expectations:

**First Run** (20 strategies tested):
- ✅ 2-4 strategies will meet criteria
- ✅ Deploy top 3
- ⏰ Takes 10-15 minutes

**After 1 Month**:
- ✅ 10-15 active strategies
- ✅ Mix of auto-generated + your manual ones
- ✅ System learning which types work best

**After 3 Months**:
- ✅ 20-30 active strategies
- ✅ Underperformers auto-deprecated
- ✅ Portfolio diversified across strategy types
- ✅ Better strategies emerging from variations

### Performance Tracking:

The system tracks:
```
Strategy: "Breakout - AutoGen_12"
Backtest Sharpe: 1.8
Live Sharpe: 1.6 ← Still good!
Performance Delta: -11% ← Acceptable
Status: ACTIVE ✅

Strategy: "Momentum - AutoGen_3"
Backtest Sharpe: 1.5
Live Sharpe: 0.4 ← Degraded!
Performance Delta: -73% ← Failed
Status: DEPRECATED ❌
```

---

## Safety Features

1. **Strict Testing**
   - 2 years of backtesting
   - Minimum 30 trades required
   - Must pass all quality checks

2. **Risk Limits**
   - Max drawdown capped at 20%
   - Stop losses always enforced
   - Position size limits

3. **Performance Monitoring**
   - Live vs backtest comparison
   - Auto-deprecation of failures
   - Allocation weight adjustment

4. **Human Override**
   - You can view all generated strategies
   - Manually enable/disable any strategy
   - Adjust parameters anytime

---

## Troubleshooting

### "No strategies met criteria"
- Lower `MIN_SHARPE_FOR_DEPLOYMENT` (try 1.0)
- Increase `STRATEGIES_PER_BATCH` (try 30-40)
- Check if market conditions are difficult

### Takes too long
- Reduce `STRATEGIES_PER_BATCH` (try 10)
- Backtesting 20 strategies takes ~15 minutes

### All strategies look similar
- This is normal at first
- After few weeks, you'll see more diversity
- Best strategies are used as templates for variations

---

## Next Steps

1. **Run First Batch**: Test with 5 strategies, low threshold
2. **Review Winners**: Check what got deployed
3. **Enable Auto-Generation**: Set `AUTO_GENERATE_STRATEGIES=true`
4. **Schedule Weekly**: Use cron/scheduler to run every Sunday
5. **Monitor Results**: Track strategy performance dashboard

---

## Advanced: Evolutionary Learning

The system learns over time by:

1. **Tracking Winners**: Which strategy types work best?
2. **Creating Variations**: Mutate parameters of top performers
3. **Cross-Breeding**: Combine elements from different winners
4. **Pruning Losers**: Remove underperformers

Example evolution:
```
Week 1: Random strategies → 3 momentum winners
Week 2: Generate variations of momentum → 2 better momentum strategies
Week 3: Cross momentum + breakout → New hybrid strategy
Week 4: Prune old strategies, keep evolving top performers
```

After 3 months, your strategy pool will look VERY different from the initial random batch - and should perform better!

---

## Questions?

This is cutting-edge algorithmic trading! The system basically runs a hedge fund strategy discovery process 24/7. Start conservatively, test thoroughly, then enable full automation when comfortable.
