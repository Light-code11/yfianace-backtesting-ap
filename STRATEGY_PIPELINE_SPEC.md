# AI Strategy Discovery & Optimization Pipeline

## Overview
Two-part system:
1. **Optimizer** — retune existing 5 strategies on 2 years of data, find best parameters
2. **AI Discovery** — use AI to research, generate, backtest, and deploy new strategies

## Part 1: Strategy Parameter Optimizer

### File: `strategy_optimizer.py`

Run the vectorized backtester across all 5 strategies with parameter grids. Find optimal params for current market.

```python
PARAM_GRIDS = {
    "trend_momentum": {
        "ema_fast": [10, 15, 20, 25],
        "ema_slow": [40, 50, 60, 75],
        "adx_threshold": [20, 25, 30],
    },
    "mean_reversion_rsi2": {
        "rsi_period": [2, 3, 5],
        "rsi_oversold": [5, 10, 15],
        "rsi_overbought": [85, 90, 95],
        "sma_trend": [100, 150, 200],
    },
    "breakout_momentum": {
        "breakout_period": [10, 15, 20, 30],
        "volume_multiplier": [1.2, 1.5, 2.0],
    },
    "vix_adaptive_momentum": {
        "momentum_period": [63, 126, 189],
        "vol_lookback": [15, 20, 30],
    },
    "overnight_anomaly": {
        "volume_threshold": [1.0, 1.2, 1.5],
        "min_gap_pct": [0.0, 0.1, 0.2],
    },
}
```

For each strategy × ticker combination:
1. Download 2y of data
2. Grid search all param combos
3. Calculate: Sharpe, total return, win rate, max DD, trade count
4. Pick params with best Sharpe (minimum 10 trades, positive return)
5. Save optimized params to `optimized_params.json`
6. Update `trading_config.py` with new params

Run on top 30 most liquid tickers (not all 119 — too slow).

### Acceptance criteria for optimized params:
- Sharpe > 0.5
- Win rate > 40%
- Max drawdown < 20%
- Min 10 trades in 2y
- Out-of-sample validation: train on first 18 months, test on last 6 months. Must be profitable in both.

### Output: `optimized_params.json`
```json
{
    "trend_momentum": {
        "ema_fast": 15,
        "ema_slow": 50,
        "adx_threshold": 25,
        "sharpe": 1.2,
        "win_rate": 0.52,
        "tickers_profitable": ["AAPL", "MSFT", "NVDA", ...],
        "tickers_unprofitable": ["NU", "COIN", ...],
    },
    ...
}
```

## Part 2: AI Strategy Discovery Pipeline

### File: `strategy_discovery.py`

Uses the existing `ai_strategy_generator.py` (GPT-4o) but adds:

1. **Market Research Phase**
   - Fetch current market conditions (regime, VIX, sector performance)
   - Download recent quant research summaries (web search)
   - Analyze which sectors/factors are working (momentum, value, quality, etc.)

2. **Strategy Generation Phase**
   - Prompt GPT-4o with market context + what we've already tried + what worked/didn't
   - Ask for 5 new strategy ideas with:
     - Clear entry/exit rules (must be implementable with yfinance data)
     - Expected holding period
     - Target Sharpe ratio
     - Which tickers it should work best on
   - Generate Python code for each strategy's signal function

3. **Auto-Backtest Phase**
   - For each generated strategy:
     - Parse the AI-generated signal function
     - Run it on 2 years of data, top 30 tickers
     - Walk-forward: train on 18mo, test on 6mo
     - Calculate all metrics

4. **Deployment Phase**
   - Strategies passing ALL gates get added to `trading_config.py`
   - Strategies are tagged as `ai_generated=True` with generation date
   - Max 3 new AI strategies deployed per week (don't flood)
   - Each new strategy starts with 50% position sizing (paper trade first)

### AI Strategy Prompt Template
```
You are a quantitative trading researcher. Generate {n} trading strategies for US equities.

Current market conditions:
- Macro regime: {regime} 
- VIX: {vix}
- Sector leaders: {sectors}
- Market breadth: {breadth}

Previously tried strategies and their results:
{past_results}

Requirements:
- Must use ONLY data available from yfinance (OHLCV, no alternative data)
- Must have clear, unambiguous entry and exit rules
- Must be implementable as a Python function that takes a pandas DataFrame and returns buy/sell signals
- Target Sharpe > 1.0 on 2-year backtest
- Prefer strategies that work in the current regime

For each strategy, provide:
1. Name and description
2. Entry conditions (specific indicator values, thresholds)
3. Exit conditions (stop loss, take profit, signal reversal)
4. Python signal function code
5. Recommended tickers (5-15)
6. Expected holding period
7. Why this should work in current conditions
```

### Integration with daily cycle

Add to `run_daily_trading.py`:
- Weekly (Sundays): run full optimization + AI discovery
- Daily: just use the current optimized params
- Track AI strategy performance separately in DB

### New DB table: `strategy_research`
```sql
CREATE TABLE strategy_research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy_name TEXT,
    source TEXT,  -- 'optimizer' or 'ai_generated'
    params_json TEXT,
    backtest_sharpe REAL,
    backtest_return_pct REAL,
    backtest_win_rate REAL,
    backtest_max_dd REAL,
    backtest_trade_count INTEGER,
    walk_forward_sharpe REAL,
    walk_forward_return_pct REAL,
    is_deployed BOOLEAN DEFAULT FALSE,
    deployment_date TIMESTAMP,
    notes TEXT
);
```

## Execution Plan
1. First: Run optimizer on existing 5 strategies (immediate)
2. Second: Build the AI discovery pipeline
3. Third: Wire into weekly cron

## Important Notes
- Use `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` for Windows
- yfinance data only — no paid APIs
- OpenAI API key is in `.env` as `OPENAI_API_KEY`
- Keep all optimization results logged for learning
- Walk-forward validation is NON-NEGOTIABLE — no strategy goes live without it
- Position sizing for new AI strategies: 50% of normal until proven (2 weeks profitable)
