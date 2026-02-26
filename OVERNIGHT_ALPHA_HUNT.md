# Overnight Alpha Hunt — Find Profitable Strategies

## Mission
Find AT LEAST 3 profitable, backtested, walk-forward validated strategies by morning. No bullshit. Real edge.

## What's Wrong With Our Current Approach
1. Basic technical indicators (RSI, MACD, EMA crossovers) are well-known — the edge is arbed away
2. We're testing on mega-cap tech (AAPL, MSFT) — most efficient market on earth, hardest to find alpha
3. No factor-based strategies (the actual proven source of alpha)
4. No cross-sectional strategies (ranking stocks against each other)
5. No regime-aware position sizing
6. Only daily timeframe

## Strategy Ideas To Research & Backtest

### Tier 1: Academically Proven (highest confidence)
1. **Cross-Sectional Momentum (Jegadeesh & Titman 1993)**
   - Buy top 20% performers of last 6-12 months, sell bottom 20%
   - Rebalance monthly
   - This has worked for 200+ years across every market
   - Use our 119 tickers, rank by 6m return, buy top quintile

2. **Short-Term Mean Reversion (Lehmann 1990)**
   - Stocks that dropped most in past week tend to bounce
   - Buy bottom 10% of weekly performers, sell after 5 days
   - Works best on liquid large-caps (our universe)

3. **Post-Earnings Announcement Drift (PEAD)**
   - Stocks that beat earnings estimates drift up for 60 days
   - We already have earnings data — use it!
   - Buy on beat, hold 20-40 trading days

4. **Low Volatility Anomaly (Ang et al. 2006)**
   - Low-vol stocks outperform high-vol stocks risk-adjusted
   - Sort by 60-day realized vol, buy bottom quintile
   - Monthly rebalance

5. **Quality Factor (Novy-Marx 2013)**
   - High gross profitability stocks outperform
   - Use yfinance fundamentals: gross_profit / total_assets
   - Quarterly rebalance

### Tier 2: Practitioner Proven
6. **Monthly Seasonality**
   - Some months historically better for certain sectors
   - "Sell in May" but more sophisticated — sector rotation by month
   
7. **Gap Fade Strategy**
   - Stocks gapping down >2% at open with no news catalyst tend to recover
   - Need intraday data (use yfinance interval="1h")

8. **Bollinger Band Squeeze → Breakout**
   - When BB width contracts to 6-month low, big move coming
   - Direction predicted by 20-day momentum
   - Tested on high-beta stocks (TSLA, NVDA, AMD, etc.)

### Tier 3: AI-Generated (experimental)
9. **VWAP Reversion (from our AI discovery)**
   - Already showed promise — test on 30+ tickers
   - Lower min trades to 5 given high win rate

10. **Regime-Adaptive Ensemble**
    - Different strategies work in different regimes
    - Bull: momentum + breakout
    - Bear: mean reversion + low vol
    - Sideways: mean reversion + pairs
    - Use our regime filter to switch

## Implementation Plan

### For each strategy:
```python
def backtest_strategy(strategy_func, tickers, period="2y"):
    """
    1. Download 2y data for all tickers
    2. Split: 18mo train, 6mo test (walk-forward)
    3. Run strategy on train set
    4. Calculate: Sharpe, return, win rate, max DD, trade count
    5. Run on test set (walk-forward)
    6. BOTH must pass: Sharpe > 0.5, return > 0%, win rate > 35%
    7. Save results
    """
```

### Acceptance criteria (strict):
- Train Sharpe > 0.5
- Train return > 5% (annualized)
- Train win rate > 35%
- Train max DD < 25%
- Train trades >= 5
- Walk-forward return > 0%
- Walk-forward Sharpe > 0
- Walk-forward trades >= 3

### Output
Save all results to `alpha_hunt_results.json` with full metrics.
Deploy passing strategies to trading_config.py.
Wire into autonomous engine.

## Cross-Sectional Momentum Implementation (most important)

This is the #1 priority because it's the most proven edge in finance.

```python
import yfinance as yf
import pandas as pd
import numpy as np

def cross_sectional_momentum(tickers, lookback_months=6, hold_months=1, top_pct=0.2):
    """
    Classic cross-sectional momentum:
    1. Calculate past N-month returns for all tickers
    2. Rank them
    3. Buy top quintile, equal weight
    4. Rebalance monthly
    5. Skip the most recent month (momentum crash protection)
    """
    # Download all data
    data = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="3y")
            if len(hist) > 200:
                data[t] = hist['Close']
        except:
            continue
    
    prices = pd.DataFrame(data)
    
    # Monthly returns for ranking (skip most recent month)
    monthly = prices.resample('M').last()
    lookback_return = monthly.pct_change(lookback_months).shift(1)  # skip recent month
    
    # Monthly forward returns (what we'd earn)
    forward_return = monthly.pct_change(hold_months).shift(-hold_months)
    
    # Backtest
    portfolio_returns = []
    for date in lookback_return.index:
        if date not in forward_return.index:
            continue
        ranks = lookback_return.loc[date].dropna()
        if len(ranks) < 10:
            continue
        
        # Top quintile
        n_stocks = max(int(len(ranks) * top_pct), 3)
        winners = ranks.nlargest(n_stocks).index
        
        # Equal-weight portfolio return
        fwd = forward_return.loc[date]
        port_ret = fwd[winners].mean()
        if not np.isnan(port_ret):
            portfolio_returns.append({"date": date, "return": port_ret})
    
    returns = pd.Series([r["return"] for r in portfolio_returns])
    
    # Metrics
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(12)  # annualized monthly Sharpe
    win_rate = (returns > 0).mean()
    max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
    
    return {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (12 / len(returns)) - 1,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "num_months": len(returns),
        "avg_monthly_return": returns.mean(),
    }
```

## Short-Term Reversal Implementation

```python
def weekly_reversal(tickers, lookback_days=5, hold_days=5, bottom_pct=0.1):
    """
    Buy the biggest weekly losers, hold 1 week.
    Classic short-term reversal strategy.
    """
    data = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2y")
            if len(hist) > 200:
                data[t] = hist['Close']
        except:
            continue
    
    prices = pd.DataFrame(data)
    
    # Weekly returns for ranking
    weekly_ret = prices.pct_change(lookback_days)
    forward_ret = prices.pct_change(hold_days).shift(-hold_days)
    
    # Sample every week
    portfolio_returns = []
    dates = weekly_ret.index[::5]  # every 5 trading days
    
    for date in dates:
        if date not in forward_ret.index:
            continue
        ranks = weekly_ret.loc[date].dropna()
        if len(ranks) < 20:
            continue
        
        # Bottom decile (biggest losers)
        n_stocks = max(int(len(ranks) * bottom_pct), 3)
        losers = ranks.nsmallest(n_stocks).index
        
        fwd = forward_ret.loc[date]
        port_ret = fwd[losers].mean()
        if not np.isnan(port_ret):
            portfolio_returns.append(port_ret)
    
    returns = pd.Series(portfolio_returns)
    total_return = (1 + returns).prod() - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(52)
    win_rate = (returns > 0).mean()
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "num_trades": len(returns),
    }
```
