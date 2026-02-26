"""
OVERNIGHT ALPHA HUNT
Find profitable, walk-forward validated strategies.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Our ticker universe
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "CRM", "NVDA",
    "ORCL", "ADBE", "UBER", "ABNB", "AMD", "INTC", "AVGO", "MRVL", "QCOM",
    "JPM", "V", "MA", "BAC", "WFC", "PYPL", "GS", "MS", "SCHW", "AXP",
    "SHOP", "MELI", "NU", "SOFI", "HOOD",
    "UNH", "JNJ", "ABBV", "LLY", "TMO", "ISRG", "GILD",
    "HD", "DIS", "PG", "WMT", "COST", "TGT", "SBUX", "MCD", "NKE",
    "LULU", "DECK", "ROST", "TJX", "BKNG", "MAR", "CMG",
    "XOM", "CVX", "SLB", "EOG", "MPC", "DVN",
    "CAT", "DE", "GE", "HON", "RTX", "LMT", "UNP", "UPS", "FDX",
    "PLTR", "CRWD", "NET", "DDOG", "ZS", "PANW", "SNOW", "MDB", "FTNT",
    "FCX", "NEM", "AMT", "PLD", "EQIX", "TMUS", "VZ", "CMCSA",
]

print(f"=== OVERNIGHT ALPHA HUNT ===")
print(f"Time: {datetime.now()}")
print(f"Tickers: {len(TICKERS)}")
print()

# Download all data once
print("Downloading 3 years of data for all tickers...")
all_prices = {}
for t in TICKERS:
    try:
        hist = yf.Ticker(t).history(period="3y", auto_adjust=True)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if len(hist) > 200:
            all_prices[t] = hist
    except Exception as e:
        pass
print(f"Got data for {len(all_prices)} tickers")
print()

# Build price matrix
close_prices = pd.DataFrame({t: d['Close'] for t, d in all_prices.items()})

# Split: train (first 2y) and test (last 1y) 
split_idx = int(len(close_prices) * 0.67)
train_prices = close_prices.iloc[:split_idx]
test_prices = close_prices.iloc[split_idx:]
print(f"Train: {train_prices.index[0].date()} to {train_prices.index[-1].date()} ({len(train_prices)} days)")
print(f"Test:  {test_prices.index[0].date()} to {test_prices.index[-1].date()} ({len(test_prices)} days)")
print()

results = {}

# ================================================================
# STRATEGY 1: Cross-Sectional Momentum (6-month lookback, monthly rebal)
# ================================================================
def momentum_strategy(prices, lookback=126, hold=21, top_pct=0.2, skip_recent=21):
    """Buy top quintile by past N-day return, skip recent month, rebalance monthly."""
    returns_data = []
    dates = prices.index[lookback + skip_recent::hold]
    
    for date in dates:
        idx = prices.index.get_loc(date)
        if idx + hold >= len(prices):
            break
        
        # Lookback returns (skip recent month)
        past_ret = {}
        for t in prices.columns:
            p = prices[t].values
            if idx - skip_recent >= 0 and idx - skip_recent - lookback >= 0:
                start_val = p[idx - skip_recent - lookback]
                end_val = p[idx - skip_recent]
                if start_val > 0 and not np.isnan(start_val) and not np.isnan(end_val):
                    past_ret[t] = end_val / start_val - 1
        
        if len(past_ret) < 10:
            continue
        
        # Top quintile
        sorted_tickers = sorted(past_ret.items(), key=lambda x: x[1], reverse=True)
        n = max(int(len(sorted_tickers) * top_pct), 3)
        winners = [t for t, _ in sorted_tickers[:n]]
        
        # Forward return (hold period)
        fwd_rets = []
        for t in winners:
            p = prices[t].values
            if idx + hold < len(p):
                entry = p[idx]
                exit_val = p[idx + hold]
                if entry > 0 and not np.isnan(entry) and not np.isnan(exit_val):
                    fwd_rets.append(exit_val / entry - 1)
        
        if fwd_rets:
            returns_data.append(np.mean(fwd_rets))
    
    return np.array(returns_data)

print("=" * 60)
print("STRATEGY 1: Cross-Sectional Momentum (6mo lookback, monthly)")
print("=" * 60)

for lookback, name in [(63, "3mo"), (126, "6mo"), (252, "12mo")]:
    for top in [0.1, 0.2, 0.3]:
        train_ret = momentum_strategy(train_prices, lookback=lookback, top_pct=top)
        test_ret = momentum_strategy(test_prices, lookback=lookback, top_pct=top)
        
        if len(train_ret) > 3 and len(test_ret) > 2:
            train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(12)
            test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(12)
            train_total = (np.prod(1 + train_ret) - 1) * 100
            test_total = (np.prod(1 + test_ret) - 1) * 100
            train_win = np.mean(train_ret > 0) * 100
            test_win = np.mean(test_ret > 0) * 100
            
            tag = ""
            if train_sharpe > 0.5 and test_sharpe > 0 and test_total > 0:
                tag = " *** PASSED ***"
            
            print(f"  {name} top{int(top*100)}%: Train={train_total:+.1f}% Sharpe={train_sharpe:.2f} Win={train_win:.0f}% | Test={test_total:+.1f}% Sharpe={test_sharpe:.2f} Win={test_win:.0f}%{tag}")
            
            if tag:
                results[f"momentum_{name}_top{int(top*100)}"] = {
                    "strategy": "cross_sectional_momentum",
                    "params": {"lookback": lookback, "top_pct": top, "hold": 21, "skip_recent": 21},
                    "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                    "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
                }

print()

# ================================================================
# STRATEGY 2: Short-Term Reversal (weekly losers bounce)
# ================================================================
def reversal_strategy(prices, lookback=5, hold=5, bottom_pct=0.1):
    """Buy biggest weekly losers, hold 1 week."""
    returns_data = []
    dates = prices.index[lookback::hold]
    
    for date in dates:
        idx = prices.index.get_loc(date)
        if idx + hold >= len(prices):
            break
        
        # Past-week returns
        past_ret = {}
        for t in prices.columns:
            p = prices[t].values
            if idx - lookback >= 0:
                start_val = p[idx - lookback]
                end_val = p[idx]
                if start_val > 0 and not np.isnan(start_val) and not np.isnan(end_val):
                    past_ret[t] = end_val / start_val - 1
        
        if len(past_ret) < 20:
            continue
        
        # Bottom decile (biggest losers)
        sorted_tickers = sorted(past_ret.items(), key=lambda x: x[1])
        n = max(int(len(sorted_tickers) * bottom_pct), 3)
        losers = [t for t, _ in sorted_tickers[:n]]
        
        # Forward return
        fwd_rets = []
        for t in losers:
            p = prices[t].values
            if idx + hold < len(p):
                entry = p[idx]
                exit_val = p[idx + hold]
                if entry > 0 and not np.isnan(entry) and not np.isnan(exit_val):
                    fwd_rets.append(exit_val / entry - 1)
        
        if fwd_rets:
            returns_data.append(np.mean(fwd_rets))
    
    return np.array(returns_data)

print("=" * 60)
print("STRATEGY 2: Short-Term Reversal (buy weekly losers)")
print("=" * 60)

for lookback in [3, 5, 10]:
    for hold in [3, 5, 10]:
        for bottom in [0.05, 0.1, 0.15]:
            train_ret = reversal_strategy(train_prices, lookback=lookback, hold=hold, bottom_pct=bottom)
            test_ret = reversal_strategy(test_prices, lookback=lookback, hold=hold, bottom_pct=bottom)
            
            if len(train_ret) > 5 and len(test_ret) > 3:
                train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(52/max(hold/5,1))
                test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(52/max(hold/5,1))
                train_total = (np.prod(1 + train_ret) - 1) * 100
                test_total = (np.prod(1 + test_ret) - 1) * 100
                train_win = np.mean(train_ret > 0) * 100
                test_win = np.mean(test_ret > 0) * 100
                
                tag = ""
                if train_sharpe > 0.5 and test_sharpe > 0 and test_total > 0 and train_win > 45:
                    tag = " *** PASSED ***"
                
                if tag or train_sharpe > 0.3:
                    print(f"  L={lookback}d H={hold}d bot{int(bottom*100)}%: Train={train_total:+.1f}% Sharpe={train_sharpe:.2f} Win={train_win:.0f}% | Test={test_total:+.1f}% Sharpe={test_sharpe:.2f} Win={test_win:.0f}%{tag}")
                
                if tag:
                    results[f"reversal_L{lookback}_H{hold}_bot{int(bottom*100)}"] = {
                        "strategy": "short_term_reversal",
                        "params": {"lookback": lookback, "hold": hold, "bottom_pct": bottom},
                        "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                        "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
                    }

print()

# ================================================================
# STRATEGY 3: Dual Momentum (absolute + relative)
# ================================================================
def dual_momentum(prices, spy_prices, lookback=126):
    """
    Dual momentum: buy stocks with positive absolute momentum 
    AND top relative momentum vs SPY. Monthly rebalance.
    """
    returns_data = []
    dates = prices.index[lookback::21]
    
    for date in dates:
        idx = prices.index.get_loc(date)
        if idx + 21 >= len(prices):
            break
        
        # Absolute + relative momentum
        candidates = []
        for t in prices.columns:
            p = prices[t].values
            if idx - lookback >= 0:
                abs_ret = p[idx] / p[idx - lookback] - 1
                if abs_ret > 0:  # Absolute momentum filter
                    candidates.append((t, abs_ret))
        
        if len(candidates) < 5:
            returns_data.append(0)  # sit in cash
            continue
        
        # Top 20% by relative momentum
        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
        n = max(int(len(sorted_cands) * 0.2), 3)
        winners = [t for t, _ in sorted_cands[:n]]
        
        # Forward return
        fwd_rets = []
        for t in winners:
            p = prices[t].values
            if idx + 21 < len(p):
                fwd_rets.append(p[idx + 21] / p[idx] - 1)
        
        if fwd_rets:
            returns_data.append(np.mean(fwd_rets))
    
    return np.array(returns_data)

print("=" * 60)
print("STRATEGY 3: Dual Momentum (absolute + relative)")
print("=" * 60)

# Get SPY for reference
spy_data = yf.Ticker("SPY").history(period="3y", auto_adjust=True)
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_data.columns = spy_data.columns.get_level_values(0)

for lookback in [63, 126, 189, 252]:
    train_ret = dual_momentum(train_prices, spy_data, lookback=lookback)
    test_ret = dual_momentum(test_prices, spy_data, lookback=lookback)
    
    if len(train_ret) > 3 and len(test_ret) > 2:
        train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(12)
        test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(12)
        train_total = (np.prod(1 + train_ret) - 1) * 100
        test_total = (np.prod(1 + test_ret) - 1) * 100
        train_win = np.mean(train_ret > 0) * 100
        test_win = np.mean(test_ret > 0) * 100
        
        tag = ""
        if train_sharpe > 0.5 and test_sharpe > 0 and test_total > 0:
            tag = " *** PASSED ***"
        
        months = {63: "3mo", 126: "6mo", 189: "9mo", 252: "12mo"}
        print(f"  {months[lookback]}: Train={train_total:+.1f}% Sharpe={train_sharpe:.2f} Win={train_win:.0f}% | Test={test_total:+.1f}% Sharpe={test_sharpe:.2f} Win={test_win:.0f}%{tag}")
        
        if tag:
            results[f"dual_momentum_{months[lookback]}"] = {
                "strategy": "dual_momentum",
                "params": {"lookback": lookback},
                "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
            }

print()

# ================================================================
# STRATEGY 4: Low Volatility + Momentum Combo
# ================================================================
def low_vol_momentum(prices, vol_lookback=60, mom_lookback=126, hold=21, n_stocks=10):
    """
    Buy stocks in bottom 50% volatility AND top 50% momentum.
    Best of both worlds: smooth ride + trending up.
    """
    returns_data = []
    dates = prices.index[max(vol_lookback, mom_lookback)::hold]
    
    for date in dates:
        idx = prices.index.get_loc(date)
        if idx + hold >= len(prices):
            break
        
        scores = {}
        for t in prices.columns:
            p = prices[t].values
            if idx - mom_lookback >= 0:
                # Volatility (lower = better)
                rets = np.diff(p[idx-vol_lookback:idx+1]) / p[idx-vol_lookback:idx]
                if len(rets) > 10 and not np.any(np.isnan(rets)):
                    vol = np.std(rets) * np.sqrt(252)
                    # Momentum (higher = better)
                    mom = p[idx] / p[idx - mom_lookback] - 1
                    if not np.isnan(mom) and vol > 0:
                        scores[t] = {"vol": vol, "mom": mom}
        
        if len(scores) < 20:
            continue
        
        # Filter: bottom 50% vol AND top 50% momentum
        med_vol = np.median([s["vol"] for s in scores.values()])
        med_mom = np.median([s["mom"] for s in scores.values()])
        
        candidates = [t for t, s in scores.items() if s["vol"] < med_vol and s["mom"] > med_mom]
        
        if len(candidates) < 3:
            continue
        
        # Top N by momentum among low-vol stocks
        candidates.sort(key=lambda t: scores[t]["mom"], reverse=True)
        selected = candidates[:n_stocks]
        
        # Forward return
        fwd_rets = []
        for t in selected:
            p = prices[t].values
            if idx + hold < len(p):
                fwd_rets.append(p[idx + hold] / p[idx] - 1)
        
        if fwd_rets:
            returns_data.append(np.mean(fwd_rets))
    
    return np.array(returns_data)

print("=" * 60)
print("STRATEGY 4: Low Volatility + Momentum Combo")
print("=" * 60)

for n in [5, 10, 15]:
    for mom_lb in [63, 126]:
        train_ret = low_vol_momentum(train_prices, mom_lookback=mom_lb, n_stocks=n)
        test_ret = low_vol_momentum(test_prices, mom_lookback=mom_lb, n_stocks=n)
        
        if len(train_ret) > 3 and len(test_ret) > 2:
            train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(12)
            test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(12)
            train_total = (np.prod(1 + train_ret) - 1) * 100
            test_total = (np.prod(1 + test_ret) - 1) * 100
            train_win = np.mean(train_ret > 0) * 100
            test_win = np.mean(test_ret > 0) * 100
            
            tag = ""
            if train_sharpe > 0.5 and test_sharpe > 0 and test_total > 0:
                tag = " *** PASSED ***"
            
            months = {63: "3mo", 126: "6mo"}
            print(f"  top{n} mom={months[mom_lb]}: Train={train_total:+.1f}% Sharpe={train_sharpe:.2f} Win={train_win:.0f}% | Test={test_total:+.1f}% Sharpe={test_sharpe:.2f} Win={test_win:.0f}%{tag}")
            
            if tag:
                results[f"low_vol_mom_{months[mom_lb]}_n{n}"] = {
                    "strategy": "low_vol_momentum",
                    "params": {"vol_lookback": 60, "mom_lookback": mom_lb, "n_stocks": n, "hold": 21},
                    "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                    "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
                }

print()

# ================================================================
# STRATEGY 5: Mean Reversion RSI(2) on Large-Cap Only 
# ================================================================
def rsi2_largecap(prices, rsi_period=2, oversold=10, hold_until_rsi=60):
    """
    Larry Connors RSI(2) on large caps.
    Buy when RSI(2) < 10, sell when RSI(2) > 60.
    Only on stocks above 200-day SMA (uptrend filter).
    """
    all_trades = []
    
    for t in prices.columns:
        p = prices[t].dropna()
        if len(p) < 250:
            continue
        
        # Calculate RSI(2)
        delta = p.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # 200-day SMA
        sma200 = p.rolling(200).mean()
        
        # Trade logic
        position = False
        entry_price = 0
        
        for i in range(200, len(p)):
            if not position:
                if rsi.iloc[i] < oversold and p.iloc[i] > sma200.iloc[i]:
                    position = True
                    entry_price = p.iloc[i]
            else:
                if rsi.iloc[i] > hold_until_rsi or i == len(p) - 1:
                    exit_price = p.iloc[i]
                    pnl = exit_price / entry_price - 1
                    all_trades.append(pnl)
                    position = False
    
    return np.array(all_trades) if all_trades else np.array([])

print("=" * 60)
print("STRATEGY 5: RSI(2) Mean Reversion (large-cap, uptrend only)")
print("=" * 60)

for oversold in [5, 10, 15]:
    for exit_rsi in [50, 60, 70, 80]:
        train_ret = rsi2_largecap(train_prices, oversold=oversold, hold_until_rsi=exit_rsi)
        test_ret = rsi2_largecap(test_prices, oversold=oversold, hold_until_rsi=exit_rsi)
        
        if len(train_ret) > 10 and len(test_ret) > 5:
            train_total = (np.prod(1 + train_ret) - 1) * 100
            test_total = (np.prod(1 + test_ret) - 1) * 100
            train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(len(train_ret) / 2)
            test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(len(test_ret))
            train_win = np.mean(train_ret > 0) * 100
            test_win = np.mean(test_ret > 0) * 100
            
            tag = ""
            if train_total > 0 and test_total > 0 and train_win > 50 and test_win > 40:
                tag = " *** PASSED ***"
            
            if tag or train_win > 55:
                print(f"  RSI<{oversold} exit>{exit_rsi}: Train={train_total:+.1f}% Win={train_win:.0f}% ({len(train_ret)} trades) | Test={test_total:+.1f}% Win={test_win:.0f}% ({len(test_ret)} trades){tag}")
            
            if tag:
                results[f"rsi2_os{oversold}_ex{exit_rsi}"] = {
                    "strategy": "rsi2_mean_reversion",
                    "params": {"rsi_period": 2, "oversold": oversold, "exit_rsi": exit_rsi, "sma_filter": 200},
                    "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                    "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
                }

print()

# ================================================================
# STRATEGY 6: Bollinger Band Squeeze Breakout
# ================================================================
def bb_squeeze(prices, bb_period=20, squeeze_percentile=10, hold=10):
    """
    When Bollinger Band width reaches 6-month low (squeeze),
    bet on breakout direction based on momentum.
    """
    all_trades = []
    
    for t in prices.columns:
        p = prices[t].dropna()
        if len(p) < 150:
            continue
        
        sma = p.rolling(bb_period).mean()
        std = p.rolling(bb_period).std()
        bb_width = (std / sma * 100)  # normalize
        
        # 6-month rolling min of BB width
        bb_min = bb_width.rolling(126).min()
        
        # Momentum (20-day)
        mom = p.pct_change(20)
        
        position = False
        cooldown = 0
        
        for i in range(130, len(p)):
            cooldown = max(0, cooldown - 1)
            
            if not position and cooldown == 0:
                # Squeeze: BB width within 10% of 6-month low
                if not np.isnan(bb_width.iloc[i]) and not np.isnan(bb_min.iloc[i]) and bb_min.iloc[i] > 0:
                    if bb_width.iloc[i] <= bb_min.iloc[i] * 1.1:
                        # Direction from momentum
                        if not np.isnan(mom.iloc[i]) and mom.iloc[i] > 0:
                            position = True
                            entry_price = p.iloc[i]
                            entry_idx = i
            
            elif position:
                if i - entry_idx >= hold or i == len(p) - 1:
                    exit_price = p.iloc[i]
                    pnl = exit_price / entry_price - 1
                    all_trades.append(pnl)
                    position = False
                    cooldown = 5
    
    return np.array(all_trades) if all_trades else np.array([])

print("=" * 60)
print("STRATEGY 6: Bollinger Band Squeeze Breakout")
print("=" * 60)

for hold in [5, 10, 15, 20]:
    train_ret = bb_squeeze(train_prices, hold=hold)
    test_ret = bb_squeeze(test_prices, hold=hold)
    
    if len(train_ret) > 10 and len(test_ret) > 5:
        train_total = (np.prod(1 + train_ret) - 1) * 100
        test_total = (np.prod(1 + test_ret) - 1) * 100
        train_win = np.mean(train_ret > 0) * 100
        test_win = np.mean(test_ret > 0) * 100
        train_sharpe = np.mean(train_ret) / (np.std(train_ret) + 1e-10) * np.sqrt(252/hold)
        test_sharpe = np.mean(test_ret) / (np.std(test_ret) + 1e-10) * np.sqrt(252/hold)
        
        tag = ""
        if train_total > 0 and test_total > 0 and train_win > 45 and test_win > 40:
            tag = " *** PASSED ***"
        
        print(f"  hold={hold}d: Train={train_total:+.1f}% Win={train_win:.0f}% Sharpe={train_sharpe:.2f} ({len(train_ret)} trades) | Test={test_total:+.1f}% Win={test_win:.0f}% Sharpe={test_sharpe:.2f} ({len(test_ret)} trades){tag}")
        
        if tag:
            results[f"bb_squeeze_h{hold}"] = {
                "strategy": "bb_squeeze_breakout",
                "params": {"bb_period": 20, "hold": hold},
                "train": {"return": train_total, "sharpe": train_sharpe, "win_rate": train_win, "trades": len(train_ret)},
                "test": {"return": test_total, "sharpe": test_sharpe, "win_rate": test_win, "trades": len(test_ret)},
            }

print()

# ================================================================
# SUMMARY
# ================================================================
print("=" * 60)
print(f" RESULTS: {len(results)} strategies passed walk-forward validation")
print("=" * 60)

if results:
    # Sort by test Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1]["test"]["sharpe"], reverse=True)
    for name, r in sorted_results:
        print(f"\n  {name}")
        print(f"    Strategy: {r['strategy']}")
        print(f"    Params: {r['params']}")
        print(f"    Train: {r['train']['return']:+.1f}% | Sharpe={r['train']['sharpe']:.2f} | Win={r['train']['win_rate']:.0f}% | Trades={r['train']['trades']}")
        print(f"    Test:  {r['test']['return']:+.1f}% | Sharpe={r['test']['sharpe']:.2f} | Win={r['test']['win_rate']:.0f}% | Trades={r['test']['trades']}")
else:
    print("  No strategies passed. Need to widen search or lower thresholds.")

# Save
with open("alpha_hunt_results.json", "w") as f:
    json.dump({"generated_at": datetime.now().isoformat(), "strategies": results}, f, indent=2, default=str)
print(f"\nResults saved to alpha_hunt_results.json")
