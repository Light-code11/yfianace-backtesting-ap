"""
Walk-Forward Backtester for Trading Strategies
================================================
Tests all 10 strategies against 3 years of historical data using walk-forward
optimization windows to detect overfitting.

Walk-forward params:
  - Train window: 252 days (1 year) â€” used to compute in-sample Sharpe
  - Test window:  63 days  (3 months) â€” out-of-sample performance
  - Roll:         63 days
  - Periods:      ~8 test windows over 3 years

Fixed position size: $10,000 per trade
Commission: $1 per trade (Alpaca approx.)
Slippage: NOT modelled â€” noted as limitation

Run:
    .venv312\\Scripts\\python.exe backtester.py
"""

import json
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Fix Windows console encoding for Unicode output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# â”€â”€â”€ Config (mirrored from trading_config.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TICKER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "CRM", "NVDA",
    "AMD", "INTC", "AVGO", "MRVL", "ARM", "QCOM", "LRCX", "KLAC", "ASML",
    "JPM", "V", "MA", "BAC", "WFC", "PYPL", "XYZ", "COIN",
    "SHOP", "MELI", "NU", "AFRM", "SOFI",
    "UNH", "JNJ", "MRNA", "REGN", "ABBV", "LLY", "TMO", "ISRG", "DXCM",
    "HD", "DIS", "PG", "WMT", "COST", "TGT", "SBUX", "MCD", "NKE", "LULU", "DECK",
    "XOM", "CVX", "SLB", "EOG", "OXY", "MPC",
    "CAT", "DE", "GE", "HON", "RTX", "LMT",
    "SMCI", "PLTR", "CRWD", "NET", "DDOG", "ZS", "PANW",
]

PAIRS = [
    ("V", "MA"),
    ("GOOGL", "META"),
    ("XOM", "CVX"),
    ("COST", "WMT"),
    ("JPM", "BAC"),
    ("AMD", "INTC"),
    ("AAPL", "MSFT"),
    ("LMT", "RTX"),
    ("CRWD", "PANW"),
    ("DDOG", "NET"),
]

SECTOR_ETF_MAP: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
    "AVGO": "XLK", "MRVL": "XLK", "ARM": "XLK", "QCOM": "XLK", "LRCX": "XLK",
    "KLAC": "XLK", "ASML": "XLK", "CRM": "XLK", "SMCI": "XLK", "PLTR": "XLK",
    "CRWD": "XLK", "NET": "XLK", "DDOG": "XLK", "ZS": "XLK", "PANW": "XLK",
    "SHOP": "XLK",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "MELI": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY", "LULU": "XLY", "DECK": "XLY",
    "WMT": "XLP", "PG": "XLP", "COST": "XLP", "TGT": "XLP", "SBUX": "XLP", "MCD": "XLP",
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF", "WFC": "XLF",
    "PYPL": "XLF", "XYZ": "XLF", "COIN": "XLF", "AFRM": "XLF", "SOFI": "XLF", "NU": "XLF",
    "UNH": "XLV", "JNJ": "XLV", "MRNA": "XLV", "REGN": "XLV", "ABBV": "XLV",
    "LLY": "XLV", "TMO": "XLV", "ISRG": "XLV", "DXCM": "XLV",
    "XOM": "XLE", "CVX": "XLE", "SLB": "XLE", "EOG": "XLE", "OXY": "XLE", "MPC": "XLE",
    "CAT": "XLI", "DE": "XLI", "GE": "XLI", "HON": "XLI", "RTX": "XLI", "LMT": "XLI",
}

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"]

STRATEGIES_CFG = [
    {
        "name": "RSI Mean Reversion",
        "key": "rsi_mean_reversion",
        "strategy_type": "mean_reversion",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 4.5,
        "take_profit_pct": 9.0,
    },
    {
        "name": "MACD Trend Following",
        "key": "macd_trend_following",
        "strategy_type": "trend_following",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 12.0,
    },
    {
        "name": "SMA Momentum",
        "key": "sma_momentum",
        "strategy_type": "momentum",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
    },
    {
        "name": "Volume Breakout",
        "key": "volume_breakout",
        "strategy_type": "breakout",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 4.0,
        "take_profit_pct": 10.0,
    },
    {
        "name": "Bollinger Band Squeeze",
        "key": "bb_squeeze",
        "strategy_type": "bb_squeeze",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 4.0,
        "take_profit_pct": 10.0,
    },
    {
        "name": "VWAP Reversion",
        "key": "vwap_reversion",
        "strategy_type": "vwap_reversion",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 3.5,
        "take_profit_pct": 7.0,
    },
    {
        "name": "Gap Fill",
        "key": "gap_fill",
        "strategy_type": "gap_fill",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 3.0,
        "take_profit_pct": 6.0,
    },
    {
        "name": "Earnings Momentum",
        "key": "earnings_momentum",
        "strategy_type": "earnings_momentum",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 6.0,
        "take_profit_pct": 15.0,
    },
    {
        "name": "Sector Rotation",
        "key": "sector_rotation",
        "strategy_type": "sector_rotation",
        "tickers": TICKER_UNIVERSE,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 12.0,
    },
    {
        "name": "Pair Trading",
        "key": "pair_trading",
        "strategy_type": "pair_trading",
        "tickers": [],
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
    },
]

POSITION_SIZE = 10_000.0   # USD per trade
COMMISSION = 1.0           # USD per trade
TRAIN_WINDOW = 252         # days
TEST_WINDOW = 63           # days
YEARS_OF_DATA = 3

# â”€â”€â”€ Technical Indicators (standalone) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()

def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s: pd.Series, fast=12, slow=26, signal_p=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=signal_p, adjust=False).mean()
    return ml, sl, ml - sl

def bollinger_bands(s: pd.Series, period=20, std_dev=2):
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    return mid + std * std_dev, mid, mid - std * std_dev

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rolling_vwap(high, low, close, volume, period=20):
    tp = (high + low + close) / 3
    num = (tp * volume).rolling(period).sum()
    den = volume.rolling(period).sum()
    vwap_s = (num / den.replace(0, np.nan))
    dev = tp - vwap_s
    std = dev.rolling(period).std()
    zscore = (dev / std.replace(0, np.nan))
    return vwap_s, zscore

def zscore_series(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - m) / std.replace(0, np.nan)

# â”€â”€â”€ Signal Generators â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def signal_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI + Bollinger band mean reversion."""
    close = df['Close']
    r = rsi(close, 14)
    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2)

    sig = pd.Series('HOLD', index=df.index)
    buy = (close <= bb_l * 1.02) & (r < 35)
    sell = (close >= bb_u * 0.98) & (r > 65)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_trend_following(df: pd.DataFrame) -> pd.Series:
    """MACD + SMA50/200 golden/death cross."""
    close = df['Close']
    s50 = sma(close, 50)
    s200 = sma(close, 200)
    ml, sl, _ = macd(close)

    sig = pd.Series('HOLD', index=df.index)
    buy = (s50 > s200) & (close > s50) & (ml > sl)
    sell = (s50 < s200) & (close < s50) & (ml < sl)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_momentum(df: pd.DataFrame) -> pd.Series:
    """SMA 20/50 crossover with RSI filter."""
    close = df['Close']
    s20 = sma(close, 20)
    s50 = sma(close, 50)
    r = rsi(close, 14)

    sig = pd.Series('HOLD', index=df.index)
    buy = (s20 > s50) & (close > s20) & (r < 70)
    sell = (s20 < s50) & (close < s20) & (r > 30)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_breakout(df: pd.DataFrame) -> pd.Series:
    """20-day high/low breakout with volume confirmation."""
    close = df['Close']
    volume = df['Volume']

    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    avg_vol = volume.rolling(20).mean()
    vol_ratio = volume / avg_vol.replace(0, np.nan)

    sig = pd.Series('HOLD', index=df.index)
    buy = (close >= high_20 * 0.995) & (vol_ratio >= 1.2)
    sell = (close <= low_20 * 1.005) & (vol_ratio >= 1.2)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_bb_squeeze(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band squeeze â†’ breakout direction."""
    close = df['Close']
    bb_u, bb_m, bb_l = bollinger_bands(close, 20, 2)
    r = rsi(close, 14)

    bandwidth = (bb_u - bb_l) / bb_m
    sq_thresh = bandwidth.rolling(50).quantile(0.20)

    in_squeeze = bandwidth <= sq_thresh
    sig = pd.Series('HOLD', index=df.index)
    buy = in_squeeze & (close > bb_u) & (r > 50)
    sell = in_squeeze & (close < bb_l) & (r < 50)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_vwap_reversion(df: pd.DataFrame) -> pd.Series:
    """Fade price when > 2 std devs from rolling VWAP."""
    close = df['Close']
    r = rsi(close, 14)

    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return pd.Series('HOLD', index=df.index)

    vwap_s, z = rolling_vwap(df['High'], df['Low'], close, df['Volume'], 20)

    sig = pd.Series('HOLD', index=df.index)
    buy = (z < -2.0) & (r < 40)
    sell = (z > 2.0) & (r > 60)
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_gap_fill(df: pd.DataFrame) -> pd.Series:
    """Fade overnight gaps > 2%."""
    close = df['Close']
    open_ = df['Open'] if 'Open' in df.columns else close

    gap_pct = (open_ - close.shift(1)) / close.shift(1) * 100

    sig = pd.Series('HOLD', index=df.index)
    buy = gap_pct <= -2.0   # gap down â†’ fade up
    sell = gap_pct >= 2.0   # gap up â†’ fade down
    sig[buy] = 'BUY'
    sig[sell] = 'SELL'
    return sig

def signal_earnings_momentum(df: pd.DataFrame) -> pd.Series:
    """PEAD â€” ride 5%+ single-day moves for up to 20 days."""
    close = df['Close']
    ret = close.pct_change()
    r = rsi(close, 14)

    sig = pd.Series('HOLD', index=df.index)
    for i in range(20, len(df)):
        window_ret = ret.iloc[i-20:i]
        large = window_ret[window_ret.abs() >= 0.05]
        if large.empty:
            continue
        latest = large.iloc[-1]
        rsi_val = r.iloc[i]
        if pd.isna(rsi_val):
            continue
        if latest > 0.05 and rsi_val < 75:
            sig.iloc[i] = 'BUY'
        elif latest < -0.05 and rsi_val > 25:
            sig.iloc[i] = 'SELL'
    return sig

def signal_sector_rotation(df: pd.DataFrame, ticker: str,
                            sector_data: Dict[str, pd.Series],
                            spy_data: pd.Series) -> pd.Series:
    """Sector outperformance vs SPY (20-day relative return)."""
    close = df['Close']
    r = rsi(close, 14)
    etf_key = SECTOR_ETF_MAP.get(ticker)

    sig = pd.Series('HOLD', index=df.index)

    if etf_key is None or etf_key not in sector_data or spy_data is None:
        return sig

    etf_close = sector_data[etf_key]

    for i in range(21, len(df)):
        idx = df.index[i]
        # find matching etf/spy index
        try:
            etf_slice = etf_close[etf_close.index <= idx].iloc[-21:]
            spy_slice = spy_data[spy_data.index <= idx].iloc[-21:]
            if len(etf_slice) < 2 or len(spy_slice) < 2:
                continue
            etf_ret = float(etf_slice.iloc[-1]) / float(etf_slice.iloc[0]) - 1.0
            spy_ret = float(spy_slice.iloc[-1]) / float(spy_slice.iloc[0]) - 1.0
            rel = etf_ret - spy_ret
            rsi_val = float(r.iloc[i]) if not pd.isna(r.iloc[i]) else 50
            if rel > 0.02 and rsi_val < 70:
                sig.iloc[i] = 'BUY'
            elif rel < -0.02 and rsi_val > 30:
                sig.iloc[i] = 'SELL'
        except Exception:
            continue
    return sig

def signal_pair_trading(df_a: pd.DataFrame, df_b: pd.DataFrame,
                         lookback: int = 60,
                         entry_z: float = 2.0,
                         exit_z: float = 0.5) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (sig_a, sig_b): BUY/SELL/HOLD for each leg.
    When z > entry: sell A, buy B. When z < -entry: buy A, sell B.
    """
    close_a = df_a['Close']
    close_b = df_b['Close']

    # Align on common dates
    common = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common]
    close_b = close_b.loc[common]

    spread = close_a - close_b
    z = zscore_series(spread, lookback)

    sig_a = pd.Series('HOLD', index=common)
    sig_b = pd.Series('HOLD', index=common)

    sig_a[z > entry_z] = 'SELL'
    sig_b[z > entry_z] = 'BUY'
    sig_a[z < -entry_z] = 'BUY'
    sig_b[z < -entry_z] = 'SELL'

    return sig_a, sig_b

# â”€â”€â”€ Trade Simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def simulate_trades(
    signals: pd.Series,
    close: pd.Series,
    stop_loss_pct: float,
    take_profit_pct: float,
    position_size: float = POSITION_SIZE,
    commission: float = COMMISSION,
    direction: str = "LONG",  # LONG or SHORT
) -> List[Dict]:
    """
    Simulate trades from a signal series.
    - Entry at close on signal day (approximation for simplicity)
    - Exit at take_profit, stop_loss, or opposite signal
    Returns list of trade dicts.
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_date = None
    trade_signal = None

    for i, (date, sig) in enumerate(signals.items()):
        price = float(close.loc[date]) if date in close.index else np.nan
        if pd.isna(price):
            continue

        if not in_trade:
            if sig in ('BUY', 'SELL'):
                in_trade = True
                entry_price = price
                entry_date = date
                trade_signal = sig
        else:
            # Check exit conditions
            if trade_signal == 'BUY':
                pnl_pct = (price - entry_price) / entry_price * 100
                hit_tp = pnl_pct >= take_profit_pct
                hit_sl = pnl_pct <= -stop_loss_pct
                exit_signal = sig == 'SELL'
            else:  # SELL / SHORT
                pnl_pct = (entry_price - price) / entry_price * 100
                hit_tp = pnl_pct >= take_profit_pct
                hit_sl = pnl_pct <= -stop_loss_pct
                exit_signal = sig == 'BUY'

            if hit_tp or hit_sl or exit_signal:
                shares = position_size / entry_price
                if trade_signal == 'BUY':
                    gross_pnl = (price - entry_price) * shares
                else:
                    gross_pnl = (entry_price - price) * shares
                net_pnl = gross_pnl - 2 * commission  # entry + exit

                duration = (date - entry_date).days if hasattr(date, 'days') else (
                    pd.Timestamp(date) - pd.Timestamp(entry_date)).days

                trades.append({
                    'entry_date': str(entry_date),
                    'exit_date': str(date),
                    'signal': trade_signal,
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(price, 4),
                    'pnl_pct': round(pnl_pct, 4),
                    'net_pnl': round(net_pnl, 4),
                    'duration_days': duration,
                    'exit_reason': 'TP' if hit_tp else ('SL' if hit_sl else 'SIGNAL'),
                })
                in_trade = False

                # Immediately check if new signal triggers a trade
                if sig in ('BUY', 'SELL') and not hit_tp and not hit_sl:
                    in_trade = True
                    entry_price = price
                    entry_date = date
                    trade_signal = sig

    return trades

# â”€â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def calc_sharpe(returns: List[float], annual_factor: float = 252) -> float:
    """Annualized Sharpe (daily returns)."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return float((arr.mean() / std) * math.sqrt(annual_factor))

def calc_max_drawdown(equity_curve: List[float]) -> float:
    """Maximum drawdown as a fraction."""
    if not equity_curve:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())

def calc_profit_factor(trades: List[Dict]) -> float:
    gross_profit = sum(t['net_pnl'] for t in trades if t['net_pnl'] > 0)
    gross_loss = abs(sum(t['net_pnl'] for t in trades if t['net_pnl'] < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 1.0
    return round(gross_profit / gross_loss, 4)

def aggregate_metrics(all_trades: List[Dict], window_label: str = "") -> Dict:
    if not all_trades:
        return {
            'window': window_label,
            'trades': 0,
            'win_rate': 0,
            'avg_return_pct': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'avg_duration_days': 0,
            'total_pnl': 0,
        }

    wins = [t for t in all_trades if t['net_pnl'] > 0]
    win_rate = len(wins) / len(all_trades)
    avg_return = np.mean([t['pnl_pct'] for t in all_trades])
    avg_duration = np.mean([t['duration_days'] for t in all_trades])
    total_pnl = sum(t['net_pnl'] for t in all_trades)

    # Build daily P&L for Sharpe / drawdown
    daily_pnl: Dict[str, float] = {}
    for t in all_trades:
        d = t['exit_date']
        daily_pnl[d] = daily_pnl.get(d, 0) + t['net_pnl']

    sorted_days = sorted(daily_pnl.keys())
    daily_returns = [daily_pnl[d] / POSITION_SIZE for d in sorted_days]

    equity = [POSITION_SIZE]
    for r in daily_returns:
        equity.append(equity[-1] * (1 + r))

    sharpe = calc_sharpe(daily_returns)
    max_dd = calc_max_drawdown(equity)
    pf = calc_profit_factor(all_trades)

    return {
        'window': window_label,
        'trades': len(all_trades),
        'win_rate': round(win_rate, 4),
        'avg_return_pct': round(float(avg_return), 4),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(float(max_dd), 4),
        'profit_factor': pf,
        'avg_duration_days': round(float(avg_duration), 1),
        'total_pnl': round(total_pnl, 2),
    }

# â”€â”€â”€ Data Downloader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def download_all_data(tickers: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers. Returns dict ticker â†’ DataFrame."""
    end = datetime.today()
    start = end - timedelta(days=365 * years + 30)

    print(f"\nðŸ“¥ Downloading {len(tickers)} tickers ({years}y)...", flush=True)

    all_data: Dict[str, pd.DataFrame] = {}
    batch_size = 20

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        print(f"  Batch {i//batch_size + 1}/{math.ceil(len(tickers)/batch_size)}: {', '.join(batch)}", flush=True)
        try:
            raw = yf.download(
                batch,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
                group_by='ticker',
            )
        except Exception as e:
            print(f"  âš ï¸  Batch failed: {e}", flush=True)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    df = raw[ticker].copy()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.dropna(subset=['Close'])
                if len(df) > 100:
                    all_data[ticker] = df
            except Exception:
                pass

    print(f"  âœ… Got data for {len(all_data)}/{len(tickers)} tickers", flush=True)
    return all_data

# â”€â”€â”€ Walk-Forward Engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_walk_forward_windows(n_days: int, train: int = TRAIN_WINDOW,
                              test: int = TEST_WINDOW) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (train_start, train_end, test_start, test_end) index tuples.
    """
    windows = []
    train_start = 0
    while True:
        train_end = train_start + train
        test_start = train_end
        test_end = test_start + test
        if test_end > n_days:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start += test  # roll forward by test window
    return windows

def run_strategy_backtest(
    strategy_cfg: Dict,
    data: Dict[str, pd.DataFrame],
    sector_data: Dict[str, pd.Series],
    spy_data: Optional[pd.Series],
) -> Dict:
    """Run full walk-forward backtest for a single (non-pair) strategy."""
    strategy_type = strategy_cfg['strategy_type']
    tickers = [t for t in strategy_cfg['tickers'] if t in data]
    stop_loss_pct = strategy_cfg['stop_loss_pct']
    take_profit_pct = strategy_cfg['take_profit_pct']

    print(f"\nðŸ”„ {strategy_cfg['name']} ({strategy_type}) â€” {len(tickers)} tickers", flush=True)

    all_window_results = []
    in_sample_trades: List[Dict] = []
    out_sample_trades: List[Dict] = []

    # Use a representative date index (union of all tickers)
    all_dates = sorted(set().union(*[set(data[t].index) for t in tickers]))
    all_dates = pd.DatetimeIndex(all_dates)
    n_days = len(all_dates)

    windows = get_walk_forward_windows(n_days)
    print(f"  Walk-forward: {len(windows)} test windows", flush=True)

    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        train_dates = all_dates[tr_s:tr_e]
        test_dates = all_dates[te_s:te_e]

        window_label = f"W{w_idx+1} {test_dates[0].date()}â†’{test_dates[-1].date()}"
        w_trades: List[Dict] = []
        w_in_sample_trades: List[Dict] = []

        for ticker in tickers:
            df = data[ticker]

            # --- IN-SAMPLE (train period) ---
            train_df = df[df.index.isin(train_dates)].copy()
            if len(train_df) < 50:
                continue

            try:
                if strategy_type == 'mean_reversion':
                    sigs = signal_mean_reversion(train_df)
                elif strategy_type == 'trend_following':
                    sigs = signal_trend_following(train_df)
                elif strategy_type == 'momentum':
                    sigs = signal_momentum(train_df)
                elif strategy_type == 'breakout':
                    sigs = signal_breakout(train_df)
                elif strategy_type == 'bb_squeeze':
                    sigs = signal_bb_squeeze(train_df)
                elif strategy_type == 'vwap_reversion':
                    sigs = signal_vwap_reversion(train_df)
                elif strategy_type == 'gap_fill':
                    sigs = signal_gap_fill(train_df)
                elif strategy_type == 'earnings_momentum':
                    sigs = signal_earnings_momentum(train_df)
                elif strategy_type == 'sector_rotation':
                    sigs = signal_sector_rotation(train_df, ticker, sector_data, spy_data)
                else:
                    sigs = pd.Series('HOLD', index=train_df.index)

                t_in = simulate_trades(sigs, train_df['Close'], stop_loss_pct, take_profit_pct)
                w_in_sample_trades.extend(t_in)
            except Exception:
                pass

            # --- OUT-OF-SAMPLE (test period) ---
            # Use train+test data for indicator calculation, only trade in test period
            combined_df = df[df.index <= test_dates[-1]].copy()
            if len(combined_df) < 50:
                continue

            try:
                if strategy_type == 'mean_reversion':
                    sigs_all = signal_mean_reversion(combined_df)
                elif strategy_type == 'trend_following':
                    sigs_all = signal_trend_following(combined_df)
                elif strategy_type == 'momentum':
                    sigs_all = signal_momentum(combined_df)
                elif strategy_type == 'breakout':
                    sigs_all = signal_breakout(combined_df)
                elif strategy_type == 'bb_squeeze':
                    sigs_all = signal_bb_squeeze(combined_df)
                elif strategy_type == 'vwap_reversion':
                    sigs_all = signal_vwap_reversion(combined_df)
                elif strategy_type == 'gap_fill':
                    sigs_all = signal_gap_fill(combined_df)
                elif strategy_type == 'earnings_momentum':
                    sigs_all = signal_earnings_momentum(combined_df)
                elif strategy_type == 'sector_rotation':
                    sigs_all = signal_sector_rotation(combined_df, ticker, sector_data, spy_data)
                else:
                    sigs_all = pd.Series('HOLD', index=combined_df.index)

                # Only use signals in test period
                test_sigs = sigs_all[sigs_all.index.isin(test_dates)]
                test_close = combined_df['Close'][combined_df.index.isin(test_dates)]
                t_out = simulate_trades(test_sigs, test_close, stop_loss_pct, take_profit_pct)
                w_trades.extend(t_out)
            except Exception:
                pass

        in_sample_trades.extend(w_in_sample_trades)
        out_sample_trades.extend(w_trades)

        w_metrics = aggregate_metrics(w_trades, window_label)
        all_window_results.append(w_metrics)
        print(f"  {window_label}: {w_metrics['trades']} trades, "
              f"WR={w_metrics['win_rate']*100:.0f}%, "
              f"Sharpe={w_metrics['sharpe']:.2f}", flush=True)

    overall = aggregate_metrics(out_sample_trades, "Overall Out-of-Sample")
    in_sample_overall = aggregate_metrics(in_sample_trades, "Overall In-Sample")

    # Overfitting detection: compare Sharpe
    oos_sharpe = overall['sharpe']
    is_sharpe = in_sample_overall['sharpe']
    if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.5:
        overfit_risk = "HIGH"
    elif is_sharpe > 0 and oos_sharpe < is_sharpe * 0.75:
        overfit_risk = "MEDIUM"
    elif oos_sharpe < 0:
        overfit_risk = "HIGH"
    else:
        overfit_risk = "LOW"

    return {
        'strategy': strategy_cfg['name'],
        'strategy_type': strategy_type,
        'window_results': all_window_results,
        'overall': overall,
        'in_sample': in_sample_overall,
        'overfit_risk': overfit_risk,
        'in_sample_sharpe': round(is_sharpe, 4),
        'out_sample_sharpe': round(oos_sharpe, 4),
    }

def run_pair_backtest(
    pair: Tuple[str, str],
    data: Dict[str, pd.DataFrame],
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_pct: float = 5.0,
    take_profit_pct: float = 10.0,
) -> Dict:
    """Walk-forward backtest for a single cointegrated pair."""
    a, b = pair
    if a not in data or b not in data:
        return {
            'pair': f"{a}/{b}",
            'error': 'Missing data',
            'overall': aggregate_metrics([]),
            'overfit_risk': 'N/A',
        }

    df_a = data[a]
    df_b = data[b]

    common = df_a.index.intersection(df_b.index)
    if len(common) < TRAIN_WINDOW + TEST_WINDOW:
        return {
            'pair': f"{a}/{b}",
            'error': f'Insufficient common data ({len(common)} days)',
            'overall': aggregate_metrics([]),
            'overfit_risk': 'N/A',
        }

    df_a = df_a.loc[common]
    df_b = df_b.loc[common]

    all_window_results = []
    out_sample_trades: List[Dict] = []
    in_sample_trades: List[Dict] = []

    windows = get_walk_forward_windows(len(common))

    for w_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        train_dates = common[tr_s:tr_e]
        test_dates = common[te_s:te_e]
        window_label = f"W{w_idx+1} {test_dates[0].date()}â†’{test_dates[-1].date()}"

        # In-sample
        t_a = df_a.loc[train_dates]
        t_b = df_b.loc[train_dates]
        sig_a, sig_b = signal_pair_trading(t_a, t_b, lookback, entry_z, exit_z)
        t_in_a = simulate_trades(sig_a, t_a['Close'], stop_loss_pct, take_profit_pct)
        t_in_b = simulate_trades(sig_b, t_b['Close'], stop_loss_pct, take_profit_pct)
        in_sample_trades.extend(t_in_a + t_in_b)

        # Out-of-sample
        oos_a_df = df_a.loc[test_dates]
        oos_b_df = df_b.loc[test_dates]

        # Calculate signals on combined data up to test end
        combined_a = df_a.loc[common[:te_e]]
        combined_b = df_b.loc[common[:te_e]]
        full_sig_a, full_sig_b = signal_pair_trading(combined_a, combined_b, lookback, entry_z, exit_z)

        oos_sig_a = full_sig_a[full_sig_a.index.isin(test_dates)]
        oos_sig_b = full_sig_b[full_sig_b.index.isin(test_dates)]

        t_out_a = simulate_trades(oos_sig_a, oos_a_df['Close'], stop_loss_pct, take_profit_pct)
        t_out_b = simulate_trades(oos_sig_b, oos_b_df['Close'], stop_loss_pct, take_profit_pct)
        w_trades = t_out_a + t_out_b

        out_sample_trades.extend(w_trades)
        w_metrics = aggregate_metrics(w_trades, window_label)
        all_window_results.append(w_metrics)

    overall = aggregate_metrics(out_sample_trades, "Overall")
    in_overall = aggregate_metrics(in_sample_trades, "In-Sample")

    # Check cointegration stability: use spread stationarity proxy
    # If std of z-score is close to 1 and mean ~0, cointegration holds
    spread = df_a['Close'] - df_b['Close']
    z_all = zscore_series(spread, lookback).dropna()
    coint_stable = bool(abs(float(z_all.mean())) < 0.3 and float(z_all.std()) < 1.5)

    oos_sharpe = overall['sharpe']
    is_sharpe = in_overall['sharpe']
    if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.5:
        overfit_risk = "HIGH"
    elif oos_sharpe < 0:
        overfit_risk = "HIGH"
    else:
        overfit_risk = "LOW"

    return {
        'pair': f"{a}/{b}",
        'ticker_a': a,
        'ticker_b': b,
        'overall': overall,
        'in_sample': in_overall,
        'window_results': all_window_results,
        'cointegration_stable': coint_stable,
        'overfit_risk': overfit_risk,
        'in_sample_sharpe': round(is_sharpe, 4),
        'out_sample_sharpe': round(oos_sharpe, 4),
    }

# â”€â”€â”€ Report Generator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_markdown_report(
    strategy_results: List[Dict],
    pair_results: List[Dict],
    generated_at: str,
) -> str:
    lines = []
    lines.append("# Trading Strategy Backtest Report")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Methodology")
    lines.append("- **Data**: 3 years daily OHLCV via yfinance")
    lines.append(f"- **Train window**: {TRAIN_WINDOW} days (1 year) â€” in-sample")
    lines.append(f"- **Test window**: {TEST_WINDOW} days (3 months) â€” out-of-sample")
    lines.append(f"- **Position size**: ${POSITION_SIZE:,.0f} per trade, fixed")
    lines.append(f"- **Commission**: ${COMMISSION:.0f} per trade (Alpaca)")
    lines.append("- **Slippage**: NOT modelled (limitation â€” results are optimistic)")
    lines.append("- **Signals**: Technical indicators computed on close prices")
    lines.append("")

    # â”€â”€ Summary Table â”€â”€
    lines.append("## Summary")
    lines.append("")
    lines.append("| Strategy | Trades | Win Rate | Sharpe | Max DD | Profit Factor | Overfit Risk |")
    lines.append("|----------|--------|----------|--------|--------|---------------|--------------|")

    for r in strategy_results:
        ov = r['overall']
        lines.append(
            f"| {r['strategy']} | {ov['trades']} | {ov['win_rate']*100:.0f}% | "
            f"{ov['sharpe']:.2f} | {ov['max_drawdown']*100:.1f}% | "
            f"{ov['profit_factor']:.2f} | {r['overfit_risk']} |"
        )
    lines.append("")

    # â”€â”€ Overfitting Analysis â”€â”€
    lines.append("## Overfitting Analysis")
    lines.append("")
    lines.append("Strategies where out-of-sample Sharpe < 50% of in-sample Sharpe are flagged **HIGH** risk.")
    lines.append("")
    lines.append("| Strategy | IS Sharpe | OOS Sharpe | OOS/IS Ratio | Overfit Risk |")
    lines.append("|----------|-----------|------------|--------------|--------------|")
    for r in strategy_results:
        is_s = r.get('in_sample_sharpe', 0)
        oos_s = r.get('out_sample_sharpe', 0)
        ratio = (oos_s / is_s) if is_s and is_s != 0 else float('nan')
        ratio_str = f"{ratio:.2f}" if not math.isnan(ratio) else "N/A"
        lines.append(
            f"| {r['strategy']} | {is_s:.2f} | {oos_s:.2f} | {ratio_str} | {r['overfit_risk']} |"
        )
    lines.append("")

    # â”€â”€ Per-Strategy Detail â”€â”€
    lines.append("## Per-Strategy Detail")
    lines.append("")
    for r in strategy_results:
        lines.append(f"### {r['strategy']}")
        lines.append("")
        ov = r['overall']
        lines.append(f"- **Total Trades**: {ov['trades']}")
        lines.append(f"- **Win Rate**: {ov['win_rate']*100:.1f}%")
        lines.append(f"- **Avg Return/Trade**: {ov['avg_return_pct']:.2f}%")
        lines.append(f"- **Sharpe (OOS)**: {ov['sharpe']:.2f}")
        lines.append(f"- **Max Drawdown**: {ov['max_drawdown']*100:.1f}%")
        lines.append(f"- **Profit Factor**: {ov['profit_factor']:.2f}")
        lines.append(f"- **Avg Trade Duration**: {ov['avg_duration_days']:.0f} days")
        lines.append(f"- **Total P&L**: ${ov['total_pnl']:,.0f}")
        lines.append(f"- **Overfitting Risk**: {r['overfit_risk']}")
        lines.append("")

        # Walk-forward window results
        windows = r.get('window_results', [])
        if windows:
            lines.append("**Walk-Forward Windows:**")
            lines.append("")
            lines.append("| Window | Trades | Win Rate | Sharpe | Max DD |")
            lines.append("|--------|--------|----------|--------|--------|")
            for w in windows:
                lines.append(
                    f"| {w['window']} | {w['trades']} | {w['win_rate']*100:.0f}% | "
                    f"{w['sharpe']:.2f} | {w['max_drawdown']*100:.1f}% |"
                )
            lines.append("")

        # Best / worst windows
        if len(windows) >= 2:
            best = max(windows, key=lambda x: x['sharpe'])
            worst = min(windows, key=lambda x: x['sharpe'])
            lines.append(f"- ðŸ† **Best period**: {best['window']} (Sharpe {best['sharpe']:.2f})")
            lines.append(f"- ðŸ“‰ **Worst period**: {worst['window']} (Sharpe {worst['sharpe']:.2f})")
            lines.append("")

        # Parameter recommendation
        if r['overfit_risk'] == 'HIGH':
            lines.append("âš ï¸ **Recommendation**: Consider looser parameters or eliminating this strategy. "
                         "Significant performance degradation out-of-sample detected.")
        elif ov['sharpe'] > 1.0 and ov['win_rate'] > 0.5:
            lines.append("âœ… **Recommendation**: Strategy shows consistent positive performance. "
                         "Consider going live with current parameters.")
        elif ov['sharpe'] > 0 and ov['profit_factor'] > 1.0:
            lines.append("ðŸ”¶ **Recommendation**: Marginal strategy. Consider adjusting parameters "
                         "(tighter stops, higher thresholds) before going live.")
        else:
            lines.append("âŒ **Recommendation**: Strategy is unprofitable out-of-sample. "
                         "Strongly consider removing.")
        lines.append("")

    # â”€â”€ Pair Trading Results â”€â”€
    lines.append("## Pair Trading Results")
    lines.append("")
    lines.append("| Pair | Trades | Win Rate | Avg Return | Sharpe | Cointegration Stable? | Overfit Risk |")
    lines.append("|------|--------|----------|------------|--------|-----------------------|--------------|")
    for p in pair_results:
        ov = p['overall']
        coint = "âœ… Yes" if p.get('cointegration_stable') else "âš ï¸ No"
        lines.append(
            f"| {p['pair']} | {ov['trades']} | {ov['win_rate']*100:.0f}% | "
            f"{ov['avg_return_pct']:.2f}% | {ov['sharpe']:.2f} | {coint} | {p.get('overfit_risk', 'N/A')} |"
        )
    lines.append("")

    # â”€â”€ Recommendations â”€â”€
    lines.append("## Recommendations")
    lines.append("")

    kill = [r['strategy'] for r in strategy_results
            if r['overall']['sharpe'] < 0 or r['overfit_risk'] == 'HIGH']
    keep = [r['strategy'] for r in strategy_results
            if r['overall']['sharpe'] > 1.0 and r['overfit_risk'] == 'LOW']
    adjust = [r['strategy'] for r in strategy_results
              if r not in kill and r not in keep
              and r['overall']['sharpe'] > 0]

    def strategy_summary(s_list):
        return ", ".join(s_list) if s_list else "None"

    if kill:
        lines.append(f"### âŒ Kill (negative Sharpe or HIGH overfit risk)")
        for s in kill:
            lines.append(f"- {s}")
        lines.append("")

    if keep:
        lines.append(f"### âœ… Keep (consistent positive Sharpe, LOW risk)")
        for s in keep:
            lines.append(f"- {s}")
        lines.append("")

    if adjust:
        lines.append(f"### ðŸ”¶ Adjust (works in some regimes â€” tune parameters)")
        for s in adjust:
            lines.append(f"- {s}")
        lines.append("")

    # Pair recommendations
    good_pairs = [p['pair'] for p in pair_results if p['overall']['sharpe'] > 0]
    bad_pairs = [p['pair'] for p in pair_results if p['overall']['sharpe'] <= 0]

    if good_pairs:
        lines.append(f"### âœ… Active Pairs (positive Sharpe)")
        for p in good_pairs:
            lines.append(f"- {p}")
        lines.append("")
    if bad_pairs:
        lines.append(f"### âŒ Inactive Pairs (non-positive Sharpe)")
        for p in bad_pairs:
            lines.append(f"- {p}")
        lines.append("")

    lines.append("---")
    lines.append("*Note: Results are approximations. Slippage not modelled. Past performance does not guarantee future results.*")
    lines.append("")

    return "\n".join(lines)

# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print("=" * 60)
    print("Walk-Forward Backtester â€” Trading Strategies")
    print("=" * 60)

    start_time = datetime.now()

    # â”€â”€ 1. Download data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    all_tickers = list(set(TICKER_UNIVERSE + [t for p in PAIRS for t in p]
                           + SECTOR_ETFS + ["SPY"]))
    # Remove invalid tickers (XYZ might cause issues)
    all_data = download_all_data(all_tickers, years=YEARS_OF_DATA)

    # Build sector ETF data references
    sector_data: Dict[str, pd.Series] = {}
    for etf in SECTOR_ETFS:
        if etf in all_data:
            sector_data[etf] = all_data[etf]['Close']

    spy_data = all_data.get('SPY', pd.DataFrame()).get('Close') if 'SPY' in all_data else None

    # â”€â”€ 2. Backtest each strategy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    strategy_results = []
    for cfg in STRATEGIES_CFG:
        if cfg['strategy_type'] == 'pair_trading':
            continue  # handled separately

        result = run_strategy_backtest(cfg, all_data, sector_data, spy_data)
        strategy_results.append(result)

    # â”€â”€ 3. Pair trading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nðŸ“Š Running Pair Trading backtests...", flush=True)
    pair_results = []
    for pair in PAIRS:
        print(f"  Pair: {pair[0]}/{pair[1]}", flush=True)
        result = run_pair_backtest(
            pair, all_data,
            lookback=60,
            entry_z=2.0,
            exit_z=0.5,
            stop_loss_pct=5.0,
            take_profit_pct=10.0,
        )
        pair_results.append(result)
        ov = result['overall']
        print(f"    {ov['trades']} trades, WR={ov['win_rate']*100:.0f}%, Sharpe={ov['sharpe']:.2f}", flush=True)

    # â”€â”€ 4. Save results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    elapsed = (datetime.now() - start_time).total_seconds()

    output = {
        'generated_at': generated_at,
        'elapsed_seconds': round(elapsed, 1),
        'config': {
            'train_window': TRAIN_WINDOW,
            'test_window': TEST_WINDOW,
            'position_size': POSITION_SIZE,
            'commission': COMMISSION,
            'years_of_data': YEARS_OF_DATA,
        },
        'strategy_results': strategy_results,
        'pair_results': pair_results,
    }

    project_dir = Path(__file__).parent
    json_path = project_dir / 'backtest_results.json'
    report_path = project_dir / 'backtest_report.md'

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nâœ… JSON results saved: {json_path}")

    report_md = generate_markdown_report(strategy_results, pair_results, generated_at)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"âœ… Markdown report saved: {report_path}")

    # Also save to OpenClaw workspace
    oc_report_dir = Path(r'C:\Users\User\.openclaw\workspace\mission-control\reports')
    oc_report_dir.mkdir(parents=True, exist_ok=True)
    oc_report_path = oc_report_dir / 'backtest-report.md'
    with open(oc_report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"âœ… Report also saved: {oc_report_path}")

    print(f"\nâ±ï¸  Total elapsed: {elapsed:.0f}s")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in strategy_results:
        ov = r['overall']
        flag = "âœ…" if ov['sharpe'] > 0.5 and r['overfit_risk'] != 'HIGH' else (
               "ðŸ”¶" if ov['sharpe'] > 0 else "âŒ")
        print(f"  {flag} {r['strategy']:30s} | Trades:{ov['trades']:4d} | "
              f"WR:{ov['win_rate']*100:.0f}% | Sharpe:{ov['sharpe']:+.2f} | "
              f"Overfit:{r['overfit_risk']}")

    print("\nPair Trading:")
    for p in pair_results:
        ov = p['overall']
        flag = "âœ…" if ov['sharpe'] > 0 else "âŒ"
        print(f"  {flag} {p['pair']:15s} | Trades:{ov['trades']:4d} | "
              f"WR:{ov['win_rate']*100:.0f}% | Sharpe:{ov['sharpe']:+.2f}")

    print(f"\nâœ… Done! Report at: {report_path}")
    return output


if __name__ == '__main__':
    main()

