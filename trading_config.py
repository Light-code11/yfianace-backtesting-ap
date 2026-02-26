"""
Runtime configuration for daily autonomous trading.
"""
import os


TICKER_UNIVERSE = [
    # ── Core Tech / Mega-cap ─────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "CRM", "NVDA",
    "ORCL", "ADBE", "IBM", "UBER", "ABNB",

    # ── Semiconductors ───────────────────────────────────────────────────────
    "AMD", "INTC", "AVGO", "MRVL", "ARM", "QCOM", "LRCX", "KLAC", "ASML",
    "MPWR", "ON", "NXPI", "TXN", "MU",

    # ── Finance / Payments ───────────────────────────────────────────────────
    "JPM", "V", "MA", "BAC", "WFC", "PYPL", "XYZ", "COIN",
    "GS", "MS", "SCHW", "AXP",

    # ── Fintech / Growth ─────────────────────────────────────────────────────
    "SHOP", "MELI", "NU", "AFRM", "SOFI", "HOOD", "SQ",

    # ── Biotech / Health ─────────────────────────────────────────────────────
    "UNH", "JNJ", "MRNA", "REGN", "ABBV", "LLY", "TMO", "ISRG", "DXCM",
    "VEEV", "ALGN", "ILMN", "GILD",

    # ── Consumer Discretionary ───────────────────────────────────────────────
    "HD", "DIS", "PG", "WMT", "COST", "TGT", "SBUX", "MCD", "NKE", "LULU", "DECK",
    "ROST", "TJX", "BKNG", "MAR", "CMG",

    # ── Energy ───────────────────────────────────────────────────────────────
    "XOM", "CVX", "SLB", "EOG", "OXY", "MPC",
    "LNG", "DVN", "FANG", "PSX",

    # ── Industrials ──────────────────────────────────────────────────────────
    "CAT", "DE", "GE", "HON", "RTX", "LMT",
    "UNP", "UPS", "FDX", "WM", "MMM",

    # ── Cybersecurity / Cloud Infra ──────────────────────────────────────────
    "SMCI", "PLTR", "CRWD", "NET", "DDOG", "ZS", "PANW",
    "SNOW", "MDB", "TEAM", "OKTA", "FTNT",

    # ── Materials / Commodities ──────────────────────────────────────────────
    "FCX", "NEM", "APD", "LIN",

    # ── REITs / Real Estate ──────────────────────────────────────────────────
    "AMT", "PLD", "EQIX",

    # ── Telecom / Media ──────────────────────────────────────────────────────
    "TMUS", "VZ", "CMCSA",
]

# ── Crypto Universe (Alpaca crypto trading, separate from equity TICKER_UNIVERSE) ──
# These trade 24/7, use different strategies, and skip EDGAR/earnings entirely.
# Alpaca format: BTC/USD, ETH/USD (slash).  yfinance format: BTC-USD, ETH-USD (dash).
CRYPTO_UNIVERSE = [
    "BTC/USD",
    "ETH/USD",
]

# ── Crypto Pair ──────────────────────────────────────────────────────────────
# BTC and ETH are highly correlated and mean-revert well against each other.
CRYPTO_PAIRS = [
    ("BTC/USD", "ETH/USD"),   # dominant pair — high corr, fast mean-reversion
]

# ── Crypto Pair Trading Parameters ───────────────────────────────────────────
CRYPTO_PAIR_PARAMS = {
    "lookback_days": 30,          # shorter than equity (crypto moves faster)
    "zscore_entry": 1.5,
    "zscore_exit": 0.3,
    "zscore_stop": 3.0,
    "max_position_pct": 0.05,     # 5% per side (crypto is volatile)
}

# ── Crypto Risk Parameters ───────────────────────────────────────────────────
CRYPTO_RISK_PARAMS = {
    "max_position_pct": 0.05,     # 5% max per crypto position
    "max_total_crypto_pct": 0.10, # 10% max total crypto exposure
    "stop_multiplier": 3.0,       # 3x ATR stops (wider than equity due to vol)
    "trailing_atr": 3.5,          # trailing stop in ATR units
    "min_volume_usd": 1_000_000,  # minimum 24h volume
}

# ── Equity Pairs (cointegrated, auto-scanned 2026-02-25) ────────────────────
EQUITY_PAIRS = [
    ("SHOP", "SOFI"),  # corr=0.95 hl=5.5d sharpe=17.73 p=0.0049 (1y)
    ("AMZN", "NVDA"),  # corr=0.85 hl=8.0d sharpe=15.36 p=0.0258 (1y)
    ("MRVL", "KLAC"),  # corr=0.69 hl=9.6d sharpe=8.04 p=0.0354 (1y)
    ("TGT", "SBUX"),  # corr=0.85 hl=7.8d sharpe=7.80 p=0.0693 (1y)
    ("REGN", "LLY"),  # corr=0.90 hl=6.4d sharpe=5.96 p=0.0112 (1y)
    ("AMD", "MRVL"),  # corr=0.79 hl=9.0d sharpe=5.39 p=0.0065 (1y)
    ("V", "PYPL"),  # corr=0.67 hl=5.8d sharpe=5.49 p=0.0105 (1y)
    ("MRVL", "TSLA"),  # corr=0.75 hl=7.9d sharpe=5.30 p=0.0097 (1y)
    ("INTC", "ASML"),  # corr=0.88 hl=14.7d sharpe=6.74 p=0.0796 (2y)
    ("AAPL", "AMD"),  # corr=0.83 hl=12.4d sharpe=6.03 p=0.0570 (1y)
    ("AMZN", "SHOP"),  # corr=0.86 hl=14.0d sharpe=3.92 p=0.0071 (2y)
    ("MRVL", "QCOM"),  # corr=0.74 hl=8.7d sharpe=5.46 p=0.0489 (1y)
    ("PLTR", "NET"),  # corr=0.97 hl=11.8d sharpe=4.70 p=0.0206 (2y)
    ("AMZN", "TSLA"),  # corr=0.87 hl=11.9d sharpe=3.29 p=0.0044 (2y)
    ("PG", "DECK"),  # corr=0.56 hl=16.5d sharpe=5.03 p=0.0689 (2y)
    ("EOG", "OXY"),  # corr=0.57 hl=17.4d sharpe=4.09 p=0.0266 (2y)
    ("LULU", "DECK"),  # corr=0.72 hl=31.5d sharpe=4.63 p=0.0922 (2y)
    ("LRCX", "ASML"),  # corr=0.99 hl=8.8d sharpe=3.66 p=0.0308 (1y)
    ("INTC", "LRCX"),  # corr=0.79 hl=24.2d sharpe=3.31 p=0.0235 (2y)
    ("SHOP", "AFRM"),  # corr=0.92 hl=14.2d sharpe=3.65 p=0.0390 (2y)
    ("AMZN", "AVGO"),  # corr=0.81 hl=7.8d sharpe=2.52 p=0.0225 (1y)
    ("AAPL", "GOOGL"),  # corr=0.91 hl=9.9d sharpe=2.48 p=0.0294 (1y)
    ("WMT", "MCD"),  # corr=0.84 hl=19.2d sharpe=2.22 p=0.0465 (2y)
    ("NVDA", "AVGO"),  # corr=0.93 hl=18.2d sharpe=1.87 p=0.0378 (2y)
    ("BAC", "WFC"),  # corr=0.97 hl=9.0d sharpe=2.61 p=0.0981 (1y)
    ("INTC", "KLAC"),  # corr=0.66 hl=30.5d sharpe=1.84 p=0.0466 (2y)
    ("AVGO", "TSLA"),  # corr=0.90 hl=9.5d sharpe=2.00 p=0.0793 (1y)
    ("AMZN", "META"),  # corr=0.81 hl=20.6d sharpe=0.95 p=0.0263 (2y)
]

# Alias for backward compatibility — engine imports PAIRS
PAIRS = EQUITY_PAIRS

PAIR_PARAMS = {
    # Fast mean-reverting (half-life < 10 days) -- tighter z-score parameters
    ("SHOP", "SOFI"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMZN", "NVDA"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("MRVL", "KLAC"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("TGT", "SBUX"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("REGN", "LLY"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMD", "MRVL"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("V", "PYPL"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("MRVL", "TSLA"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("MRVL", "QCOM"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("LRCX", "ASML"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMZN", "AVGO"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("AAPL", "GOOGL"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("BAC", "WFC"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    ("AVGO", "TSLA"): {"zscore_entry": 1.5, "zscore_exit": 0.3, "zscore_stop": 3.5, "lookback_days": 60},
    # Slow mean-reverting (half-life 10-30 days) -- wider z-score parameters
    ("INTC", "ASML"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AAPL", "AMD"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMZN", "SHOP"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("PLTR", "NET"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMZN", "TSLA"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("PG", "DECK"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("EOG", "OXY"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("LULU", "DECK"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("INTC", "LRCX"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("SHOP", "AFRM"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("WMT", "MCD"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("NVDA", "AVGO"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("INTC", "KLAC"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AVGO", "NVDA"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AMZN", "META"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
    ("AVGO", "MRVL"): {"zscore_entry": 2.5, "zscore_exit": 0.5, "zscore_stop": 3.5, "lookback_days": 60},
}

PAIR_TRADING_PARAMS = {
    "lookback_days": 60,
    "zscore_entry": 2.0,
    "zscore_exit": 0.5,
    "zscore_stop": 3.5,
    "min_correlation": 0.4,  # lowered from 0.7 — pairs validated via cointegration (stronger test)
    "cointegration_pvalue": 0.05,
    "max_position_pct": 0.10,   # 10% of portfolio per side
}


# ── Strategy Definitions ─────────────────────────────────────────────────────
#
# AUDIT (2026-02-25): Generic strategies (RSI14, MACD, SMA cross, etc.) all
# showed NEGATIVE OOS Sharpe in walk-forward backtests. Root causes:
#   - RSI(14) mean reversion: exits too slow (uses profit target, not RSI exit)
#   - MACD trend: lags too much; whipsaws in volatile single-name stocks
#   - SMA 20/50: similar lag problem; no volume/ADX filter
#   - Volume breakout: 20-day high = too short; no consolidation filter
#   - BB squeeze: direction ambiguous without trend filter
#   - VWAP reversion: needs intraday data; daily VWAP proxy is noisy
#   - Gap fill: too dependent on market regime; high failure rate
#   - Earnings momentum: proxy (5% move) misfires outside earnings season
#   - Sector rotation: high transaction costs; signal too coarse at daily freq
#
# REPLACEMENT: optimized_strategies.py (5 focused strategies with academic basis)
# Old strategies DISABLED (not deleted) for reference.
#
# ── Disabled Generic Strategies (kept for reference) ─────────────────────────

_DISABLED_STRATEGIES = [
    # ── 1. RSI / Bollinger Mean Reversion ────────────────────────────────────
    {
        "name": "cfg_rsi_mean_reversion",
        "description": "RSI/Bollinger mean reversion on liquid US equities",
        "enabled": False,   # DISABLED: negative OOS Sharpe — exits too slow (RSI14 not RSI2)
        "strategy_type": "mean_reversion",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "RSI", "period": 14},
            {"name": "BB", "period": 20, "std_dev": 2},
            {"name": "ATR", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 4.5,
            "take_profit_pct": 9.0,
            "position_size_pct": 12.0,
        },
        "atr_stop_multiplier": 2.0,
    },

    # ── 2. MACD Trend Following ───────────────────────────────────────────────
    {
        "name": "cfg_macd_trend_following",
        "description": "MACD + long trend alignment strategy",
        "enabled": False,   # DISABLED: negative OOS Sharpe — MACD lags, whipsaws in single names
        "strategy_type": "trend_following",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "MACD"},
            {"name": "SMA", "period": 50},
            {"name": "SMA", "period": 200},
            {"name": "ATR", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 12.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
    },

    # ── 3. SMA Momentum ──────────────────────────────────────────────────────
    {
        "name": "cfg_sma_momentum",
        "description": "SMA crossover momentum with RSI filter",
        "enabled": False,   # DISABLED: negative OOS Sharpe — no ADX/volume filter; whipsaws
        "strategy_type": "momentum",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "SMA", "period": 20},
            {"name": "SMA", "period": 50},
            {"name": "RSI", "period": 14},
            {"name": "ATR", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "position_size_pct": 12.0,
        },
        "atr_stop_multiplier": 2.0,
    },

    # ── 4. Volume Breakout ───────────────────────────────────────────────────
    {
        "name": "cfg_volume_breakout",
        "description": "20-day breakout with volume expansion confirmation",
        "enabled": False,   # DISABLED: 20-day high too short; no consolidation filter
        "strategy_type": "breakout",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "SMA", "period": 20},
            {"name": "ATR", "period": 14},
            {"name": "VOLUME", "period": 20, "multiplier": 1.5},
        ],
        "risk_management": {
            "stop_loss_pct": 4.0,
            "take_profit_pct": 10.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
    },

    # ── 5. Bollinger Band Squeeze ─────────────────────────────────────────────
    {
        "name": "cfg_bb_squeeze",
        "description": "Bollinger Band low-bandwidth squeeze → expansion breakout",
        "enabled": False,   # DISABLED: directionally ambiguous without trend/ADX filter
        "strategy_type": "bb_squeeze",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "BB", "period": 20, "std_dev": 2},
            {"name": "ATR", "period": 14},
            {"name": "RSI", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 4.0,
            "take_profit_pct": 10.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
    },

    # ── 6. VWAP Reversion ────────────────────────────────────────────────────
    {
        "name": "cfg_vwap_reversion",
        "description": "Fade price when >2 std devs from rolling VWAP",
        "enabled": False,   # DISABLED: daily VWAP proxy is noisy; needs intraday data
        "strategy_type": "vwap_reversion",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "VWAP", "period": 20},
            {"name": "ATR", "period": 14},
            {"name": "RSI", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 3.5,
            "take_profit_pct": 7.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 1.5,
    },

    # ── 7. Gap Fill ───────────────────────────────────────────────────────────
    {
        "name": "cfg_gap_fill",
        "description": "Fade overnight gaps >2% expecting partial fill",
        "enabled": False,   # DISABLED: highly regime dependent; too many false positives
        "strategy_type": "gap_fill",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "ATR", "period": 14},
            {"name": "RSI", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 3.0,
            "take_profit_pct": 6.0,
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 1.5,
        "gap_threshold_pct": 2.0,
    },

    # ── 8. Earnings Momentum (PEAD) ───────────────────────────────────────────
    {
        "name": "cfg_earnings_momentum",
        "description": "Post-earnings announcement drift — ride 5%+ earnings surprise for 20-60 days",
        "enabled": False,   # DISABLED: proxy fires outside earnings season; use alpha_strategies_v2 Pre-Earnings
        "strategy_type": "earnings_momentum",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "ATR", "period": 14},
            {"name": "RSI", "period": 14},
            {"name": "SMA", "period": 20},
        ],
        "risk_management": {
            "stop_loss_pct": 6.0,
            "take_profit_pct": 15.0,
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 2.5,
        "earnings_surprise_threshold_pct": 5.0,
    },

    # ── 9. Sector Rotation ────────────────────────────────────────────────────
    {
        "name": "cfg_sector_rotation",
        "description": "Overweight strongest sector ETFs, underweight weakest vs SPY",
        "enabled": False,   # DISABLED: high turnover; signal too coarse at daily freq
        "strategy_type": "sector_rotation",
        "tickers": TICKER_UNIVERSE,
        "sector_etfs": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"],
        "indicators": [
            {"name": "SMA", "period": 20},
            {"name": "ATR", "period": 14},
            {"name": "RSI", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 12.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
        "rotation_lookback_days": 20,
    },
]

# ── Active Strategies (replaces the old STRATEGIES list) ─────────────────────
#
# Five evidence-based strategies from optimized_strategies.py:
#   1. opt_trend_momentum      — EMA 20/50 + ADX + pullback entry
#   2. opt_mean_reversion_rsi2 — RSI(2) + BB%B (Larry Connors)
#   3. opt_breakout_momentum   — 52-week high + consolidation + 2x volume
#   4. opt_overnight_anomaly   — Buy at close / sell at open (0.54 OOS Sharpe)
#   5. opt_vix_adaptive        — VIX regime switching (0.50 OOS Sharpe)
#   6. cfg_pair_trading        — Statistical arbitrage (keep as-is)

STRATEGIES = [
    # ── 1. Trend Momentum ─────────────────────────────────────────────────────
    {
        "name": "opt_trend_momentum",
        "description": "EMA 20/50 crossover + ADX>25 + pullback to EMA20. Multi-week trend holds.",
        "enabled": True,
        "strategy_type": "trend_momentum",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "EMA", "period": 20},
            {"name": "EMA", "period": 50},
            {"name": "EMA", "period": 200},
            {"name": "ADX", "period": 14},
            {"name": "ATR", "period": 14},
            {"name": "VOLUME", "period": 20},
        ],
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 15.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
        "regime_fit": ["BULL", "STRONG_BULL"],
        # Expected: 35-45% win rate, 2.5:1 R/R → positive expectancy
    },

    # ── 2. Mean Reversion RSI(2) ───────────────────────────────────────────────
    {
        "name": "opt_mean_reversion_rsi2",
        "description": "RSI(2)<10 + BB%B<0 + volume capitulation in uptrend. Larry Connors RSI(2).",
        "enabled": True,
        "strategy_type": "mean_reversion_rsi2",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "RSI", "period": 2},
            {"name": "BB", "period": 20, "std_dev": 2},
            {"name": "SMA", "period": 200},
            {"name": "ATR", "period": 14},
            {"name": "VOLUME", "period": 20},
        ],
        "risk_management": {
            "stop_loss_pct": 4.0,
            "take_profit_pct": 6.0,
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 1.5,
        "regime_fit": ["BULL", "STRONG_BULL", "NEUTRAL"],
        # Expected: 65-75% win rate, 1.2:1 R/R → high consistency
    },

    # ── 3. Breakout Momentum ───────────────────────────────────────────────────
    # NOTE: disabled pending proper ATR trailing-stop backtest.
    # Signal-based backtest (SMA50 exit) shows -8 Sharpe but only 3 trades — not meaningful.
    # Live trading with asymmetric_exits.py trailing stops should perform as intended.
    # Re-enable after backtesting with trailing-stop exit simulation.
    {
        "name": "opt_breakout_momentum",
        "description": "52-week high breakout + tight consolidation (<15% range) + 1.8x volume.",
        "enabled": False,  # pending trailing-stop backtest validation
        "strategy_type": "breakout_momentum",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "HIGH_52W", "period": 252},
            {"name": "SMA", "period": 50},
            {"name": "ATR", "period": 14},
            {"name": "VOLUME", "period": 20},
        ],
        "risk_management": {
            "stop_loss_pct": 7.0,
            "take_profit_pct": 30.0,
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 2.5,
        "regime_fit": ["BULL", "STRONG_BULL"],
        # Expected: 30-40% win rate, 4:1 R/R → explosive outlier payoff
    },

    # ── 4. Overnight Anomaly ───────────────────────────────────────────────────
    {
        "name": "opt_overnight_anomaly",
        "description": "Persistent overnight return anomaly. Tech/high-beta focus. OOS Sharpe 0.54.",
        "enabled": True,
        "strategy_type": "overnight_anomaly",
        "tickers": [
            "NVDA", "AMD", "TSLA", "META", "SHOP", "NFLX", "CRM",
            "MRVL", "AVGO", "PLTR", "CRWD", "NET", "DDOG",
            "AAPL", "MSFT", "GOOGL", "AMZN",
        ],
        "indicators": [
            {"name": "OVERNIGHT_RETURN", "period": 20},
            {"name": "VOLUME", "period": 20},
        ],
        "risk_management": {
            "stop_loss_pct": 3.0,
            "take_profit_pct": 5.0,
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 1.5,
        "regime_fit": ["BULL", "STRONG_BULL", "NEUTRAL"],
        # V1 OOS Sharpe: 0.54 (proven)
    },

    # ── 5. VIX Adaptive Momentum ───────────────────────────────────────────────
    {
        "name": "opt_vix_adaptive_momentum",
        "description": "VIX regime-switching: momentum low-VIX / reversion high-VIX. OOS Sharpe 0.50.",
        "enabled": True,
        "strategy_type": "vix_adaptive_momentum",
        "tickers": TICKER_UNIVERSE,
        "indicators": [
            {"name": "VIX"},
            {"name": "RSI", "period": 14},
            {"name": "MOM", "period": 126},
            {"name": "ATR", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 6.0,
            "take_profit_pct": 12.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
        "regime_fit": ["BULL", "STRONG_BULL", "NEUTRAL", "BEAR"],
        # V1 OOS Sharpe: 0.50 (proven)
    },

    # ── 6. Pair Trading ──────────────────────────────────────────────────────
    {
        "name": "cfg_pair_trading",
        "description": "Statistical arbitrage on cointegrated pairs via z-score mean reversion",
        "enabled": True,
        "strategy_type": "pair_trading",
        # Tickers are sourced from PAIRS config — leave empty for normal scanner
        "tickers": [],
        "indicators": [],
        "risk_management": {
            "stop_loss_pct": 5.0,
            "take_profit_pct": 10.0,
            "position_size_pct": 10.0,
        },
        "atr_stop_multiplier": 2.0,
    },
]


# ── Crypto Strategies ─────────────────────────────────────────────────────────
#
# Separate from equity STRATEGIES. These use wider parameters tuned for crypto's
# higher volatility. No EDGAR insiders, no earnings catalyst — crypto-specific only.
#
# Key differences vs equity strategies:
#   - ATR stop multiplier: 3.0× (vs 1.5× equity) — crypto has 5-10% daily swings
#   - RSI thresholds: oversold at 20 (vs 30), overbought at 80 (vs 70)
#   - Larger stop losses (10-15% vs 4-7%)
#   - No market hours check (24/7 trading)
#   - Uses CRYPTO_UNIVERSE, not TICKER_UNIVERSE

CRYPTO_STRATEGIES = [
    # ── 1. Crypto Trend Momentum ─────────────────────────────────────────────
    # EMA crossover + ADX on BTC/ETH. Crypto trends are strong and persistent.
    {
        "name": "crypto_trend_momentum",
        "description": "EMA 20/50 crossover + ADX>20 on BTC/ETH. Crypto trends run far.",
        "enabled": True,
        "strategy_type": "crypto_trend_momentum",
        "tickers": CRYPTO_UNIVERSE,
        "is_crypto": True,
        "indicators": [
            {"name": "EMA", "period": 20},
            {"name": "EMA", "period": 50},
            {"name": "EMA", "period": 200},
            {"name": "ADX", "period": 14},
            {"name": "ATR", "period": 14},
        ],
        "risk_management": {
            "stop_loss_pct": 12.0,      # 2× equity — crypto can drop 10% intraday
            "take_profit_pct": 30.0,    # crypto trends run further
            "position_size_pct": 8.0,
        },
        "atr_stop_multiplier": 3.0,     # 3× ATR initial stop (vs 1.5× equity)
        "rsi_oversold": 20,             # more extreme oversold threshold
        "rsi_overbought": 80,           # more extreme overbought threshold
        "regime_fit": ["CRYPTO_BULL", "CRYPTO_NEUTRAL"],
    },

    # ── 2. Crypto Mean Reversion RSI ────────────────────────────────────────
    # RSI(14) deep oversold / overbought on crypto. Crypto fear spikes are buyable.
    {
        "name": "crypto_mean_reversion",
        "description": "RSI(14)<20 capitulation buy / RSI>80 fade on BTC/ETH. Crypto fear = opportunity.",
        "enabled": True,
        "strategy_type": "crypto_mean_reversion",
        "tickers": CRYPTO_UNIVERSE,
        "is_crypto": True,
        "indicators": [
            {"name": "RSI", "period": 14},
            {"name": "BB", "period": 20, "std_dev": 2},
            {"name": "ATR", "period": 14},
            {"name": "SMA", "period": 200},
        ],
        "risk_management": {
            "stop_loss_pct": 10.0,
            "take_profit_pct": 15.0,
            "position_size_pct": 6.0,   # smaller — contrarian in volatile asset
        },
        "atr_stop_multiplier": 3.0,
        "rsi_oversold": 20,
        "rsi_overbought": 80,
        "regime_fit": ["CRYPTO_BULL", "CRYPTO_NEUTRAL", "CRYPTO_FEAR"],
    },

    # ── 3. Crypto Pair Trading ────────────────────────────────────────────────
    # BTC/ETH spread mean-reversion. Highly correlated, fast-reverting pair.
    {
        "name": "crypto_pair_trading",
        "description": "BTC/ETH spread z-score mean reversion. 24/7 pair arb.",
        "enabled": True,
        "strategy_type": "crypto_pair_trading",
        "tickers": [],   # sourced from CRYPTO_PAIRS
        "is_crypto": True,
        "indicators": [],
        "risk_management": {
            "stop_loss_pct": 8.0,
            "take_profit_pct": 12.0,
            "position_size_pct": 5.0,   # smaller per leg — 2 positions open
        },
        "atr_stop_multiplier": 3.0,
        "regime_fit": ["CRYPTO_BULL", "CRYPTO_NEUTRAL", "CRYPTO_BEAR", "CRYPTO_FEAR"],
    },
]


# ── Global Risk Parameters ────────────────────────────────────────────────────

RISK_PARAMS = {
    "max_position_size_pct": 20.0,
    "max_daily_loss_pct": 5.0,
    "max_positions": 10,
    "correlation_limit": 0.70,
    "min_signal_confidence": "MEDIUM",
    "min_conviction_score": 55.0,
    "auto_trading_enabled": True,
}


ALPACA_CONFIG = {
    "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    "request_timeout_sec": 15,
    "connect_timeout_sec": 5,
    "read_timeout_sec": 20,
}

# AUTO-GENERATED OPTIMIZER PARAMS START
OPTIMIZED_STRATEGY_PARAMS = {
    "trend_momentum": {}
}
# AUTO-GENERATED OPTIMIZER PARAMS END
