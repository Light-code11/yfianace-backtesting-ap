"""
Runtime configuration for daily autonomous trading.
"""
import os


TICKER_UNIVERSE = [
    # ── Core Tech / Mega-cap ─────────────────────────────────────────────────
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "CRM", "NVDA",

    # ── Semiconductors ───────────────────────────────────────────────────────
    "AMD", "INTC", "AVGO", "MRVL", "ARM", "QCOM", "LRCX", "KLAC", "ASML",

    # ── Finance / Payments ───────────────────────────────────────────────────
    "JPM", "V", "MA", "BAC", "WFC", "PYPL", "XYZ", "COIN",

    # ── Fintech / Growth ─────────────────────────────────────────────────────
    "SHOP", "MELI", "NU", "AFRM", "SOFI",

    # ── Biotech / Health ─────────────────────────────────────────────────────
    "UNH", "JNJ", "MRNA", "REGN", "ABBV", "LLY", "TMO", "ISRG", "DXCM",

    # ── Consumer Discretionary ───────────────────────────────────────────────
    "HD", "DIS", "PG", "WMT", "COST", "TGT", "SBUX", "MCD", "NKE", "LULU", "DECK",

    # ── Energy ───────────────────────────────────────────────────────────────
    "XOM", "CVX", "SLB", "EOG", "OXY", "MPC",

    # ── Industrials ──────────────────────────────────────────────────────────
    "CAT", "DE", "GE", "HON", "RTX", "LMT",

    # ── Cybersecurity / Cloud Infra ──────────────────────────────────────────
    "SMCI", "PLTR", "CRWD", "NET", "DDOG", "ZS", "PANW",
]


# ── Pair Trading Configuration ───────────────────────────────────────────────

PAIRS = [
    ("V", "MA"),        # Payment networks
    ("GOOGL", "META"),  # Digital advertising
    ("XOM", "CVX"),     # Oil majors
    ("COST", "WMT"),    # Retail giants
    ("JPM", "BAC"),     # Big banks
    ("AMD", "INTC"),    # Chip rivals
    ("AAPL", "MSFT"),   # Tech giants
    ("LMT", "RTX"),     # Defense
    ("CRWD", "PANW"),   # Cybersecurity
    ("DDOG", "NET"),    # Cloud infra
]

PAIR_TRADING_PARAMS = {
    "lookback_days": 60,
    "zscore_entry": 2.0,
    "zscore_exit": 0.5,
    "zscore_stop": 3.5,
    "min_correlation": 0.7,
    "cointegration_pvalue": 0.05,
    "max_position_pct": 0.10,   # 10% of portfolio per side
}


# ── Strategy Definitions ─────────────────────────────────────────────────────

STRATEGIES = [
    # ── 1. RSI / Bollinger Mean Reversion ────────────────────────────────────
    {
        "name": "cfg_rsi_mean_reversion",
        "description": "RSI/Bollinger mean reversion on liquid US equities",
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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
        "enabled": True,
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

    # ── 10. Pair Trading ──────────────────────────────────────────────────────
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


# ── Global Risk Parameters ────────────────────────────────────────────────────

RISK_PARAMS = {
    "max_position_size_pct": 20.0,
    "max_daily_loss_pct": 5.0,
    "max_positions": 10,
    "correlation_limit": 0.70,
    "min_signal_confidence": "MEDIUM",
    "auto_trading_enabled": True,
}


ALPACA_CONFIG = {
    "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    "request_timeout_sec": 15,
    "connect_timeout_sec": 5,
    "read_timeout_sec": 20,
}
