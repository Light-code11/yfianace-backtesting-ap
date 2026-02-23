"""
Runtime configuration for daily autonomous trading.
"""
import os


TICKER_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN", "TSLA", "META", "NFLX",
    "JPM", "V", "MA", "UNH", "HD", "PG", "JNJ", "XOM", "CVX", "BAC", "WFC",
    "DIS", "CRM", "PYPL", "SQ", "COIN",
]


STRATEGIES = [
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
]


RISK_PARAMS = {
    "max_position_size_pct": 20.0,
    "max_daily_loss_pct": 5.0,
    "max_positions": 8,
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
