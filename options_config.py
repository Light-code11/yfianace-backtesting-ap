"""
Options trading configuration.

Defines the universe of optionable tickers, strategy parameters, and risk controls
for the Alpaca paper-trading options module.
"""
import os

# ── Optionable Tickers (high-liquidity, tight spreads) ───────────────────────

OPTIONS_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX",
    # Finance / Payments
    "JPM", "V", "MA",
    # Healthcare
    "UNH",
    # Consumer
    "HD", "PG", "COST", "WMT",
    # Semiconductors
    "AMD", "NVDA",
    # Index ETFs (most liquid options market)
    "SPY", "QQQ", "IWM",
    # Energy
    "XOM", "CVX",
]

# ── Strategy Configurations ───────────────────────────────────────────────────

OPTIONS_STRATEGIES = {
    "wheel": {
        "enabled": True,
        "description": (
            "Sell cash-secured puts on stocks we want to own; "
            "sell covered calls if assigned"
        ),
        # Target delta for put leg (negative = put)
        "target_delta": -0.30,          # 30-delta put → ~70 % prob OTM
        "min_premium_pct": 0.01,        # Minimum 1 % of strike as premium
        "dte_range": (21, 45),           # 21-45 days to expiration
        "max_positions": 3,
    },

    "iron_condor": {
        "enabled": True,
        "description": "Sell iron condors on range-bound / high-IV stocks",
        "short_delta": 0.16,            # ~84 % probability of profit
        "wing_width": 5,                # $5 wide wings
        "dte_range": (30, 45),
        "max_positions": 5,
        "min_iv_rank": 30,              # Only trade when IV rank > 30 %
    },

    "credit_spread": {
        "enabled": True,
        "description": "Directional credit spreads driven by ML signals",
        "short_delta": 0.30,
        "spread_width": 5,              # $5 wide spread
        "dte_range": (21, 45),
        "max_positions": 5,
        "max_risk_per_trade_pct": 0.02, # Max 2 % of portfolio at risk per trade
    },

    "earnings_play": {
        "enabled": True,
        "description": "Sell iron condors ahead of earnings to capture IV crush",
        "dte_range": (1, 7),            # Close to earnings for maximum IV crush
        "short_delta": 0.20,
        "wing_width": 5,
        "max_positions": 3,
    },

    "protective_puts": {
        "enabled": True,
        "description": "Buy protective puts to hedge existing equity positions",
        "target_delta": -0.20,          # OTM puts for cheap hedge
        "dte_range": (30, 60),
        "hedge_pct": 0.50,              # Hedge 50 % of equity exposure
    },
}

# ── Risk Parameters ───────────────────────────────────────────────────────────

RISK_PARAMS = {
    "max_total_options_allocation_pct": 0.40,  # Max 40 % of portfolio in options
    "max_single_underlying_pct": 0.10,         # Max 10 % risk on any one name
    "min_open_interest": 100,                  # Liquidity filter
    "min_volume": 50,                          # Daily volume filter
    "max_bid_ask_spread_pct": 0.05,            # Max 5 % bid-ask spread
    # Position management thresholds
    "winner_close_pct": 0.50,                  # Close at 50 % of max profit
    "loser_close_pct": 2.00,                   # Stop at 2× premium received
    "roll_dte_threshold": 7,                   # Roll / close when DTE ≤ 7
}

# ── Alpaca Connection (shared with equity bot) ───────────────────────────────

ALPACA_CONFIG = {
    "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    "connect_timeout_sec": 5,
    "read_timeout_sec": 20,
}
