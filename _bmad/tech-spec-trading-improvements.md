---
title: 'Trading Platform Improvements - RL + Risk + OpenClaw'
slug: 'trading-improvements'
created: '2026-02-21'
status: 'ready-for-dev'
---

# Tech-Spec: Trading Platform Improvements

## Overview
Three improvements to the existing autonomous trading engine:
1. Reinforcement Learning agent for strategy allocation
2. Risk management hardening
3. OpenClaw reporting integration (daily P&L cron)

## Task 1: Reinforcement Learning Strategy Allocator

### File: `rl_strategy_allocator.py` (NEW)

Build a PPO-based RL agent that learns to allocate capital between trading strategies based on market regime.

**Environment:**
- State space: [regime_probabilities(3), strategy_performances(n), portfolio_metrics(5), market_features(10)]
  - Regime probs from HMM (bull/bear/consolidation)
  - Per-strategy: recent win rate, avg return, sharpe ratio, max drawdown
  - Portfolio: total PnL, drawdown, cash ratio, position count, correlation
  - Market: VIX level, SPY trend, volume ratio, breadth, momentum
- Action space: Continuous allocation weights across strategies (sum to 1.0)
- Reward: risk-adjusted returns (Sharpe ratio of portfolio over episode)
- Episode: 20 trading days

**Implementation:**
- Use `stable-baselines3` PPO agent (pip install stable-baselines3)
- Custom Gym environment wrapping the backtesting engine
- Train on historical data (walk-forward: train on year N, test on N+1)
- Save trained model to `./rl_models/`
- Inference method: `get_allocation(current_state) -> Dict[str, float]`
- Retraining: weekly cron, retrain on last 2 years of data
- Fallback: if RL model not available, use equal-weight allocation

**Integration with autonomous_trading_engine.py:**
- In `_evaluate_signal()`, weight the signal strength by the RL allocation for that strategy
- If RL allocates 0 to a strategy, skip its signals entirely
- Log RL allocations to database

### File: `rl_environment.py` (NEW)

Custom Gym environment:
```python
class TradingAllocEnv(gym.Env):
    # observation_space: Box(low=-inf, high=inf, shape=(state_dim,))
    # action_space: Box(low=0, high=1, shape=(num_strategies,))
    # step(): advance 1 day, apply allocation, calculate reward
    # reset(): start new episode from random historical point
```

### File: `train_rl_model.py` (NEW)

CLI script to train/retrain the RL model:
```bash
python train_rl_model.py --episodes 1000 --eval-freq 100
```

## Task 2: Risk Management Hardening

### File: `risk_manager.py` (NEW)

Centralized risk management that the autonomous engine must consult before EVERY trade.

**Rules:**
1. **Max portfolio drawdown**: If drawdown > 10% from peak, halt all new trades until recovery to -5%
2. **Max daily loss**: If daily PnL < -2%, halt trading for the rest of the day
3. **Max position size**: No single position > 5% of portfolio value
4. **Max total exposure**: Total positions < 80% of portfolio (keep 20% cash minimum)
5. **Correlation check**: Don't enter new position if correlation > 0.8 with existing positions (use 30-day rolling correlation)
6. **Circuit breaker**: If 3 consecutive losing trades, pause for 4 hours
7. **Max positions**: No more than 10 concurrent positions
8. **Volatility scaling**: Reduce position size when VIX > 25 (scale by 25/VIX)
9. **Sector concentration**: No more than 30% in any single sector

**Implementation:**
```python
class RiskManager:
    def __init__(self, config: RiskConfig):
        ...
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed (drawdown, daily loss, circuit breaker)"""
    
    def validate_order(self, ticker, side, qty, price) -> Tuple[bool, str]:
        """Check if specific order passes all risk rules"""
    
    def calculate_position_size(self, ticker, signal_strength, account_value) -> float:
        """Size position considering Kelly, volatility, correlation"""
    
    def update_state(self, trade_result):
        """Update internal state after trade execution"""
    
    def get_risk_report(self) -> Dict:
        """Current risk metrics for dashboard"""
```

**Integration with autonomous_trading_engine.py:**
- Replace existing `_calculate_position_size()` with `risk_manager.calculate_position_size()`
- Add `risk_manager.can_trade()` check at the start of each scan cycle
- Add `risk_manager.validate_order()` before every `alpaca.place_order()`
- Log risk events to database

### File: `risk_config.py` (NEW)

Configurable risk parameters:
```python
@dataclass
class RiskConfig:
    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 2.0
    max_position_pct: float = 5.0
    max_exposure_pct: float = 80.0
    max_correlation: float = 0.8
    consecutive_loss_limit: int = 3
    pause_hours: int = 4
    max_positions: int = 10
    vix_threshold: float = 25.0
    max_sector_pct: float = 30.0
```

## Task 3: OpenClaw Daily P&L Report Script

### File: `daily_report.py` (NEW)

Script that generates a daily trading summary. Designed to be called by OpenClaw cron.

**Output (JSON):**
```python
{
    "date": "2026-02-21",
    "account": {
        "equity": 100500.00,
        "cash": 45000.00,
        "buying_power": 90000.00
    },
    "daily_pnl": 500.00,
    "daily_pnl_pct": 0.5,
    "total_pnl": 500.00,
    "total_pnl_pct": 0.5,
    "positions": [
        {"ticker": "AAPL", "qty": 10, "entry": 180.50, "current": 182.00, "pnl": 15.00, "pnl_pct": 0.83}
    ],
    "trades_today": [
        {"ticker": "AAPL", "side": "buy", "qty": 10, "price": 180.50, "strategy": "momentum_breakout"}
    ],
    "risk_status": {
        "drawdown_pct": 0.5,
        "daily_loss_pct": 0.0,
        "exposure_pct": 55.0,
        "positions_count": 3,
        "circuit_breaker": false
    },
    "regime": "BULL",
    "rl_allocations": {"momentum": 0.4, "mean_reversion": 0.3, "breakout": 0.3},
    "alerts": []
}
```

**Also generates a human-readable summary string** for the OpenClaw cron to send via Telegram.

### Integration:
- Add `"report": "python daily_report.py"` to scripts
- Reads from Alpaca API (positions, account) + local DB (trades, strategies)
- If Alpaca not configured, reads from paper trading simulator state

## Task 4: Update autonomous_trading_engine.py

Integrate all three improvements into the main engine:

1. Import and initialize RiskManager
2. Import and use RLStrategyAllocator (with fallback)
3. Add risk checks before every trade
4. Weight signals by RL allocation
5. Log RL allocations and risk events
6. Update daily_report integration

## Task 5: Requirements update

### File: `requirements-trading-platform.txt` (UPDATE)

Add:
```
stable-baselines3>=2.0.0
gymnasium>=0.29.0
torch>=2.0.0
```

## Task 6: Build verification

- Verify all new files import correctly: `python -c "from rl_strategy_allocator import RLStrategyAllocator; from risk_manager import RiskManager; from daily_report import generate_daily_report"`
- Verify no syntax errors in modified autonomous_trading_engine.py
- Git commit with message: `feat: RL strategy allocator, risk management hardening, daily P&L reporting`

## Notes
- RL training requires GPU for speed but works on CPU (just slower)
- Risk manager should be conservative — better to miss trades than blow up
- Daily report must work even without Alpaca keys (uses paper trading state)
- All new modules should have graceful fallbacks if dependencies are missing
