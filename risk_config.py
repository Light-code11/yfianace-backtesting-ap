"""
Configurable risk management parameters.
"""
from dataclasses import dataclass


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