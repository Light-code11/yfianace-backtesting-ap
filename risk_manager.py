"""
Centralized risk management for autonomous trading.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from risk_config import RiskConfig


class RiskManager:
    """
    Conservative rule-based risk manager.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

        self.account_value: float = 0.0
        self.cash: float = 0.0
        self.peak_equity: float = 0.0
        self.daily_start_equity: float = 0.0
        self.last_equity_date: Optional[str] = None

        self.current_vix: float = 20.0
        self.open_positions: List[Dict[str, Any]] = []
        self.position_correlations: Dict[str, float] = {}

        self.drawdown_halted: bool = False
        self.pause_until: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.recent_trade_results: List[Dict[str, Any]] = []

    def update_market_context(self, vix: Optional[float] = None):
        if vix is not None and vix > 0:
            self.current_vix = float(vix)

    def update_portfolio_state(
        self,
        account_value: float,
        cash: float,
        positions: Optional[List[Dict[str, Any]]] = None,
        correlations: Optional[Dict[str, float]] = None,
    ):
        self.account_value = max(float(account_value), 0.0)
        self.cash = max(float(cash), 0.0)

        if self.peak_equity <= 0:
            self.peak_equity = self.account_value
        else:
            self.peak_equity = max(self.peak_equity, self.account_value)

        today = datetime.utcnow().date().isoformat()
        if self.last_equity_date != today:
            self.daily_start_equity = self.account_value
            self.last_equity_date = today

        self.open_positions = positions or []
        self.position_correlations = correlations or {}

        drawdown = self._drawdown_pct()
        if drawdown > self.config.max_drawdown_pct:
            self.drawdown_halted = True
        elif self.drawdown_halted and drawdown <= (self.config.max_drawdown_pct / 2.0):
            # Resume after recovery from -10% halt to at least -5%
            self.drawdown_halted = False

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check global trading gates: drawdown, daily loss, circuit breaker.
        """
        if self.drawdown_halted:
            return False, f"Drawdown halt active ({self._drawdown_pct():.2f}%)"

        daily_pnl_pct = self._daily_pnl_pct()
        if daily_pnl_pct <= -abs(self.config.max_daily_loss_pct):
            return False, f"Daily loss limit exceeded ({daily_pnl_pct:.2f}%)"

        if self.pause_until and datetime.utcnow() < self.pause_until:
            return False, f"Circuit breaker active until {self.pause_until.isoformat()}"

        return True, "OK"

    def validate_order(
        self,
        ticker: str,
        side: str,
        qty: float,
        price: float,
        sector: Optional[str] = None,
        correlation_map: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """
        Validate order against position, exposure, correlation, and concentration rules.
        """
        side_lower = (side or "").lower()
        qty = float(qty or 0)
        price = float(price or 0)

        if qty <= 0 or price <= 0:
            return False, "Invalid order quantity/price"

        # Sells are generally allowed for risk reduction
        if side_lower == "sell":
            return True, "OK"

        if self.account_value <= 0:
            return False, "Account value unavailable"

        order_notional = qty * price
        order_pct = (order_notional / self.account_value) * 100.0

        if order_pct > self.config.max_position_pct:
            return False, f"Position too large ({order_pct:.2f}% > {self.config.max_position_pct:.2f}%)"

        if len(self.open_positions) >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        projected_exposure = self._exposure_pct() + order_pct
        if projected_exposure > self.config.max_exposure_pct:
            return False, f"Exposure limit exceeded ({projected_exposure:.2f}% > {self.config.max_exposure_pct:.2f}%)"

        correlations = correlation_map or self.position_correlations or {}
        high_corr = correlations.get(ticker)
        if high_corr is not None and float(high_corr) > self.config.max_correlation:
            return False, f"Correlation too high ({float(high_corr):.2f} > {self.config.max_correlation:.2f})"

        if sector:
            sector_pct = self._sector_exposure_pct(sector)
            projected_sector = sector_pct + order_pct
            if projected_sector > self.config.max_sector_pct:
                return False, (
                    f"Sector concentration exceeded ({projected_sector:.2f}% > "
                    f"{self.config.max_sector_pct:.2f}%)"
                )

        return True, "OK"

    def calculate_position_size(
        self,
        ticker: str,
        signal_strength: float,
        account_value: float,
        volatility: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ) -> float:
        """
        Calculate conservative position size in USD with Kelly and volatility scaling.
        """
        if account_value <= 0:
            return 0.0

        strength = max(0.0, min(float(signal_strength), 1.0))
        if strength <= 0:
            return 0.0

        base_pct = self.config.max_position_pct * strength

        # Fractional Kelly for safety
        kelly = 0.5
        if kelly_fraction is not None:
            kelly = max(0.1, min(float(kelly_fraction), 1.0))
        base_pct *= kelly

        # Volatility reduction (higher vol -> smaller size)
        if volatility is not None and volatility > 0:
            vol_scale = max(0.2, min(1.0, 0.3 / float(volatility)))
            base_pct *= vol_scale

        # VIX scaling when fear is elevated
        if self.current_vix > self.config.vix_threshold and self.current_vix > 0:
            base_pct *= self.config.vix_threshold / self.current_vix

        # Respect remaining exposure headroom
        current_exposure = self._exposure_pct()
        remaining = max(0.0, self.config.max_exposure_pct - current_exposure)
        capped_pct = min(base_pct, self.config.max_position_pct, remaining)

        return max(0.0, account_value * (capped_pct / 100.0))

    def update_state(self, trade_result: Dict[str, Any]):
        """
        Update risk state after trade execution/fill.
        """
        pnl = float(trade_result.get("pnl", 0.0))
        timestamp = trade_result.get("timestamp") or datetime.utcnow().isoformat()

        self.recent_trade_results.append(
            {
                "timestamp": timestamp,
                "ticker": trade_result.get("ticker"),
                "pnl": pnl,
                "win": pnl > 0,
            }
        )
        self.recent_trade_results = self.recent_trade_results[-100:]

        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.config.consecutive_loss_limit:
            self.pause_until = datetime.utcnow() + timedelta(hours=self.config.pause_hours)

    def get_risk_report(self) -> Dict[str, Any]:
        """
        Snapshot of current risk metrics.
        """
        now = datetime.utcnow()
        return {
            "timestamp": now.isoformat(),
            "config": asdict(self.config),
            "account_value": self.account_value,
            "cash": self.cash,
            "drawdown_pct": self._drawdown_pct(),
            "daily_pnl_pct": self._daily_pnl_pct(),
            "exposure_pct": self._exposure_pct(),
            "positions_count": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker": bool(self.pause_until and now < self.pause_until),
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "drawdown_halted": self.drawdown_halted,
            "vix": self.current_vix,
        }

    def _drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.account_value) / self.peak_equity
        return max(0.0, dd * 100.0)

    def _daily_pnl_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return ((self.account_value - self.daily_start_equity) / self.daily_start_equity) * 100.0

    def _exposure_pct(self) -> float:
        if self.account_value <= 0:
            return 0.0
        total_notional = 0.0
        for pos in self.open_positions:
            mv = pos.get("market_value")
            if mv is None:
                qty = float(pos.get("qty", 0.0))
                price = float(pos.get("current_price", 0.0))
                mv = qty * price
            total_notional += abs(float(mv))
        return (total_notional / self.account_value) * 100.0

    def _sector_exposure_pct(self, sector: str) -> float:
        if self.account_value <= 0:
            return 0.0
        sector = (sector or "UNKNOWN").upper()
        sector_notional = 0.0
        for pos in self.open_positions:
            pos_sector = str(pos.get("sector") or "UNKNOWN").upper()
            if pos_sector != sector:
                continue

            mv = pos.get("market_value")
            if mv is None:
                qty = float(pos.get("qty", 0.0))
                price = float(pos.get("current_price", 0.0))
                mv = qty * price
            sector_notional += abs(float(mv))

        return (sector_notional / self.account_value) * 100.0