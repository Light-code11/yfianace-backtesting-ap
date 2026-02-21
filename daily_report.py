"""
Daily P&L reporting for OpenClaw cron integration.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore
    YFINANCE_AVAILABLE = False

from database import AutoTradingState, PaperTrade, SessionLocal, TradeExecution

try:
    from database import RLAllocationLog, RiskEventLog
except ImportError:
    RLAllocationLog = None  # type: ignore
    RiskEventLog = None  # type: ignore


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_alpaca_snapshot() -> Dict[str, Any]:
    try:
        from alpaca_client import AlpacaClient

        client = AlpacaClient()
        account_resp = client.get_account()
        pos_resp = client.get_positions()

        if not account_resp.get("success"):
            return {"success": False, "error": account_resp.get("error")}

        account = account_resp.get("account", {})
        positions = pos_resp.get("positions", []) if pos_resp.get("success") else []

        return {
            "success": True,
            "account": {
                "equity": _safe_float(account.get("equity")),
                "cash": _safe_float(account.get("cash")),
                "buying_power": _safe_float(account.get("buying_power")),
                "last_equity": _safe_float(account.get("last_equity")),
            },
            "positions": positions,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


def _paper_position_snapshot(db) -> Dict[str, Any]:
    open_positions = db.query(PaperTrade).filter(PaperTrade.is_open == True).all()
    closed_trades = db.query(PaperTrade).filter(PaperTrade.is_open == False).all()

    total_closed_pnl = sum(_safe_float(t.profit_loss_usd) for t in closed_trades)
    open_value = 0.0
    positions_payload = []

    for pos in open_positions:
        current_price = _safe_float(pos.entry_price)
        if YFINANCE_AVAILABLE:
            try:
                market = yf.Ticker(pos.ticker).history(period="1d")
                if not market.empty:
                    current_price = _safe_float(market["Close"].iloc[-1], current_price)
            except Exception:
                pass

        qty = _safe_float(pos.quantity)
        entry = _safe_float(pos.entry_price)
        pnl = (current_price - entry) * qty
        pnl_pct = ((current_price - entry) / entry * 100.0) if entry > 0 else 0.0

        open_value += current_price * qty
        positions_payload.append(
            {
                "ticker": pos.ticker,
                "qty": qty,
                "entry": entry,
                "current": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    initial_capital = 100000.0
    equity = initial_capital + total_closed_pnl + open_value
    cash = max(0.0, initial_capital + total_closed_pnl)

    return {
        "account": {
            "equity": equity,
            "cash": cash,
            "buying_power": cash,
            "last_equity": initial_capital,
        },
        "positions": positions_payload,
        "total_closed_pnl": total_closed_pnl,
    }


def _format_alpaca_positions(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for pos in positions:
        qty = _safe_float(pos.get("qty"))
        entry = _safe_float(pos.get("avg_entry_price"))
        current = _safe_float(pos.get("current_price"), entry)
        pnl = _safe_float(pos.get("unrealized_pl"))
        pnl_pct = _safe_float(pos.get("unrealized_plpc")) * 100.0
        payload.append(
            {
                "ticker": pos.get("symbol"),
                "qty": qty,
                "entry": entry,
                "current": current,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )
    return payload


def _build_summary(report: Dict[str, Any]) -> str:
    account = report.get("account", {})
    risk = report.get("risk_status", {})
    alerts = report.get("alerts", [])

    return (
        f"{report.get('date')} | Equity ${account.get('equity', 0):,.2f} | "
        f"Daily PnL {report.get('daily_pnl_pct', 0):+.2f}% (${report.get('daily_pnl', 0):+,.2f}) | "
        f"Open positions {len(report.get('positions', []))} | "
        f"Exposure {risk.get('exposure_pct', 0):.2f}% | "
        f"Regime {report.get('regime', 'UNKNOWN')} | Alerts {len(alerts)}"
    )


def generate_daily_report() -> Dict[str, Any]:
    """
    Generate daily trading report JSON and text summary.
    """
    db = SessionLocal()
    now = datetime.utcnow()
    start_of_day = datetime(now.year, now.month, now.day)

    try:
        alpaca = _load_alpaca_snapshot()
        if alpaca.get("success"):
            account = alpaca["account"]
            positions = _format_alpaca_positions(alpaca.get("positions", []))
            data_source = "alpaca"
        else:
            paper = _paper_position_snapshot(db)
            account = paper["account"]
            positions = paper["positions"]
            data_source = "paper"

        equity = _safe_float(account.get("equity"))
        last_equity = _safe_float(account.get("last_equity"))
        daily_pnl = equity - last_equity
        daily_pnl_pct = ((daily_pnl / last_equity) * 100.0) if last_equity > 0 else 0.0

        trades_today = db.query(TradeExecution).filter(TradeExecution.created_at >= start_of_day).all()
        trade_rows = []
        for trade in trades_today:
            trade_rows.append(
                {
                    "ticker": trade.ticker,
                    "side": trade.side,
                    "qty": _safe_float(trade.qty),
                    "price": _safe_float(trade.signal_price),
                    "strategy": trade.strategy_name,
                }
            )

        state = db.query(AutoTradingState).order_by(AutoTradingState.updated_at.desc()).first()

        risk_status = {
            "drawdown_pct": 0.0,
            "daily_loss_pct": abs(min(daily_pnl_pct, 0.0)),
            "exposure_pct": 0.0,
            "positions_count": len(positions),
            "circuit_breaker": False,
        }

        if state:
            risk_status["drawdown_pct"] = _safe_float(getattr(state, "daily_pnl_pct", 0.0))
            risk_status["positions_count"] = int(getattr(state, "num_positions", len(positions)) or len(positions))
            if equity > 0:
                exposure = ((equity - _safe_float(getattr(state, "cash_balance", 0.0))) / equity) * 100.0
                risk_status["exposure_pct"] = max(0.0, exposure)
            risk_status["circuit_breaker"] = bool(getattr(state, "circuit_breaker_triggered", False))

        rl_allocations = {}
        regime = "UNKNOWN"
        if RLAllocationLog is not None:
            latest_alloc = db.query(RLAllocationLog).order_by(RLAllocationLog.created_at.desc()).first()
            if latest_alloc:
                rl_allocations = latest_alloc.allocations or {}
                regime = latest_alloc.regime or "UNKNOWN"

        alerts: List[str] = []
        if risk_status["daily_loss_pct"] > 2.0:
            alerts.append("Daily loss exceeded 2%")
        if risk_status["circuit_breaker"]:
            alerts.append("Circuit breaker active")

        if RiskEventLog is not None:
            recent_risk_events = db.query(RiskEventLog).filter(
                RiskEventLog.created_at >= start_of_day
            ).order_by(RiskEventLog.created_at.desc()).limit(5).all()
            for evt in recent_risk_events:
                alerts.append(f"{evt.event_type}: {evt.message}")

        total_pnl = 0.0
        if data_source == "paper":
            closed = db.query(PaperTrade).filter(PaperTrade.is_open == False).all()
            total_pnl = sum(_safe_float(t.profit_loss_usd) for t in closed)
        else:
            total_pnl = daily_pnl

        total_pnl_pct = ((total_pnl / last_equity) * 100.0) if last_equity > 0 else 0.0

        report = {
            "date": now.date().isoformat(),
            "source": data_source,
            "account": {
                "equity": equity,
                "cash": _safe_float(account.get("cash")),
                "buying_power": _safe_float(account.get("buying_power")),
            },
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": positions,
            "trades_today": trade_rows,
            "risk_status": risk_status,
            "regime": regime,
            "rl_allocations": rl_allocations,
            "alerts": alerts,
        }
        report["summary"] = _build_summary(report)

        return report
    finally:
        db.close()


if __name__ == "__main__":
    payload = generate_daily_report()
    print(json.dumps(payload, indent=2, default=str))
    print("\nSUMMARY:")
    print(payload.get("summary", ""))
