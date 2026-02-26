"""
Deterministic trade justification generation and logging.
"""
import json
from typing import Any, Dict, Optional

from database import SessionLocal, TradeJustification


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def generate_justification(decision_factors: Dict[str, Any], action: str, ticker: str) -> str:
    """
    Generate a compact 2-3 sentence justification from structured decision factors.
    """
    factors = decision_factors or {}
    strategy = factors.get("strategy_name") or "unknown_strategy"
    conviction = _to_float(factors.get("conviction_score"))
    conviction_tier = factors.get("conviction_tier")
    regime = factors.get("macro_regime") or factors.get("regime") or "UNKNOWN"
    kelly_fraction = _to_float(factors.get("kelly_fraction"))
    kelly_pct = (kelly_fraction * 100.0) if kelly_fraction is not None and abs(kelly_fraction) <= 1 else kelly_fraction
    confluence = _to_int(factors.get("confluence_count"))
    earnings_days = _to_int(factors.get("earnings_days_away"))
    rsi = _to_float(factors.get("trend_rsi14") or factors.get("rsi14") or factors.get("rsi"))
    price = _to_float(factors.get("trend_price") or factors.get("price"))
    sma50 = _to_float(factors.get("trend_sma50") or factors.get("sma50"))
    rejection_reason = factors.get("rejection_reason")
    size_pct = _to_float(factors.get("adjusted_position_size_pct") or factors.get("position_size_pct"))
    stop_price = _to_float(factors.get("dynamic_stop_loss") or factors.get("initial_stop") or factors.get("stop_loss"))

    if action.upper() == "REJECT":
        sentence_1 = f"REJECTED {ticker} from {strategy}: {rejection_reason or 'entry criteria not satisfied'}."
    else:
        sentence_1 = f"{action.upper()} {ticker} via {strategy} under {regime} regime."

    sentence_2 = (
        f"Conviction {conviction:.1f}" if conviction is not None else "Conviction n/a"
    ) + (
        f" ({conviction_tier})" if conviction_tier else ""
    ) + f", Kelly {_fmt_pct(kelly_pct)}, confluence {confluence if confluence is not None else 'n/a'}."

    trend_clause = "Trend data unavailable"
    if price is not None and sma50 is not None:
        trend_side = "above" if price >= sma50 else "below"
        trend_clause = f"Price {trend_side} SMA50 ({price:.2f} vs {sma50:.2f})"
    elif rsi is not None:
        trend_clause = f"RSI {rsi:.1f}"

    earnings_clause = ""
    if earnings_days is not None:
        earnings_clause = f", earnings in {earnings_days} day(s)"

    risk_clause = ""
    if size_pct is not None or stop_price is not None:
        risk_parts = []
        if size_pct is not None:
            risk_parts.append(f"size {size_pct:.2f}%")
        if stop_price is not None:
            risk_parts.append(f"stop ${stop_price:.2f}")
        risk_clause = ". Risk: " + ", ".join(risk_parts) + "."

    sentence_3 = f"{trend_clause}{earnings_clause}.{risk_clause}".replace("..", ".")
    return " ".join([sentence_1, sentence_2, sentence_3]).strip()


def log_trade_decision(
    ticker: str,
    action: str,
    strategy_name: Optional[str],
    decision_factors: Optional[Dict[str, Any]] = None,
    rejection_reason: Optional[str] = None,
    trade_execution_id: Optional[int] = None,
) -> Optional[int]:
    """
    Persist a trade decision (execution or rejection) to trade_justifications.
    """
    factors = dict(decision_factors or {})
    if strategy_name and "strategy_name" not in factors:
        factors["strategy_name"] = strategy_name
    if rejection_reason:
        factors["rejection_reason"] = rejection_reason

    justification = generate_justification(factors, action=action, ticker=ticker)

    kelly_fraction = _to_float(factors.get("kelly_fraction"))
    if kelly_fraction is not None and abs(kelly_fraction) <= 1:
        kelly_pct = kelly_fraction * 100.0
    else:
        kelly_pct = kelly_fraction

    trend_price = _to_float(factors.get("trend_price"))
    trend_sma50 = _to_float(factors.get("trend_sma50"))
    sma50_position = None
    if trend_price is not None and trend_sma50 is not None:
        sma50_position = "above" if trend_price >= trend_sma50 else "below"

    db = SessionLocal()
    try:
        row = TradeJustification(
            trade_execution_id=trade_execution_id,
            ticker=ticker,
            action=action.upper(),
            strategy_name=strategy_name,
            justification=justification,
            conviction_score=_to_float(factors.get("conviction_score")),
            kelly_pct=kelly_pct,
            regime=factors.get("macro_regime") or factors.get("regime"),
            earnings_days_away=_to_int(factors.get("earnings_days_away")),
            rsi=_to_float(factors.get("trend_rsi14") or factors.get("rsi14") or factors.get("rsi")),
            sma50_position=sma50_position,
            confluence_count=_to_int(factors.get("confluence_count")),
            rejection_reason=rejection_reason,
            decision_factors_json=json.dumps(factors, default=str),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()
