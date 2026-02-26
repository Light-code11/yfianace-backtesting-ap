"""
AI strategy discovery pipeline.

Usage:
    python strategy_discovery.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

from database import SessionLocal, Strategy, StrategyResearch, init_db
from trading_config import TICKER_UNIVERSE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ALLOWED_STRATEGY_TYPES = {
    "momentum",
    "mean_reversion",
    "breakout",
    "trend_following",
    "bb_squeeze",
    "vwap_reversion",
    "gap_fill",
    "sector_rotation",
}

PASS_GATES = {
    "min_sharpe": 0.5,
    "min_win_rate": 0.40,
    "max_drawdown": 20.0,
    "min_trades": 10,
}


def _parse_json_block(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.find("```", start)
        raw = raw[start:end].strip()
    elif raw.startswith("```"):
        raw = raw.strip("`")
    return json.loads(raw)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"]) if not df.empty else df


def _pick_top_liquid_tickers(limit: int = 30) -> List[str]:
    subset = TICKER_UNIVERSE[:80]
    end = datetime.utcnow()
    start = end - timedelta(days=120)
    raw = yf.download(
        tickers=subset,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return TICKER_UNIVERSE[:limit]

    liquidity: List[Tuple[str, float]] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in subset:
            try:
                close = raw[("Close", t)].dropna()
                vol = raw[("Volume", t)].dropna()
                if close.empty or vol.empty:
                    continue
                liquidity.append((t, float((close * vol).tail(30).mean())))
            except Exception:
                continue

    liquidity.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in liquidity[:limit]] if liquidity else TICKER_UNIVERSE[:limit]


def _market_context() -> Dict[str, Any]:
    spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
    vix = yf.Ticker("^VIX").history(period="3mo", auto_adjust=True)

    regime = "unknown"
    breadth = None
    if not spy.empty:
        close = spy["Close"]
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        ret_21 = float((close.iloc[-1] / close.iloc[-22]) - 1) if len(close) > 22 else 0.0
        if close.iloc[-1] > sma200.iloc[-1] and ret_21 > 0.02:
            regime = "strong_bull"
        elif close.iloc[-1] > sma50.iloc[-1]:
            regime = "bull"
        elif close.iloc[-1] < sma200.iloc[-1] and ret_21 < -0.02:
            regime = "strong_bear"
        elif close.iloc[-1] < sma50.iloc[-1]:
            regime = "bear"
        else:
            regime = "sideways"

    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLP", "XLY"]
    sector_raw = yf.download(sector_etfs, period="3mo", auto_adjust=True, progress=False, threads=True)
    sector_perf: Dict[str, float] = {}
    if not sector_raw.empty and isinstance(sector_raw.columns, pd.MultiIndex):
        for etf in sector_etfs:
            try:
                c = sector_raw[("Close", etf)].dropna()
                if len(c) > 21:
                    sector_perf[etf] = round(float((c.iloc[-1] / c.iloc[-22]) - 1), 4)
            except Exception:
                continue
    leaders = [k for k, _ in sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)[:3]]

    universe = _pick_top_liquid_tickers(30)
    prices = yf.download(universe, period="6mo", auto_adjust=True, progress=False, threads=True)
    if isinstance(prices.columns, pd.MultiIndex):
        above_50 = 0
        total = 0
        for t in universe:
            try:
                c = prices[("Close", t)].dropna()
                if len(c) < 55:
                    continue
                total += 1
                if c.iloc[-1] > c.rolling(50).mean().iloc[-1]:
                    above_50 += 1
            except Exception:
                continue
        breadth = round(above_50 / total, 4) if total > 0 else None

    return {
        "macro_regime": regime,
        "vix": round(float(vix["Close"].iloc[-1]), 2) if not vix.empty else None,
        "sector_leaders": leaders,
        "sector_performance_21d": sector_perf,
        "market_breadth": breadth,
    }


def _load_past_results(limit: int = 20) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        rows = (
            db.query(StrategyResearch)
            .order_by(StrategyResearch.created_at.desc())
            .limit(limit)
            .all()
        )
        out = []
        for r in rows:
            out.append(
                {
                    "strategy_name": r.strategy_name,
                    "source": r.source,
                    "backtest_sharpe": r.backtest_sharpe,
                    "walk_forward_sharpe": r.walk_forward_sharpe,
                    "is_deployed": bool(r.is_deployed),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
        return out
    finally:
        db.close()


def _build_prompt(context: Dict[str, Any], past_results: List[Dict[str, Any]], n: int) -> str:
    return f"""
You are a quantitative trading researcher. Generate {n} trading strategies for US equities.

Current market conditions:
- Macro regime: {context.get('macro_regime')}
- VIX: {context.get('vix')}
- Sector leaders: {context.get('sector_leaders')}
- Market breadth: {context.get('market_breadth')}

Previously tried strategies and their results:
{json.dumps(past_results, indent=2)}

Requirements:
- Must use ONLY data available from yfinance (OHLCV)
- Must have clear entry and exit rules
- strategy_type must be one of: {sorted(ALLOWED_STRATEGY_TYPES)}
- Provide a Python function: def signal(df: pd.DataFrame) -> pd.Series with values BUY/SELL/HOLD
- The code must run with only pandas/numpy and the input dataframe
- Target Sharpe > 1.0 on 2-year backtest

Return ONLY valid JSON in this schema:
{{
  "strategies": [
    {{
      "name": "string",
      "description": "string",
      "strategy_type": "string",
      "entry_conditions": ["string"],
      "exit_conditions": ["string"],
      "signal_function_code": "python code string",
      "recommended_tickers": ["AAPL", "MSFT"],
      "holding_period_days": 5,
      "target_sharpe": 1.0,
      "why_it_should_work": "string",
      "indicators": [{{"name": "SMA", "period": 20}}],
      "risk_management": {{"stop_loss_pct": 4.0, "take_profit_pct": 8.0, "position_size_pct": 8.0}}
    }}
  ]
}}
""".strip()


def _compile_signal_fn(code: str):
    namespace: Dict[str, Any] = {}
    safe_globals = {"pd": pd, "np": np, "__builtins__": __builtins__}
    exec(code, safe_globals, namespace)
    fn = namespace.get("signal") or safe_globals.get("signal")
    if fn is None:
        raise ValueError("signal function not defined")
    return fn


def _simulate(df: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
    close = df["Close"].astype(float)
    sig = signals.reindex(df.index).fillna("HOLD")

    position = 0
    entry = 0.0
    returns: List[float] = []
    equity = [1.0]

    prev = "HOLD"
    for i, px in enumerate(close):
        s = str(sig.iloc[i])
        if position == 0 and s == "BUY" and prev != "BUY":
            entry = float(px)
            position = 1
        elif position == 1 and (s == "SELL" or i == len(close) - 1):
            ret = (float(px) - entry) / entry if entry > 0 else 0.0
            returns.append(ret)
            equity.append(equity[-1] * (1.0 + ret))
            position = 0
            entry = 0.0
        prev = s

    if not returns:
        return {
            "sharpe": 0.0,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
        }

    arr = np.array(returns, dtype=float)
    sharpe = float((arr.mean() / arr.std()) * np.sqrt(252)) if arr.std() > 0 else 0.0
    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks

    return {
        "sharpe": round(sharpe, 4),
        "total_return_pct": round((eq[-1] - 1.0) * 100.0, 4),
        "win_rate": round(float(np.mean(arr > 0)), 4),
        "max_drawdown": round(abs(float(np.min(dd))) * 100.0 if len(dd) else 0.0, 4),
        "trade_count": int(len(arr)),
    }


def _walk_forward(df: pd.DataFrame, fn) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    split = max(len(df) - 126, int(len(df) * 0.75))
    split = min(max(split, 60), len(df) - 30)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    full_metrics = _simulate(df, fn(df))
    train_metrics = _simulate(train, fn(train))
    test_metrics = _simulate(test, fn(test))
    return full_metrics, train_metrics, test_metrics


def _passes(full_m: Dict[str, Any], train_m: Dict[str, Any], test_m: Dict[str, Any]) -> bool:
    return (
        full_m["sharpe"] > PASS_GATES["min_sharpe"]
        and full_m["win_rate"] > PASS_GATES["min_win_rate"]
        and full_m["max_drawdown"] < PASS_GATES["max_drawdown"]
        and full_m["trade_count"] >= PASS_GATES["min_trades"]
        and train_m["total_return_pct"] > 0
        and test_m["total_return_pct"] > 0
    )


def _sanitize_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return slug.strip("_")[:40] or "ai_strategy"


def _deploy_strategy(idea: Dict[str, Any]) -> str:
    db = SessionLocal()
    try:
        base_name = f"ai_{datetime.utcnow().strftime('%Y%m%d')}_{_sanitize_name(idea.get('name', 'strategy'))}"
        name = base_name
        i = 1
        while db.query(Strategy).filter(Strategy.name == name).first() is not None:
            i += 1
            name = f"{base_name}_{i}"

        stype = str(idea.get("strategy_type", "momentum")).strip().lower()
        if stype not in ALLOWED_STRATEGY_TYPES:
            stype = "momentum"

        risk = dict(idea.get("risk_management") or {})
        base_position = float(risk.get("position_size_pct", 8.0))
        half_position = max(1.0, round(base_position * 0.5, 2))

        row = Strategy(
            name=name,
            description=idea.get("description", "AI-generated strategy"),
            tickers=idea.get("recommended_tickers") or _pick_top_liquid_tickers(10),
            entry_conditions={
                "rules": idea.get("entry_conditions", []),
                "ai_generated": True,
                "generation_date": datetime.utcnow().date().isoformat(),
                "signal_function_code": idea.get("signal_function_code", ""),
            },
            exit_conditions={"rules": idea.get("exit_conditions", [])},
            stop_loss_pct=float(risk.get("stop_loss_pct", 4.0)),
            take_profit_pct=float(risk.get("take_profit_pct", 8.0)),
            position_size_pct=half_position,
            holding_period_days=int(idea.get("holding_period_days", 5) or 5),
            rationale=idea.get("why_it_should_work", ""),
            market_analysis="AI strategy discovery deployment",
            risk_assessment="Starts at 50% normal position sizing (paper trading)",
            strategy_type=stype,
            indicators=idea.get("indicators") or [],
            is_active=True,
            is_paper_trading=True,
        )
        db.add(row)
        db.commit()
        return name
    finally:
        db.close()


def _log_result(
    name: str,
    idea: Dict[str, Any],
    full_m: Dict[str, Any],
    test_m: Dict[str, Any],
    deployed: bool,
    deployment_name: Optional[str],
    note: str,
) -> None:
    db = SessionLocal()
    try:
        row = StrategyResearch(
            strategy_name=name,
            source="ai_generated",
            params_json=json.dumps({
                "strategy_type": idea.get("strategy_type"),
                "indicators": idea.get("indicators", []),
                "risk_management": idea.get("risk_management", {}),
                "recommended_tickers": idea.get("recommended_tickers", []),
            }),
            backtest_sharpe=full_m.get("sharpe"),
            backtest_return_pct=full_m.get("total_return_pct"),
            backtest_win_rate=full_m.get("win_rate"),
            backtest_max_dd=full_m.get("max_drawdown"),
            backtest_trade_count=full_m.get("trade_count"),
            walk_forward_sharpe=test_m.get("sharpe"),
            walk_forward_return_pct=test_m.get("total_return_pct"),
            is_deployed=deployed,
            deployment_date=datetime.utcnow() if deployed else None,
            notes=f"{note}; deployed_name={deployment_name}" if deployment_name else note,
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def run_pipeline(num_strategies: int = 5, max_weekly_deploy: int = 3) -> Dict[str, Any]:
    init_db()

    context = _market_context()
    past_results = _load_past_results(limit=20)

    client = OpenAI()
    prompt = _build_prompt(context=context, past_results=past_results, n=num_strategies)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.6,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": "You are a quantitative strategy researcher. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    ideas = _parse_json_block(response.choices[0].message.content).get("strategies", [])

    top30 = _pick_top_liquid_tickers(30)
    deploy_count = 0
    results: List[Dict[str, Any]] = []

    for idx, idea in enumerate(ideas, start=1):
        raw_name = idea.get("name") or f"ai_strategy_{idx}"
        print(f"Evaluating AI idea {idx}/{len(ideas)}: {raw_name}")

        stype = str(idea.get("strategy_type", "momentum")).strip().lower()
        if stype not in ALLOWED_STRATEGY_TYPES:
            stype = "momentum"
            idea["strategy_type"] = stype

        code = str(idea.get("signal_function_code", "")).strip()
        if not code:
            _log_result(raw_name, idea, {}, {}, False, None, "missing signal code")
            results.append({"name": raw_name, "status": "failed", "reason": "missing signal code"})
            continue

        try:
            fn = _compile_signal_fn(code)
        except Exception as exc:
            _log_result(raw_name, idea, {}, {}, False, None, f"compile error: {exc}")
            results.append({"name": raw_name, "status": "failed", "reason": f"compile error: {exc}"})
            continue

        eval_tickers = [t for t in (idea.get("recommended_tickers") or []) if t in top30]
        if not eval_tickers:
            eval_tickers = top30

        full_all: List[Dict[str, Any]] = []
        train_all: List[Dict[str, Any]] = []
        test_all: List[Dict[str, Any]] = []

        for ticker in eval_tickers:
            try:
                df = _normalize_ohlcv(yf.Ticker(ticker).history(period="2y", auto_adjust=True))
                if len(df) < 260:
                    continue
                full_m, train_m, test_m = _walk_forward(df, fn)
                full_all.append(full_m)
                train_all.append(train_m)
                test_all.append(test_m)
            except Exception:
                continue

        if not full_all:
            _log_result(raw_name, idea, {}, {}, False, None, "no valid backtest runs")
            results.append({"name": raw_name, "status": "failed", "reason": "no valid backtest runs"})
            continue

        full_avg = {
            "sharpe": round(float(np.mean([m["sharpe"] for m in full_all])), 4),
            "total_return_pct": round(float(np.mean([m["total_return_pct"] for m in full_all])), 4),
            "win_rate": round(float(np.mean([m["win_rate"] for m in full_all])), 4),
            "max_drawdown": round(float(np.mean([m["max_drawdown"] for m in full_all])), 4),
            "trade_count": int(round(float(np.mean([m["trade_count"] for m in full_all])))),
        }
        train_avg = {
            "total_return_pct": round(float(np.mean([m["total_return_pct"] for m in train_all])), 4)
        }
        test_avg = {
            "sharpe": round(float(np.mean([m["sharpe"] for m in test_all])), 4),
            "total_return_pct": round(float(np.mean([m["total_return_pct"] for m in test_all])), 4),
        }

        passed = _passes(full_avg, train_avg, test_avg)
        deployed_name = None
        if passed and deploy_count < max_weekly_deploy:
            deployed_name = _deploy_strategy(idea)
            deploy_count += 1

        _log_result(
            raw_name,
            idea,
            full_avg,
            test_avg,
            deployed=bool(deployed_name),
            deployment_name=deployed_name,
            note=f"passed={passed}; tickers={len(eval_tickers)}",
        )

        results.append(
            {
                "name": raw_name,
                "strategy_type": stype,
                "passed": passed,
                "deployed_name": deployed_name,
                "full_metrics": full_avg,
                "walk_forward": test_avg,
                "tickers_evaluated": eval_tickers,
            }
        )

    out = {
        "generated_at": datetime.utcnow().isoformat(),
        "market_context": context,
        "deployed_count": deploy_count,
        "results": results,
    }
    Path("ai_discovery_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved AI discovery results to ai_discovery_results.json")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI strategy discovery pipeline")
    parser.add_argument("--num-strategies", type=int, default=5)
    parser.add_argument("--max-weekly-deploy", type=int, default=3)
    args = parser.parse_args()

    run_pipeline(num_strategies=args.num_strategies, max_weekly_deploy=args.max_weekly_deploy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
