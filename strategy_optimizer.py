"""
Strategy parameter optimizer with walk-forward validation.

Usage:
    python strategy_optimizer.py --tickers AAPL,MSFT,NVDA --strategies trend_momentum
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from database import SessionLocal, StrategyResearch, init_db
from optimized_strategies import (
    TrendMomentumStrategy,
    MeanReversionRSI2Strategy,
    BreakoutMomentumStrategy,
    OvernightAnomalyOptimized,
    VIXAdaptiveMomentumOptimized,
)
from trading_config import TICKER_UNIVERSE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "trend_momentum": {
        "ema_fast": [10, 15, 20, 25],
        "ema_slow": [40, 50, 60, 75],
        "adx_threshold": [20, 25, 30],
    },
    "mean_reversion_rsi2": {
        "rsi_period": [2, 3, 5],
        "rsi_oversold": [5, 10, 15],
        "rsi_overbought": [85, 90, 95],
        "sma_trend": [100, 150, 200],
    },
    "breakout_momentum": {
        "breakout_period": [10, 15, 20, 30],
        "volume_multiplier": [1.2, 1.5, 2.0],
    },
    "vix_adaptive_momentum": {
        "momentum_period": [63, 126, 189],
        "vol_lookback": [15, 20, 30],
    },
    "overnight_anomaly": {
        "volume_threshold": [1.0, 1.2, 1.5],
        "min_gap_pct": [0.0, 0.1, 0.2],
    },
}


@dataclass
class Metrics:
    sharpe: float
    total_return_pct: float
    win_rate: float
    max_drawdown: float
    trade_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "total_return_pct": self.total_return_pct,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
        }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"]) if not df.empty else df


def _param_combos(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*(param_grid[k] for k in keys)):
        combo = {k: v for k, v in zip(keys, values)}
        if combo.get("ema_fast", 0) >= combo.get("ema_slow", 10**9):
            continue
        combos.append(combo)
    return combos


def _build_strategy(strategy_type: str, params: Dict[str, Any]):
    if strategy_type == "trend_momentum":
        return TrendMomentumStrategy(
            ema_fast=int(params["ema_fast"]),
            ema_slow=int(params["ema_slow"]),
            adx_threshold=float(params["adx_threshold"]),
        )
    if strategy_type == "mean_reversion_rsi2":
        _ = params.get("rsi_period", 2)  # retained for grid traceability
        return MeanReversionRSI2Strategy(
            rsi2_oversold=float(params["rsi_oversold"]),
            rsi2_overbought=float(params["rsi_overbought"]),
            sma_trend_period=int(params["sma_trend"]),
        )
    if strategy_type == "breakout_momentum":
        return BreakoutMomentumStrategy(
            high_lookback=int(params["breakout_period"]),
            volume_breakout_mult=float(params["volume_multiplier"]),
        )
    if strategy_type == "vix_adaptive_momentum":
        return VIXAdaptiveMomentumOptimized(
            momentum_lookback=int(params["momentum_period"]),
            reversal_lookback=int(params["vol_lookback"]),
        )
    if strategy_type == "overnight_anomaly":
        return OvernightAnomalyOptimized(
            volume_min_ratio=float(params["volume_threshold"]),
            threshold_pct=float(params["min_gap_pct"]),
        )
    raise ValueError(f"Unsupported strategy type: {strategy_type}")


def _simulate_from_signals(df: pd.DataFrame, signals: pd.Series) -> Metrics:
    close = df["Close"].astype(float)
    sig = signals.reindex(df.index).fillna("HOLD")

    position = 0
    entry_price = 0.0
    trade_returns: List[float] = []
    equity_curve: List[float] = [1.0]
    prev = "HOLD"

    for i, price in enumerate(close):
        signal = str(sig.iloc[i])
        if position == 0 and signal == "BUY" and prev != "BUY":
            position = 1
            entry_price = float(price)
        elif position == 1 and (signal == "SELL" or i == len(close) - 1):
            ret = (float(price) - entry_price) / entry_price if entry_price > 0 else 0.0
            trade_returns.append(ret)
            equity_curve.append(equity_curve[-1] * (1.0 + ret))
            position = 0
            entry_price = 0.0
        prev = signal

    if not trade_returns:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0)

    rets = np.array(trade_returns, dtype=float)
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets))
    sharpe = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0
    win_rate = float(np.mean(rets > 0))
    total_return_pct = (equity_curve[-1] - 1.0) * 100.0

    eq = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    max_dd = abs(float(np.min(drawdowns))) * 100.0 if len(drawdowns) else 0.0

    return Metrics(
        sharpe=round(sharpe, 4),
        total_return_pct=round(total_return_pct, 4),
        win_rate=round(win_rate, 4),
        max_drawdown=round(max_dd, 4),
        trade_count=len(trade_returns),
    )


def _evaluate(df: pd.DataFrame, strategy_type: str, params: Dict[str, Any]) -> Metrics:
    strat = _build_strategy(strategy_type, params)
    signals = strat.generate_signals(df)
    return _simulate_from_signals(df, signals)


def _walk_forward_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = max(len(df) - 126, int(len(df) * 0.75))
    split = min(max(split, 60), len(df) - 30)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _qualifies(train: Metrics, test: Metrics, full: Metrics) -> bool:
    return (
        full.sharpe > 0.5
        and full.win_rate > 0.40
        and full.max_drawdown < 20.0
        and full.trade_count >= 10
        and train.total_return_pct > 0
        and test.total_return_pct > 0
    )


def _pick_top_liquid_tickers(limit: int = 30) -> List[str]:
    subset = TICKER_UNIVERSE[:80]
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=120)
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

    scores: List[Tuple[str, float]] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in subset:
            try:
                c = raw[("Close", t)].dropna()
                v = raw[("Volume", t)].dropna()
                if c.empty or v.empty:
                    continue
                dollar_vol = float((c * v).tail(30).mean())
                scores.append((t, dollar_vol))
            except Exception:
                continue
    else:
        return TICKER_UNIVERSE[:limit]

    scores.sort(key=lambda x: x[1], reverse=True)
    tickers = [t for t, _ in scores[:limit]]
    return tickers if tickers else TICKER_UNIVERSE[:limit]


def _update_trading_config_params(repo_root: Path, optimized_payload: Dict[str, Any]) -> None:
    target = repo_root / "trading_config.py"
    text = target.read_text(encoding="utf-8")

    params_only = {
        k: {kk: vv for kk, vv in v.items() if kk in PARAM_GRIDS.get(k, {})}
        for k, v in optimized_payload.items()
    }
    block = (
        "\n# AUTO-GENERATED OPTIMIZER PARAMS START\n"
        f"OPTIMIZED_STRATEGY_PARAMS = {json.dumps(params_only, indent=4)}\n"
        "# AUTO-GENERATED OPTIMIZER PARAMS END\n"
    )

    pattern = re.compile(
        r"\n# AUTO-GENERATED OPTIMIZER PARAMS START\n.*?# AUTO-GENERATED OPTIMIZER PARAMS END\n",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        text = text.rstrip() + "\n" + block

    target.write_text(text, encoding="utf-8")


def _log_research(strategy_type: str, params: Dict[str, Any], full: Metrics, test: Metrics, deployed: bool, note: str) -> None:
    db = SessionLocal()
    try:
        row = StrategyResearch(
            strategy_name=strategy_type,
            source="optimizer",
            params_json=json.dumps(params),
            backtest_sharpe=full.sharpe,
            backtest_return_pct=full.total_return_pct,
            backtest_win_rate=full.win_rate,
            backtest_max_dd=full.max_drawdown,
            backtest_trade_count=full.trade_count,
            walk_forward_sharpe=test.sharpe,
            walk_forward_return_pct=test.total_return_pct,
            is_deployed=deployed,
            deployment_date=datetime.utcnow() if deployed else None,
            notes=note,
        )
        db.add(row)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def optimize(
    tickers: List[str],
    strategies: List[str],
    save_path: Path,
    update_config: bool,
) -> Dict[str, Any]:
    init_db()

    data_cache: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
            if df.empty:
                df = yf.download(ticker, period="2y", auto_adjust=True, progress=False, threads=False)
            df = _normalize_ohlcv(df)
            if len(df) >= 180:
                data_cache[ticker] = df
        except Exception:
            continue

    output: Dict[str, Any] = {}

    for strategy_type in strategies:
        grid = PARAM_GRIDS[strategy_type]
        combos = _param_combos(grid)
        print(f"\n[{strategy_type}] testing {len(combos)} parameter combinations across {len(data_cache)} tickers")

        best_by_ticker: Dict[str, Dict[str, Any]] = {}
        combo_rollup: Dict[str, Dict[str, Any]] = {}

        for ticker, df in data_cache.items():
            train_df, test_df = _walk_forward_split(df)
            best_score = -1e9
            best_record: Optional[Dict[str, Any]] = None

            for params in combos:
                try:
                    train_m = _evaluate(train_df, strategy_type, params)
                    test_m = _evaluate(test_df, strategy_type, params)
                    full_m = _evaluate(df, strategy_type, params)
                except Exception:
                    continue

                if full_m.trade_count < 10:
                    continue
                if train_m.total_return_pct <= 0 or test_m.total_return_pct <= 0:
                    continue

                score = test_m.sharpe * 0.7 + train_m.sharpe * 0.3
                if score > best_score:
                    best_score = score
                    best_record = {
                        "params": params,
                        "train": train_m,
                        "test": test_m,
                        "full": full_m,
                    }

            if best_record is None:
                continue

            best_by_ticker[ticker] = best_record
            key = json.dumps(best_record["params"], sort_keys=True)
            bucket = combo_rollup.setdefault(key, {"params": best_record["params"], "tickers": [], "test_sharpes": []})
            bucket["tickers"].append(ticker)
            bucket["test_sharpes"].append(best_record["test"].sharpe)

        if not best_by_ticker:
            output[strategy_type] = {
                "error": "No parameter set met minimum trade/profitability constraints.",
                "tickers_profitable": [],
                "tickers_unprofitable": sorted(list(data_cache.keys())),
            }
            continue

        picked = sorted(
            combo_rollup.values(),
            key=lambda x: (len(x["tickers"]), float(np.mean(x["test_sharpes"]))),
            reverse=True,
        )[0]
        selected_params = picked["params"]

        profitable: List[str] = []
        unprofitable: List[str] = []
        full_metrics: List[Metrics] = []
        test_metrics: List[Metrics] = []

        for ticker, df in data_cache.items():
            train_df, test_df = _walk_forward_split(df)
            train_m = _evaluate(train_df, strategy_type, selected_params)
            test_m = _evaluate(test_df, strategy_type, selected_params)
            full_m = _evaluate(df, strategy_type, selected_params)
            ok = _qualifies(train_m, test_m, full_m)
            if ok:
                profitable.append(ticker)
            else:
                unprofitable.append(ticker)
            full_metrics.append(full_m)
            test_metrics.append(test_m)

        avg_full = Metrics(
            sharpe=round(float(np.mean([m.sharpe for m in full_metrics])), 4),
            total_return_pct=round(float(np.mean([m.total_return_pct for m in full_metrics])), 4),
            win_rate=round(float(np.mean([m.win_rate for m in full_metrics])), 4),
            max_drawdown=round(float(np.mean([m.max_drawdown for m in full_metrics])), 4),
            trade_count=int(round(float(np.mean([m.trade_count for m in full_metrics])))),
        )
        avg_test = Metrics(
            sharpe=round(float(np.mean([m.sharpe for m in test_metrics])), 4),
            total_return_pct=round(float(np.mean([m.total_return_pct for m in test_metrics])), 4),
            win_rate=round(float(np.mean([m.win_rate for m in test_metrics])), 4),
            max_drawdown=round(float(np.mean([m.max_drawdown for m in test_metrics])), 4),
            trade_count=int(round(float(np.mean([m.trade_count for m in test_metrics])))),
        )

        payload = {
            **selected_params,
            "sharpe": avg_full.sharpe,
            "win_rate": avg_full.win_rate,
            "max_drawdown": avg_full.max_drawdown,
            "trade_count": avg_full.trade_count,
            "walk_forward_sharpe": avg_test.sharpe,
            "walk_forward_return_pct": avg_test.total_return_pct,
            "tickers_profitable": sorted(profitable),
            "tickers_unprofitable": sorted(unprofitable),
            "selection_coverage": len(picked["tickers"]),
        }
        output[strategy_type] = payload

        _log_research(
            strategy_type=strategy_type,
            params=selected_params,
            full=avg_full,
            test=avg_test,
            deployed=len(profitable) > 0,
            note=f"coverage={len(picked['tickers'])}/{len(data_cache)}",
        )

        print(
            f"  selected={selected_params} | sharpe={avg_full.sharpe:+.3f} | "
            f"wf={avg_test.sharpe:+.3f} | profitable={len(profitable)}/{len(data_cache)}"
        )

    save_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved optimized params to {save_path}")

    if update_config:
        _update_trading_config_params(save_path.parent, output)
        print("Updated trading_config.py with OPTIMIZED_STRATEGY_PARAMS block")

    return output


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize strategy parameters with walk-forward validation")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers. Default: top 30 liquid from config.")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(PARAM_GRIDS.keys()),
        help=f"Comma-separated strategies. Choices: {','.join(PARAM_GRIDS.keys())}",
    )
    parser.add_argument("--output", type=str, default="optimized_params.json")
    parser.add_argument("--no-config-update", action="store_true", help="Do not modify trading_config.py")
    args = parser.parse_args()

    selected_strategies = _parse_csv_list(args.strategies)
    invalid = [s for s in selected_strategies if s not in PARAM_GRIDS]
    if invalid:
        print(f"Invalid strategies: {invalid}")
        return 1

    if args.tickers.strip():
        tickers = _parse_csv_list(args.tickers)
    else:
        tickers = _pick_top_liquid_tickers(limit=30)

    if not tickers:
        print("No tickers selected.")
        return 1

    print(f"Tickers: {','.join(tickers)}")
    print(f"Strategies: {','.join(selected_strategies)}")

    optimize(
        tickers=tickers,
        strategies=selected_strategies,
        save_path=Path(args.output).resolve(),
        update_config=not args.no_config_update,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
