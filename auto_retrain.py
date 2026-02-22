"""
Weekly auto-retraining utility for ensemble ML models.

Can be executed via cron every Sunday or imported by other modules.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_price_predictor import MLPricePredictor


DEFAULT_TOP20_SP500 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "TSLA",
    "LLY",
    "AVGO",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "HD",
    "PG",
    "JNJ",
    "ABBV",
]


def _parse_tickers(raw_tickers: Optional[str]) -> List[str]:
    if not raw_tickers:
        return DEFAULT_TOP20_SP500
    cleaned = [x.strip().upper() for x in raw_tickers.split(",") if x.strip()]
    return cleaned or DEFAULT_TOP20_SP500


def _load_model_trained_at(model_path: Path) -> Optional[datetime]:
    if not model_path.exists():
        return None
    try:
        import pickle

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        trained_at = model_data.get("trained_at")
        if not trained_at:
            return None
        parsed = datetime.fromisoformat(trained_at)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _snapshot_model(model_path: Path, ticker: str) -> Optional[str]:
    if not model_path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot = model_path.with_name(f"{ticker}_model_{ts}.pkl")
    shutil.copy2(model_path, snapshot)
    return str(snapshot)


def run_weekly_auto_retraining(
    tickers: Optional[List[str]] = None,
    model_dir: str = "./ml_models",
    lookback_days: int = 60,
    max_model_age_days: int = 7,
    train_period: str = "6mo",
) -> Dict[str, Any]:
    """
    Run weekly retraining cycle for configured tickers.

    Retrains if:
    - no existing model
    - model is older than max_model_age_days
    - feature drift detection flags retraining
    """
    universe = tickers or DEFAULT_TOP20_SP500
    predictor = MLPricePredictor(model_dir=model_dir)

    results: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(model_dir).resolve()),
        "tickers_processed": len(universe),
        "tickers_retrained": [],
        "ticker_details": [],
        "errors": [],
    }

    now = datetime.now(timezone.utc)

    for ticker in universe:
        ticker = ticker.upper()
        detail: Dict[str, Any] = {
            "ticker": ticker,
            "retrained": False,
            "reason": [],
            "drift_score": None,
            "needs_retraining_drift": False,
            "model_age_days": None,
            "validation_accuracy": None,
            "error": None,
            "snapshot_path": None,
        }

        try:
            model_path = Path(model_dir) / f"{ticker}_model.pkl"
            trained_at = _load_model_trained_at(model_path)

            if trained_at is None:
                detail["reason"].append("missing_or_unreadable_model")
            else:
                age_days = (now - trained_at).total_seconds() / 86400.0
                detail["model_age_days"] = round(age_days, 2)
                if age_days > max_model_age_days:
                    detail["reason"].append(f"model_older_than_{max_model_age_days}_days")

            drift_result = predictor.detect_feature_drift(ticker=ticker, lookback_days=lookback_days)
            if drift_result.get("success"):
                detail["drift_score"] = drift_result.get("drift_score")
                detail["needs_retraining_drift"] = bool(drift_result.get("needs_retraining", False))
                if detail["needs_retraining_drift"]:
                    detail["reason"].append("feature_drift_detected")
            else:
                detail["reason"].append("drift_check_unavailable")

            should_retrain = len(detail["reason"]) > 0

            if should_retrain:
                train_result = predictor.train_model(ticker=ticker, period=train_period, test_size=0.2, horizon=1)
                if train_result.get("success"):
                    detail["retrained"] = True
                    detail["validation_accuracy"] = train_result.get("validation_accuracy")
                    detail["snapshot_path"] = _snapshot_model(model_path, ticker)
                    results["tickers_retrained"].append(ticker)
                else:
                    err = train_result.get("error", "Unknown training error")
                    detail["error"] = err
                    results["errors"].append(f"{ticker}: {err}")
            results["ticker_details"].append(detail)
        except Exception as e:
            detail["error"] = str(e)
            results["ticker_details"].append(detail)
            results["errors"].append(f"{ticker}: {str(e)}")

    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["retrained_count"] = len(results["tickers_retrained"])
    results["error_count"] = len(results["errors"])
    return results


def _print_summary(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("WEEKLY ENSEMBLE MODEL RETRAINING SUMMARY")
    print("=" * 80)
    print(f"Started:   {summary.get('started_at')}")
    print(f"Completed: {summary.get('completed_at')}")
    print(f"Processed: {summary.get('tickers_processed')}")
    print(f"Retrained: {summary.get('retrained_count')}")
    print(f"Errors:    {summary.get('error_count')}")
    print()

    print("Retrained Tickers:")
    if summary.get("tickers_retrained"):
        for ticker in summary["tickers_retrained"]:
            print(f"  - {ticker}")
    else:
        print("  - None")
    print()

    print("Per-Ticker Details:")
    for d in summary.get("ticker_details", []):
        drift_score = d.get("drift_score")
        drift_str = f"{drift_score:.4f}" if isinstance(drift_score, (int, float)) else "n/a"
        acc = d.get("validation_accuracy")
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "n/a"
        reason = ", ".join(d.get("reason", [])) if d.get("reason") else "no_retrain_needed"
        print(
            f"  - {d.get('ticker')}: retrained={d.get('retrained')}, "
            f"drift={drift_str}, accuracy={acc_str}, reason={reason}"
        )
        if d.get("error"):
            print(f"    error={d['error']}")
    print()

    if summary.get("errors"):
        print("Errors:")
        for err in summary["errors"]:
            print(f"  - {err}")
        print()

    print("JSON Summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly ensemble model retraining")
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers. Default: top 20 S&P 500 symbols",
    )
    parser.add_argument("--model-dir", type=str, default="./ml_models")
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--max-model-age-days", type=int, default=7)
    parser.add_argument("--train-period", type=str, default="6mo")
    args = parser.parse_args()

    tickers = _parse_tickers(args.tickers)
    summary = run_weekly_auto_retraining(
        tickers=tickers,
        model_dir=args.model_dir,
        lookback_days=args.lookback_days,
        max_model_age_days=args.max_model_age_days,
        train_period=args.train_period,
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
