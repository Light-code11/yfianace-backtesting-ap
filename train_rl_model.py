"""
CLI utility to train or retrain the RL strategy allocator.

Example:
python train_rl_model.py --episodes 1000 --eval-freq 100
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

from rl_strategy_allocator import RLStrategyAllocator


def _synthetic_state_data(strategy_names: List[str], days: int = 800) -> List[Dict[str, Any]]:
    """
    Create synthetic training data when no historical file is provided.
    """
    base_date = datetime.utcnow() - timedelta(days=days)
    rows: List[Dict[str, Any]] = []

    for i in range(days):
        regime_probs = np.random.dirichlet(np.array([1.5, 1.2, 1.3]))
        strategy_perf = np.random.normal(0.0, 1.0, size=len(strategy_names) * 4)
        portfolio_metrics = np.random.normal(0.0, 1.0, size=5)
        market_features = np.random.normal(0.0, 1.0, size=10)

        state_vector = np.concatenate([
            regime_probs,
            strategy_perf,
            portfolio_metrics,
            market_features,
        ]).astype(float)

        strategy_returns = {
            name: float(np.random.normal(0.0005, 0.01)) for name in strategy_names
        }

        rows.append(
            {
                "date": (base_date + timedelta(days=i)).date().isoformat(),
                "state_vector": state_vector.tolist(),
                "strategy_returns": strategy_returns,
            }
        )

    return rows


def _load_state_data(path: str, strategy_names: List[str]) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Training data must be a JSON list of state rows")

    # Minimal normalization for expected keys
    rows: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue

        state_vector = row.get("state_vector") or []
        strategy_returns = row.get("strategy_returns") or {}
        date_val = row.get("date") or datetime.utcnow().date().isoformat()

        if not isinstance(state_vector, list):
            continue
        if not isinstance(strategy_returns, dict):
            strategy_returns = {}

        normalized_returns = {
            s: float(strategy_returns.get(s, 0.0)) for s in strategy_names
        }

        rows.append(
            {
                "date": str(date_val),
                "state_vector": state_vector,
                "strategy_returns": normalized_returns,
            }
        )

    return rows


def main():
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required for RL training script")

    parser = argparse.ArgumentParser(description="Train RL strategy allocation model")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--eval-freq", type=int, default=100, help="Evaluation frequency")
    parser.add_argument(
        "--strategies",
        type=str,
        default="momentum,mean_reversion,breakout",
        help="Comma-separated strategy names",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="",
        help="Optional JSON file containing historical state rows",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./rl_models",
        help="Directory for trained RL model",
    )

    args = parser.parse_args()

    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    allocator = RLStrategyAllocator(strategy_names=strategy_names, model_dir=args.model_dir)

    if args.data_file:
        training_data = _load_state_data(args.data_file, strategy_names)
    else:
        training_data = _synthetic_state_data(strategy_names)

    result = allocator.train(
        historical_states=training_data,
        episodes=args.episodes,
        eval_freq=args.eval_freq,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required to generate synthetic RL training data")
