"""
Reinforcement Learning strategy allocator using PPO.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

from rl_environment import GYM_AVAILABLE, TradingAllocEnv

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    PPO = None  # type: ignore
    SB3_AVAILABLE = False


class RLStrategyAllocator:
    """
    PPO-based allocator for multi-strategy capital distribution.
    """

    def __init__(
        self,
        strategy_names: Sequence[str],
        model_dir: str = "./rl_models",
        model_name: str = "strategy_allocator_ppo",
    ):
        self.strategy_names = list(strategy_names)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.model_path = self.model_dir / f"{self.model_name}.zip"
        self.meta_path = self.model_dir / f"{self.model_name}.meta.json"

        self.model = None
        self.training_info: Dict[str, Any] = {}

        self._load_model_if_available()

    @staticmethod
    def _normalize_weights(raw: Any) -> Any:
        if not NUMPY_AVAILABLE:
            return raw
        clipped = np.clip(raw, 0.0, 1.0)
        total = float(np.sum(clipped))
        if total <= 0:
            return np.ones_like(clipped, dtype=np.float32) / len(clipped)
        return clipped / total

    def _equal_weight(self) -> Dict[str, float]:
        if not self.strategy_names:
            return {}
        weight = 1.0 / len(self.strategy_names)
        return {name: weight for name in self.strategy_names}

    def _load_model_if_available(self):
        if not SB3_AVAILABLE:
            return

        if self.model_path.exists():
            try:
                self.model = PPO.load(str(self.model_path))
                if self.meta_path.exists():
                    self.training_info = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                self.model = None

    def model_available(self) -> bool:
        return self.model is not None and SB3_AVAILABLE and GYM_AVAILABLE and NUMPY_AVAILABLE

    def get_allocation(self, current_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Return allocation weights by strategy, sum to 1.0.
        """
        if not self.strategy_names:
            return {}

        if not self.model_available():
            return self._equal_weight()

        if not NUMPY_AVAILABLE:
            return self._equal_weight()

        state_vector = np.array(current_state.get("state_vector", []), dtype=np.float32)
        if state_vector.size == 0:
            return self._equal_weight()

        try:
            action, _ = self.model.predict(state_vector, deterministic=True)
            weights = self._normalize_weights(np.asarray(action, dtype=np.float32))
            return {
                strategy: float(weights[idx])
                for idx, strategy in enumerate(self.strategy_names)
            }
        except Exception:
            return self._equal_weight()

    def train(
        self,
        historical_states: List[Dict[str, Any]],
        episodes: int = 1000,
        eval_freq: int = 100,
        episode_length: int = 20,
    ) -> Dict[str, Any]:
        """
        Train PPO model on historical state/return data.
        """
        if not SB3_AVAILABLE:
            return {
                "success": False,
                "error": "stable-baselines3 not available. Install stable-baselines3.",
            }

        if not GYM_AVAILABLE:
            return {
                "success": False,
                "error": "gymnasium/gym not available. Install gymnasium.",
            }

        if not historical_states:
            return {
                "success": False,
                "error": "No historical states provided for training.",
            }

        try:
            env = TradingAllocEnv(
                historical_states=historical_states,
                strategy_names=self.strategy_names,
                episode_length=episode_length,
            )

            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                gamma=0.99,
            )

            total_timesteps = max(episodes * episode_length, episode_length)
            self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
            self.model.save(str(self.model_path))

            self.training_info = {
                "trained_at": datetime.utcnow().isoformat(),
                "episodes": int(episodes),
                "eval_freq": int(eval_freq),
                "episode_length": int(episode_length),
                "timesteps": int(total_timesteps),
                "strategy_names": list(self.strategy_names),
                "data_points": int(len(historical_states)),
            }
            self.meta_path.write_text(json.dumps(self.training_info, indent=2), encoding="utf-8")

            return {
                "success": True,
                "model_path": str(self.model_path),
                "training_info": self.training_info,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"RL training failed: {str(exc)}",
            }

    def walk_forward_train(
        self,
        yearly_data: Dict[int, List[Dict[str, Any]]],
        episodes: int = 300,
        eval_freq: int = 50,
    ) -> Dict[str, Any]:
        """
        Walk-forward training: train on year N, evaluate on N+1.
        """
        years = sorted(yearly_data.keys())
        if len(years) < 2:
            return {
                "success": False,
                "error": "Need at least 2 years for walk-forward training.",
            }

        runs = []
        for idx in range(len(years) - 1):
            train_year = years[idx]
            test_year = years[idx + 1]
            train_result = self.train(
                historical_states=yearly_data.get(train_year, []),
                episodes=episodes,
                eval_freq=eval_freq,
            )
            runs.append(
                {
                    "train_year": train_year,
                    "test_year": test_year,
                    "train_success": train_result.get("success", False),
                    "error": train_result.get("error"),
                }
            )

        return {
            "success": any(run["train_success"] for run in runs),
            "runs": runs,
        }

    def retrain_on_last_two_years(
        self,
        dated_states: List[Dict[str, Any]],
        episodes: int = 500,
        eval_freq: int = 100,
    ) -> Dict[str, Any]:
        """
        Weekly retrain helper using only the last 2 years of data.
        """
        cutoff = datetime.utcnow() - timedelta(days=730)
        recent = []

        for row in dated_states:
            ts = row.get("date")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt >= cutoff:
                recent.append(row)

        return self.train(recent, episodes=episodes, eval_freq=eval_freq)
        if not NUMPY_AVAILABLE:
            return {
                "success": False,
                "error": "numpy not available. Install numpy.",
            }
