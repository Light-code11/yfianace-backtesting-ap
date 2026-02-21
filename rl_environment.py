"""
Custom RL environment for strategy allocation.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
        GYM_AVAILABLE = True
    except ImportError:
        gym = object  # type: ignore
        spaces = None  # type: ignore
        GYM_AVAILABLE = False


class TradingAllocEnv(gym.Env if GYM_AVAILABLE and NUMPY_AVAILABLE else object):
    """
    RL environment for multi-strategy capital allocation.

    Expected historical state format:
    {
        "state_vector": [...],
        "strategy_returns": {"momentum": 0.01, "mean_reversion": -0.004}
    }
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        historical_states: List[Dict[str, Any]],
        strategy_names: Sequence[str],
        episode_length: int = 20,
        random_seed: int = 42,
    ):
        self.historical_states = historical_states or []
        self.strategy_names = list(strategy_names)
        self.num_strategies = max(len(self.strategy_names), 1)
        self.episode_length = max(1, episode_length)
        self.random = random.Random(random_seed)

        self.current_idx = 0
        self.steps_taken = 0
        self.episode_returns: List[float] = []

        self.state_dim = self._infer_state_dim()

        if GYM_AVAILABLE and NUMPY_AVAILABLE and spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.state_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_strategies,),
                dtype=np.float32,
            )

    def _infer_state_dim(self) -> int:
        if not self.historical_states:
            return self.num_strategies + 18

        first = self.historical_states[0]
        vector = first.get("state_vector")
        if isinstance(vector, list) and vector:
            return len(vector)

        return self.num_strategies + 18

    def _state_to_vector(self, state: Dict[str, Any]):
        if not NUMPY_AVAILABLE:
            return [0.0] * self.state_dim

        vector = state.get("state_vector")
        if isinstance(vector, list) and vector:
            arr = np.array(vector, dtype=np.float32)
        else:
            arr = np.zeros(self.state_dim, dtype=np.float32)

        if arr.shape[0] != self.state_dim:
            if arr.shape[0] > self.state_dim:
                arr = arr[: self.state_dim]
            else:
                pad = np.zeros(self.state_dim - arr.shape[0], dtype=np.float32)
                arr = np.concatenate([arr, pad])

        return arr

    @staticmethod
    def _normalize_weights(action):
        if not NUMPY_AVAILABLE:
            total = sum(max(0.0, min(float(x), 1.0)) for x in action)
            if total <= 0:
                return [1.0 / len(action)] * len(action)
            return [max(0.0, min(float(x), 1.0)) / total for x in action]

        clipped = np.clip(action, 0.0, 1.0)
        total = float(np.sum(clipped))
        if total <= 0:
            return np.ones_like(clipped, dtype=np.float32) / len(clipped)
        return clipped / total

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.random.seed(seed)

        self.steps_taken = 0
        self.episode_returns = []

        if len(self.historical_states) <= self.episode_length + 1:
            self.current_idx = 0
        else:
            max_start = len(self.historical_states) - self.episode_length - 1
            self.current_idx = self.random.randint(0, max_start)

        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required to run TradingAllocEnv")

        if not self.historical_states:
            obs = np.zeros(self.state_dim, dtype=np.float32)
        else:
            obs = self._state_to_vector(self.historical_states[self.current_idx])

        if GYM_AVAILABLE:
            return obs, {}
        return obs

    def step(self, action):
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required to run TradingAllocEnv")

        if not self.historical_states:
            obs = np.zeros(self.state_dim, dtype=np.float32)
            if GYM_AVAILABLE:
                return obs, 0.0, True, False, {"portfolio_return": 0.0}
            return obs, 0.0, True, {"portfolio_return": 0.0}

        weights = self._normalize_weights(np.asarray(action, dtype=np.float32))
        current_state = self.historical_states[self.current_idx]

        strategy_returns = current_state.get("strategy_returns", {})
        weighted_return = 0.0

        for idx, strategy in enumerate(self.strategy_names):
            weighted_return += float(strategy_returns.get(strategy, 0.0)) * float(weights[idx])

        self.episode_returns.append(weighted_return)
        self.steps_taken += 1
        self.current_idx += 1

        if self.current_idx >= len(self.historical_states):
            self.current_idx = len(self.historical_states) - 1

        reward = self._calculate_reward()
        terminated = self.steps_taken >= self.episode_length

        next_obs = self._state_to_vector(self.historical_states[self.current_idx])
        info = {
            "portfolio_return": weighted_return,
            "weights": {s: float(weights[i]) for i, s in enumerate(self.strategy_names)},
        }

        if GYM_AVAILABLE:
            return next_obs, reward, terminated, False, info
        return next_obs, reward, terminated, info

    def _calculate_reward(self) -> float:
        if not self.episode_returns:
            return 0.0

        if not NUMPY_AVAILABLE:
            return sum(self.episode_returns) / max(len(self.episode_returns), 1)

        returns = np.asarray(self.episode_returns, dtype=np.float32)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))

        if std_return <= 1e-9:
            return mean_return

        sharpe = mean_return / std_return
        return float(sharpe)

    def render(self):
        if not self.episode_returns:
            return
        if not NUMPY_AVAILABLE:
            print(f"Episode steps={self.steps_taken}, mean_return={sum(self.episode_returns)/len(self.episode_returns):.6f}")
            return
        print(
            f"Episode steps={self.steps_taken}, mean_return={np.mean(self.episode_returns):.6f}, "
            f"std_return={np.std(self.episode_returns):.6f}"
        )
