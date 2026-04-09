"""Sparse-reward grid navigation (Gymnasium)."""

from __future__ import annotations

from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CurriculumGridWorldEnv(gym.Env):
    """
    Agent must reach a goal in a square grid. Rewards are sparse:
    small step penalty, large bonus at the goal, optional curriculum hook.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        size: int = 9,
        max_steps: int = 200,
        step_penalty: float = -0.01,
        goal_reward: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if size < 3:
            raise ValueError("size must be at least 3")
        self.size = size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        # Normalized [ax, ay, gx, gy] in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self._rng: np.random.Generator = np.random.default_rng()
        self._agent_pos: tuple[int, int] = (0, 0)
        self._goal_pos: tuple[int, int] = (0, 0)
        self._steps: int = 0

    def _obs(self) -> np.ndarray:
        ax, ay = self._agent_pos
        gx, gy = self._goal_pos
        s = float(self.size - 1)
        return np.array(
            [ax / s, ay / s, gx / s, gy / s],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        opts = options or {}
        goal_fixed = opts.get("goal_pos")
        if goal_fixed is not None:
            self._goal_pos = cast(tuple[int, int], tuple(goal_fixed))
        else:
            self._goal_pos = (
                int(self._rng.integers(0, self.size)),
                int(self._rng.integers(0, self.size)),
            )

        for _ in range(64):
            ax = int(self._rng.integers(0, self.size))
            ay = int(self._rng.integers(0, self.size))
            if (ax, ay) != self._goal_pos:
                self._agent_pos = (ax, ay)
                break
        else:
            self._agent_pos = (0, 0)
            if self._goal_pos == (0, 0):
                self._goal_pos = (self.size - 1, self.size - 1)

        self._steps = 0
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        x, y = self._agent_pos
        if action == 0:
            y = min(self.size - 1, y + 1)
        elif action == 1:
            x = min(self.size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            x = max(0, x - 1)
        else:
            raise ValueError(f"Invalid action {action}")
        self._agent_pos = (x, y)
        self._steps += 1

        terminated = self._agent_pos == self._goal_pos
        truncated = self._steps >= self.max_steps
        reward = self.goal_reward if terminated else self.step_penalty

        return self._obs(), float(reward), terminated, truncated, {}
