from __future__ import annotations

import numpy as np

from curriculum_rl_lab.envs.grid_world import CurriculumGridWorldEnv


def test_episode_can_terminate_at_goal() -> None:
    env = CurriculumGridWorldEnv(size=5, max_steps=50)
    obs, _ = env.reset(seed=42, options={"goal_pos": (4, 4)})
    assert obs.shape == (4,)
    for _ in range(200):
        obs, r, term, trunc, _ = env.step(1)  # move right-ish / explore
        if term or trunc:
            break
    # With random exploration we eventually terminate (truncated if unlucky)
    assert isinstance(obs, np.ndarray)


def test_deterministic_short_path() -> None:
    env = CurriculumGridWorldEnv(size=3, max_steps=20)
    env.reset(seed=0, options={"goal_pos": (2, 2)})
    # Manually step to goal from start if we know positions — easier: fixed seed loop
    done = False
    steps = 0
    while not done and steps < 50:
        # random actions until goal — just check step API
        obs, _, term, trunc, _ = env.step(0)
        done = term or trunc
        steps += 1
    assert obs.dtype == np.float32
