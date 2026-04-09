"""Train PPO on CurriculumGridWorldEnv and persist model + metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from curriculum_rl_lab.envs.grid_world import CurriculumGridWorldEnv
from curriculum_rl_lab.metrics import write_metrics


def _make_vec_env(size: int, max_episode_steps: int) -> DummyVecEnv:
    def _init() -> TimeLimit:
        base = CurriculumGridWorldEnv(size=size, max_steps=max_episode_steps)
        return TimeLimit(base, max_episode_steps=max_episode_steps)

    return DummyVecEnv([_init])


def run_training(
    out_dir: Path,
    total_timesteps: int,
    seed: int,
    *,
    grid_size: int = 9,
    max_episode_steps: int = 200,
    learning_rate: float = 3e-4,
    n_eval_episodes: int = 5,
) -> dict[str, Any]:
    """Train PPO; write model.zip and metrics.json under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    train_env = _make_vec_env(size=grid_size, max_episode_steps=max_episode_steps)
    eval_env = _make_vec_env(size=grid_size, max_episode_steps=max_episode_steps)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    model_path = out_dir / "model.zip"
    model.save(str(model_path))

    payload: dict[str, Any] = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "grid_size": grid_size,
        "max_episode_steps": max_episode_steps,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "model_path": str(model_path.resolve()),
    }
    write_metrics(out_dir / "metrics.json", payload)
    train_env.close()
    eval_env.close()
    return payload
