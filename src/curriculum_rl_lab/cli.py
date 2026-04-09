"""CLI entrypoints: train and report."""

from __future__ import annotations

import argparse
from pathlib import Path

from curriculum_rl_lab.reporting import emit_report
from curriculum_rl_lab.training import run_training


def main_train() -> None:
    p = argparse.ArgumentParser(description="Train PPO on CurriculumGridWorldEnv")
    p.add_argument("--out", type=Path, default=Path("runs/latest"), help="Output directory")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-size", type=int, default=9)
    p.add_argument("--max-episode-steps", type=int, default=200)
    args = p.parse_args()
    payload = run_training(
        args.out,
        total_timesteps=args.timesteps,
        seed=args.seed,
        grid_size=args.grid_size,
        max_episode_steps=args.max_episode_steps,
    )
    print("Training complete.")
    print(f"Mean reward: {payload['mean_reward']:.4f} ± {payload['std_reward']:.4f}")


def main_report() -> None:
    p = argparse.ArgumentParser(description="Generate lab report from metrics.json")
    p.add_argument("--metrics", type=Path, default=Path("runs/latest/metrics.json"))
    p.add_argument("--out", type=Path, default=None, help="Optional path to write report.md")
    args = p.parse_args()
    text = emit_report(args.metrics, args.out)
    print(text)
