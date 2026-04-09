from __future__ import annotations

from pathlib import Path

from curriculum_rl_lab.training import run_training


def test_training_writes_metrics(tmp_path: Path) -> None:
    out = tmp_path / "run"
    payload = run_training(out, total_timesteps=2048, seed=1, grid_size=7, n_eval_episodes=2)
    assert (out / "model.zip").is_file()
    assert (out / "metrics.json").is_file()
    assert "mean_reward" in payload
    assert payload["total_timesteps"] == 2048
