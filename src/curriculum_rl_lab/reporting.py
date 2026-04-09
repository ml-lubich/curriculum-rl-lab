"""Human-readable lab reports: MiniMax (from .env), optional CrewAI, or rule-based."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from curriculum_rl_lab.env_bootstrap import load_minimax_env
from curriculum_rl_lab.metrics import read_metrics


def rule_based_report(metrics: dict[str, Any]) -> str:
    mean_r = float(metrics.get("mean_reward", 0.0))
    std_r = float(metrics.get("std_reward", 0.0))
    ts = int(metrics.get("total_timesteps", 0))
    lines = [
        "## Lab report (rule-based)",
        "",
        f"- **Mean return** (eval): {mean_r:.3f} ± {std_r:.3f}",
        f"- **Timesteps**: {ts}",
        "",
        "Interpretation: higher mean return means the policy reaches the goal more reliably.",
        "",
    ]
    return "\n".join(lines)


def lab_report(metrics_path: Path) -> str:
    """Prefer MiniMax (`MINIMAX_API_KEY` in `.env`), then CrewAI + OpenAI, else rule-based."""
    load_minimax_env()
    metrics = read_metrics(metrics_path)

    if os.environ.get("MINIMAX_API_KEY", "").strip():
        try:
            from curriculum_rl_lab.reporting_minimax import minimax_report

            return minimax_report(metrics)
        except ImportError as exc:
            raise ImportError(
                "Install the `openai` package to use MiniMax reports: pip install openai"
            ) from exc

    try:
        from curriculum_rl_lab.reporting_crew import crew_report_llm
    except ImportError:
        return rule_based_report(metrics)
    return crew_report_llm(metrics)


def emit_report(metrics_path: Path, out_path: Path | None) -> str:
    text = lab_report(metrics_path)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return text
