"""Human-readable lab reports: rule-based (default) or CrewAI (optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def crew_report(metrics_path: Path) -> str:
    """If CrewAI + OPENAI_API_KEY are available, return an LLM report; else rule-based."""
    metrics = read_metrics(metrics_path)
    try:
        from curriculum_rl_lab.reporting_crew import crew_report_llm
    except ImportError:
        return rule_based_report(metrics)
    return crew_report_llm(metrics)


def emit_report(metrics_path: Path, out_path: Path | None) -> str:
    text = crew_report(metrics_path)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return text
