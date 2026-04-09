"""CrewAI-powered narrative over metrics (optional dependency)."""

from __future__ import annotations

import os
from typing import Any

from curriculum_rl_lab.reporting import rule_based_report


def crew_report_llm(metrics: dict[str, Any]) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return rule_based_report(metrics) + "\n*(CrewAI skipped: set OPENAI_API_KEY for an LLM narrative.)*\n"

    from crewai import Agent, Crew, Process, Task  # type: ignore[import-untyped]

    summary = (
        f"mean_reward={metrics.get('mean_reward')}, "
        f"std_reward={metrics.get('std_reward')}, "
        f"total_timesteps={metrics.get('total_timesteps')}"
    )

    analyst = Agent(
        role="RL metrics analyst",
        goal="Explain whether training results look healthy for a sparse-reward grid task.",
        backstory="You write concise technical summaries for ML engineers.",
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=(
            "Given these metrics, write a short markdown section with bullets: "
            "stability, likely success at reaching the goal, and one next experiment. "
            f"Metrics: {summary}"
        ),
        expected_output="Markdown with 3-6 bullet points.",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    text = str(result).strip()
    header = "## Lab report (CrewAI)\n\n"
    return header + text + "\n"
