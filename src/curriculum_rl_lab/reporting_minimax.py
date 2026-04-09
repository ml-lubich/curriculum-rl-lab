"""Lab report via MiniMax OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from curriculum_rl_lab.env_bootstrap import load_minimax_env


def _strip_thinking_blocks(text: str) -> str:
    """Remove model reasoning / scratch blocks some providers wrap in XML-like tags."""
    # MiniMax may emit `<think>...</think>` or `<redacted_thinking>...</redacted_thinking>`.
    tag = r"think(?:ing)?"
    cleaned = re.sub(
        rf"<[^>]*{tag}[^>]*>.*?</[^>]*{tag}[^>]*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def minimax_report(metrics: dict[str, Any]) -> str:
    """Call MiniMax chat completions to summarize training metrics."""
    load_minimax_env()
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise ValueError("MINIMAX_API_KEY is not set")

    from openai import OpenAI

    base = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1").strip()
    model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2").strip()

    client = OpenAI(base_url=base, api_key=key)
    payload = json.dumps(metrics, indent=2)
    user = (
        "You are an RL metrics analyst. Given the following JSON metrics from a PPO run "
        "on a sparse-reward grid world, write a short Markdown section with 3-6 bullets: "
        "stability, goal-reaching success, and one suggested next experiment.\n\n"
        f"```json\n{payload}\n```"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write concise technical Markdown for engineers."},
            {"role": "user", "content": user},
        ],
    )
    choice = completion.choices[0].message
    text = _strip_thinking_blocks((choice.content or "").strip())
    return "## Lab report (MiniMax)\n\n" + text + "\n"
