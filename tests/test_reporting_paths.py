from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from curriculum_rl_lab.metrics import write_metrics
from curriculum_rl_lab.reporting import lab_report
from curriculum_rl_lab.reporting_minimax import _strip_thinking_blocks


def test_lab_report_without_api_keys(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Avoid reading the developer's real `.env` during this test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_a, **_k: None)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mpath = tmp_path / "metrics.json"
    write_metrics(
        mpath,
        {"mean_reward": 0.5, "std_reward": 0.2, "total_timesteps": 1000},
    )
    text = lab_report(mpath)
    assert "rule-based" in text


def test_strip_thinking_blocks_removes_tagged_sections() -> None:
    raw = "<redacted_thinking>secret</redacted_thinking>\n## Visible"
    out = _strip_thinking_blocks(raw)
    assert "secret" not in out
    assert "Visible" in out
