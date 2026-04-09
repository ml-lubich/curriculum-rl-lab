"""Load/save run metrics as JSON for reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_metrics(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("metrics file must contain a JSON object")
    return data
