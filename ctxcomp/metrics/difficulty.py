"""Per-task difficulty, read from AppWorld's own task metadata.

`$APPWORLD_ROOT/data/tasks/<task_id>/ground_truth/metadata.json` carries
`{"difficulty": 1|2|3}`. Reading it here rather than reimplementing a partition
means the Easy/Medium/Hard columns are the environment's own labels, and the
counts (57 / 48 / 63 on test_normal) come out as a check on the wiring.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

LEVELS = {"1": "Easy", "2": "Medium", "3": "Hard"}
_cache: dict[str, str] = {}


def appworld_root() -> Path:
    return Path(os.environ.get("APPWORLD_ROOT",
                               "/z5s/morph/home/sjk/Agent/datasets/appworld-0.1.0"))


def clean_id(task_id: str) -> str:
    return task_id[5:] if task_id.startswith("task_") else task_id


def difficulty(task_id: str) -> str | None:
    tid = clean_id(task_id)
    if tid in _cache:
        return _cache[tid]
    meta = appworld_root() / "data" / "tasks" / tid / "ground_truth" / "metadata.json"
    level = None
    if meta.exists():
        try:
            d = json.loads(meta.read_text()).get("difficulty")
            if str(d) in LEVELS:
                level = str(d)
        except Exception:
            level = None
    _cache[tid] = level
    return level
