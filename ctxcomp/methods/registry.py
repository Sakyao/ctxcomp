"""Build policies from methods.yml.

The registry is the only place that knows how a manifest row becomes an object, so
run_suite.py, compute_table.py and check_methods.py can all agree on what a method
is without hard-coding a method name.

Dispatch is on (axis, policy), not on policy alone: the same manifest entry shape
describes a history row and an observation row, and the mechanism differs between
them. The axis also selects the threshold, because T_hist and T_obs are different
numbers and borrowing the wrong one would silently change when a policy fires.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import CompressionPolicy
from .builtin import (FifoPolicy, LlmLinguaPolicy, NoCompressionPolicy,
                      ObsFifoPolicy, ObsLlmLinguaPolicy)
from .prompted import PromptedPolicy
from ..llm import ChatClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or REPO / "methods.yml").read_text())


def prompt_dir_for(row: dict, name: str) -> Path:
    spec = row.get("prompts") or {}
    d = spec.get("dir") or name
    p = Path(d)
    return p if p.is_absolute() else (REPO / "ctxcomp" / "prompts" / p)


def threshold_for(row: dict, man: dict | None = None) -> int:
    man = man or load_manifest()
    axis = row.get("axis", "history")
    if axis == "none":
        return 0
    return int((man.get("thresholds") or {}).get(axis, 4096))


def build_policy(row: dict, llm: ChatClient | None = None,
                 budget: int | None = None, man: dict | None = None) -> CompressionPolicy | None:
    """None means "no policy", i.e. the uncompressed reference row."""
    axis = row.get("axis", "history")
    if axis == "none":
        return NoCompressionPolicy()

    budget = budget or threshold_for(row, man)
    kind = row["policy"]

    if kind == "builtin":
        table = {
            ("history", "hist_fifo"): lambda: FifoPolicy(keep_last=int(row.get("keep_last", 5))),
            ("history", "hist_llmlingua"): lambda: LlmLinguaPolicy(keep_rate=float(row.get("keep_rate", 0.30))),
            ("observation", "obs_fifo"): lambda: ObsFifoPolicy(),
            ("observation", "obs_llmlingua"): lambda: ObsLlmLinguaPolicy(keep_rate=float(row.get("keep_rate", 0.30))),
        }
        try:
            return table[(axis, row["name"])]()
        except KeyError:
            raise KeyError(f"builtin policy {row['name']!r} on axis {axis!r} is not implemented") from None

    if kind == "prompt":
        spec = row.get("prompts") or {}
        return PromptedPolicy(
            name=row["name"],
            prompt_dir=prompt_dir_for(row, row["name"]),
            llm=llm or ChatClient(),
            axis=axis,
            system=spec.get("system", "system"),
            first=spec.get("first"),
            update=spec.get("update"),
            obs=spec.get("obs", "obs"),
            prefix=spec.get("prefix", "prefix"),
            budget=budget,
            max_chars=row.get("max_chars"),
        )

    raise KeyError(f"unknown policy kind {kind!r} for row {row['name']!r}")
