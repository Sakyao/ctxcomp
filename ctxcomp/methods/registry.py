"""Build policies from methods.yml.

The registry is the only place that knows how a manifest row becomes an object.
Keeping it here means make_configs.py, run_suite.sh and compute_table.py can all
agree on what a method *is* without any of them hard-coding a method name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import CompressionPolicy
from .builtin import FifoPolicy, LlmLinguaPolicy, NoCompressionPolicy
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


def build_policy(row: dict, llm: ChatClient | None = None,
                 budget: int = 4096) -> CompressionPolicy | None:
    """None means "no policy", i.e. the full-history row."""
    kind = row["kind"]
    if kind == "none":
        return NoCompressionPolicy()

    policy = row.get("policy")
    if policy == "builtin":
        if row["name"] == "fifo":
            return FifoPolicy(keep_last=int(row.get("keep_last", 5)), budget=budget)
        if row["name"] == "llmlingua":
            return LlmLinguaPolicy(keep_rate=float(row.get("keep_rate", 0.30)))
        raise KeyError(f"builtin policy {row['name']!r} is not implemented")

    if policy == "prompt":
        spec = row.get("prompts") or {}
        return PromptedPolicy(
            name=row["name"],
            prompt_dir=prompt_dir_for(row, row["name"]),
            llm=llm or ChatClient(),
            system=spec.get("system", "system"),
            first=spec.get("first"),
            update=spec.get("update"),
            prefix=spec.get("prefix", "prefix") if (prompt_dir_for(row, row["name"]) / "prefix.jinja").exists() else None,
            budget=budget,
            max_chars=row.get("max_chars"),
        )

    raise KeyError(f"unknown policy kind {policy!r} for row {row['name']!r}")
