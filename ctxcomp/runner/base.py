"""Engine boundary: anything that can run one AppWorld task under a policy.

The runner is an interface rather than a fixed loop so that the compression policy
stays the only thing under study. A different execution backend plugs in behind
this protocol without touching methods/ or metrics/.

The two budgets are separate because the two axes are: T_hist and T_obs are
different numbers in the paper's setup, and giving an observation policy the
history threshold would make it fire 4x less often than the method it claims to be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..methods.base import CompressionPolicy


@dataclass
class EngineConfig:
    model: str = "deepseek-v4.1-flash"
    max_iter: int = 1000
    budget_hist: int = 4096
    budget_obs: int = 1024
    agent_prompt: str | None = None
    experiment_name: str = "ctxcomp"
    temperature: float = 0.8
    api_docs_mode: str = "full"          # full | instruction-only
    output_dir: str | None = None        # ctxcomp's own artefacts (metrics)
    save_appworld_artifacts: bool = True # dbs/logs/version under the experiment dir


@dataclass
class TaskOutcome:
    task_id: str
    success: bool | None = None          # filled by the evaluator, not the engine
    steps: int = 0
    hit_cap: bool = False
    error: str | None = None
    compactions_hist: int = 0
    compactions_obs: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    def run_task(self, task_id: str, policy: CompressionPolicy | None,
                 cfg: EngineConfig) -> TaskOutcome:
        ...
