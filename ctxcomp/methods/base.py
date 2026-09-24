"""The one interface every method implements.

A policy sees the compressible history and returns the text that replaces it.
Nothing else about the agent is visible to it, which is what makes the methods
comparable: the agent, its prompt, its tools and its decoding are frozen, and the
only thing that differs between rows is the representation handed back.

Adding the project's own method means adding one class here. It does not mean
touching the runner, the metrics or the config generator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Turn:
    """One agent step. `observation` is the environment reply, `action` the code."""
    index: int
    action: str
    observation: str


@dataclass
class CompressionRequest:
    task: str
    history: list[Turn]
    prev_summary: str = ""
    budget: int = 4096
    is_first: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    text: str
    strategy: str
    meta: dict[str, Any] = field(default_factory=dict)
    dropped_turns: int = 0


class CompressionPolicy(Protocol):
    """Replace the compressible history with a bounded text representation.

    Implementations must be pure with respect to `history` and must never mutate
    it: the runner keeps the raw history for the reliability metrics.
    """

    name: str

    def compress(self, req: CompressionRequest) -> CompressionResult:
        ...
