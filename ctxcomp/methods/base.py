"""The one interface every method implements.

A policy sees the compressible input and returns the text that replaces it.
Nothing else about the agent is visible to it, which is what makes the methods
comparable: the agent, its prompt, its tools and its decoding are frozen, and the
only thing that differs between rows is the representation handed back.

There are TWO axes, and they are not interchangeable:

    history      replace the accumulated interaction history   -> `summary`
    observation  replace the current observation               -> `observation`

ACON equations (3) and (4). They take different inputs, produce different shapes
of output, trip at different thresholds, and are optimised separately (P_hist vs
P_obs), which is why they are separate rows in the manifest and why a policy must
declare which one it is replacing. A policy that returned a summary where the
runner expected an observation would silently corrupt the trajectory.

Adding the project's own method means adding one class here. It does not mean
touching the runner, the metrics or the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Axis = Literal["history", "observation"]


@dataclass
class Turn:
    """One agent step. `observation` is the environment reply, `action` the code."""
    index: int
    action: str
    observation: str


@dataclass
class CompressionRequest:
    """Everything a policy is allowed to see.

    `axis` tells the policy which part of the context it is replacing, and
    therefore which threshold brought it here and which input is populated:

        axis == "history"      -> `history` is long,        `observation` is ""
        axis == "observation"  -> `observation` is long,    `history` is the
                                  PREVIOUS history (h_{t-1}), per equation (4)

    Equation (4) compresses the current observation *conditioned on* what came
    before, so an observation policy does get the history. It must not rewrite it.
    """
    task: str
    axis: Axis = "history"
    history: list[Turn] = field(default_factory=list)
    observation: str = ""
    prev_summary: str = ""
    budget: int = 4096
    is_first: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """The replacement text, and what it replaces.

    `replaces` is not redundant with the request's axis: a policy is allowed to
    refuse (return the input unchanged) but not to change what it is replacing.
    """
    text: str
    strategy: str
    replaces: Axis = "history"
    meta: dict[str, Any] = field(default_factory=dict)
    dropped_turns: int = 0


class CompressionPolicy(Protocol):
    """Replace the compressible input with a bounded text representation.

    Implementations must be pure with respect to `history` and `observation` and
    must never mutate them: the runner keeps the raw trajectory for the
    reliability metrics and for the evaluator.
    """

    name: str
    axis: Axis

    def compress(self, req: CompressionRequest) -> CompressionResult:
        ...
