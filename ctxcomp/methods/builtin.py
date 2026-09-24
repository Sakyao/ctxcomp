"""Policies that need no LLM, on both axes.

Four mechanisms, two per axis:

    history      fifo              drop whole turns, oldest first
                 llmlingua         token-level extractive pruning

    observation  obs_fifo          truncate one over-long observation
                 obs_llmlingua     token-level pruning of one observation

The observation variants are not "fifo over observations". By ACON equation (4)
the observation axis compresses the CURRENT observation when that observation
alone exceeds T_obs, so the mechanism has to be a length reduction of a single
observation. A recency window over observations is a different method and is not
implemented here.
"""
from __future__ import annotations

import json
import urllib.request

from .base import CompressionRequest, CompressionResult, Turn


def render_turns(turns: list[Turn]) -> str:
    """Canonical text form of a span of turns, shared by every policy.

    Kept in one place so FIFO, LLMLingua and the prompted policies all produce the
    same surface format; otherwise a difference between two rows could come from
    formatting rather than from the policy.
    """
    out = []
    for t in turns:
        out.append(f"ACTION:\n{t.action}\n\nOBSERVATION:\n{t.observation}")
    return "\n\n".join(out)


def render_history(turns: list[Turn], summary: str = "") -> str:
    parts = []
    if summary:
        parts.append(f"PRIOR SUMMARY:\n{summary}")
    if turns:
        parts.append(render_turns(turns))
    return "\n\n".join(parts)


def truncate_middle(text: str, budget_chars: int, marker: str = "\n...[truncated]...\n") -> str:
    """Keep the head and the tail, drop the middle.

    Both ends are kept because these are tool observations: the head carries the
    shape of the payload (status, keys, ids) and the tail carries the end of the
    listing, which is where pagination and error messages live. Keeping only the
    head is the common mistake -- it looks like it preserves structure while
    silently dropping the failures the agent needs to see.
    """
    if len(text) <= budget_chars:
        return text
    room = max(0, budget_chars - len(marker))
    head = room * 2 // 3
    tail = room - head
    return text[:head] + marker + text[-tail:] if tail else text[:head] + marker


class NoCompressionPolicy:
    """Full history. Present so the interface has a no-op member; the runner never
    calls it, it just leaves the context alone."""

    name = "no_compression"
    axis = "history"

    def compress(self, req: CompressionRequest) -> CompressionResult:
        return CompressionResult(text=render_history(req.history, req.prev_summary),
                                 strategy="none")


class FifoPolicy:
    """history axis: recency-only window over whole turns.

    Deliberately keeps *entire* turns rather than masking observations, because
    that is what the baseline does: the difference between "drop the turn" and
    "keep the action, drop the observation" is the difference between this and
    observation masking, and conflating them would make the row uninterpretable.
    """

    name = "hist_fifo"
    axis = "history"

    def __init__(self, keep_last: int = 5):
        self.keep_last = keep_last

    def compress(self, req: CompressionRequest) -> CompressionResult:
        keep = req.history[-self.keep_last:]
        dropped = len(req.history) - len(keep)
        return CompressionResult(text=render_turns(keep), strategy="fifo",
                                 replaces="history", dropped_turns=dropped,
                                 meta={"keep_last": self.keep_last})


class ObsFifoPolicy:
    """observation axis: truncate one over-long observation, keeping both ends."""

    name = "obs_fifo"
    axis = "observation"

    def __init__(self, keep_last: int = 5):
        self.keep_last = keep_last          # unused; kept so the manifest is uniform

    def compress(self, req: CompressionRequest) -> CompressionResult:
        text = truncate_middle(req.observation, req.budget * 4)
        return CompressionResult(
            text=text, strategy="obs_fifo", replaces="observation",
            meta={"in_chars": len(req.observation), "out_chars": len(text),
                  "budget_chars": req.budget * 4})


class LlmLinguaPolicy:
    """history axis: token-level extractive pruning.

    Delegates to a local LLMLingua-2 server rather than importing the library, so
    that the compressor's weights and version are pinned outside this repo and are
    recorded with the run. Contract: POST {"text", "rate"} -> {"text"}.
    """

    name = "hist_llmlingua"
    axis = "history"

    def __init__(self, keep_rate: float = 0.30, endpoint: str | None = None):
        self.keep_rate = keep_rate
        self.endpoint = endpoint or "http://127.0.0.1:9999/compress"

    def _prune(self, text: str) -> str:
        body = json.dumps({"text": text, "rate": self.keep_rate}).encode()
        rq = urllib.request.Request(self.endpoint, data=body,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=300) as r:
            out = json.loads(r.read())
        return out.get("text") or out.get("compressed") or ""

    def compress(self, req: CompressionRequest) -> CompressionResult:
        text = render_history(req.history, req.prev_summary)
        pruned = self._prune(text)
        return CompressionResult(text=pruned, strategy="llmlingua",
                                 replaces="history",
                                 meta={"keep_rate": self.keep_rate,
                                       "endpoint": self.endpoint,
                                       "in_chars": len(text), "out_chars": len(pruned)})


class ObsLlmLinguaPolicy(LlmLinguaPolicy):
    """observation axis: the same pruner, applied to the current observation."""

    name = "obs_llmlingua"
    axis = "observation"

    def compress(self, req: CompressionRequest) -> CompressionResult:
        pruned = self._prune(req.observation)
        return CompressionResult(text=pruned, strategy="obs_llmlingua",
                                 replaces="observation",
                                 meta={"keep_rate": self.keep_rate,
                                       "endpoint": self.endpoint,
                                       "in_chars": len(req.observation),
                                       "out_chars": len(pruned)})
