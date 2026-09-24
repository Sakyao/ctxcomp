"""Policies that need no LLM: full history, recency truncation, token pruning."""
from __future__ import annotations

import json
import urllib.request

from .base import CompressionRequest, CompressionResult, Turn


def render_turns(turns: list[Turn]) -> str:
    """Canonical text form of a span of turns, shared by every policy.

    Kept in one place so that FIFO, LLMLingua and the prompted policies all
    produce the same surface format; otherwise a difference between two rows
    could come from formatting rather than from the policy.
    """
    out = []
    for t in turns:
        out.append(f"ACTION:\n{t.action}\n\nOBSERVATION:\n{t.observation}")
    return "\n\n".join(out)


class NoCompressionPolicy:
    """Full history. Present so the interface has a no-op member; the runner
    never calls it, it just leaves the context alone."""

    name = "no_compression"

    def compress(self, req: CompressionRequest) -> CompressionResult:
        return CompressionResult(text=render_turns(req.history), strategy="none")


class FifoPolicy:
    """Recency-only window.

    Drops whole turns oldest-first. Deliberately keeps *entire* turns rather than
    masking observations, because that is what the baseline does: the difference
    between "drop the turn" and "keep the action, drop the observation" is the
    difference between this and observation masking, and conflating them would
    make the row uninterpretable.
    """

    def __init__(self, keep_last: int = 5, budget: int = 4096):
        self.name = "fifo"
        self.keep_last = keep_last
        self.budget = budget

    def compress(self, req: CompressionRequest) -> CompressionResult:
        keep = req.history[-self.keep_last:]
        dropped = len(req.history) - len(keep)
        return CompressionResult(text=render_turns(keep), strategy="fifo",
                                 dropped_turns=dropped,
                                 meta={"keep_last": self.keep_last})


class LlmLinguaPolicy:
    """Token-level extractive pruning.

    Delegates to a local LLMLingua-2 server rather than importing the library, so
    that the compressor's weights and version are pinned outside this repo and are
    recorded with the run. Contract: POST {"text", "rate"} -> {"text"}.
    """

    def __init__(self, keep_rate: float = 0.30, endpoint: str | None = None):
        self.name = "llmlingua"
        self.keep_rate = keep_rate
        self.endpoint = endpoint or "http://127.0.0.1:9999/compress"

    def compress(self, req: CompressionRequest) -> CompressionResult:
        text = render_turns(req.history)
        body = json.dumps({"text": text, "rate": self.keep_rate}).encode()
        rq = urllib.request.Request(self.endpoint, data=body,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=300) as r:
            out = json.loads(r.read())
        pruned = out.get("text") or out.get("compressed") or ""
        return CompressionResult(text=pruned, strategy="llmlingua",
                                 meta={"keep_rate": self.keep_rate,
                                       "endpoint": self.endpoint,
                                       "in_chars": len(text),
                                       "out_chars": len(pruned)})
