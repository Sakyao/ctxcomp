"""Steps, Peak and Dep -- recomputed from the trajectory, not read from a log.

Formulas follow the ACON paper's Table 1:

    Steps   mean environment interactions per task
    Peak    max input context length over all steps, INCLUDING the system prompt
            (reported in 10^3 tokens)
    Dep.    sum_t ((n_i + 2*n_o) * n_o) / 2, where n_i EXCLUDES the system prompt
            (reported in 10^6)

Dep is recomputed from the stored per-call message lists rather than trusted from
a counter, so it is available for any past run and can be re-derived if the
tokenizer or the definition changes. That matters because Dep is the one metric
whose definition is easy to get subtly wrong -- whether the system prompt counts
changes the number by a constant factor, and the two papers print it in different
units from Peak.
"""
from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

_CACHE: dict = {}


def _encoding(model: str = "gpt-4.1"):
    if model in _CACHE:
        return _CACHE[model]
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
    _CACHE[model] = enc
    return enc


def _count(enc, text: str) -> int:
    if not text:
        return 0
    return len(enc.encode(text)) if enc is not None else max(1, len(text) // 4)


def task_steps(task_dir: Path) -> int | None:
    """Environment interactions, i.e. how many times the env was called."""
    p = task_dir / "env_history.json"
    if not p.exists():
        return None
    try:
        h = json.loads(p.read_text(errors="replace"))
    except Exception:
        return None
    return len(h) if isinstance(h, list) else None


def task_peak_dep(task_dir: Path, enc=None) -> tuple[int, float] | None:
    """(peak_input_tokens, dependency) for one task."""
    p = task_dir / "llm_history.json"
    if not p.exists():
        return None
    try:
        sessions = json.loads(p.read_text(errors="replace"))
    except Exception:
        return None
    if not isinstance(sessions, list):
        return None
    enc = enc or _encoding()
    peak = 0
    dep = 0.0
    for session in sessions:
        if not isinstance(session, list):
            continue
        msgs = [m for m in session if isinstance(m, dict)]
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            out_text = str(m.get("content") or "")
            prior = msgs[:i]
            in_all = "\n".join(f"{x.get('role','')}: {x.get('content','')}" for x in prior)
            in_nosys = "\n".join(f"{x.get('role','')}: {x.get('content','')}"
                                 for x in prior if x.get("role") != "system")
            n_i = _count(enc, in_all)
            n_i_ns = _count(enc, in_nosys)
            n_o = _count(enc, out_text)
            peak = max(peak, n_i)
            if n_o > 0:
                dep += ((n_i_ns + 2 * n_o) * n_o) / 2.0
    return peak, dep


def run_token_stats(run_dir: Path, split: str) -> dict:
    """Aggregate Steps / Peak / Dep over the tasks of one run."""
    tasks_dir = run_dir / split
    if not tasks_dir.is_dir():
        tasks_dir = run_dir
    enc = _encoding()
    steps, peaks, deps = [], [], []
    for t in sorted(os.listdir(tasks_dir)):
        td = tasks_dir / t
        if not td.is_dir() or not t.startswith("task_"):
            continue
        s = task_steps(td)
        if s is not None:
            steps.append(s)
        pd = task_peak_dep(td, enc)
        if pd is not None:
            peaks.append(pd[0])
            deps.append(pd[1])
    return {
        "tasks": len(steps),
        "avg_steps": st.mean(steps) if steps else None,
        "avg_peak_tokens": st.mean(peaks) if peaks else None,
        "avg_dependency": st.mean(deps) if deps else None,
    }
