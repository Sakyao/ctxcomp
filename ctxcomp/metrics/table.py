"""Assemble the results table.

Run directory contract, produced by run_suite.py:

    runs/<stamp>/<method>__<served_model>[__r<k>]/
        run.json                          method, model, fingerprint, verdict, acc
        evaluations/<split>.json          AppWorld's scorer output
        tasks/<task_id>/                  trajectory artefacts, beside dbs/

Everything here is derived from those artefacts. Nothing is read from a log,
because a log line can be written by the wrong process -- which has already
happened once in this project, when a redeployed endpoint silently changed the
model partway through a row.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .difficulty import LEVELS, difficulty
from .tokens import run_token_stats

REPEAT_RE = re.compile(r"^r(\d+)$")


def parse_run_name(name: str) -> tuple[str, str | None, int]:
    """(method, served_model, repeat) from "<method>__<model>[__r<k>]".

    Split on the double underscore rather than matched with a pattern: model ids
    contain hyphens and dots, and a method name is a prefix of no other field, so a
    regexp would be a worse description of the format than a split.
    """
    parts = name.split("__")
    if len(parts) == 1:
        return parts[0], None, 1
    rep = 1
    if REPEAT_RE.match(parts[-1]):
        rep = int(parts[-1][1:])
        parts = parts[:-1]
    return parts[0], "__".join(parts[1:]) or None, rep


def load_runs(runs_root: Path, stamp: str) -> dict[str, list[Path]]:
    """method -> [run dirs], ordered by repeat index."""
    out: dict[str, list[Path]] = {}
    base = runs_root / stamp
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir(), key=lambda p: parse_run_name(p.name)[2]):
        if not d.is_dir():
            continue
        method, _, _ = parse_run_name(d.name)
        out.setdefault(method, []).append(d)
    return out


def successes(run_dir: Path, split: str) -> dict[str, bool]:
    f = run_dir / "evaluations" / f"{split}.json"
    if not f.exists():
        return {}
    try:
        ind = json.loads(f.read_text())["individual"]
    except Exception:
        return {}
    return {k: bool(v.get("success")) for k, v in ind.items()}


def run_meta(run_dir: Path) -> dict:
    f = run_dir / "run.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _pct(hit: int, tot: int) -> str:
    return f"{100 * hit / tot:.1f}" if tot else "-"


def summarise(run_dirs: list[Path], split: str) -> dict:
    """All metrics for one method, joining its repeats."""
    per = [successes(r, split) for r in run_dirs]
    per = [p for p in per if p]
    if not per:
        return {"ok": False}
    tasks = sorted(set().union(*[set(p) for p in per]))
    reps = len(per)
    n = len(tasks)
    accs = [sum(p.get(t, False) for t in tasks) / n for p in per]
    p2 = sum(all(p.get(t, False) for p in per) for t in tasks) / n
    pa2 = sum(any(p.get(t, False) for p in per) for t in tasks) / n

    tot = 0
    per_level = {k: [0, 0] for k in LEVELS}
    for t in tasks:
        lv = difficulty(t)
        if lv is None:
            continue
        per_level[lv][0] += 1
        per_level[lv][1] += any(p.get(t, False) for p in per) if reps > 1 else per[0].get(t, False)
        tot += 1

    tok = run_token_stats(run_dirs[-1], split)
    metas = [run_meta(r) for r in run_dirs]
    # Report the serving backbone alongside the verdict, because it is the one
    # thing a reader needs to judge whether a row is comparable with its
    # neighbours: the endpoint here has been redeployed four times in a day.
    backbones = {(m.get("served_model"), m.get("endpoint_fingerprint"))
                 for m in metas if m.get("served_model")}
    verdicts = {m.get("verdict") for m in metas}
    return {
        "ok": True, "reps": reps, "tasks": n,
        "acc": 100 * sum(accs) / len(accs),
        "pass2": 100 * p2 if reps > 1 else None,
        "pass_at2": 100 * pa2 if reps > 1 else None,
        "steps": tok.get("avg_steps"),
        "peak": tok.get("avg_peak_tokens"),
        "dep": tok.get("avg_dependency"),
        "levels": {k: (_pct(v[1], v[0]), v[0]) for k, v in per_level.items()},
        "backbones": sorted(f"{s} ({f})" for s, f in backbones),
        "contaminated": "CONTAMINATED" in verdicts,
    }


def render(rows: list[tuple[str, dict]], stamp: str, split: str) -> str:
    head = ["Method", "Avg Acc", "Pass^2", "Pass@2", "Steps", "Peak", "Dep.",
            "Easy", "Medium", "Hard"]
    lines = [f"# Results — {stamp}, {split}", "",
             "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for label, s in rows:
        if not s.get("ok"):
            lines.append(f"| {label} |" + " - |" * (len(head) - 1))
            continue
        f2 = lambda v: "-" if v is None else f"{v:.1f}"
        fnum = lambda v, sc=1.0: "-" if v is None else f"{v * sc:.2f}"
        cells = [
            label, f"{s['acc']:.1f}", f2(s["pass2"]), f2(s["pass_at2"]),
            fnum(s["steps"]), fnum(s["peak"], 1e-3), fnum(s["dep"], 1e-6),
        ] + [s["levels"][k][0] for k in ("1", "2", "3")]
        mark = " ⚠️" if s.get("contaminated") else ""
        lines.append("| " + " | ".join(cells) + mark + " |")
    lines += ["", "```",
              "Pass^2 (P^k, k=2)  fraction solved in ALL runs      -> reliability",
              "Avg Acc            mean single-run success rate   (already an average over runs)",
              "Pass@2  (P@k, k=2) fraction solved in AT LEAST ONE -> capability",
              "invariant:  Pass^2 <= Avg Acc <= Pass@2",
              "```",
              "",
              "_Peak in 10^3, Dep. in 10^6. Steps = environment interactions. "
              "Pass^2 / Pass@2 both read the same repeats, so one doubling of cost "
              "fills both; with repeats = 1 they collapse onto Acc and are shown as -._",
              "",
              "## backbone and provenance", ""]
    for label, s in rows:
        if s.get("ok"):
            lines.append(f"- {label}: {', '.join(s['backbones']) or 'unknown'} "
                         f"· {s['reps']} run(s) · {s['tasks']} tasks"
                         + (" · **CONTAMINATED**" if s.get("contaminated") else ""))
    return "\n".join(lines) + "\n"
