#!/usr/bin/env python3
"""Assemble the results table from finished rows.

Columns match ACON Table 1 + TRACE Table 1, in the order the two papers print
them:

    Method | Avg Acc | Steps | Peak | Dep. | Easy | Medium | Hard
    and, when repeats >= 2, Pass^2 / Pass@2 for the average and each difficulty.

Acc and the difficulty breakdown come from AppWorld's own evaluator
(evaluations/<split>.json). Steps / Peak / Dep come from
analysis_tools.analyze_experiment_tokens_v2, which walks llm_history with
tiktoken -- so those three are recomputed from the trajectory rather than trusted
from a log line, and are therefore available for every run including old ones.

Rows are discovered by globbing `<kind>-<method>__*__sb-*__<stamp>*`, which is
also how the serving backbone is read off: two rows are only comparable when
their `sb-` segment matches. A row whose model_signature.json says CONTAMINATED
is marked in the table instead of being silently averaged in.

Usage
    python compute_table.py --stamp 20260924_v2
    python compute_table.py --stamp 20260924_v2 --methods trace,fifo
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

SUITE = Path(__file__).resolve().parent
# The runs and the analysis tools are both in the ACON checkout: Steps/Peak/Dep come
# from analysis_tools.analyze_experiment_tokens_v2, which is the implementation the
# numbers were measured with, and Acc comes from AppWorld's own evaluator output
# sitting beside each run. Nothing here recomputes a metric its own way.
ACON = Path(os.environ.get("ACON_ROOT", "/z5s/morph/home/sjk/Agent/datasets/repos/acon"))
REPO = ACON
sys.path.insert(0, str(ACON))

SPLIT_COUNTS = {"1": ("Easy", 57), "2": ("Medium", 48), "3": ("Hard", 63)}


def load_methods() -> list[tuple[str, str]]:
    import yaml
    m = yaml.safe_load((SUITE / "methods.yml").read_text())
    return [(r["name"], r["label"]) for r in m["rows"]]


def find_runs(stamp: str, name: str) -> list[Path]:
    # AppWorld names the run directory "<model>_<tag>" and the tag begins with the
    # row name, so the pattern allows an arbitrary prefix. It deliberately does not
    # require a "-" before the name the way upstream's did: the row name is already
    # axis-qualified (hist_*, obs_*), and demanding that hyphen matches nothing and
    # prints a table of dashes that reads like every row failed.
    pat = str(REPO / "experiments" / "appworld" / "experiments" / "outputs"
              / f"*{name}__*__sb-*__{stamp}*")
    return sorted(Path(p) for p in glob.glob(pat) if Path(p).is_dir())


def read_eval(run: Path, split: str) -> dict[str, bool]:
    f = run / "evaluations" / f"{split}.json"
    if not f.exists():
        return {}
    try:
        ind = json.loads(f.read_text())["individual"]
    except Exception:
        return {}
    return {t: bool(r.get("success")) for t, r in ind.items()}


def difficulty_of(task_id: str, cache: dict) -> str | None:
    if task_id in cache:
        return cache[task_id]
    try:
        from experiments.analysis_tools.utils import get_task_difficulty
        d = get_task_difficulty(task_id)
    except Exception:
        d = None
    cache[task_id] = d
    return d


def token_stats(run_id: str, split: str) -> tuple[float, float, float] | None:
    try:
        from experiments.analysis_tools.utils import analyze_experiment_tokens_v2
    except Exception as e:
        print(f"  [warn] analysis_tools unavailable: {e}", file=sys.stderr)
        return None
    try:
        res = analyze_experiment_tokens_v2(run_id, folds=[split])
        agent = res["by_fold"][split]["agent"]
        return (agent.get("avg_steps"), agent.get("avg_peak_tokens"),
                agent.get("avg_dependency"))
    except Exception as e:
        print(f"  [warn] token stats failed for {run_id}: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--split", default="test_normal")
    ap.add_argument(
        "--eval-dataset", default=None,
        help="Which dataset the evaluator walked. Defaults to --split. A smoke run "
             "evaluated against a subset writes evaluations/<eval-dataset>.json, "
             "while Steps/Peak/Dep still come from the --split output directory.",
    )
    ap.add_argument("--methods", default=None)
    args = ap.parse_args()
    eval_dataset = args.eval_dataset or args.split
    want = set(args.methods.split(",")) if args.methods else None

    hdr = ["Method", "Avg Acc", "Pass^2", "Pass@2", "Steps", "Peak", "Dep.",
           "Easy", "Medium", "Hard"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    dcache: dict = {}
    ledger = []

    for name, label in load_methods():
        if want and name not in want:
            continue
        runs = find_runs(args.stamp, name)
        if not runs:
            out.append(f"| {label} |" + " - |" * (len(hdr) - 1))
            ledger.append((label, "no runs"))
            continue

        per_run = [read_eval(r, eval_dataset) for r in runs]
        sigs = set()
        verdicts = []
        for r in runs:
            ms = r / "model_signature.json"
            if ms.exists():
                try:
                    d = json.loads(ms.read_text())
                    sigs.add(d.get("sig_after"))
                    verdicts.append(d.get("verdict"))
                except Exception:
                    pass
        tasks = sorted(set().union(*[set(p) for p in per_run])) if per_run else []
        if not tasks:
            out.append(f"| {label} |" + " - |" * (len(hdr) - 1))
            ledger.append((label, "no evaluation"))
            continue

        n = len(tasks)
        accs = [sum(p.get(t, False) for t in tasks) / n for p in per_run]
        allok = [all(p.get(t, False) for p in per_run) for t in tasks]
        anyok = [any(p.get(t, False) for p in per_run) for t in tasks]
        p2 = sum(allok) / n
        pa2 = sum(anyok) / n
        reps = len(per_run)

        st = token_stats(runs[-1].name, args.split)
        steps, peak, dep = st if st else (None, None, None)

        by_diff = defaultdict(lambda: [0, 0])
        for t, ok in zip(tasks, anyok if reps > 1 else [p.get(t, False) for t in tasks for p in per_run[:1]]):
            d = difficulty_of(t, dcache)
            if d is None:
                continue
            by_diff[str(d)][0] += 1
            by_diff[str(d)][1] += ok
        diff_cells = []
        for k in ("1", "2", "3"):
            nm, _ = SPLIT_COUNTS[k]
            tot, hit = by_diff.get(k, [0, 0])
            diff_cells.append(f"{100*hit/tot:.1f}" if tot else "-")

        def fnum(v, scale=1.0, nd=2):
            return "-" if v is None else f"{v*scale:.{nd}f}"

        row = [label,
               f"{100*sum(accs)/len(accs):.1f}",
               f"{100*p2:.1f}" if reps > 1 else "-",
               f"{100*pa2:.1f}" if reps > 1 else "-",
               fnum(steps, 1, 2),
               fnum(peak, 1e-3, 2),
               fnum(dep, 1e-6, 2)] + diff_cells
        mark = ""
        if "CONTAMINATED" in verdicts:
            mark = " ⚠️"
        out.append("| " + " | ".join(row) + mark + " |")
        ledger.append((label, f"sigs={sorted(sigs)} reps={reps} {','.join(verdicts)}"))

    print(f"# Results — stamp {args.stamp}, split {args.split}\n")
    print("\n".join(out))
    print("\n_Peak in 10^3 tokens, Dep. in 10^6. Steps = environment interactions. "
          "Pass^2/Pass@2 need repeats >= 2._\n")
    print("## run ledger")
    for label, note in ledger:
        print(f"- {label}: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
