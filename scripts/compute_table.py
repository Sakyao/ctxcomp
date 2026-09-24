#!/usr/bin/env python3
"""Print the results table for a stamp.

    python scripts/compute_table.py --stamp 20260924_full1000

Columns follow the two source papers so the numbers line up with theirs:
Acc / Steps / Peak / Dep. as in ACON Table 1, Pass^2 / Pass@2 as in TRACE Table 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctxcomp.methods import load_manifest  # noqa: E402
from ctxcomp.metrics import load_runs, render, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", default=None,
                    help="comma-separated subset, default all")
    args = ap.parse_args()

    man = load_manifest()
    split = args.split or man["split"]
    want = set(args.methods.split(",")) if args.methods else None

    runs = load_runs(Path(args.runs_root), args.stamp)
    if not runs:
        print(f"no runs found under {args.runs_root}/{args.stamp}", file=sys.stderr)

    rows = []
    for row in man["rows"]:
        name, label = row["name"], row["label"]
        if want and name not in want:
            continue
        dirs = runs.get(name, [])
        rows.append((label, summarise(dirs, split) if dirs else {"ok": False}))

    text = render(rows, args.stamp, split)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
