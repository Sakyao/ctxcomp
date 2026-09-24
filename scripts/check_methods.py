#!/usr/bin/env python3
"""Validate that every row in methods.yml can actually be built.

Run this before a suite run. The failure it is designed to catch is the quiet
one: a missing prompt template does not raise anywhere in the execution path --
the compressor falls back to something generic, the row still produces numbers,
and the table looks fine while the row means nothing. Here it raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctxcomp.methods import build_policy, load_manifest  # noqa: E402


def main() -> int:
    man = load_manifest()
    print(f"methods.yml: {len(man['rows'])} rows, model={man['model']}, "
          f"max_iter={man['max_iter']}, repeats={man['repeats']}")
    bad = []
    for row in man["rows"]:
        try:
            p = build_policy(row)
            extra = ""
            if p is not None and hasattr(p, "prompt_dir"):
                extra = f"  prompts={p.prompt_dir.name}"
            print(f"  ok    {row['name']:16} {type(p).__name__:20}{extra}")
        except Exception as e:  # noqa: BLE001
            bad.append(row["name"])
            print(f"  FAIL  {row['name']:16} {type(e).__name__}: {e}")
    if bad:
        print(f"\n{len(bad)} row(s) cannot run: {', '.join(bad)}")
        return 1
    print(f"\nall {len(man['rows'])} rows build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
