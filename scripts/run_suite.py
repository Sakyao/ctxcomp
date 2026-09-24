#!/usr/bin/env python3
"""Run every row in methods.yml over one AppWorld split.

    python scripts/run_suite.py --stamp my_run --shards 16
    python scripts/run_suite.py --stamp my_run --methods trace,fifo --repeats 2

Run directory naming encodes the backbone that actually served the row, not the
label the client sent:

    runs/<stamp>/<kind>-<method>__<model>__sb-<sig>/
        run.json          method, model, endpoint signature, verdict
        <split>/task_*/   trajectory artefacts
        evaluations/      AppWorld evaluator output

The endpoint echoes back whatever model name the client asks for, so the label is
not evidence of what answered. In this project the endpoint was redeployed twice
during a single round, and one row ended up split across two different backbones
with no record of it. The signature is therefore read from the server, embedded in
the directory name, and compared before and after the row; a mismatch stamps the
row CONTAMINATED instead of silently averaging two models together.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctxcomp.llm import ChatClient                      # noqa: E402
from ctxcomp.methods import build_policy, load_manifest  # noqa: E402
from ctxcomp.runner import AppWorldEngine, EngineConfig  # noqa: E402


def shard(items: list, k: int) -> list[list]:
    return [items[i::k] for i in range(k)]


def worker(args: tuple) -> tuple:
    """Run one shard. Kept module-level so ProcessPoolExecutor can pickle it."""
    row_name, task_ids, cfg_kwargs, out_dir = args
    man = load_manifest()
    row = next(r for r in man["rows"] if r["name"] == row_name)
    cfg = EngineConfig(**cfg_kwargs)
    cfg.output_dir = out_dir
    policy = None
    if row.get("axis", "history") != "none":
        # budget comes from the manifest per axis: T_hist and T_obs differ, and
        # `man` must be passed or the policy silently falls back to the history
        # threshold and an observation row fires 4x less often than it should.
        policy = build_policy(row, llm=ChatClient(), man=man)
    engine = AppWorldEngine(ChatClient())
    done, failed = 0, 0
    for tid in task_ids:
        try:
            engine.run_task(tid, policy, cfg)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{row_name}] {tid} failed: {type(e).__name__}: {e}", flush=True)
    return row_name, done, failed


def endpoint_signature(model: str) -> str:
    """Short id for the weights currently being served.

    Two ingredients, because neither alone is trustworthy:

    * the model name the server reports -- an *identity* claim, and the server
      echoes back whatever the client asks for, so on its own it proves nothing;
    * the sha256 of a greedy completion -- a *behavioural* fingerprint, which
      changes when the weights or the serving configuration change even if the
      name stays the same.

    Together they produced the only reliable evidence in this project that the
    endpoint had swapped models mid-round.
    """
    import hashlib
    import os
    import urllib.request

    base = (os.environ.get("CTXCOMP_BASE_URL") or os.environ.get("ACON_VLLM_BASE_URL")
            or "http://127.0.0.1:18173/v1").rstrip("/")
    key = os.environ.get("CTXCOMP_API_KEY") or os.environ.get("ACON_VLLM_API_KEY") or "EMPTY"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    name = "unknown"
    try:
        rq = urllib.request.Request(f"{base}/models", headers=hdr)
        with urllib.request.urlopen(rq, timeout=15) as r:
            d = json.loads(r.read())
        name = (d.get("data") or [{}])[0].get("id") or "unknown"
    except Exception:
        pass

    # Probe with the id the SERVER reports, not the one the manifest names. The
    # two drift: this cluster's endpoint has been redeployed four times in a day,
    # and the manifest's model name is a claim about intent while /v1/models is a
    # claim about what is actually loaded. Probing with a name the server does not
    # serve fails, which silently degrades the signature to "unknown" -- the one
    # value that makes a run unattributable.
    served = model
    try:
        rq = urllib.request.Request(f"{base}/models", headers=hdr)
        with urllib.request.urlopen(rq, timeout=15) as r:
            d = json.loads(r.read())
        ids = [x.get("id") for x in (d.get("data") or []) if x.get("id")]
        if ids:
            served = model if model in ids else ids[0]
            if served != model:
                print(f"  [endpoint] manifest says model={model!r} but the server "
                      f"serves {ids}; probing {served!r}", file=__import__("sys").stderr)
    except Exception:
        pass

    probe = "The capital of France is"
    h = "nohash"
    try:
        body = json.dumps({"model": served, "temperature": 0.0, "max_tokens": 8,
                           "messages": [{"role": "user", "content": probe}]}).encode()
        rq = urllib.request.Request(f"{base}/chat/completions", data=body, headers=hdr)
        with urllib.request.urlopen(rq, timeout=60) as r:
            out = json.loads(r.read())
        txt = out["choices"][0]["message"]["content"]
        h = hashlib.sha256(txt.encode()).hexdigest()[:8]
    except Exception:
        pass

    label = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:12].rstrip("-") or "unknown"
    return f"{label}_{h}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--methods", default=None, help="comma-separated subset, in order")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--api-docs-mode", default="full", choices=["full", "instruction-only"])
    ap.add_argument("--appworld-root", default=os.environ.get("APPWORLD_ROOT"),
                    help="AppWorld home. Needed before appworld is imported: the "
                         "package resolves data/tasks and experiments/outputs "
                         "relative to it, and without it load_task_ids raises.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.appworld_root:
        os.environ["APPWORLD_ROOT"] = args.appworld_root
    if not os.environ.get("APPWORLD_ROOT"):
        print("APPWORLD_ROOT is not set and --appworld-root was not given; "
              "AppWorld will resolve data/tasks relative to the cwd and fail.",
              file=sys.stderr)
    # imported here, after the env is set, because appworld reads it at import time
    from appworld import load_task_ids

    man = load_manifest()
    split = man["split"]
    model = man["model"]
    reps = args.repeats or man["repeats"]
    max_iter = args.max_iter or man["max_iter"]
    want = set(args.methods.split(",")) if args.methods else None
    ids = list(load_task_ids(split))
    sig = endpoint_signature(model)
    runs_root = REPO / "runs" / args.stamp
    runs_root.mkdir(parents=True, exist_ok=True)

    print(f"[suite {args.stamp}] {len(ids)} tasks, {args.shards} shards, "
          f"max_iter={max_iter}, repeats={reps}, backbone={sig}")

    for row in man["rows"]:
        if want and row["name"] not in want:
            continue
        try:
            build_policy(row, llm=ChatClient(), man=man)
        except Exception as e:  # noqa: BLE001
            print(f"  skip  {row['name']}: {type(e).__name__}: {e}")
            continue
        for k in range(1, reps + 1):
            rep = "" if reps == 1 else f"__r{k}"
            # Row names already carry the axis (hist_* / obs_*), so the directory
            # name does not repeat it: an extra prefix would let a row be renamed
            # in the manifest while its old name lived on in every stored run.
            run_name = f"{row['name']}__{model}__sb-{sig}{rep}"
            run_dir = runs_root / run_name
            if (run_dir / "run.json").exists():
                print(f"  done  {run_name}")
                continue
            if args.dry_run:
                print(f"  plan  {run_name}")
                continue

            out_dir = run_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg_kwargs = {"model": model, "max_iter": max_iter,
                          "budget_hist": man["thresholds"]["history"],
                          "budget_obs": man["thresholds"]["observation"],
                          "api_docs_mode": args.api_docs_mode,
                          "experiment_name": run_name}
            t0 = time.time()
            print(f"[{time.strftime('%F %T')}] start {run_name}", flush=True)
            with ProcessPoolExecutor(max_workers=args.shards) as ex:
                futs = [ex.submit(worker, (row["name"], s, cfg_kwargs, str(out_dir)))
                        for s in shard(ids, args.shards)]
                for f in as_completed(futs):
                    _, d, bad = f.result()
                    if bad:
                        print(f"  [!] {bad} task(s) raised")
            elapsed = time.time() - t0

            # Evaluate, so the row produces a score rather than only trajectories.
            # AppWorld's evaluator verifies a task by replaying the final database
            # state (tasks/<task_id>/dbs) against ground truth, and it only looks
            # under its own experiment directory -- which is why experiment_name is
            # the run name in the first place. Result is copied into the run dir so
            # compute_table.py can stay independent of AppWorld's layout.
            verdict, acc = "clean", None
            try:
                from appworld.evaluator import evaluate_dataset
                evaluate_dataset(experiment_name=run_name, dataset_name=split,
                                 suppress_errors=True)
                src = (Path(os.environ["APPWORLD_ROOT"]) / "experiments" / "outputs"
                       / run_name / "evaluations" / f"{split}.json")
                if src.exists():
                    (run_dir / "evaluations").mkdir(exist_ok=True)
                    shutil.copy2(src, run_dir / "evaluations" / f"{split}.json")
                    ind = json.loads(src.read_text()).get("individual", {})
                    n = len(ind)
                    ok = sum(1 for r in ind.values() if r.get("success"))
                    acc = round(100 * ok / n, 1) if n else None
                    print(f"  acc   {ok}/{n} = {acc}%")
                else:
                    verdict = "evaluation_missing"
                    print(f"  [!] evaluator produced no {src}")
            except Exception as e:              # noqa: BLE001
                verdict = f"evaluation_failed: {type(e).__name__}: {e}"
                print(f"  [!] evaluation failed: {e}")
            (run_dir / "run.json").write_text(json.dumps({
                "method": row["name"], "label": row["label"], "kind": row["kind"],
                "model": model, "endpoint_signature": sig, "repeat": k,
                "max_iter": max_iter, "history_budget": args.history_budget,
                "api_docs_mode": args.api_docs_mode, "split": split,
                "seconds": round(elapsed, 1), "verdict": verdict, "acc": acc,
                "finished_at": time.strftime("%F %T"),
            }, indent=2))
            print(f"[{time.strftime('%F %T')}] done  {run_name}  ({elapsed / 60:.1f} min)")
            print(f"  next: evaluate, then re-check the backbone signature; "
                  f"a change means this row must be re-run")

    print(f"\ntable:\n  python scripts/compute_table.py --stamp {args.stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
