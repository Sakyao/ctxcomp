#!/usr/bin/env python3
"""Run every row in methods.yml over one AppWorld split.

    python scripts/run_suite.py --shards 16                    # stamp = timestamp
    python scripts/run_suite.py --stamp round3 --methods trace,fifo --repeats 2

One method-run's results live in exactly one folder:

    runs/<timestamp>/<method>__<served_model>/
        run.json                     method, model, served id, fingerprint, verdict
        evaluations/<split>.{json,txt}    AppWorld's scorer output, including Acc
        tasks/<task_id>/
            env_history.json         one entry per environment interaction
            llm_history.json         per-call messages; Peak and Dep read this
            token_usage_and_cost.json
            results.json             steps, cap, compaction counts, config echo
            dbs/  logs/  checkpoints/  misc/  version/    AppWorld's own

The metric files sit beside `dbs/` rather than in a tree of our own, because they
are the same run: ACON's harness scattered them across two roots and bridged them
with a symlink, and Acc / Steps / Peak / Dep could not be read from one place.

AppWorld's evaluator will only look under <APPWORLD_ROOT>/experiments/outputs, and
`experiment_name` may be a relative path, so runs are named "<stamp>/<run_name>"
and a symlink at <APPWORLD_ROOT>/experiments/outputs/<stamp> resolves back into the
repository. The results are stored once; the second path is a link.

The directory name carries the model the endpoint actually serves, not what the
manifest asked for -- the server rejects names it has not loaded. The behavioural
fingerprint of the backbone (sha256 of a greedy completion) is recorded in run.json
instead of the path: it is evidence of which weights answered, but it changes
whenever the serving configuration does, and a directory that renamed itself
mid-experiment would be unusable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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


def served_model_id(model: str, base: str | None = None, hdr: dict | None = None) -> str:
    """The model id the endpoint actually serves.

    The manifest names a model; the server decides what it has loaded, and the two
    drift -- this cluster's endpoint has been redeployed four times in a day under
    three different names. A request for a name the server does not have fails, so
    the served id is what goes in every request and in the run directory name.
    """
    import os
    import urllib.request
    base = (base or os.environ.get("CTXCOMP_BASE_URL") or os.environ.get("ACON_VLLM_BASE_URL")
            or "http://127.0.0.1:18173/v1").rstrip("/")
    key = os.environ.get("CTXCOMP_API_KEY") or os.environ.get("ACON_VLLM_API_KEY") or "EMPTY"
    hdr = hdr or {"Authorization": f"Bearer {key}"}
    try:
        rq = urllib.request.Request(f"{base}/models", headers=dict(hdr))
        with urllib.request.urlopen(rq, timeout=15) as r:
            ids = [x.get("id") for x in (json.loads(r.read()).get("data") or []) if x.get("id")]
    except Exception:
        return model
    if not ids:
        return model
    if model in ids:
        return model
    print(f"  [endpoint] manifest says model={model!r} but the server serves {ids}; "
          f"using {ids[0]!r}", file=sys.stderr)
    return ids[0]


def endpoint_fingerprint(model: str) -> str:
    """sha256 of a greedy completion: evidence of WHICH weights answered.

    Recorded in run.json, never in the directory name. It is the only value that
    distinguished two redeployments that shared a model name, so it is kept as
    evidence; it is not an identifier, because it changes whenever the serving
    configuration does and a directory that renamed itself mid-experiment would be
    unusable.
    """
    import hashlib
    import os
    import urllib.request
    base = (os.environ.get("CTXCOMP_BASE_URL") or os.environ.get("ACON_VLLM_BASE_URL")
            or "http://127.0.0.1:18173/v1").rstrip("/")
    key = os.environ.get("CTXCOMP_API_KEY") or os.environ.get("ACON_VLLM_API_KEY") or "EMPTY"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    served = served_model_id(model, base, {"Authorization": f"Bearer {key}"})
    try:
        body = json.dumps({"model": served, "temperature": 0.0, "max_tokens": 8,
                           "messages": [{"role": "user",
                                         "content": "The capital of France is"}]}).encode()
        rq = urllib.request.Request(f"{base}/chat/completions", data=body, headers=hdr)
        with urllib.request.urlopen(rq, timeout=60) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
        return hashlib.sha256(txt.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default=None,
                    help="Name of the top-level results folder. Defaults to a "
                         "timestamp, so successive rounds never overwrite each other.")
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
    stamp = args.stamp or time.strftime("%Y%m%d_%H%M%S")
    served = served_model_id(model)
    fp = endpoint_fingerprint(served)
    runs_root = REPO / "runs" / stamp
    runs_root.mkdir(parents=True, exist_ok=True)

    # One location, reached two ways. AppWorld's evaluator will only look under
    # <APPWORLD_ROOT>/experiments/outputs, and experiment_name may be a relative
    # path, so naming runs "<stamp>/<run_name>" puts AppWorld's dbs/ and logs/ into
    # the same folder as our trajectory files. The symlink makes that path resolve
    # back into the repository so the results are not stored twice.
    aw_outputs = Path(os.environ["APPWORLD_ROOT"]) / "experiments" / "outputs"
    aw_outputs.mkdir(parents=True, exist_ok=True)
    bridge = aw_outputs / stamp
    if not bridge.exists():
        bridge.symlink_to(runs_root)
    else:
        print(f"  [bridge] {bridge} already exists "
              f"({'symlink ok' if bridge.is_symlink() else 'NOT a symlink, leaving it'})",
              file=sys.stderr)

    print(f"[suite {stamp}] {len(ids)} tasks, {args.shards} shards, "
          f"max_iter={max_iter}, repeats={reps}")
    print(f"  results   {runs_root}")
    print(f"  backbone  {served}  (fingerprint {fp})")

    required = {"name", "label", "axis", "policy"}
    missing = [(r.get("name", "?"), sorted(required - set(r))) for r in man["rows"]
               if required - set(r)]
    if missing:
        for name, keys in missing:
            print(f"  malformed row {name}: missing {keys}", file=sys.stderr)
        raise SystemExit("fix methods.yml before running")

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
            run_name = f"{row['name']}__{served}{rep}"
            run_dir = runs_root / run_name
            if (run_dir / "run.json").exists():
                print(f"  done  {run_name}")
                continue
            if args.dry_run:
                print(f"  plan  {run_name}")
                continue

            # <- AppWorld writes tasks/<task_id>/{dbs,logs,...} here too, and the
            #    metric files land beside them: one method-run's results in one place.
            out_dir = run_dir / "tasks"
            out_dir.mkdir(parents=True, exist_ok=True)
            experiment_name = f"{stamp}/{run_name}"

            # Evaluate every field the run record needs BEFORE any task starts.
            # Twice now a manifest field was renamed and one write site was missed,
            # which raises only when the row finishes -- hours in. Writing the same
            # expressions to a provisional file moves that failure to second 0. It
            # is deliberately not `run.json`: that file's presence is what makes a
            # re-run skip a completed row, so writing it early would break resume.
            provisional = {
                "method": row["name"], "label": row["label"], "axis": row["axis"],
                "model": model, "served_model": served,
                "endpoint_fingerprint": fp, "repeat": k,
                "max_iter": max_iter,
                "budget_hist": man["thresholds"]["history"],
                "budget_obs": man["thresholds"]["observation"],
                "api_docs_mode": args.api_docs_mode, "split": split,
                "verdict": "running",
            }
            (run_dir / "run.inprogress.json").write_text(json.dumps(provisional, indent=2))

            cfg_kwargs = {"model": model, "max_iter": max_iter,
                          "budget_hist": man["thresholds"]["history"],
                          "budget_obs": man["thresholds"]["observation"],
                          "api_docs_mode": args.api_docs_mode,
                          "experiment_name": experiment_name}
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
                evaluate_dataset(experiment_name=experiment_name,
                                 dataset_name=split, suppress_errors=True)
                src = run_dir / "evaluations" / f"{split}.json"
                if src.exists():
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
                "method": row["name"], "label": row["label"], "axis": row["axis"],
                "model": model, "served_model": served,
                "endpoint_fingerprint": fp, "repeat": k,
                "max_iter": max_iter,
                "budget_hist": man["thresholds"]["history"],
                "budget_obs": man["thresholds"]["observation"],
                "api_docs_mode": args.api_docs_mode, "split": split,
                "seconds": round(elapsed, 1), "verdict": verdict, "acc": acc,
                "finished_at": time.strftime("%F %T"),
            }, indent=2))
            (run_dir / "run.inprogress.json").unlink(missing_ok=True)
            print(f"[{time.strftime('%F %T')}] done  {run_name}  ({elapsed / 60:.1f} min)")
            print(f"  next: evaluate, then re-check the backbone signature; "
                  f"a change means this row must be re-run")

    print(f"\ntable:\n  python scripts/compute_table.py --stamp {args.stamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
