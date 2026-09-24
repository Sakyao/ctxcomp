#!/usr/bin/env bash
#
# Run every row in methods.yml, on appworld, in run order.
#
# Naming. Run ids encode the model that actually served the row, not the label
# the client sent -- the endpoint echoes back any name it is given, so the label
# is not evidence. On 2026-09-24 :18173 was redeployed twice mid-round
# (V4-Flash-0731 -> V4.1-Flash at 07:29, back at 11:37) and one row ended up
# split across both backbones. Hence:
#
#     <kind>-<method>__<model>__sb-<sig8>__<stamp>[__r<k>]
#
# Two runs are comparable only if their sb- matches.
#
# Contamination guard. Each row fingerprints the endpoint before and after it
# runs and writes model_signature.json next to the row. If the signature moved
# the row is stamped CONTAMINATED. That is detection, not repair: the row still
# has to be re-run, but the failure is visible in the artefacts rather than only
# in a log.
#
# Usage
#   bash run_suite.sh
#   bash run_suite.sh --methods hist_trace,hist_fifo --max-iter 300 --shards 16
#   bash run_suite.sh --methods obs_trace --task-ids 3d9a636_1 --max-iter 3 --stamp smoke
#   bash run_suite.sh --dry-run
#
# --task-ids is the single-task path: one row, one task, the same run_all.py call the
# shard loop makes, for checking that a change still runs end to end without
# spending a whole round.
#
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
# This repository owns the suite, not the harness. Every row is executed by the code
# in the ACON checkout it was audited against, so a baseline cannot drift from its
# own implementation by being re-implemented here. Point ACON_ROOT elsewhere to
# reproduce against a different checkout; nothing else in this file assumes a path.
ACON="${ACON_ROOT:-/z5s/morph/home/sjk/Agent/datasets/repos/acon}"
REPO="$ACON"
EXP="$REPO/experiments/appworld"
PY="${PY:-/z5s/morph/home/sjk/Agent/envs/ci1/bin/python}"
LOGDIR="${ACON_LOGDIR:-/z5s/morph/home/sjk/Agent/datasets/logs}"
AW="${APPWORLD_ROOT:-/z5s/morph/home/sjk/Agent/datasets/appworld-0.1.0}"

export ACON_VLLM_BASE_URL="${ACON_VLLM_BASE_URL:-http://192.168.1.13:18173/v1}"
export ACON_VLLM_API_KEY="${ACON_VLLM_API_KEY:-alice_glm5_xofe72789}"
export APPWORLD_ROOT="$AW"
export APPWORLD_CACHE="${APPWORLD_CACHE:-$AW/cache}"
export TMPDIR=/z5s/morph/home/sjk/tmp
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export ACON_LOCAL_EMBEDDING_URL="${ACON_LOCAL_EMBEDDING_URL:-http://127.0.0.1:9200/v1}"
export ACON_LLM_RETRIES="${ACON_LLM_RETRIES:-6}"
export ACON_LLM_BACKOFF="${ACON_LLM_BACKOFF:-1.5}"

STAMP=""; FILTER=""; SHARDS=""; MAX_ITER=""; REPEATS=""; TASK_IDS=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stamp)    STAMP="$2"; shift 2 ;;
    --methods)  FILTER="$2"; shift 2 ;;
    --shards)   SHARDS="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --repeats)  REPEATS="$2"; shift 2 ;;
    --task-ids) TASK_IDS="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

MAN="$HERE/methods.yml"
[ -f "$MAN" ] || { echo "missing $MAN"; exit 1; }
mkdir -p "$LOGDIR"

set -- $("$PY" - "$MAN" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
print(m["model"], m["split"], m["max_iter"], m["repeats"], m["prompt_file"])
PY
)
MODEL="$1"; SPLIT="$2"; DEF_ITER="$3"; DEF_REP="$4"; AGENT_PROMPT="$5"
[ -n "$MAX_ITER" ] || MAX_ITER="$DEF_ITER"
[ -n "$REPEATS" ] || REPEATS="$DEF_REP"
[ -n "$STAMP" ] || STAMP="$(date '+%Y%m%d_%H%M%S')"
[ -n "$SHARDS" ] || SHARDS=16

rows() {
  "$PY" - "$MAN" "$HERE" <<'PY'
import sys, yaml, pathlib
m = yaml.safe_load(open(sys.argv[1])); here = pathlib.Path(sys.argv[2])
for r in m["rows"]:
    cfg = here / "configs" / r["name"] / f"{r['name']}.yaml"
    print("\t".join([r["kind"], r["name"], str(cfg) if cfg.exists() else "-"]))
PY
}

in_filter() {
  [ -z "$FILTER" ] && return 0
  case ",$FILTER," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

sig() {
  "$PY" - "$REPO/experiments/repro/fingerprint_endpoint.py" "$LOGDIR/suite_fingerprints.jsonl" <<'PY'
import json, pathlib, subprocess, sys
fp, out = sys.argv[1], sys.argv[2]
subprocess.run([sys.executable, fp, "--out", out, "--note", "run_suite"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import os
last = None
if os.path.exists(out):
    for line in open(out):
        line = line.strip()
        if line:
            try: last = json.loads(line)
            except Exception: pass
if not last:
    print("unreachable"); raise SystemExit
wd = last.get("weight_dir")
if wd:
    print(pathlib.Path(wd).name)
else:
    sha = ((last.get("greedy_sha256_16") or ["", ""])[1] or "")[:8]
    label = (last.get("served_model_name") or "unlabeled").replace(".", "")
    print(sha or label)
PY
}

echo "[suite $STAMP] model=$MODEL split=$SPLIT shards=$SHARDS max_iter=$MAX_ITER repeats=$REPEATS"
SIG="$(sig)"
echo "[suite $STAMP] serving backbone signature: $SIG"

# Which dataset the evaluator walks. It evaluates every task the dataset lists, so a
# run that covered only the tasks named by --task-ids has to be evaluated against a
# dataset listing just those. Pointed at the full split it stops at the first task
# that was never run and reports that task's missing dbs, which reads like the row
# failed when in fact the row was never asked to cover it.
EVAL_DATASET="$SPLIT"
if [ -n "$TASK_IDS" ]; then
  EVAL_DATASET="ctxc_${STAMP}"
  mkdir -p "$AW/data/datasets"
  printf '%s\n' $TASK_IDS > "$AW/data/datasets/${EVAL_DATASET}.txt"
  echo "[suite $STAMP] evaluating against dataset ${EVAL_DATASET}"
fi

while IFS=$'\t' read -r kind name cfg; do
  [ -z "$name" ] && continue
  in_filter "$name" || continue
  if [ "$cfg" = "-" ] && [ "$kind" != "none" ]; then
    echo "  skip  $name (no config -- run make_configs.py)"; continue
  fi
  [ "$cfg" = "-" ] && cfg=""

  for k in $(seq 1 "$REPEATS"); do
    REP=""; [ "$REPEATS" -gt 1 ] && REP="__r${k}"
    # The row name is already unique and axis-qualified (hist_*, obs_*); the kind
    # selects which compressor config to hand to run_all.py and is not repeated
    # into the tag, which has to stay a single unambiguous string.
    run_tag="${name}__${MODEL}__sb-${SIG}__${STAMP}${REP}"
    # AppWorld builds the run directory as "<model>_<tag>" and its evaluator is
    # addressed by that directory name. Addressing it by the bare tag is a silent
    # failure: the run succeeds, evaluation cannot find the database and reports
    # "<...>/tasks/<id>/dbs does not exist", and the table then shows a dash for a
    # row that really did execute. `run_id` is the directory, `run_tag` is the tag.
    run_id="${MODEL//\//_}_${run_tag}"
    marker="$EXP/experiments/outputs/$run_id/evaluations/${EVAL_DATASET}.json"
    if [ -f "$marker" ]; then echo "  done  $run_id"; continue; fi
    if [ "$DRY" = 1 ]; then echo "  plan  $run_id  cfg=${cfg:-none}"; continue; fi

    sig_before="$(sig)"
    echo "[$(date '+%F %T')] start $run_id  backbone=$sig_before"

    shard_dir="$EXP/shards"; mkdir -p "$shard_dir"
    if [ ! -f "$shard_dir/${SPLIT}_k${SHARDS}_s0.txt" ]; then
      ( cd "$EXP" && "$PY" - "$SPLIT" "$SHARDS" "$shard_dir" <<'PY'
import sys
from appworld import load_task_ids
split, k, sd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ids = load_task_ids(split)
for i in range(k):
    open(f"{sd}/{split}_k{k}_s{i}.txt", "w").write("\n".join(ids[i::k]) + "\n")
print(f"[shards] {split}: {len(ids)} tasks -> {k} shards")
PY
      )
    fi

    extra=(); [ -n "$cfg" ] && extra=(--co_config_path "$cfg")
    pids=()
    if [ -n "$TASK_IDS" ]; then
      # Single-task path, for smoke-testing the invocation end to end. It issues the
      # same run_all.py command the shard loop does, so what is verified here is the
      # real call and not a rehearsal of it.
      ( cd "$EXP" && "$PY" run_all.py --split "$SPLIT" --tag "$run_tag" \
          --model_name "$MODEL" --max_iter "$MAX_ITER" \
          --prompt_file "$AGENT_PROMPT" \
          --task_ids $TASK_IDS "${extra[@]}" ) \
        > "$LOGDIR/suite_${STAMP}_${name}${REP}_s0.log" 2>&1 &
      pids+=($!)
    else
      for i in $(seq 0 $((SHARDS-1))); do
        ids=$(tr '\n' ' ' < "$shard_dir/${SPLIT}_k${SHARDS}_s${i}.txt")
        ( cd "$EXP" && "$PY" run_all.py --split "$SPLIT" --tag "$run_tag" \
            --model_name "$MODEL" --max_iter "$MAX_ITER" \
            --prompt_file "$AGENT_PROMPT" \
            --task_ids $ids "${extra[@]}" ) \
          > "$LOGDIR/suite_${STAMP}_${name}${REP}_s${i}.log" 2>&1 &
        pids+=($!)
      done
    fi
    fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ "$fail" != 0 ] && echo "  [!] some shards failed for $run_id"

    sig_after="$(sig)"
    verdict="clean"; [ "$sig_before" != "$sig_after" ] && verdict="CONTAMINATED"
    "$PY" - "$EXP" "$run_id" "$SPLIT" "$sig_before" "$sig_after" "$verdict" <<'PY'
import json, os, sys, pathlib
exp, run_id, split, before, after, verdict = sys.argv[1:7]
rec = {"run_id": run_id, "split": split, "sig_before": before,
       "sig_after": after, "verdict": verdict}
for base in (pathlib.Path(exp, "outputs", run_id),
             pathlib.Path(os.environ["APPWORLD_ROOT"], "experiments", "outputs", run_id)):
    if base.parent.exists():
        base.mkdir(parents=True, exist_ok=True)
        (base / "model_signature.json").write_text(json.dumps(rec, indent=2))
print(f"  signature {before} -> {after}  [{verdict}]")
PY

    aw_out="$AW/experiments/outputs/$run_id"
    link="$EXP/experiments/outputs/$run_id"
    if [ -d "$aw_out" ] && [ ! -e "$link" ] && [ ! -L "$link" ]; then
      mkdir -p "$EXP/experiments/outputs"; ln -s "$aw_out" "$link"
    fi
    ( cd "$EXP" && "$PY" -m appworld.cli evaluate "$run_id" "$EVAL_DATASET" ) \
        > "$LOGDIR/suite_${STAMP}_${name}${REP}_eval.log" 2>&1 \
      || echo "  [!] evaluation failed for $run_id"

    "$PY" - "$EXP" "$run_id" "$SPLIT" <<'PY' || true
import json, os, sys
exp, run_id, split = sys.argv[1:4]
td = os.path.join(exp, "outputs", run_id, split)
cap = tot = 0
if os.path.isdir(td):
    for t in os.listdir(td):
        p = os.path.join(td, t, "results.json")
        if not os.path.exists(p): continue
        tot += 1
        try: d = json.load(open(p))
        except Exception: continue
        if (d.get("info") or {}).get("reason") == "max_interactions": cap += 1
if tot:
    print(f"  cap   {cap}/{tot} hit max_iter = {100*cap/tot:.1f}%")
PY
    echo "[$(date '+%F %T')] done  $run_id"
  done
done < <(rows)

echo
echo "[suite $STAMP] finished. table:"
echo "  $PY $HERE/compute_table.py --stamp $STAMP"
