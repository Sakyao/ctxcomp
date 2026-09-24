#!/usr/bin/env bash
# Run the ACON paper's baseline set (paper pipeline Phase 1) on the local backbone
# served by the remote OpenAI-compatible endpoint, and record wall-clock timings.
#
# Usage (in tmux):
#   tmux new -s acon -d 'bash scripts/run_paper_baselines.sh 2>&1 | tee /path/paper_run.log'
#
# Env overrides: SPLIT, SHARDS, MAXITER, MODEL, METHODS, CLEAN,
#                ACON_VLLM_BASE_URL, ACON_VLLM_API_KEY
#
# Methods are run SEQUENTIALLY on purpose: running them concurrently would make
# them contend for the same inference endpoint and corrupt the per-dataset timings.
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

PY=/z5s/morph/home/sjk/Agent/envs/ci1/bin/python
AW=/z5s/morph/home/sjk/Agent/envs/ci1/bin/appworld
export ACON_VLLM_BASE_URL="${ACON_VLLM_BASE_URL:-http://192.168.1.13:18173/v1}"
export ACON_VLLM_API_KEY="${ACON_VLLM_API_KEY:-alice_glm5_xofe72789}"
export APPWORLD_CACHE=/z5s/morph/home/sjk/Agent/datasets/appworld-0.1.0/cache
export TMPDIR=/z5s/morph/home/sjk/tmp

LOGDIR=/z5s/morph/home/sjk/Agent/datasets/logs
TIMINGS="$LOGDIR/timings.tsv"
SPLIT="${SPLIT:-test_normal}"
SHARDS="${SHARDS:-6}"
MAXITER="${MAXITER:-50}"
MODEL="${MODEL:-deepseek-v4.1-flash}"
METHODS="${METHODS:-nocomp fifo prompting}"
CLEAN="${CLEAN:-0}"

mkdir -p "$LOGDIR" shards

if [ ! -f "$TIMINGS" ]; then
  printf "dataset\tmethod\tshards\tstart\tend\twall_sec\ttasks\tclaimed_success\tmodel\n" > "$TIMINGS"
fi

co_path_for () {
  case "$1" in
    nocomp)    echo "" ;;
    fifo)      echo "configs/context_opt/local_fifo_keep5.yaml" ;;
    prompting) echo "configs/context_opt/local_deepseek-v4.1-flash_history.yaml" ;;
    *)         echo "" ;;
  esac
}

# ---- build shards once ----
if [ ! -f "shards/${SPLIT}_s0.txt" ]; then
  "$PY" - "$SPLIT" "$SHARDS" <<'EOF'
import sys
from appworld import load_task_ids
split, k = sys.argv[1], int(sys.argv[2])
ids = load_task_ids(split)
for i in range(k):
    open(f"shards/{split}_s{i}.txt", "w").write("\n".join(ids[i::k]) + "\n")
print(f"[shards] {split}: {len(ids)} tasks -> {k} shards")
EOF
fi

run_method () {
  local tag="$1" co="$2"
  if [ "$CLEAN" = "1" ]; then
    rm -rf "outputs/${MODEL}_${tag}" "experiments/outputs/${MODEL}_${tag}"
  fi
  local t0 t1
  t0=$(date +%s)
  echo "[$(date '+%F %T')] ===== START ${tag} (co=${co:-none}) ====="

  local extra_str=""
  [ -n "$co" ] && extra_str="--co_config_path $co"

  local pids=()
  for i in $(seq 0 $((SHARDS-1))); do
    local ids
    ids=$(tr '\n' ' ' < "shards/${SPLIT}_s${i}.txt")
    local endf="$LOGDIR/paper_${tag}_s${i}.end"
    rm -f "$endf"
    # each shard records its own finish timestamp, so per-shard wall time is exact
    nohup bash -c "$PY run_all.py --split $SPLIT --task_ids $ids --model_name $MODEL --tag $tag --max_iter $MAXITER $extra_str ; date +%s > '$endf'" \
        > "$LOGDIR/paper_${tag}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done

  t1=$(date +%s)
  echo "[$(date '+%F %T')] ===== DONE ${tag} (wall $((t1-t0))s) ====="

  "$PY" - "$MODEL" "$tag" "$SPLIT" "$SHARDS" "$t0" "$t1" "$TIMINGS" <<'EOF'
import glob, json, sys
model, tag, split, shards, t0, t1, timings = sys.argv[1:8]
n = ok = steps = in_tok = 0
for d in glob.glob(f"outputs/{model}_{tag}/{split}/task_*/results.json"):
    r = json.load(open(d)); n += 1
    ok += bool(r["success"])
    steps += r.get("iterations", 0)
    in_tok += r["token_usage"]["total_input_tokens"]
print(f"[summary] {tag}: tasks={n} steps={steps} in_tokens={in_tok} claimed_success={ok}")
with open(timings, "a") as f:
    f.write(f"{split}\t{tag}\t{shards}\t{t0}\t{t1}\t{int(t1)-int(t0)}\t{n}\t{ok}\t{model}\n")
EOF
  local sh="" i ef
  for i in $(seq 0 $((SHARDS-1))); do
    ef="$LOGDIR/paper_${tag}_s${i}.end"
    if [ -f "$ef" ]; then
      sh="$sh s$i=$(( $(cat "$ef") - t0 ))s"
    else
      sh="$sh s$i=NA"
    fi
  done
  echo "[shard-wall] $tag:$sh"
}

for m in $METHODS; do
  run_method "$m" "$(co_path_for "$m")"
done

echo "[$(date '+%F %T')] ===== OFFICIAL EVALUATION (task_goal_completion) ====="
for m in $METHODS; do
  echo "--- $m"
  "$AW" evaluate "${MODEL}_${m}" "$SPLIT" 2>&1 | tail -14
done

echo "[$(date '+%F %T')] ===== ALL DONE ====="
echo "timings -> $TIMINGS"
cat "$TIMINGS"
