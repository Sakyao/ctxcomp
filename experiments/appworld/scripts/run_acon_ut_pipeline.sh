#!/usr/bin/env bash
# ACON method (UT = Utility Maximization) reproduction on the local backbone.
#
# Reproduces ACON's guideline-optimisation loop:
#   0. wait for the train-split contrast runs (nocomp / initial guidelines)
#   1. analysis  : find regressions (baseline OK, compressed FAIL) -> root causes
#   2. update    : write N improved guideline candidates (jinja)
#   3. selection : run every candidate on train, pick the best  -> ACON UT
#   4. final     : run the selected guideline on test_normal (168) and evaluate
#
# Usage:  nohup bash scripts/run_acon_ut_pipeline.sh > LOG 2>&1 &
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"     # experiments/appworld
EXPDIR="$(cd "$HERE/.." && pwd)"             # experiments/appworld -> experiments
REPO="$(cd "$EXPDIR/.." && pwd)"             # experiments -> repo root

PY=/z5s/morph/home/sjk/Agent/envs/ci1/bin/python
AWBIN=/z5s/morph/home/sjk/Agent/envs/ci1/bin
export PATH="$AWBIN:$PATH"
export ACON_VLLM_BASE_URL="${ACON_VLLM_BASE_URL:-http://192.168.1.13:18173/v1}"
export ACON_VLLM_API_KEY="${ACON_VLLM_API_KEY:-alice_glm5_xofe72789}"
export APPWORLD_CACHE=/z5s/morph/home/sjk/Agent/datasets/appworld-0.1.0/cache
export TMPDIR=/z5s/morph/home/sjk/tmp

LOGDIR=/z5s/morph/home/sjk/Agent/datasets/logs
MODEL="${MODEL:-deepseek-v4.1-flash}"
SPLIT_SEL="${SPLIT_SEL:-train}"
SPLIT_FINAL="${SPLIT_FINAL:-test_normal}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
SHARDS="${SHARDS:-6}"
TAG="${TAG:-ut1}"
OPT_OUT="$EXPDIR/prompt_optimizer/outputs/${TAG}"
PHASE1_LOG="$LOGDIR/ut_phase1_train.log"
TIMINGS="$LOGDIR/timings_acon_ut.tsv"

echo "[$(date '+%F %T')] ========== ACON UT PIPELINE START (tag=$TAG) =========="

# ------------------------------------------------------------------ stage 0
echo "[$(date '+%F %T')] stage 0: waiting for train-split contrast runs..."
for _ in $(seq 1 480); do
  if grep -q "ALL DONE" "$PHASE1_LOG" 2>/dev/null; then break; fi
  sleep 60
done
if ! grep -q "ALL DONE" "$PHASE1_LOG" 2>/dev/null; then
  echo "[$(date '+%F %T')] ERROR: phase-1 train runs never finished; aborting."; exit 1
fi
echo "[$(date '+%F %T')] stage 0 done."

# ------------------------------------------------------------------ stage 1
echo "[$(date '+%F %T')] stage 1: analysis (regressions on $SPLIT_SEL)"
cd "$EXPDIR/prompt_optimizer"
T0=$(date +%s)
"$PY" unified_update_history_prompt.py \
  --phase analysis --benchmark appworld --task-split "$SPLIT_SEL" \
  --baseline-run "${MODEL}_nocomp" --optimized-run "${MODEL}_prompting" \
  --analysis-model "$MODEL" --output-dir "$OPT_OUT" --log-level INFO
T1=$(date +%s)
echo "[$(date '+%F %T')] stage 1 done in $((T1-T0))s"
[ -f "$OPT_OUT/aggregated_history_regressions.json" ] || { echo "ERROR: no aggregated regressions"; exit 1; }

# ------------------------------------------------------------------ stage 2
echo "[$(date '+%F %T')] stage 2: update -> $NUM_PROMPTS guideline candidates"
T0=$(date +%s)
"$PY" unified_update_history_prompt.py \
  --phase update --benchmark appworld --task-split "$SPLIT_SEL" \
  --baseline-run "${MODEL}_nocomp" --optimized-run "${MODEL}_prompting" \
  --update-model "$MODEL" \
  --aggregated-history-regressions "$OPT_OUT/aggregated_history_regressions.json" \
  --output-dir "$OPT_OUT" --num-prompts "$NUM_PROMPTS" \
  --base-prompt-template ../appworld/prompts/context_opt/prompt_history_v2.jinja \
  --log-level INFO
T1=$(date +%s)
echo "[$(date '+%F %T')] stage 2 done in $((T1-T0))s"
ls -1 "$OPT_OUT/optimized_prompts"/

# ------------------------------------------------------------------ stage 3
echo "[$(date '+%F %T')] stage 3: evaluate candidates on $SPLIT_SEL and select best"
cd "$HERE"
T0=$(date +%s)
"$PY" run_ctxopt_pipeline.py \
  --ctxopt-type history \
  --prompts-dir "$OPT_OUT/optimized_prompts" \
  --model-name "$MODEL" --main-model-name "$MODEL" \
  --split "$SPLIT_SEL" --tag "$TAG" --opt-version 1
T1=$(date +%s)
echo "[$(date '+%F %T')] stage 3 done in $((T1-T0))s"

BEST=$(ls -1 configs/context_opt/*/best_*.yaml 2>/dev/null | tail -1)
if [ -z "$BEST" ]; then BEST=$(find configs/context_opt -name 'best_*.yaml' 2>/dev/null | tail -1); fi
echo "[$(date '+%F %T')] selected best config: ${BEST:-<none found>}"

# ------------------------------------------------------------------ stage 4
if [ -z "$BEST" ]; then
  echo "[$(date '+%F %T')] ERROR: no best_*.yaml produced; stopping before final run."
  exit 1
fi
echo "[$(date '+%F %T')] stage 4: final run of ACON UT on $SPLIT_FINAL"
if [ ! -f "shards/${SPLIT_FINAL}_s0.txt" ]; then
  "$PY" - "$SPLIT_FINAL" "$SHARDS" <<'EOF'
import sys
from appworld import load_task_ids
split, k = sys.argv[1], int(sys.argv[2])
ids = load_task_ids(split)
for i in range(k):
    open(f"shards/{split}_s{i}.txt", "w").write("\n".join(ids[i::k]) + "\n")
print(f"[shards] {split}: {len(ids)} tasks -> {k} shards")
EOF
fi
T0=$(date +%s)
pids=()
for i in $(seq 0 $((SHARDS-1))); do
  ids=$(tr '\n' ' ' < "shards/${SPLIT_FINAL}_s${i}.txt")
  endf="$LOGDIR/acon_ut_final_s${i}.end"; rm -f "$endf"
  nohup bash -c "$PY run_all.py --split $SPLIT_FINAL --task_ids $ids --model_name $MODEL --tag ${TAG}_final --max_iter 50 --co_config_path $BEST ; date +%s > '$endf'" \
      > "$LOGDIR/acon_ut_final_s${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
T1=$(date +%s)
echo "[$(date '+%F %T')] stage 4 runs done in $((T1-T0))s"

echo "[$(date '+%F %T')] stage 5: official evaluation of ACON UT on $SPLIT_FINAL"
"$AWBIN" evaluate "${MODEL}_${TAG}_final" "$SPLIT_FINAL" 2>&1 | tail -14

if [ ! -f "$TIMINGS" ]; then
  printf "dataset\tmethod\tstage\tstart\tend\twall_sec\tmodel\n" > "$TIMINGS"
fi
printf "%s\tACON-UT(final)\tstage4\t%s\t%s\t%s\t%s\n" "$SPLIT_FINAL" "$T0" "$T1" "$((T1-T0))" "$MODEL" >> "$TIMINGS"
echo "[$(date '+%F %T')] ========== ACON UT PIPELINE DONE =========="
