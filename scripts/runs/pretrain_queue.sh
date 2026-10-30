#!/usr/bin/env bash
# Main pretraining queue: every method uses TVE's released pretraining config
# (81 epochs x 1000 steps on one GPU, batch 512, AdamW lr 1e-3, StepLR x0.1 at epoch 50), the same
# shared subsample of (entity, t) pairs, and its FINAL checkpoint for fine-tuning.
# Runs inside the container (scripts/dk.sh). LOSS selects RelJEPA's ℓ (chosen in the pilot).
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
OUT=/workspace/logs/main
mkdir -p "$OUT"
LOSS="${LOSS:-smooth_l1}"
ROOTS="${ROOTS:-user item}"
export EVAL_NUM_WORKERS=0     # val/test loaders run in the main process

run() {  # name script config
  local name=$1 script=$2 cfg=$3
  if grep -q "^.* $name exit=0 " "$OUT/summary.log" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) $name skipped (already done)" >> "$OUT/summary.log"; return
  fi
  set_commit
  RUN_LOGDIR_FILE="$OUT/logdir_$name.txt" python "src/$script.py" -c "/workspace/configs/rel-hm/$cfg.yaml" \
    --num_runs 1 --num_workers 3 --gpus '[0]' --name "$name" --note "main pretraining" \
    > "$OUT/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit=$? logdir=$(cat "$OUT/logdir_$name.txt" 2>/dev/null)" >> "$OUT/summary.log"
}

for root in $ROOTS; do
  run "pt-$root-tve1" train_tve "$root-tve-1-hop"
  run "pt-$root-mae" train_mae "$root-mae"
  run "pt-$root-reljepa" train_reljepa "$root-reljepa-ema-$LOSS"
  run "pt-$root-reljepa-tve" train_reljepa "$root-reljepa-hybrid-$LOSS"
done
echo "$(date -u +%FT%TZ) QUEUE DONE" >> "$OUT/summary.log"
exit $?
}
