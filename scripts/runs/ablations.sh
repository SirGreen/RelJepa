#!/usr/bin/env bash
# Anti-collapse ablation, user root, 10 x 1000 steps each:
#   ema10  : RelJEPA-EMA (reference at the same short budget)
#   sigreg : shared encoder, no EMA/stop-gradient, (1 - 0.02) L_JEPA + 0.02 SIGReg(z_ctx)
#   noac   : shared encoder, no EMA/stop-gradient, no regulariser (expected to collapse; fail-fast)
#   ema10-noempty   : ema10 without the empty-window BCE term
#   ema10-emptyonly : ema10 with the empty-window BCE term only, no latent regression
# Each is followed by the linear probe on full user-churn used in the pilot (frozen backbone).
# Runs inside the container.
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
OUT=/workspace/logs/ablations
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
export EVAL_NUM_WORKERS=0 SAVE_FT_CHECKPOINTS=0
for v in ema10 sigreg noac ema10-noempty ema10-emptyonly; do
  name="abl-user-reljepa-$v"
  grep -q " $name exit=0" "$OUT/summary.log" 2>/dev/null && continue
  set_commit
  RUN_LOGDIR_FILE="$OUT/logdir_$name.txt" python src/train_reljepa.py -c "$CFG/user-reljepa-$v-cosine.yaml" \
    --num_runs 1 --num_workers 3 --gpus '[0]' --name "$name" --note "anti-collapse ablation" > "$OUT/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) $name exit=$?" >> "$OUT/summary.log"
  ckpt="$(cat "$OUT/logdir_$name.txt")/run1-final.pth"
  set_commit
  EVAL_NUM_WORKERS=2 FREEZE_BACKBONE=1 python src/train_baseline.py -c "$CFG/pilot-probe-user-churn-128.yaml" \
    --num_runs 1 --num_workers 3 --gpus '[0]' --checkpoint "$ckpt" --name "probe-$name" \
    --note "ablation probe" > "$OUT/probe-$name.log" 2>&1
  echo "$(date -u +%FT%TZ) probe-$name exit=$?" >> "$OUT/summary.log"
done
echo "$(date -u +%FT%TZ) ABLATIONS DONE" >> "$OUT/summary.log"
exit $?
}
