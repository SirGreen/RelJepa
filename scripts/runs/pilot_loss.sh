#!/usr/bin/env bash
# RelJEPA pilot: choose ℓ (smooth_l1 vs cosine) by VALIDATION AUROC on
# user-churn-top100-spending after a short (8k-step) user-root pretraining. Test numbers are
# recorded but not used for the choice. Runs inside the container (scripts/dk.sh).
set -uo pipefail
cd /workspace/third_party/task-vector-estimation
OUT=/workspace/logs/pilot
mkdir -p "$OUT"
FT_CFG=/workspace/configs/rel-hm/ft-user-churn-top100-spending-128.yaml

for loss in smooth_l1 cosine; do
  RUN_LOGDIR_FILE="$OUT/logdir_pt_$loss.txt" python src/train_reljepa.py \
    -c /workspace/configs/rel-hm/user-reljepa-ema-${loss}-pilot.yaml \
    --num_runs 1 --num_workers 3 --gpus '[0]' --name "pilot-pt-ema-$loss" --note "pilot loss choice" \
    > "$OUT/pt_$loss.log" 2>&1
  echo "pretrain $loss exit=$?" >> "$OUT/summary.log"
  ckpt="$(cat "$OUT/logdir_pt_$loss.txt")/run1-final.pth"
  RUN_LOGDIR_FILE="$OUT/logdir_ft_$loss.txt" python src/train_baseline.py -c "$FT_CFG" \
    --lr 0.005 --warmup_steps 1 --num_runs 3 --num_workers 2 --gpus '[0]' --checkpoint "$ckpt" \
    --name "pilot-ft-$loss" --note "pilot finetune top100" > "$OUT/ft_$loss.log" 2>&1
  echo "finetune $loss exit=$?" >> "$OUT/summary.log"
done

RUN_LOGDIR_FILE="$OUT/logdir_ft_scratch.txt" python src/train_baseline.py -c "$FT_CFG" \
  --lr 0.005 --warmup_steps 1 --num_runs 3 --num_workers 2 --gpus '[0]' \
  --name "pilot-ft-scratch" --note "pilot from scratch" > "$OUT/ft_scratch.log" 2>&1
echo "finetune scratch exit=$?" >> "$OUT/summary.log"
echo "PILOT DONE" >> "$OUT/summary.log"
