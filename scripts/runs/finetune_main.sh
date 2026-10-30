#!/usr/bin/env bash
# Main fine-tuning queue for one pretraining root.
# Every method (scratch + 4 pretrained backbones) gets the same protocol:
#   limited : limited-label tasks, LR grid {1e-3, 5e-3} (in-process sweep), 3 seeds, 20 epochs
#   probe   : linear probing (frozen backbone, FREEZE_BACKBONE=1) on the same tasks and grid
#   rec_low : (user root) recommendation, 10% of training users, ID-GNN, 3 seeds
#   full    : full-data task with TVE's fixed LR, 10 epochs x 1000 steps, 3 seeds
#   repro   : (scratch only) full-data task with TVE's original 20-epoch protocol, 2 seeds
# Selection by validation only; test recorded at the selected configuration. Per-example
# predictions are saved for bootstrap intervals; fine-tuned weights are not kept (disk).
# Usage (inside the container): ROOT=user STAGES="limited probe rec_low" bash finetune_main.sh
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
ROOT="${ROOT:-user}"
STAGES="${STAGES:-limited probe rec_low}"
SEEDS="${SEEDS:-3}"
METHODS="${METHODS:-scratch mae tve1 reljepa reljepa-tve}"
PT=/workspace/logs/main
OUT="${OUT:-/workspace/logs/finetune}"   # item_retry.sh uses its own log directory
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
SWEEP=/workspace/configs/sweeps/lr2
export SAVE_FT_CHECKPOINTS=0

FINAL_EPOCH=$([[ $ROOT == user ]] && echo 81 || echo 41)
ckpt() {  # method -> checkpoint path ("" for scratch)
  case "$1" in
    scratch) echo "" ;;
    mae)          echo "$(cat "$PT/logdir_pt-$ROOT-mae.txt")/epoch${FINAL_EPOCH}.pth" ;;
    tve1)         echo "$(cat "$PT/logdir_pt-$ROOT-tve1.txt")/epoch${FINAL_EPOCH}.pth" ;;
    reljepa)      echo "$(cat "$PT/logdir_pt-$ROOT-reljepa.txt")/run1-final.pth" ;;
    reljepa-tve)  echo "$(cat "$PT/logdir_pt-$ROOT-reljepa-tve.txt")/run1-final.pth" ;;
  esac
}

done_already() { grep -q " $1 exit=0" "$OUT/summary.log" 2>/dev/null; }
record() { echo "$(date -u +%FT%TZ) $1 exit=$2" >> "$OUT/summary.log"; }

if [[ $ROOT == user ]]; then
  LIMITED="user-churn-top50-spending:ft-user-churn-top50-spending-128 user-churn-top100-spending:ft-user-churn-top100-spending-128"
  FULL="user-churn"
else
  LIMITED="item-sales-top200-spending:ft-item-sales-top200-spending-128-e20"
  FULL="item-sales"
fi

for stage in $STAGES; do
  for method in $METHODS; do
    c="$(ckpt $method)"
    if [[ -n "$c" && ! -f "$c" ]]; then record "missing-checkpoint-$ROOT-$method" 1; continue; fi
    ck=(); [[ -n "$c" ]] && ck=(--checkpoint "$c")
    case "$stage" in
      limited|probe)
        for entry in $LIMITED; do
          task="${entry%%:*}"; cfg="${entry#*:}"
          name="ft-$stage-$task-$method"
          done_already "$name" && continue
          if ! sweep_ready "$CFG/$cfg.yaml"; then record "$name-config-without-lr-variable" 1; continue; fi
          set_commit
          FREEZE_BACKBONE=$([[ $stage == probe ]] && echo 1 || echo 0) \
          python src/train_baseline.py -c "$CFG/$cfg.yaml" -s "$SWEEP" \
            --num_runs "$SEEDS" --num_workers 3 --gpus '[0]' "${ck[@]}" \
            --name "$name" --note "$stage $method" > "$OUT/$name.log" 2>&1
          rc=$?; sweep_failed "$OUT/$name.log" && rc=97   # sweep runner exits 0 on errors
          record "$name" $rc
        done ;;
      rec_low)
        [[ $ROOT == user ]] || continue
        name="ft-rec-low-$method"
        done_already "$name" && continue
        set_commit
        python src/train_idgnn.py -c "$CFG/ft-user-item-purchase-128.yaml" --user_fraction 0.1 \
          --num_runs "$SEEDS" --num_workers 3 --gpus '[0]' "${ck[@]}" \
          --name "$name" --note "rec low-label $method" > "$OUT/$name.log" 2>&1
        record "$name" $? ;;
      full)
        name="ft-full-$FULL-$method"
        done_already "$name" && continue
        set_commit
        python src/train_baseline.py -c "$CFG/ft-$FULL-128-e10.yaml" \
          --num_runs "$SEEDS" --num_workers 3 --gpus '[0]' "${ck[@]}" \
          --name "$name" --note "full $method" > "$OUT/$name.log" 2>&1
        record "$name" $? ;;
      repro)
        [[ $method == scratch ]] || continue
        name="ft-repro-$FULL-scratch"
        done_already "$name" && continue
        set_commit
        python src/train_baseline.py -c "$CFG/ft-$FULL-128.yaml" \
          --num_runs 2 --num_workers 3 --gpus '[0]' \
          --name "$name" --note "reproduction check" > "$OUT/$name.log" 2>&1
        record "$name" $? ;;
    esac
  done
done
record "QUEUE-DONE-$ROOT-$(echo $STAGES | tr ' ' '+')" 0
exit $?
}
