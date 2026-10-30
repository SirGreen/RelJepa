#!/usr/bin/env bash
# Full-data linear probes for every method.
# Frozen backbone (FREEZE_BACKBONE=1) + linear head, 3 epochs x 500 steps, LR grid {1e-3, 5e-3}
# selected on validation, 3 seeds, final pretraining checkpoints; scratch = frozen random encoder.
# Test predictions are saved (train_baseline.py) for the paired bootstrap in analyze_results.py.
# Waits for "CHAIN DONE" in logs/chain.log so it never shares the GPU or RAM with the chain.
# Usage (inside the container): bash /workspace/scripts/runs/probes_full.sh
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
PT=/workspace/logs/main
OUT=/workspace/logs/probes_full
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
SWEEP=/workspace/configs/sweeps/lr2
METHODS="${METHODS:-scratch mae tve1 reljepa reljepa-tve}"
export SAVE_FT_CHECKPOINTS=0 FREEZE_BACKBONE=1 EVAL_NUM_WORKERS=2   # as the pilot/ablation probes

done_already() { grep -q " $1 exit=0" "$OUT/summary.log" 2>/dev/null; }
record() { echo "$(date -u +%FT%TZ) $1 exit=$2" >> "$OUT/summary.log"; }
ckpt() {  # root method -> checkpoint path ("" for scratch); same rule as finetune_main.sh
  local final; final=$([[ $1 == user ]] && echo 81 || echo 41)
  case "$2" in
    scratch) echo "" ;;
    mae|tve1) echo "$(cat "$PT/logdir_pt-$1-$2.txt")/epoch${final}.pth" ;;
    reljepa|reljepa-tve) echo "$(cat "$PT/logdir_pt-$1-$2.txt")/run1-final.pth" ;;
  esac
}

until grep -q "CHAIN DONE" /workspace/logs/chain.log 2>/dev/null; do sleep 300; done
record "chain-finished-start-probes" 0

for entry in user:user-churn item:item-sales; do
  root="${entry%%:*}"; task="${entry#*:}"
  for method in $METHODS; do
    name="ft-probe-$task-$method"
    done_already "$name" && continue
    c="$(ckpt "$root" "$method")"
    if [[ -n "$c" && ! -f "$c" ]]; then record "missing-checkpoint-$name" 1; continue; fi
    ck=(); [[ -n "$c" ]] && ck=(--checkpoint "$c")
    set_commit
    python src/train_baseline.py -c "$CFG/probe-$task-128.yaml" -s "$SWEEP" \
      --num_runs 3 --num_workers 3 --gpus '[0]' "${ck[@]}" \
      --name "$name" --note "full-data linear probe $method" > "$OUT/$name.log" 2>&1
    record "$name" $?
  done
done
record "PROBES-FULL-DONE" 0
exit $?
}
